use crate::lit::Lit;

/// Offset into `ClauseArena.data` where this clause's header begins. Treated
/// opaquely by callers — it's not the clause's ordinal position but a byte-
/// ish word index into a packed arena.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ClauseRef(pub u32);

/// All clauses live in one big `Vec<u32>`. Per clause:
///
/// ```text
/// word 0        : flags  (bit 0 = learned, bit 1 = deleted)
/// word 1        : LBD
/// word 2..=3    : activity (f64, low u32 first)
/// word 4        : length (number of literals)
/// word 5..5+len : literals (each Lit stored as its .0)
/// ```
///
/// This eliminates the per-clause heap allocation of `Vec<Lit>` and packs
/// every live clause into contiguous memory, which is a big cache win for
/// unit propagation (the hottest loop in the solver).
pub struct ClauseArena {
    data: Vec<u32>,
}

const HDR: usize = 5;
const FLAG_LEARNED: u32 = 1;
const FLAG_DELETED: u32 = 2;

impl ClauseArena {
    pub fn new() -> Self {
        ClauseArena { data: Vec::new() }
    }

    /// Reserve `extra_words` of raw storage. Each clause needs `5 + len` words.
    pub fn reserve(&mut self, extra_words: usize) {
        self.data.reserve(extra_words);
    }

    /// Total words in the arena, including live + deleted clauses and headers.
    pub fn word_size(&self) -> usize {
        self.data.len()
    }

    /// Append a new clause. Returns its ref (the offset of its header word).
    pub fn alloc(&mut self, lits: &[Lit], learned: bool) -> ClauseRef {
        let cref = ClauseRef(self.data.len() as u32);
        self.data.push(if learned { FLAG_LEARNED } else { 0 });
        self.data.push(0); // lbd
        self.data.push(0); // activity lo
        self.data.push(0); // activity hi
        self.data.push(lits.len() as u32);
        for l in lits {
            self.data.push(l.0);
        }
        cref
    }

    #[inline]
    pub fn learned(&self, c: ClauseRef) -> bool {
        (self.data[c.0 as usize] & FLAG_LEARNED) != 0
    }

    #[inline]
    pub fn deleted(&self, c: ClauseRef) -> bool {
        (self.data[c.0 as usize] & FLAG_DELETED) != 0
    }

    #[inline]
    pub fn mark_deleted(&mut self, c: ClauseRef) {
        self.data[c.0 as usize] |= FLAG_DELETED;
    }

    #[inline]
    pub fn lbd(&self, c: ClauseRef) -> u32 {
        self.data[c.0 as usize + 1]
    }

    #[inline]
    pub fn set_lbd(&mut self, c: ClauseRef, lbd: u32) {
        self.data[c.0 as usize + 1] = lbd;
    }

    #[inline]
    pub fn activity(&self, c: ClauseRef) -> f64 {
        let h = c.0 as usize;
        let lo = self.data[h + 2] as u64;
        let hi = self.data[h + 3] as u64;
        f64::from_bits(lo | (hi << 32))
    }

    #[inline]
    pub fn set_activity(&mut self, c: ClauseRef, a: f64) {
        let h = c.0 as usize;
        let bits = a.to_bits();
        self.data[h + 2] = bits as u32;
        self.data[h + 3] = (bits >> 32) as u32;
    }

    // The next four accessors are on the unit-propagation hot path (called
    // per clause visit) and use `get_unchecked`: a live `ClauseRef` always
    // has its header + `len` body words in bounds by construction (see
    // `alloc`), and callers pass `i < len`. Safe indexing here compiled to
    // an `Index → Deref → &[u32] → bounds-check` sequence that showed up as
    // `<Vec as Deref>::deref` in propagate profiles; dropping the bounds
    // check also lets the compiler keep `data`'s base pointer in a register
    // across the loop. Pure plumbing — identical behaviour, no propagation-
    // order change. `debug_assert`s keep the invariant checked in dev/test.

    // These read the arena through `data.as_ptr()` + offset rather than
    // `data.get_unchecked(idx)`. Both are unchecked, but `get_unchecked` is
    // a *slice* method, so it first calls `<Vec as Deref>::deref` to
    // materialize a `&[u32]` — which showed up as a real `deref` frame in
    // propagate profiles. `Vec::as_ptr` is an *inherent* method that hands
    // back the buffer pointer with no deref. Same codegen intent, one fewer
    // call in the hot path. Indices are in bounds for any live clause by
    // construction (see `alloc`); `debug_assert`s keep that checked in dev.

    #[inline]
    pub fn len(&self, c: ClauseRef) -> usize {
        let idx = c.0 as usize + 4;
        debug_assert!(idx < self.data.len(), "clause len read out of bounds");
        // SAFETY: a live clause header occupies words [c.0 .. c.0 + HDR).
        unsafe { *self.data.as_ptr().add(idx) as usize }
    }

    #[inline]
    pub fn get_lit(&self, c: ClauseRef, i: usize) -> Lit {
        let idx = c.0 as usize + HDR + i;
        debug_assert!(idx < self.data.len(), "clause lit read out of bounds");
        // SAFETY: i < len ⇒ header + i is within the clause body.
        Lit(unsafe { *self.data.as_ptr().add(idx) })
    }

    /// Borrow the clause's literals as a slice. Zero-copy — the underlying
    /// storage is a `[u32]` but `Lit` is `#[repr(transparent)]` around `u32`.
    #[inline]
    pub fn lits(&self, c: ClauseRef) -> &[Lit] {
        let h = c.0 as usize;
        debug_assert!(h + 4 < self.data.len(), "clause header out of bounds");
        let base = self.data.as_ptr();
        // SAFETY: header word `h+4` is in bounds for any live clause.
        let len = unsafe { *base.add(h + 4) as usize };
        let start = h + HDR;
        debug_assert!(start + len <= self.data.len(), "clause body out of bounds");
        // SAFETY: Lit is #[repr(transparent)] around u32, so a &[u32] can be
        // reinterpreted as &[Lit] with identical layout and alignment; the
        // body words [start .. start+len) are in bounds by construction.
        unsafe { std::slice::from_raw_parts(base.add(start) as *const Lit, len) }
    }

    #[inline]
    pub fn swap_lits(&mut self, c: ClauseRef, a: usize, b: usize) {
        let base = c.0 as usize + HDR;
        let (ia, ib) = (base + a, base + b);
        debug_assert!(ia < self.data.len() && ib < self.data.len(), "swap out of bounds");
        // SAFETY: both a and b are < len, so ia/ib are within the clause body.
        unsafe {
            let p = self.data.as_mut_ptr();
            std::ptr::swap(p.add(ia), p.add(ib));
        }
    }

    /// Walk every non-deleted clause and pass its literals (as a slice) to
    /// the closure. Used by preprocessing passes — preprocessing is one-shot
    /// and doesn't need to be on the propagate hot path, so this just walks
    /// the arena linearly and skips deleted entries by reading their length.
    pub fn for_each_clause<F: FnMut(&[Lit])>(&self, mut f: F) {
        let mut pos = 0usize;
        while pos + HDR <= self.data.len() {
            let flags = self.data[pos];
            let len = self.data[pos + 4] as usize;
            let body_start = pos + HDR;
            let end = body_start + len;
            if end > self.data.len() {
                break;
            }
            if (flags & FLAG_DELETED) == 0 {
                // SAFETY: Lit is #[repr(transparent)] around u32, and `end`
                // is bounded by self.data.len(), checked above.
                let lits = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr().add(body_start) as *const Lit,
                        len,
                    )
                };
                f(lits);
            }
            pos = end;
        }
    }

    /// Hint the CPU to start loading this clause's header into L1.
    /// Caller is expected to read the clause body soon afterwards; the
    /// prefetch overlaps the cache fill with intervening work. A no-op on
    /// architectures we don't have an intrinsic for.
    #[inline(always)]
    pub fn prefetch(&self, c: ClauseRef) {
        let idx = c.0 as usize;
        if idx >= self.data.len() {
            return;
        }
        // SAFETY: bounded above; the prefetch instructions are pure hints
        // and tolerate any pointer that doesn't cause an access violation.
        unsafe {
            let ptr = self.data.as_ptr().add(idx);
            #[cfg(target_arch = "x86_64")]
            {
                core::arch::x86_64::_mm_prefetch(
                    ptr as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                core::arch::asm!(
                    "prfm pldl1keep, [{p}]",
                    p = in(reg) ptr,
                    options(readonly, nostack, preserves_flags),
                );
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let _ = ptr;
            }
        }
    }
}

impl Default for ClauseArena {
    fn default() -> Self {
        Self::new()
    }
}
