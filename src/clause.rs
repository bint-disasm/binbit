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

    #[inline]
    pub fn len(&self, c: ClauseRef) -> usize {
        self.data[c.0 as usize + 4] as usize
    }

    #[inline]
    pub fn get_lit(&self, c: ClauseRef, i: usize) -> Lit {
        Lit(self.data[c.0 as usize + HDR + i])
    }

    /// Borrow the clause's literals as a slice. Zero-copy — the underlying
    /// storage is a `[u32]` but `Lit` is `#[repr(transparent)]` around `u32`.
    #[inline]
    pub fn lits(&self, c: ClauseRef) -> &[Lit] {
        let h = c.0 as usize;
        let len = self.data[h + 4] as usize;
        let start = h + HDR;
        // SAFETY: Lit is #[repr(transparent)] around u32, so a &[u32] can be
        // reinterpreted as &[Lit] with identical layout and alignment.
        unsafe {
            std::slice::from_raw_parts(self.data.as_ptr().add(start) as *const Lit, len)
        }
    }

    #[inline]
    pub fn swap_lits(&mut self, c: ClauseRef, a: usize, b: usize) {
        let base = c.0 as usize + HDR;
        self.data.swap(base + a, base + b);
    }
}

impl Default for ClauseArena {
    fn default() -> Self {
        Self::new()
    }
}
