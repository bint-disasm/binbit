use std::collections::VecDeque;

use crate::clause::{ClauseArena, ClauseRef};
use crate::lit::{LBool, Lit, Var};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SolveResult {
    Sat,
    Unsat,
}

/// What caused a literal to be assigned.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Reason {
    Decision,
    // A clause from the arena (size >= 3 or original binaries represented as
    // Clause objects; in this solver binaries are stored inline — see below).
    Clause(ClauseRef),
    // The implicit binary clause (propagated_lit, other). The propagated
    // literal sits at the head of its trail entry; `other` is what we resolve
    // against during conflict analysis.
    Binary(Lit),
}

/// A unified watcher entry. Used for both binary and long clauses, kept in a
/// single per-literal list. The high bit of `cref` is the "binary" flag:
///
/// - **Long clause** (flag clear): `cref` is the arena offset and `blocker`
///   is MiniSat's standard hint — if it's true we know the clause is
///   satisfied without touching its body.
/// - **Binary clause** (flag set): `cref`'s low bits are unused and
///   `blocker` is the *partner* literal `q` in the clause `(false_lit ∨ q)`.
///   For a binary entry `blocker` doubles as both the "is the clause
///   satisfied" hint and the literal we may need to force / conflict on,
///   so a single lookup of `value_of(blocker)` resolves the whole entry.
///
/// Packing binary entries into the same list saves an entire watch-list
/// traversal (and its `mem::take`/restore round-trip) per propagated
/// literal — MiniSat 2.2+'s standard layout.
#[derive(Copy, Clone, Debug)]
struct Watcher {
    cref: ClauseRef,
    blocker: Lit,
}

const WATCH_BINARY_FLAG: u32 = 1u32 << 31;

impl Watcher {
    #[inline]
    fn long(cref: ClauseRef, blocker: Lit) -> Self {
        debug_assert!(cref.0 & WATCH_BINARY_FLAG == 0, "clause ref overflows watcher flag bit");
        Watcher { cref, blocker }
    }
    #[inline]
    fn binary(partner: Lit) -> Self {
        Watcher {
            cref: ClauseRef(WATCH_BINARY_FLAG),
            blocker: partner,
        }
    }
    #[inline]
    fn is_binary(self) -> bool {
        self.cref.0 & WATCH_BINARY_FLAG != 0
    }
    #[inline]
    fn long_cref(self) -> ClauseRef {
        debug_assert!(!self.is_binary());
        self.cref
    }
}

/// Glucose-style adaptive restart policy. Replaces fixed Luby scheduling with
/// one driven by the quality of recently learned clauses.
///
/// Core idea: track a short sliding window of LBD scores and compare its
/// average against the long-run average. When recent clauses are noticeably
/// "worse" (higher LBD) than average, the current search tree is going
/// nowhere — restart. Separately, if the trail is growing beyond its running
/// average (solver is likely close to a model), block the restart.
///
/// On industrial/BMC-style formulas this usually beats Luby by a meaningful
/// margin; on pure combinatorial UNSAT it's roughly comparable.
struct RestartState {
    // Short window over recent LBDs (last `lbd_window_cap` conflicts).
    lbd_window: VecDeque<u32>,
    lbd_window_sum: u64,
    // Long-run total across the entire solve.
    lbd_global_sum: u64,
    lbd_global_count: u64,
    // Short window over trail sizes at conflict time. Used only by the
    // "block restart" heuristic.
    trail_window: VecDeque<u64>,
    trail_window_sum: u64,
    // Number of conflicts since the last restart (or start of solve).
    conflicts_since_restart: u64,
}

impl RestartState {
    const LBD_WINDOW_CAP: usize = 50;
    const TRAIL_WINDOW_CAP: usize = 5000;
    // Trigger restart when short-window LBD avg × K exceeds long-run avg.
    // Lower K = more aggressive restarts.
    const K: f64 = 0.8;
    // Block restart if current trail > R × long-run trail avg.
    const R: f64 = 1.4;
    // Don't restart more often than this many conflicts apart.
    const MIN_CONFLICTS: u64 = 50;

    fn new() -> Self {
        RestartState {
            lbd_window: VecDeque::with_capacity(Self::LBD_WINDOW_CAP),
            lbd_window_sum: 0,
            lbd_global_sum: 0,
            lbd_global_count: 0,
            trail_window: VecDeque::with_capacity(Self::TRAIL_WINDOW_CAP),
            trail_window_sum: 0,
            conflicts_since_restart: 0,
        }
    }

    fn on_new_solve(&mut self) {
        // Keep the running long-term averages across solve calls so
        // incremental queries benefit from accumulated history. Only
        // reset the short window and the per-call counter.
        self.lbd_window.clear();
        self.lbd_window_sum = 0;
        self.conflicts_since_restart = 0;
    }

    fn on_conflict(&mut self, lbd: u32, trail_size: u64) {
        self.conflicts_since_restart += 1;

        if self.lbd_window.len() == Self::LBD_WINDOW_CAP {
            let old = self.lbd_window.pop_front().unwrap();
            self.lbd_window_sum -= old as u64;
        }
        self.lbd_window.push_back(lbd);
        self.lbd_window_sum += lbd as u64;

        self.lbd_global_sum += lbd as u64;
        self.lbd_global_count += 1;

        if self.trail_window.len() == Self::TRAIL_WINDOW_CAP {
            let old = self.trail_window.pop_front().unwrap();
            self.trail_window_sum -= old;
        }
        self.trail_window.push_back(trail_size);
        self.trail_window_sum += trail_size;
    }

    /// True if the solver should restart now.
    fn should_restart(&self, current_trail_size: u64) -> bool {
        if self.lbd_window.len() < Self::LBD_WINDOW_CAP {
            return false;
        }
        if self.conflicts_since_restart < Self::MIN_CONFLICTS {
            return false;
        }

        let short_avg = self.lbd_window_sum as f64 / self.lbd_window.len() as f64;
        let long_avg = self.lbd_global_sum as f64 / self.lbd_global_count.max(1) as f64;

        if short_avg * Self::K <= long_avg {
            return false; // recent clauses are good enough — keep going
        }

        // Block if the trail is notably longer than average — probably close
        // to finding a model, don't throw away the progress.
        if self.trail_window.len() >= Self::TRAIL_WINDOW_CAP {
            let avg_trail =
                self.trail_window_sum as f64 / self.trail_window.len() as f64;
            if (current_trail_size as f64) > avg_trail * Self::R {
                return false;
            }
        }

        true
    }

    fn on_restart(&mut self) {
        self.conflicts_since_restart = 0;
        self.lbd_window.clear();
        self.lbd_window_sum = 0;
    }
}

/// Indexed binary max-heap ordered by variable activity. `pos[v] == -1` means
/// variable v is not currently in the heap.
struct OrderHeap {
    heap: Vec<u32>,
    pos: Vec<i32>,
}

impl OrderHeap {
    fn new() -> Self {
        OrderHeap {
            heap: Vec::new(),
            pos: Vec::new(),
        }
    }

    fn reserve(&mut self, n: usize) {
        self.heap.reserve(n);
        self.pos.reserve(n);
    }

    fn new_var(&mut self) {
        self.pos.push(-1);
    }

    #[inline]
    fn contains(&self, v: u32) -> bool {
        let vi = v as usize;
        vi < self.pos.len() && self.pos[vi] >= 0
    }

    fn insert(&mut self, v: u32, activity: &[f64]) {
        if self.contains(v) {
            return;
        }
        let idx = self.heap.len();
        self.heap.push(v);
        self.pos[v as usize] = idx as i32;
        self.sift_up(idx, activity);
    }

    /// Called when v's activity just increased. If v is in the heap, move it up.
    fn decrease(&mut self, v: u32, activity: &[f64]) {
        if !self.contains(v) {
            return;
        }
        let idx = self.pos[v as usize] as usize;
        self.sift_up(idx, activity);
    }

    fn pop(&mut self, activity: &[f64]) -> Option<u32> {
        if self.heap.is_empty() {
            return None;
        }
        let top = self.heap[0];
        self.pos[top as usize] = -1;
        if self.heap.len() == 1 {
            self.heap.pop();
            return Some(top);
        }
        let last = self.heap.pop().unwrap();
        self.heap[0] = last;
        self.pos[last as usize] = 0;
        self.sift_down(0, activity);
        Some(top)
    }

    fn sift_up(&mut self, mut i: usize, activity: &[f64]) {
        let x = self.heap[i];
        let xa = activity[x as usize];
        while i > 0 {
            let parent = (i - 1) / 2;
            let pv = self.heap[parent];
            if activity[pv as usize] < xa {
                self.heap[i] = pv;
                self.pos[pv as usize] = i as i32;
                i = parent;
            } else {
                break;
            }
        }
        self.heap[i] = x;
        self.pos[x as usize] = i as i32;
    }

    fn sift_down(&mut self, mut i: usize, activity: &[f64]) {
        let len = self.heap.len();
        let x = self.heap[i];
        let xa = activity[x as usize];
        loop {
            let left = 2 * i + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            let child = if right < len
                && activity[self.heap[right] as usize] > activity[self.heap[left] as usize]
            {
                right
            } else {
                left
            };
            let cv = self.heap[child];
            if activity[cv as usize] > xa {
                self.heap[i] = cv;
                self.pos[cv as usize] = i as i32;
                i = child;
            } else {
                break;
            }
        }
        self.heap[i] = x;
        self.pos[x as usize] = i as i32;
    }
}

pub struct Solver {
    // === Hot section — touched on every propagate() iteration. ===
    // Field order here is deliberately packed for cache locality: the inner
    // propagation loop reads clauses, lit_value, assigns, level, reason,
    // trail, qhead, and watches together.
    clauses: ClauseArena,
    // Per-literal assignment table — `lit_value[lit.0]` returns the truth
    // value of `lit` directly, with no negate branch. When var v is set to
    // True, lit_value[2v]=True and lit_value[2v+1]=False; unassigning resets
    // both to Undef. This is the *only* assignment store — a per-variable
    // `assigns` view used to live alongside but was redundant: a variable's
    // value is exactly `lit_value[2v]` (the positive literal's value).
    lit_value: Vec<LBool>,
    // Per-variable state (indexed by Var.0).
    level: Vec<i32>,
    reason: Vec<Reason>,
    // Trail: assignments in propagation order.
    trail: Vec<Lit>,
    qhead: usize,
    trail_lim: Vec<usize>,
    // Unified watch lists: one list per literal, holding both binary and
    // long-clause watchers (see `Watcher` for the layout). Replaces the
    // older split `watches` / `bin_watches` pair.
    watches: Vec<Vec<Watcher>>,

    // === Warm — touched during analyze() and branch picking. ===
    activity: Vec<f64>,
    polarity: Vec<bool>, // phase saving
    order_heap: OrderHeap,
    seen: Vec<bool>,
    lbd_stamp: Vec<u64>,

    // Refs to still-live learned clauses (ones that haven't been deleted by
    // reduce_db). Kept separately from the arena so we can sort + prune
    // without scanning the whole arena.
    learnts: Vec<ClauseRef>,

    // === Cold — restart policy, DB reduction tuning, VSIDS scaling. ===
    // VSIDS.
    var_inc: f64,
    var_decay: f64,
    // Clause activity + decay for DB reduction scoring.
    cla_inc: f64,
    cla_decay: f64,
    // When num_learnts exceeds this, reduce_db runs. Grows each time.
    max_learnts: f64,
    // Glucose-style adaptive restart policy.
    restart: RestartState,

    // === Incremental solving state. ===
    // Literals that must hold for the current solve. Each becomes a pseudo-
    // decision at its own level (level 1 = assumptions[0], level 2 =
    // assumptions[1], etc.). Real decisions from VSIDS stack on top.
    assumptions: Vec<Lit>,
    // After an UNSAT result under assumptions, holds a subset of those
    // assumption literals that jointly cause the contradiction. The first
    // element is the assumption whose negation fired the conflict.
    conflict_core: Vec<Lit>,

    // === Scratch buffers used only inside analyze() / lit_redundant(). ===
    // Every variable whose seen[] was set during the current conflict
    // analysis (including minimization). Walked once at the end to reset
    // seen[] to false in bulk.
    analyze_toclear: Vec<Lit>,
    // DFS stack reused by lit_redundant during clause minimization.
    analyze_stack: Vec<Lit>,
    lbd_counter: u64,

    // === Stats. ===
    // Total number of clauses ever allocated — just for stats display.
    num_clauses_total: u64,
    pub stats_conflicts: u64,
    pub stats_decisions: u64,
    pub stats_propagations: u64,
    pub stats_restarts: u64,
    pub stats_learned: u64,
    pub stats_deleted: u64,
    pub stats_reductions: u64,
    pub stats_min_removed: u64,

    // Sticky "this formula is already UNSAT" flag. Set whenever `add_clause`
    // detects trivial unsatisfiability (empty clause, or a unit whose value
    // is already forced to the opposite polarity). Once set, every future
    // `solve*` call returns UNSAT without actually searching. Without this
    // we'd silently ignore clauses that filter down to "empty" and report
    // spurious SAT.
    dead: bool,
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            clauses: ClauseArena::new(),
            lit_value: Vec::new(),
            level: Vec::new(),
            reason: Vec::new(),
            trail: Vec::new(),
            qhead: 0,
            trail_lim: Vec::new(),
            watches: Vec::new(),
            activity: Vec::new(),
            polarity: Vec::new(),
            order_heap: OrderHeap::new(),
            seen: Vec::new(),
            lbd_stamp: Vec::new(),
            learnts: Vec::new(),
            var_inc: 1.0,
            var_decay: 0.95,
            cla_inc: 1.0,
            cla_decay: 0.999,
            max_learnts: 0.0,
            restart: RestartState::new(),
            assumptions: Vec::new(),
            conflict_core: Vec::new(),
            analyze_toclear: Vec::new(),
            analyze_stack: Vec::new(),
            lbd_counter: 0,
            num_clauses_total: 0,
            stats_conflicts: 0,
            stats_decisions: 0,
            stats_propagations: 0,
            stats_restarts: 0,
            stats_learned: 0,
            stats_deleted: 0,
            stats_reductions: 0,
            stats_min_removed: 0,
            dead: false,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.lit_value.len() >> 1
    }
    pub fn num_clauses(&self) -> usize {
        self.num_clauses_total as usize
    }
    pub fn num_learnts(&self) -> usize {
        self.learnts.len()
    }

    /// Pre-allocate capacity for a problem of roughly `num_vars` variables
    /// and `num_clauses` input clauses. Avoids the geometric regrowth that
    /// would otherwise happen as variables and clauses stream in.
    pub fn reserve(&mut self, num_vars: usize, num_clauses: usize) {
        // Per-variable arrays.
        self.level.reserve(num_vars);
        self.reason.reserve(num_vars);
        self.activity.reserve(num_vars);
        self.polarity.reserve(num_vars);
        self.seen.reserve(num_vars);
        self.lbd_stamp.reserve(num_vars);
        self.order_heap.reserve(num_vars);

        // Per-literal arrays — two entries per variable.
        self.lit_value.reserve(2 * num_vars);
        self.watches.reserve(2 * num_vars);

        // Trail holds at most num_vars entries; trail_lim at most num_vars.
        self.trail.reserve(num_vars);
        self.trail_lim.reserve(num_vars);

        // Conflict-analysis scratch buffers — bounded by trail depth.
        self.analyze_toclear.reserve(num_vars);
        self.analyze_stack.reserve(num_vars);

        // Clause arena: reserve raw word capacity. Each clause takes 5 header
        // words plus its literals (~3 for random 3-SAT, more on average for
        // learned clauses). Conservative estimate: ~10 words per clause.
        self.clauses.reserve(num_clauses * 10);
        self.learnts.reserve(num_clauses / 4);
    }

    pub fn new_var(&mut self) -> Var {
        let v = Var((self.lit_value.len() >> 1) as u32);
        // Per-literal value table — two entries per variable, both Undef.
        self.lit_value.push(LBool::Undef);
        self.lit_value.push(LBool::Undef);
        self.level.push(-1);
        self.reason.push(Reason::Decision);
        self.activity.push(0.0);
        self.polarity.push(false);
        self.seen.push(false);
        self.lbd_stamp.push(0);
        // One unified watch list per literal — two per variable. Prime each
        // with a small capacity; most literals accumulate several watchers
        // quickly, and reallocating from 0 → 1 → 2 → 4 adds up.
        self.watches.push(Vec::with_capacity(4));
        self.watches.push(Vec::with_capacity(4));
        self.order_heap.new_var();
        self.order_heap.insert(v.0, &self.activity);
        v
    }

    pub fn ensure_var(&mut self, v: Var) {
        while self.num_vars() <= v.idx() {
            self.new_var();
        }
    }

    #[inline]
    pub fn value_of_var(&self, v: Var) -> LBool {
        // A variable's value is exactly the value of its positive literal,
        // which lives at `lit_value[2v]`.
        self.lit_value[(v.0 as usize) << 1]
    }

    #[inline]
    pub fn value_of(&self, l: Lit) -> LBool {
        // Branch-free: the per-literal table already encodes negation, so
        // both `lit` and `!lit` are direct loads. Maintained in lockstep
        // with `assigns` at every assign / unassign site.
        self.lit_value[l.0 as usize]
    }

    #[inline]
    pub fn decision_level(&self) -> i32 {
        self.trail_lim.len() as i32
    }

    /// Add an input clause. Safe to call between solve invocations — it
    /// automatically cancels any lingering decision stack first. Returns
    /// false iff the formula is trivially unsatisfiable as a result. When
    /// that happens, the unsat condition is recorded persistently so that
    /// later `solve*` calls also return UNSAT.
    pub fn add_clause(&mut self, lits: Vec<Lit>) -> bool {
        if self.dead {
            return false;
        }
        let ok = self.add_clause_inner(lits);
        if !ok {
            self.dead = true;
        }
        ok
    }

    fn add_clause_inner(&mut self, mut lits: Vec<Lit>) -> bool {
        self.cancel_until(0);

        // Sort + filter: drop false-at-level-0 lits, detect taut/satisfied.
        lits.sort_by_key(|l| l.0);
        let mut j = 0usize;
        let mut i = 0usize;
        while i < lits.len() {
            let li = lits[i];
            if i + 1 < lits.len() && li.var() == lits[i + 1].var() {
                if li.is_negated() != lits[i + 1].is_negated() {
                    return true; // tautology
                }
                i += 1;
                continue;
            }
            match self.value_of(li) {
                LBool::True => return true,
                LBool::False => {
                    i += 1;
                    continue;
                }
                LBool::Undef => {
                    lits[j] = li;
                    j += 1;
                    i += 1;
                }
            }
        }
        lits.truncate(j);

        match lits.len() {
            0 => false,
            1 => self.enqueue(lits[0], Reason::Decision), // unit forced at level 0
            2 => {
                // Binary: skip the arena, store inline as binary watchers
                // in the same unified list used for long clauses.
                self.watches[lits[0].idx()].push(Watcher::binary(lits[1]));
                self.watches[lits[1].idx()].push(Watcher::binary(lits[0]));
                true
            }
            _ => {
                let w0 = lits[0];
                let w1 = lits[1];
                let cref = self.clauses.alloc(&lits, false);
                self.num_clauses_total += 1;
                self.watches[w0.idx()].push(Watcher::long(cref, w1));
                self.watches[w1.idx()].push(Watcher::long(cref, w0));
                true
            }
        }
    }

    /// Assign `lit` to true at the current decision level with the given reason.
    #[inline]
    fn enqueue(&mut self, lit: Lit, reason: Reason) -> bool {
        match self.value_of(lit) {
            LBool::True => true,
            LBool::False => false,
            LBool::Undef => {
                let vi = lit.var_idx();
                self.lit_value[lit.0 as usize] = LBool::True;
                self.lit_value[(lit.0 ^ 1) as usize] = LBool::False;
                self.level[vi] = self.decision_level();
                self.reason[vi] = reason;
                self.trail.push(lit);
                true
            }
        }
    }

    /// Unit propagation via two-watched literals over a single unified
    /// per-literal watch list (MiniSat 2.2+ layout). Binary and long
    /// clauses share one traversal: a binary entry's `blocker` IS the
    /// partner literal, so a single `value_of(blocker)` resolves it; a
    /// long entry's `blocker` is the standard hint.
    ///
    /// Several layers of optimization stack here:
    /// - **Disjoint field references**: `self` is destructured once so each
    ///   hot field is its own `&mut` reference, letting the compiler see
    ///   non-aliasing across the loop body.
    /// - **Lazy two-watched management**: we don't eagerly swap `lits[0]`
    ///   and `lits[1]` on every visit; instead we read both and remember
    ///   which slot held `false_lit`. The arena write only happens when we
    ///   actually migrate a watch.
    /// - **Raw-pointer two-cursor iteration**: the watch list is compacted
    ///   in place with `read`/`write` pointers — no per-element bound
    ///   checks on the inner `ws[i]` / `ws[j]` accesses.
    /// - **`get_unchecked` on indexed accesses** whose bounds are program
    ///   invariants (every `Lit.0 < 2 * num_vars == lit_value.len()`,
    ///   every variable index is < `level.len() / reason.len()`, etc.).
    /// - **Software prefetch** of the next watcher's clause header.
    ///
    /// Returns `Some` on conflict, tagging binary vs long-clause.
    fn propagate(&mut self) -> Option<PropConflict> {
        // Destructure `self` into disjoint mutable references. The compiler
        // can prove non-aliasing between distinct fields, so each access
        // inside the loop becomes a clean pointer/index op without the
        // implicit "borrow checker barrier" of `self.field` projections.
        let Solver {
            clauses,
            lit_value,
            level,
            reason,
            trail,
            trail_lim,
            watches,
            qhead,
            stats_propagations,
            ..
        } = self;

        // Decision level is invariant across one propagate() call.
        let dl = trail_lim.len() as i32;
        // Snapshot for stats — single accumulator update at the end.
        let qhead_start = *qhead;
        let mut conflict: Option<PropConflict> = None;

        'queue: while *qhead < trail.len() {
            // SAFETY: the loop guard just established `*qhead < trail.len()`.
            let p = unsafe { *trail.get_unchecked(*qhead) };
            *qhead += 1;

            // p is now true, so !p is false. Visit every watcher of !p.
            let false_lit = !p;
            let wl_idx = false_lit.idx();

            // Detach this slot so long-clause migrations can push to other
            // slots of `watches` without aliasing it. Binaries never
            // migrate — they always get copied through to the write cursor.
            //
            // SAFETY: every literal on the trail is a valid Lit, so its
            // index is < 2 * num_vars == watches.len().
            let mut ws = std::mem::take(unsafe { watches.get_unchecked_mut(wl_idx) });

            // Two-cursor compaction: `read` walks forward consuming entries,
            // `write` lags behind recording the survivors. Final length is
            // `write.offset_from(start)`.
            let start_ptr = ws.as_mut_ptr();
            let mut read = start_ptr;
            let mut write = start_ptr;
            // SAFETY: ws.len() is the valid extent of the buffer; `add` stays
            // within (or one past) the allocation as required by the API.
            let end = unsafe { start_ptr.add(ws.len()) };

            'watches: while read < end {
                // SAFETY: `read < end` and `end == start + len`, so `read`
                // points to an initialized Watcher within the buffer.
                let w = unsafe { *read };
                // SAFETY: after this advance, `read <= end`. Subsequent
                // reads are guarded by the `read < end` loop condition.
                read = unsafe { read.add(1) };

                // Software prefetch of the *next* watcher's clause body
                // AND of its blocker's `lit_value` slot. Both are off the
                // critical path of this iteration but on the critical path
                // of the next, so a hint here overlaps the cache fill with
                // the current iteration's work. `lit_value` is large
                // (2 × num_vars bytes) so its lookups regularly miss L1/L2
                // on big formulas — pulling the next slot in early is
                // measurable on long watch lists.
                if read < end {
                    let nw = unsafe { *read };
                    if !nw.is_binary() {
                        clauses.prefetch(nw.long_cref());
                    }
                    prefetch_lit_value_slot(lit_value, nw.blocker.0 as usize);
                }

                // One lookup serves both clause shapes: for binary entries
                // `blocker` IS the partner; for long entries it's a hint.
                // SAFETY: blocker is a valid Lit, so blocker.0 < lit_value.len().
                let bv = unsafe { *lit_value.get_unchecked(w.blocker.0 as usize) };

                if w.is_binary() {
                    // Binary path: blocker == q. Binaries never migrate, so
                    // unconditionally copy through to the write cursor.
                    let q = w.blocker;
                    // SAFETY: `write <= read - 1 < end`.
                    unsafe { *write = w };
                    write = unsafe { write.add(1) };

                    match bv {
                        LBool::True => {} // satisfied
                        LBool::Undef => {
                            let vq = q.var_idx();
                            // SAFETY: q.0 < lit_value.len() and vq < level.len() / reason.len().
                            unsafe {
                                *lit_value.get_unchecked_mut(q.0 as usize) = LBool::True;
                                *lit_value.get_unchecked_mut((q.0 ^ 1) as usize) = LBool::False;
                                *level.get_unchecked_mut(vq) = dl;
                                *reason.get_unchecked_mut(vq) = Reason::Binary(false_lit);
                            }
                            trail.push(q);
                        }
                        LBool::False => {
                            // Both !p and q false — binary conflict. Copy
                            // remaining watchers and exit the queue loop.
                            while read < end {
                                unsafe {
                                    *write = *read;
                                    read = read.add(1);
                                    write = write.add(1);
                                }
                            }
                            // SAFETY: write is between start_ptr and end, all
                            // entries in [start, write) are initialized.
                            let new_len = unsafe { write.offset_from(start_ptr) } as usize;
                            unsafe { ws.set_len(new_len) };
                            unsafe { *watches.get_unchecked_mut(wl_idx) = ws };
                            conflict = Some(PropConflict::Binary(false_lit, q));
                            break 'queue;
                        }
                    }
                    continue 'watches;
                }

                // Long-clause path. If the blocker is true, the clause is
                // satisfied — keep the watch verbatim.
                if bv == LBool::True {
                    unsafe { *write = w };
                    write = unsafe { write.add(1) };
                    continue 'watches;
                }

                let cref = w.long_cref();

                // Ensure lits[1] == false_lit so lits[0] is the "other"
                // watch. We do this eagerly — a lazy variant was tried but
                // changed the cross-visit arena state enough to shift
                // propagation order, pushing one instance over the SMT-bench
                // timeout cliff. Eager keeps propagation behavior identical
                // to the pre-Tier 3 baseline.
                if clauses.get_lit(cref, 0) == false_lit {
                    clauses.swap_lits(cref, 0, 1);
                }

                let (first, first_val, found) = {
                    let lits = clauses.lits(cref);
                    // SAFETY: every long clause has length >= 3 by construction.
                    let first = unsafe { *lits.get_unchecked(0) };
                    // SAFETY: first is a valid Lit.
                    let first_val =
                        unsafe { *lit_value.get_unchecked(first.0 as usize) };

                    if first_val == LBool::True {
                        // Skip the inner scan — clause is satisfied by `first`.
                        (first, first_val, None)
                    } else {
                        let mut found: Option<(usize, Lit)> = None;
                        let n = lits.len();
                        let mut k = 2usize;
                        while k < n {
                            // SAFETY: k < n == lits.len().
                            let lk = unsafe { *lits.get_unchecked(k) };
                            // SAFETY: lk.0 < lit_value.len().
                            let lkv = unsafe { *lit_value.get_unchecked(lk.0 as usize) };
                            if lkv != LBool::False {
                                found = Some((k, lk));
                                break;
                            }
                            k += 1;
                        }
                        (first, first_val, found)
                    }
                };

                if first_val == LBool::True {
                    // Refresh blocker hint and keep watch.
                    unsafe { *write = Watcher::long(cref, first) };
                    write = unsafe { write.add(1) };
                    continue 'watches;
                }

                if let Some((k, lk)) = found {
                    // Migrate: lits[1] holds false_lit (post-eager-swap);
                    // swap with k so lk takes the watch and false_lit
                    // moves into the body.
                    clauses.swap_lits(cref, 1, k);
                    // SAFETY: lk is a valid Lit, lk.idx() < watches.len().
                    unsafe {
                        watches
                            .get_unchecked_mut(lk.idx())
                            .push(Watcher::long(cref, first));
                    }
                    continue 'watches;
                }

                // No replacement watch — keep this one with refreshed blocker.
                unsafe { *write = Watcher::long(cref, first) };
                write = unsafe { write.add(1) };

                if first_val == LBool::False {
                    // Long-clause conflict. Copy remainder and exit.
                    while read < end {
                        unsafe {
                            *write = *read;
                            read = read.add(1);
                            write = write.add(1);
                        }
                    }
                    let new_len = unsafe { write.offset_from(start_ptr) } as usize;
                    unsafe { ws.set_len(new_len) };
                    unsafe { *watches.get_unchecked_mut(wl_idx) = ws };
                    conflict = Some(PropConflict::Clause(cref));
                    break 'queue;
                } else {
                    // Unit prop on `first`. The eager swap above already
                    // put `first` at lits[0], which is the invariant that
                    // `analyze` and `lit_redundant` rely on when walking
                    // a reason clause (they skip lits[0] as the pivot).
                    let vi = first.var_idx();
                    unsafe {
                        *lit_value.get_unchecked_mut(first.0 as usize) = LBool::True;
                        *lit_value.get_unchecked_mut((first.0 ^ 1) as usize) =
                            LBool::False;
                        *level.get_unchecked_mut(vi) = dl;
                        *reason.get_unchecked_mut(vi) = Reason::Clause(cref);
                    }
                    trail.push(first);
                }
            }

            let new_len = unsafe { write.offset_from(start_ptr) } as usize;
            unsafe { ws.set_len(new_len) };
            unsafe { *watches.get_unchecked_mut(wl_idx) = ws };
        }

        *stats_propagations += (*qhead - qhead_start) as u64;
        conflict
    }

    /// 1UIP conflict analysis. Returns (learned clause, backtrack level, LBD).
    /// Learned clause layout: position 0 is the asserting literal, position 1
    /// is the second-watch (highest-level literal among the rest).
    fn analyze(&mut self, confl: PropConflict) -> (Vec<Lit>, i32, u32) {
        // Start with enough capacity that typical learned clauses (10-30 lits)
        // fit without any regrowth. Each analyze call allocates a fresh Vec
        // because the storage gets handed off into the clause arena; sizing
        // it up front avoids 2-3 intermediate doublings per conflict.
        let mut learned: Vec<Lit> = Vec::with_capacity(32);
        learned.push(Lit(0)); // placeholder, overwritten with !UIP at the end

        // analyze_toclear accumulates every var whose seen[] we set, across
        // the main 1UIP walk AND the minimization phase. We reset all of them
        // at the end of this function so seen[] re-enters as all-false.
        self.analyze_toclear.clear();

        let mut path_c: i32 = 0;
        let current_level = self.decision_level();
        let mut trail_idx = self.trail.len() as isize - 1;
        let mut uip: Option<Lit> = None;

        // The conflict source on the first iteration is special: for a binary
        // conflict we iterate its two literals; for a clause conflict we
        // iterate the clause body.
        let mut current: AnalyzeSrc = match confl {
            PropConflict::Clause(cr) => AnalyzeSrc::Clause(cr),
            PropConflict::Binary(a, b) => AnalyzeSrc::Binary(a, b),
        };
        let mut first_iter = true;

        loop {
            // Walk the current resolvent's literals. On the first iteration
            // include everything (the conflict source); afterwards skip the
            // pivot literal which resolution cancels.
            match current {
                AnalyzeSrc::Clause(cr) => {
                    self.bump_clause_activity(cr);
                    let start = if first_iter { 0 } else { 1 };
                    let clen = self.clauses.len(cr);
                    for i in start..clen {
                        let q = self.clauses.get_lit(cr, i);
                        self.analyze_touch(q, current_level, &mut path_c, &mut learned);
                    }
                }
                AnalyzeSrc::Binary(a, b) => {
                    // The reason "clause" is {a, b}. For non-first iterations
                    // one of them is the pivot and we're expected to process
                    // only the other; we handle that at the call site.
                    if first_iter {
                        self.analyze_touch(a, current_level, &mut path_c, &mut learned);
                        self.analyze_touch(b, current_level, &mut path_c, &mut learned);
                    } else {
                        // `b` is the "other" literal for a binary reason.
                        self.analyze_touch(b, current_level, &mut path_c, &mut learned);
                    }
                }
            }

            first_iter = false;

            while trail_idx >= 0 && !self.seen[self.trail[trail_idx as usize].var_idx()] {
                trail_idx -= 1;
            }
            if trail_idx < 0 {
                break;
            }
            let p = self.trail[trail_idx as usize];
            let pvi = p.var_idx();
            self.seen[pvi] = false;
            path_c -= 1;
            trail_idx -= 1;

            if path_c <= 0 {
                uip = Some(p);
                break;
            }

            current = match self.reason[pvi] {
                Reason::Clause(cr) => AnalyzeSrc::Clause(cr),
                Reason::Binary(other) => AnalyzeSrc::Binary(p, other),
                Reason::Decision => {
                    panic!("analyze walked past a decision without reaching UIP")
                }
            };
        }

        let uip = uip.expect("conflict analysis must produce a UIP");
        learned[0] = !uip;

        // === Recursive clause minimization ===
        // Build the level abstraction: a 64-bit bitmask where bit (level & 63)
        // is set for every decision level represented in the learned clause.
        // Used by lit_redundant as a cheap level-locality reject filter.
        let mut abstract_levels: u64 = 0;
        for i in 1..learned.len() {
            abstract_levels |= 1u64 << (self.level[learned[i].var_idx()] & 63);
        }

        let pre_min_len = learned.len();
        let mut write = 1;
        for read in 1..learned.len() {
            let l = learned[read];
            let vi = l.var_idx();
            // A decision literal has no reason graph to walk, and is never
            // implied by the other clause literals — keep it.
            let keep = matches!(self.reason[vi], Reason::Decision)
                || !self.lit_redundant(l, abstract_levels);
            if keep {
                learned[write] = l;
                write += 1;
            }
        }
        learned.truncate(write);
        self.stats_min_removed += (pre_min_len - learned.len()) as u64;

        // Reset seen[] for every variable we touched anywhere in this analyze
        // (original walk + minimization). Idempotent: the UIP's seen was
        // already cleared during the pop, and resetting to false again is fine.
        for i in 0..self.analyze_toclear.len() {
            let l = self.analyze_toclear[i];
            self.seen[l.var_idx()] = false;
        }
        self.analyze_toclear.clear();

        // Move the highest-level non-UIP literal to position 1 (second watch)
        // and compute the backtrack level.
        let btlevel = if learned.len() == 1 {
            0
        } else {
            let mut max_i = 1;
            let mut max_l = self.level[learned[1].var_idx()];
            for i in 2..learned.len() {
                let lv = self.level[learned[i].var_idx()];
                if lv > max_l {
                    max_l = lv;
                    max_i = i;
                }
            }
            learned.swap(1, max_i);
            max_l
        };

        let lbd = self.compute_lbd(&learned);
        (learned, btlevel, lbd)
    }

    #[inline]
    fn analyze_touch(
        &mut self,
        q: Lit,
        current_level: i32,
        path_c: &mut i32,
        learned: &mut Vec<Lit>,
    ) {
        let vq = q.var_idx();
        if !self.seen[vq] && self.level[vq] > 0 {
            self.bump_var_activity(q.var());
            self.seen[vq] = true;
            self.analyze_toclear.push(q);
            if self.level[vq] >= current_level {
                *path_c += 1;
            } else {
                learned.push(q);
            }
        }
    }

    /// Recursive self-subsumption check used for learned-clause minimization.
    /// A literal `start` can be removed from the learned clause if every
    /// literal reachable via its reason graph (transitively) is either
    /// already in the learned clause (seen == true), at decision level 0, or
    /// itself recursively reducible to such literals.
    ///
    /// Visited variables are marked seen=true and recorded in `analyze_toclear`
    /// so they don't get re-examined and so that bulk cleanup resets them.
    /// On failure, any seen[] markers set during THIS call are rolled back,
    /// so later calls can still reach those variables and decide independently.
    fn lit_redundant(&mut self, start: Lit, abstract_levels: u64) -> bool {
        self.analyze_stack.clear();
        self.analyze_stack.push(start);
        let top = self.analyze_toclear.len();

        while let Some(p) = self.analyze_stack.pop() {
            // Extract p's reason into one of two shapes: Clause (many lits,
            // indices 1..n are the "other" lits) or Binary (single partner).
            let (cref_opt, binary_other): (Option<ClauseRef>, Option<Lit>) =
                match self.reason[p.var_idx()] {
                    // SAFETY: the caller filters decision literals, and we
                    // only push literals onto analyze_stack when we've
                    // verified their reason is not a Decision. Reaching this
                    // arm would be a logic bug elsewhere in the solver.
                    Reason::Decision => unsafe {
                        debug_assert!(
                            false,
                            "stacked lit must have non-decision reason"
                        );
                        std::hint::unreachable_unchecked()
                    },
                    Reason::Clause(cr) => (Some(cr), None),
                    Reason::Binary(other) => (None, Some(other)),
                };

            let n = match binary_other {
                Some(_) => 1,
                None => self.clauses.len(cref_opt.unwrap()) - 1,
            };

            for i in 0..n {
                let q = match binary_other {
                    Some(other) => other,
                    None => self.clauses.get_lit(cref_opt.unwrap(), i + 1),
                };
                let vq = q.var_idx();
                if self.seen[vq] || self.level[vq] <= 0 {
                    // Already in learned / visited, or at level 0 (implied).
                    continue;
                }
                // To recurse, q must itself have a reason (not be a decision)
                // AND its level must match one of the levels present in the
                // learned clause's 64-bit abstraction — a cheap reject filter
                // that avoids walking into subgraphs whose levels are absent.
                let can_recurse = !matches!(self.reason[vq], Reason::Decision)
                    && (abstract_levels & (1u64 << (self.level[vq] & 63))) != 0;
                if can_recurse {
                    self.seen[vq] = true;
                    self.analyze_stack.push(q);
                    self.analyze_toclear.push(q);
                } else {
                    // Roll back everything this call marked so later
                    // minimization attempts can still examine these vars.
                    for j in top..self.analyze_toclear.len() {
                        self.seen[self.analyze_toclear[j].var_idx()] = false;
                    }
                    self.analyze_toclear.truncate(top);
                    return false;
                }
            }
        }
        true
    }

    /// Build an UNSAT core from a "final" conflict — a literal `p` that the
    /// formula + earlier assumptions force to be true, but whose negation is
    /// the assumption we were about to install. Called as `analyze_final(!a)`
    /// where `a` is that current assumption.
    ///
    /// The core is produced in caller-friendly form: every element is an
    /// assumption literal *as the user originally passed it*. Position 0 is
    /// `a` (== `!p`), the assumption that clashed; the rest are the earlier
    /// assumptions whose implication chain forced `!a`. These are jointly
    /// UNSAT — there exists no model in which all of them are true together.
    fn analyze_final(&mut self, p: Lit) {
        self.conflict_core.clear();
        self.conflict_core.push(!p); // the assumption we tried to install

        if self.decision_level() == 0 {
            // No assumptions installed — the formula itself implies !p. The
            // single-element core "a is unsat with the formula" is complete.
            return;
        }

        self.seen[p.var_idx()] = true;

        // Walk the trail from top down to the first assumption's level
        // (trail_lim[0]). Anything below that is a root-level fact and
        // can be dropped from the core.
        let start = self.trail_lim[0];
        for i in (start..self.trail.len()).rev() {
            let x = self.trail[i];
            let vx = x.var_idx();
            if !self.seen[vx] {
                continue;
            }
            match self.reason[vx] {
                Reason::Decision => {
                    // Decisions at this stage of solve are assumptions (no
                    // VSIDS decisions happen inside the assumption prefix).
                    // `x` is the trail entry = the literal the caller passed.
                    self.conflict_core.push(x);
                }
                Reason::Clause(cr) => {
                    let clen = self.clauses.len(cr);
                    for j in 1..clen {
                        let lj = self.clauses.get_lit(cr, j);
                        if self.level[lj.var_idx()] > 0 {
                            self.seen[lj.var_idx()] = true;
                        }
                    }
                }
                Reason::Binary(other) => {
                    if self.level[other.var_idx()] > 0 {
                        self.seen[other.var_idx()] = true;
                    }
                }
            }
            self.seen[vx] = false;
        }
        self.seen[p.var_idx()] = false;
    }

    /// LBD = number of distinct decision levels among the clause's literals.
    fn compute_lbd(&mut self, lits: &[Lit]) -> u32 {
        self.lbd_counter = self.lbd_counter.wrapping_add(1);
        let stamp = self.lbd_counter;
        let mut count = 0u32;
        for &l in lits {
            let lvl = self.level[l.var_idx()];
            if lvl < 0 {
                continue;
            }
            let li = lvl as usize;
            while self.lbd_stamp.len() <= li {
                self.lbd_stamp.push(0);
            }
            if self.lbd_stamp[li] != stamp {
                self.lbd_stamp[li] = stamp;
                count += 1;
            }
        }
        count
    }

    /// Undo assignments back down to `level`. Phase-save, re-insert freed
    /// variables into the order heap so we can pick them again.
    fn cancel_until(&mut self, level: i32) {
        if self.decision_level() <= level {
            return;
        }
        let target = self.trail_lim[level as usize];
        for i in (target..self.trail.len()).rev() {
            let lit = self.trail[i];
            let vi = lit.var_idx();
            self.polarity[vi] = !lit.is_negated();
            // Reset both polarities in the per-literal table.
            let pos = (lit.0 & !1) as usize;
            self.lit_value[pos] = LBool::Undef;
            self.lit_value[pos | 1] = LBool::Undef;
            self.level[vi] = -1;
            self.reason[vi] = Reason::Decision;
            self.order_heap.insert(lit.var().0, &self.activity);
        }
        self.trail.truncate(target);
        self.trail_lim.truncate(level as usize);
        self.qhead = target;
    }

    #[inline]
    fn bump_var_activity(&mut self, v: Var) {
        let vi = v.idx();
        self.activity[vi] += self.var_inc;
        if self.activity[vi] > 1e100 {
            self.rescale_var_activity();
        }
        self.order_heap.decrease(v.0, &self.activity);
    }

    /// Bump a variable's VSIDS activity from outside the solver — same
    /// mechanics as a conflict-driven bump. Intended for higher-level layers
    /// (e.g. an SMT bitblaster) that want to bias the search toward
    /// variables they know are structurally important, like ITE selectors.
    pub fn boost_var_activity(&mut self, v: Var) {
        self.bump_var_activity(v);
    }

    /// Rescales every variable activity to keep them out of the floating-point
    /// stratosphere. Fires maybe once per billion-or-so conflicts — isolate it
    /// so the hot `bump_var_activity` path stays branch-prediction-friendly.
    #[cold]
    #[inline(never)]
    fn rescale_var_activity(&mut self) {
        for a in self.activity.iter_mut() {
            *a *= 1e-100;
        }
        self.var_inc *= 1e-100;
    }

    fn decay_var_activity(&mut self) {
        self.var_inc /= self.var_decay;
    }

    #[inline]
    fn bump_clause_activity(&mut self, cref: ClauseRef) {
        if !self.clauses.learned(cref) {
            return;
        }
        let new_act = self.clauses.activity(cref) + self.cla_inc;
        self.clauses.set_activity(cref, new_act);
        if new_act > 1e20 {
            self.rescale_clause_activity();
        }
    }

    /// Same idea as `rescale_var_activity` — cold branch, hoisted out.
    #[cold]
    #[inline(never)]
    fn rescale_clause_activity(&mut self) {
        for i in 0..self.learnts.len() {
            let cr = self.learnts[i];
            let a = self.clauses.activity(cr);
            self.clauses.set_activity(cr, a * 1e-20);
        }
        self.cla_inc *= 1e-20;
    }

    fn decay_clause_activity(&mut self) {
        self.cla_inc /= self.cla_decay;
    }

    /// Is this clause currently the reason for some assigned variable?
    #[inline]
    fn locked(&self, cref: ClauseRef) -> bool {
        if self.clauses.len(cref) == 0 {
            return false;
        }
        let first = self.clauses.get_lit(cref, 0);
        if self.value_of(first) != LBool::True {
            return false;
        }
        matches!(self.reason[first.var_idx()], Reason::Clause(r) if r == cref)
    }

    /// Drop low-quality learned clauses. Keep clauses with LBD <= 2 (glue),
    /// locked clauses, and the upper half by quality (LBD first, activity
    /// second). Marks the rest as deleted in the arena and purges watch
    /// entries pointing at them. The arena words themselves stay allocated
    /// until a full compaction is triggered (not implemented here).
    fn reduce_db(&mut self) {
        self.stats_reductions += 1;

        // Sort learnts best-first: low LBD, then high activity.
        self.learnts.sort_by(|&a, &b| {
            let la = self.clauses.lbd(a);
            let lb = self.clauses.lbd(b);
            let aa = self.clauses.activity(a);
            let ab = self.clauses.activity(b);
            la.cmp(&lb).then_with(|| {
                ab.partial_cmp(&aa).unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        // We delete from the back half of the sorted list.
        let start = self.learnts.len() / 2;
        let mut new_learnts: Vec<ClauseRef> = Vec::with_capacity(self.learnts.len());
        new_learnts.extend_from_slice(&self.learnts[..start]);

        for i in start..self.learnts.len() {
            let cref = self.learnts[i];
            let keep = self.clauses.lbd(cref) <= 2 || self.locked(cref);
            if keep {
                new_learnts.push(cref);
            } else {
                self.clauses.mark_deleted(cref);
                self.stats_deleted += 1;
            }
        }
        self.learnts = new_learnts;

        // Clean up watch lists: drop watchers pointing at deleted clauses.
        // Binary watchers don't refer to the arena at all (their cref is a
        // sentinel), so they're always preserved.
        let clauses = &self.clauses;
        for wl in &mut self.watches {
            wl.retain(|w| w.is_binary() || !clauses.deleted(w.long_cref()));
        }
    }

    fn pick_branch_lit(&mut self) -> Option<Lit> {
        while let Some(v) = self.order_heap.pop(&self.activity) {
            if self.lit_value[(v as usize) << 1] == LBool::Undef {
                let lit = Lit::new(Var(v), !self.polarity[v as usize]);
                return Some(lit);
            }
            // Otherwise this variable was assigned since we last saw it; keep
            // popping. (It'll re-enter on the next backtrack.)
        }
        None
    }

    /// Jeroslow-Wang one-sided heuristic. For each variable v, score its
    /// positive and negative literal as `J(l) = Σ_{C ∋ l} 2^-|C|` — short
    /// clauses contribute exponentially more, since they're more constraining.
    /// The polarity with the higher score is the one we pick first when the
    /// var gets selected by VSIDS.
    ///
    /// **Not wired in by default.** Empirically didn't help (and on some
    /// catchconv SAT traces it hurt by ~80%) — kept around as an opt-in for
    /// callers who want to experiment. Best called once, after all input
    /// clauses are loaded but before the first `solve*` — phase saving will
    /// then carry the seed forward across incremental queries.
    pub fn init_polarity_jw(&mut self) {
        let n = self.polarity.len();
        if n == 0 {
            return;
        }
        let mut pos_score: Vec<f64> = vec![0.0; n];
        let mut neg_score: Vec<f64> = vec![0.0; n];

        // Long clauses live in the arena. Walk them.
        let pos_ref = &mut pos_score;
        let neg_ref = &mut neg_score;
        self.clauses.for_each_clause(|lits| {
            let w = 2f64.powi(-(lits.len() as i32));
            for &l in lits {
                let v = l.var_idx();
                if l.is_negated() {
                    neg_ref[v] += w;
                } else {
                    pos_ref[v] += w;
                }
            }
        });

        // Binary clauses are stored inline in the watch lists with the binary
        // flag set on the watcher. Each binary appears twice (once per
        // literal's slot); dedupe by counting it only when the slot's literal
        // sorts below its partner.
        let bin_w = 0.25_f64; // 2^-2
        for i in 0..self.watches.len() {
            let lit_a = Lit(i as u32);
            for w in &self.watches[i] {
                if !w.is_binary() {
                    continue;
                }
                let lit_b = w.blocker;
                if lit_a.0 >= lit_b.0 {
                    continue;
                }
                let va = lit_a.var_idx();
                if lit_a.is_negated() {
                    neg_score[va] += bin_w;
                } else {
                    pos_score[va] += bin_w;
                }
                let vb = lit_b.var_idx();
                if lit_b.is_negated() {
                    neg_score[vb] += bin_w;
                } else {
                    pos_score[vb] += bin_w;
                }
            }
        }

        // Classical JW: prefer the more-frequent polarity (the one expected
        // to satisfy more clauses per assignment).
        for v in 0..n {
            let p = pos_score[v];
            let q = neg_score[v];
            if p > q {
                self.polarity[v] = true;
            } else if q > p {
                self.polarity[v] = false;
            }
        }
    }

    /// Solve the current formula. Equivalent to `solve_under_assumptions(&[])`.
    pub fn solve(&mut self) -> SolveResult {
        self.solve_under_assumptions(&[])
    }

    /// Solve the formula under a set of assumption literals — each `assumptions[i]`
    /// must be true in any produced model. If the formula is UNSAT under these
    /// assumptions, [`unsat_core`] returns a subset of assumption literals that
    /// jointly caused the contradiction.
    ///
    /// Safe to call repeatedly with different assumption sets; the solver
    /// keeps learned clauses, variable activities, and LBD history across
    /// calls, which is exactly what an SMT frontend wants.
    pub fn solve_under_assumptions(&mut self, assumptions: &[Lit]) -> SolveResult {
        // Pass a termination predicate that always returns false. Because
        // `solve_bounded_inner` is generic over the predicate type, Rust
        // monomorphizes this call into a version that inlines the closure
        // and dead-codes the stop-check — so the hot loop pays zero
        // overhead compared to pre-bounded-solve days.
        self.solve_bounded_inner(assumptions, |_, _| false)
            .expect("unbounded solve always resolves")
    }

    /// Bounded variant of [`solve_under_assumptions`]. Returns `None` once
    /// `max_conflicts` conflicts have accumulated during this call — used
    /// by callers who want "a quick answer if one exists, otherwise give
    /// up" semantics (e.g. symbolic execution probing branch feasibility
    /// under a tight time budget). A `max_conflicts` of `0` means unbounded.
    ///
    /// A `Some(SolveResult::Unsat)` from a bounded call is a real UNSAT
    /// proof derived entirely from formula + assumption clauses; the budget
    /// only ever converts "still-searching" into `None`, never a soundness
    /// relaxation.
    pub fn solve_under_assumptions_bounded(
        &mut self,
        assumptions: &[Lit],
        max_conflicts: u64,
    ) -> Option<SolveResult> {
        if max_conflicts == 0 {
            return Some(self.solve_under_assumptions(assumptions));
        }
        self.solve_bounded_inner(assumptions, move |s, start| {
            s.stats_conflicts - start >= max_conflicts
        })
    }

    /// Wall-clock-bounded variant of [`solve_under_assumptions`]. Returns
    /// `None` once `timeout` has elapsed since the call began. Like the
    /// conflict-bounded form, `Some(Unsat)` is a real proof and the solver
    /// is left consistent on timeout — callers can retry with a larger
    /// deadline or different assumptions.
    ///
    /// The deadline is checked every 256 conflicts (cheap bitmask test,
    /// then an `Instant::now()` on the hit). Worst-case overshoot is
    /// therefore "however long 256 conflicts take" — typically a few
    /// milliseconds at 10k-100k conflicts/sec. Callers with tighter
    /// budgets (sub-millisecond) should prefer the conflict-budget variant.
    pub fn solve_under_assumptions_timed(
        &mut self,
        assumptions: &[Lit],
        timeout: std::time::Duration,
    ) -> Option<SolveResult> {
        let deadline = std::time::Instant::now() + timeout;
        // Checking `Instant::now()` on every conflict is measurable on
        // fast-conflict workloads (≈25ns × 100k conflicts/sec = 2.5ms/sec).
        // A bitmask throttle skips the clock query 255 times out of 256;
        // the amortised cost is under 0.1ns per conflict.
        self.solve_bounded_inner(assumptions, move |s, start| {
            let conflicts_since = s.stats_conflicts - start;
            if conflicts_since & 0xFF != 0 {
                return false;
            }
            std::time::Instant::now() >= deadline
        })
    }

    fn solve_bounded_inner<F>(
        &mut self,
        assumptions: &[Lit],
        should_stop: F,
    ) -> Option<SolveResult>
    where
        F: Fn(&Self, u64) -> bool,
    {
        if self.dead {
            return Some(SolveResult::Unsat);
        }
        let start_conflicts = self.stats_conflicts;

        // Reset to a clean search state and install the new assumptions.
        self.cancel_until(0);
        self.assumptions.clear();
        self.assumptions.extend_from_slice(assumptions);
        self.conflict_core.clear();
        self.restart.on_new_solve();

        // Fresh DB reduction budget sized by the current (post-learning)
        // clause count. Running this per call lets long incremental sessions
        // expand the budget naturally as they accumulate clauses.
        self.max_learnts = (self.num_clauses_total as f64 / 3.0).max(1000.0);

        // Propagate level-0 units once up front.
        if self.propagate().is_some() {
            return Some(SolveResult::Unsat);
        }

        loop {
            if let Some(confl) = self.propagate() {
                self.stats_conflicts += 1;

                if self.decision_level() == 0 {
                    // Root-level conflict: UNSAT regardless of assumptions.
                    return Some(SolveResult::Unsat);
                }

                let (learned, btlevel, lbd) = self.analyze(confl);
                self.restart.on_conflict(lbd, self.trail.len() as u64);
                self.stats_learned += 1;
                self.cancel_until(btlevel);

                match learned.len() {
                    1 => {
                        self.enqueue(learned[0], Reason::Decision);
                    }
                    2 => {
                        let a = learned[0];
                        let b = learned[1];
                        self.watches[a.idx()].push(Watcher::binary(b));
                        self.watches[b.idx()].push(Watcher::binary(a));
                        self.enqueue(a, Reason::Binary(b));
                    }
                    _ => {
                        let w0 = learned[0];
                        let w1 = learned[1];
                        let cref = self.clauses.alloc(&learned, true);
                        self.num_clauses_total += 1;
                        self.clauses.set_lbd(cref, lbd);
                        self.learnts.push(cref);
                        self.watches[w0.idx()].push(Watcher::long(cref, w1));
                        self.watches[w1.idx()].push(Watcher::long(cref, w0));
                        self.bump_clause_activity(cref);
                        self.enqueue(w0, Reason::Clause(cref));
                    }
                }

                self.decay_var_activity();
                self.decay_clause_activity();

                // Termination check lives here — *after* a conflict has
                // been analyzed and the learned clause added, so every
                // `None` return leaves the solver in a consistent state
                // (trail cancelled to level 0, learned clauses durably in
                // the DB). Both conflict-budget and wall-clock callers
                // share this hook through the termination predicate.
                if should_stop(self, start_conflicts) {
                    self.cancel_until(0);
                    return None;
                }
            } else {
                // No conflict. Consider DB reduction, then restart, then
                // pick a decision (or declare SAT).
                if self.num_learnts() as f64 > self.max_learnts + self.trail.len() as f64 {
                    self.reduce_db();
                    self.max_learnts *= 1.1;
                }

                if self.restart.should_restart(self.trail.len() as u64) {
                    self.stats_restarts += 1;
                    self.cancel_until(0);
                    self.restart.on_restart();
                }

                // Pick next decision: installed assumptions take priority.
                // Each assumption becomes its own decision level; if one is
                // already forced-true at a lower level we just push an
                // empty level so the 1-level-per-assumption invariant holds.
                let mut next: Option<Lit> = None;
                while (self.decision_level() as usize) < self.assumptions.len() {
                    let a = self.assumptions[self.decision_level() as usize];
                    match self.value_of(a) {
                        LBool::True => {
                            self.trail_lim.push(self.trail.len());
                        }
                        LBool::False => {
                            // Assumption contradicts existing implications.
                            // Build an UNSAT core out of it.
                            self.analyze_final(!a);
                            return Some(SolveResult::Unsat);
                        }
                        LBool::Undef => {
                            next = Some(a);
                            break;
                        }
                    }
                }

                let chosen = next.or_else(|| self.pick_branch_lit());
                match chosen {
                    None => return Some(SolveResult::Sat),
                    Some(lit) => {
                        self.stats_decisions += 1;
                        self.trail_lim.push(self.trail.len());
                        self.enqueue(lit, Reason::Decision);
                    }
                }
            }
        }
    }

    /// The UNSAT core from the most recent [`solve_under_assumptions`] call
    /// that returned [`SolveResult::Unsat`] because of an assumption clash.
    /// Empty if the last solve was SAT, or was UNSAT at level 0 (the
    /// formula is unconditionally UNSAT and no assumptions were needed).
    pub fn unsat_core(&self) -> &[Lit] {
        &self.conflict_core
    }

    pub fn model_dimacs(&self) -> Vec<i32> {
        (0..self.num_vars())
            .map(|v| {
                let sign = match self.lit_value[v << 1] {
                    LBool::True => 1,
                    LBool::False => -1,
                    LBool::Undef => 1,
                };
                sign * (v as i32 + 1)
            })
            .collect()
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

/// The source of a conflict detected during propagation.
enum PropConflict {
    Clause(ClauseRef),
    // a, b with both literals false — the implicit binary clause is {a, b}.
    Binary(Lit, Lit),
}

/// The source being resolved in a given iteration of analyze().
enum AnalyzeSrc {
    Clause(ClauseRef),
    // Binary reason: resolving on literal at position 0, keeping position 1.
    // Stored as (pivot, other) so we know which to skip.
    Binary(Lit, Lit),
}

/// Hint the CPU to start loading the `lit_value` byte at `idx` into L1.
/// Companion to `ClauseArena::prefetch` — used during `propagate` to
/// overlap the next watcher's blocker-value fetch with this iteration's
/// work. No-op on architectures without a prefetch intrinsic we know.
#[inline(always)]
fn prefetch_lit_value_slot(lv: &[LBool], idx: usize) {
    if idx >= lv.len() {
        return;
    }
    // SAFETY: bounded above; prefetch is a pure CPU hint that tolerates
    // any pointer not causing an access violation.
    unsafe {
        let ptr = lv.as_ptr().add(idx);
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
