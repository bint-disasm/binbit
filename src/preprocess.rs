//! CNF preprocessing: subsumption, self-subsumption (strengthening) and
//! bounded variable elimination (SatELite / MiniSat-simp style).
//!
//! Operates on a standalone clause "soup" — the batch of clauses produced by
//! one bitblast flush — *before* they are committed to the CDCL core. This
//! sidesteps the classic incrementality problem: variables that appear only
//! in the current batch (freshly-allocated Tseitin gate outputs) can be
//! resolved away freely because nothing outside the batch can ever mention
//! them again — the SMT layer drops its AIG-node → SAT-lit binding for every
//! eliminated variable, so any later re-use of the same AIG node simply
//! re-materializes it under a fresh variable with fresh defining clauses.
//!
//! Eager bitblasting is the textbook workload for bounded VE: most gate
//! variables have a handful of occurrences (their 3-4 defining clauses plus
//! one or two uses), and resolving them out both shrinks the formula and
//! shortens the implication chains the SAT solver has to walk. This is the
//! same simplification bitwuzla/z3-class solvers get from handing their CNF
//! to CaDiCaL-style inprocessing.
//!
//! Storage follows the house arena pattern (ClauseArena / WatchArena / the
//! cnfmap Plan): every literal lives in ONE flat buffer and clauses are
//! `(off, len)` headers into it. The flush pipeline hands the emission
//! buffer straight in, resolvents append to the same buffer through a
//! reused scratch, and survivors go back out as slices — no per-clause
//! `Vec` exists anywhere on the path (the per-clause allocate/free pair
//! was ~20% of preprocess-bound instance profiles).
//!
//! Everything here is deterministic: candidate orders are index-based, and
//! all limits are fixed constants.
//!
//! Equivalent-literal substitution (SCC over the binary implication
//! graph) was built and measured here twice on 2026-08-05 — once before
//! VE, once after — and removed both times. The soup genuinely carries
//! equivalences (bench_5906: ~2.7k matched binary pairs reach the SAT
//! core even post-VE; ~158k pre-VE), and merging them shrinks the CNF,
//! but the corpus outcome was a per-instance lottery in both placements:
//! post-VE, merging just EIGHTEEN variables on bench_16728 took it
//! 2.0s → 8.3s (conflicts ×3), and eventlogadm tripled, while other
//! spear instances halved. Same verdict as reuse-trail / reason-side
//! bumping / vivification: on this workload, CNF perturbations that
//! don't pay in massively reduced propagation work re-roll the
//! trajectory dice, and the corpus punishes that symmetrically.

use crate::lit::Lit;

/// Upper bound on resolvent length. A candidate elimination that would
/// produce a clause longer than this is rejected (MiniSat's `cl-lim`).
const CLAUSE_LIM: usize = 20;

/// Skip variable elimination for variables whose positive×negative
/// occurrence product exceeds this — the pair enumeration alone would be
/// too expensive, and high-occurrence variables essentially never satisfy
/// the non-increasing bound anyway.
const VE_PRODUCT_LIM: usize = 100;

/// Pair budget for gate-driven substitution (plan B). Substitution
/// enumerates `def_pos × neg + pos × def_neg` pairs — linear in the
/// environment for AND/OR gates (definition side is one long clause +
/// its binaries) — so heavily shared gate outputs stay affordable well
/// past the quadratic full-VE product limit.
const VE_SUB_PAIR_LIM: usize = 400;

/// Don't use a clause as a backward-subsumption candidate if its
/// least-occurring literal still occurs more often than this.
const SUB_OCC_LIM: usize = 1_000;

/// Result of one preprocessing run. Clause storage stays flat: surviving
/// clauses (units included) are `(off, len)` ranges into `data`, in
/// deterministic order. On `unsat`, `clauses` holds a single empty range
/// so the SAT core's `add_clause` records the dead state through its
/// normal path.
pub struct SimplifyResult {
    pub data: Vec<Lit>,
    pub clauses: Vec<(u32, u32)>,
    /// Variable indices eliminated by VE. The caller must ensure these can
    /// never be referenced by later clauses (see module docs).
    pub eliminated: Vec<u32>,
    /// Number of clauses removed by (self-)subsumption.
    pub subsumed: usize,
    /// Number of literals removed by strengthening.
    pub strengthened: usize,
    /// True if preprocessing derived the empty clause — formula is UNSAT.
    pub unsat: bool,
    /// The recycled per-flush storage — hand it to the next
    /// [`Preprocessor::from_flat`] call.
    pub pool: PreprocessPool,
}

impl SimplifyResult {
    /// Materialize the surviving clauses as owned vectors — test/debug
    /// convenience; the production path consumes the flat form.
    #[cfg(test)]
    pub fn clause_vecs(&self) -> Vec<Vec<Lit>> {
        self.clauses
            .iter()
            .map(|&(off, len)| self.data[off as usize..(off + len) as usize].to_vec())
            .collect()
    }
}

/// Clause header into the shared literal arena. `len` shrinks in place on
/// strengthening; freed tail words simply become arena waste for the rest
/// of the (single-flush) run.
struct Clause {
    off: u32,
    len: u32,
    sig: u64,
    deleted: bool,
}

/// A detected Tseitin gate definition for a VE pivot (see
/// [`Preprocessor::find_gate`]): masks over both occurrence lists
/// marking the definition clauses. Substitution resolvents are
/// `def_pos × env_neg ∪ env_pos × def_neg` — environment × environment
/// pairs are implied by those and dropped (Eén-Biere).
struct GateDef {
    def_pos: Vec<bool>,
    def_neg: Vec<bool>,
    /// Marked-clause counts per side (for the pair-budget check).
    n_def_pos: usize,
    n_def_neg: usize,
}

#[inline]
fn sig_of(lits: &[Lit]) -> u64 {
    lits.iter().fold(0u64, |s, l| s | 1u64 << (l.var_idx() & 63))
}

/// Reusable storage for one [`Preprocessor`] run. Kept by the caller
/// across flushes: every per-flush array lives here, sized to the largest
/// batch seen, and each run clears only the prefix it actually uses (the
/// compact variable remap makes that prefix exactly `2 × batch vars`, so
/// reset cost stays O(batch) — the same property that fixed the
/// quadratic-incremental preprocessing bug). Steady-state flushes
/// allocate nothing.
#[derive(Default)]
pub struct PreprocessPool {
    occ: Vec<Vec<u32>>,
    n_occ: Vec<u32>,
    assign: Vec<u8>,
    eliminated: Vec<bool>,
    clauses: Vec<Clause>,
    unit_queue: Vec<Lit>,
    touched: Vec<u32>,
    sub_scratch: Vec<u32>,
    res_scratch: Vec<Lit>,
    sub_order: Vec<u32>,
    sub_queued: Vec<bool>,
    ve_queued: Vec<bool>,
    ve_fail: Vec<u32>,
    ve_seed: Vec<std::cmp::Reverse<(u32, u32)>>,
    /// Rides along for the caller: freeze-flag buffer reused per flush.
    pub frozen: Vec<bool>,
}

pub struct Preprocessor {
    /// Allow the gate-substitution fallback (plan B in `try_eliminate`).
    /// Disabled when the AIG is already being minimized by two-level
    /// rewriting: stacking the two on Sage2-class instances measured
    /// +120% (bench_6554) and +122% (bench_16728, 6.3s → 14.0s under
    /// --aig2) — the same stacked-minimizer interference as the
    /// cnfmap × aig2 arc. Plain full VE is unaffected.
    gate_subst: bool,
    /// Number of (compact) variables in this batch — bounds every
    /// per-variable loop. The pool arrays may be longer (sized by an
    /// earlier, bigger batch) and their tails hold stale data that this
    /// run must never read.
    num_vars: usize,
    /// Flat literal arena. Clause bodies live at their header's range;
    /// strengthened clauses shrink in place, resolvents append at the end.
    data: Vec<Lit>,
    clauses: Vec<Clause>,
    /// Occurrence lists: `occ[lit.0]` = indices of live clauses containing
    /// `lit`. May contain stale entries (deleted / strengthened clauses);
    /// consumers re-verify membership.
    occ: Vec<Vec<u32>>,
    /// Exact live-occurrence counts per literal.
    n_occ: Vec<u32>,
    /// Level-0 assignments discovered in the soup: 0 = unassigned,
    /// 1 = lit true, 2 = lit false (indexed per-literal like `occ`).
    assign: Vec<u8>,
    /// Variables that must not be eliminated (inputs, activation literals,
    /// anything visible outside this batch).
    frozen: Vec<bool>,
    eliminated: Vec<bool>,
    unit_queue: Vec<Lit>,
    /// Vars whose occurrence profile changed since last drained — used by
    /// the VE worklist to re-enqueue only affected candidates instead of
    /// re-scanning every variable each round.
    touched: Vec<u32>,
    /// Reused candidate buffer for the subsumption pass.
    sub_scratch: Vec<u32>,
    /// Reused resolvent build buffer (see `resolve_into`).
    res_scratch: Vec<Lit>,
    /// Reused subsumption / VE worklist storage (pooled across flushes).
    sub_order: Vec<u32>,
    sub_queued: Vec<bool>,
    ve_queued: Vec<bool>,
    ve_fail: Vec<u32>,
    ve_seed: Vec<std::cmp::Reverse<(u32, u32)>>,
    unsat: bool,
    stats_subsumed: usize,
    stats_strengthened: usize,
}

impl Preprocessor {
    /// Flat-input constructor — the production path. `data` holds every
    /// batch clause back to back; `ends[i]` is the exclusive end of clause
    /// i (clause i occupies `ends[i-1]..ends[i]`). Takes ownership of
    /// `data` (it becomes the run's arena and is returned in the result);
    /// `ends` is only read.
    ///
    /// `num_vars` bounds the variable indices appearing in the clauses.
    /// `frozen[v]` marks variables that must survive.
    pub fn from_flat(
        mut data: Vec<Lit>,
        ends: &[u32],
        num_vars: usize,
        frozen: Vec<bool>,
        mut pool: PreprocessPool,
    ) -> Self {
        assert_eq!(frozen.len(), num_vars);
        // Size the pool arrays for this batch, then reset exactly the
        // prefix this run will touch. Anything past `2 * num_vars` /
        // `num_vars` is stale from a bigger earlier batch and is never
        // read (every loop below is bounded by `num_vars`).
        let nl = 2 * num_vars;
        if pool.occ.len() < nl {
            pool.occ.resize_with(nl, Vec::new);
        }
        if pool.n_occ.len() < nl {
            pool.n_occ.resize(nl, 0);
        }
        if pool.assign.len() < nl {
            pool.assign.resize(nl, 0);
        }
        if pool.eliminated.len() < num_vars {
            pool.eliminated.resize(num_vars, false);
        }
        for l in &mut pool.occ[..nl] {
            l.clear();
        }
        pool.n_occ[..nl].fill(0);
        pool.assign[..nl].fill(0);
        pool.eliminated[..num_vars].fill(false);
        pool.clauses.clear();
        pool.unit_queue.clear();
        pool.touched.clear();
        pool.sub_scratch.clear();
        pool.res_scratch.clear();
        pool.sub_order.clear();
        pool.sub_queued.clear();
        pool.ve_queued.clear();
        pool.ve_fail.clear();
        pool.ve_seed.clear();
        let mut p = Preprocessor {
            gate_subst: true,
            num_vars,
            data: Vec::new(),
            clauses: std::mem::take(&mut pool.clauses),
            occ: std::mem::take(&mut pool.occ),
            n_occ: std::mem::take(&mut pool.n_occ),
            assign: std::mem::take(&mut pool.assign),
            frozen,
            eliminated: std::mem::take(&mut pool.eliminated),
            unit_queue: std::mem::take(&mut pool.unit_queue),
            touched: std::mem::take(&mut pool.touched),
            sub_scratch: std::mem::take(&mut pool.sub_scratch),
            res_scratch: std::mem::take(&mut pool.res_scratch),
            sub_order: std::mem::take(&mut pool.sub_order),
            sub_queued: std::mem::take(&mut pool.sub_queued),
            ve_queued: std::mem::take(&mut pool.ve_queued),
            ve_fail: std::mem::take(&mut pool.ve_fail),
            ve_seed: std::mem::take(&mut pool.ve_seed),
            unsat: false,
            stats_subsumed: 0,
            stats_strengthened: 0,
        };
        let mut start = 0usize;
        for &e in ends {
            let end = e as usize;
            let range = &mut data[start..end];
            // Sort + dedup in place (freed dedup tail stays as arena
            // garbage — headers never point at it).
            range.sort_by_key(|l| l.0);
            let mut w = 0usize;
            for r in 0..range.len() {
                if w == 0 || range[w - 1] != range[r] {
                    range[w] = range[r];
                    w += 1;
                }
            }
            let lits = &range[..w];
            // Tautology: drop at intake.
            if lits.windows(2).any(|p| p[0].var() == p[1].var()) {
                start = end;
                continue;
            }
            if lits.is_empty() {
                p.unsat = true;
                start = end;
                continue;
            }
            if lits.len() == 1 {
                p.unit_queue.push(lits[0]);
            }
            let idx = p.clauses.len() as u32;
            for &l in lits.iter() {
                p.occ[l.0 as usize].push(idx);
                p.n_occ[l.0 as usize] += 1;
            }
            p.clauses.push(Clause {
                off: start as u32,
                len: w as u32,
                sig: sig_of(lits),
                deleted: false,
            });
            start = end;
        }
        p.data = data;
        p
    }

    /// Enable/disable the gate-substitution fallback (see the field doc).
    pub fn set_gate_substitution(&mut self, on: bool) {
        self.gate_subst = on;
    }

    /// Owned-clause constructor (tests / ad-hoc callers): flattens the
    /// soup and defers to [`Self::from_flat`].
    pub fn new(clauses: Vec<Vec<Lit>>, num_vars: usize, frozen: Vec<bool>) -> Self {
        let mut data = Vec::with_capacity(clauses.iter().map(|c| c.len()).sum());
        let mut ends = Vec::with_capacity(clauses.len());
        for c in &clauses {
            data.extend_from_slice(c);
            ends.push(data.len() as u32);
        }
        Self::from_flat(data, &ends, num_vars, frozen, PreprocessPool::default())
    }

    #[inline]
    fn lits(&self, ci: u32) -> &[Lit] {
        let c = &self.clauses[ci as usize];
        &self.data[c.off as usize..(c.off + c.len) as usize]
    }

    /// Run the full pipeline: unit propagation → subsumption fixpoint →
    /// equivalent-literal substitution → bounded VE (with local
    /// subsumption on resolvents) → final sweep.
    pub fn run(mut self) -> SimplifyResult {
        self.propagate_units();
        if !self.unsat {
            self.subsumption_pass();
        }
        if !self.unsat {
            self.eliminate_vars();
        }
        let mut out: Vec<(u32, u32)> = Vec::new();
        if self.unsat {
            out.push((0, 0));
        } else {
            for c in &self.clauses {
                if !c.deleted {
                    out.push((c.off, c.len));
                }
            }
            // Re-emit discovered units (propagate_units removes them from
            // clause form once applied). Appended to the arena tail.
            // Bounded by THIS batch's literal count — the pool arrays may
            // carry a longer stale tail from an earlier, bigger batch.
            for li in 0..2 * self.num_vars {
                if self.assign[li] == 1 {
                    let off = self.data.len() as u32;
                    self.data.push(Lit(li as u32));
                    out.push((off, 1));
                }
            }
        }
        let eliminated = self
            .eliminated
            .iter()
            .take(self.num_vars)
            .enumerate()
            .filter(|&(_, &e)| e)
            .map(|(v, _)| v as u32)
            .collect();
        SimplifyResult {
            data: self.data,
            clauses: out,
            eliminated,
            subsumed: self.stats_subsumed,
            strengthened: self.stats_strengthened,
            unsat: self.unsat,
            pool: PreprocessPool {
                occ: self.occ,
                n_occ: self.n_occ,
                assign: self.assign,
                eliminated: self.eliminated,
                clauses: self.clauses,
                unit_queue: self.unit_queue,
                touched: self.touched,
                sub_scratch: self.sub_scratch,
                res_scratch: self.res_scratch,
                sub_order: self.sub_order,
                sub_queued: self.sub_queued,
                ve_queued: self.ve_queued,
                ve_fail: self.ve_fail,
                ve_seed: self.ve_seed,
                frozen: self.frozen,
            },
        }
    }

    // ---------- unit propagation over the soup ----------

    fn propagate_units(&mut self) {
        while let Some(u) = self.unit_queue.pop() {
            let ui = u.0 as usize;
            let ni = (u.0 ^ 1) as usize;
            match self.assign[ui] {
                1 => continue,          // already true
                2 => {
                    self.unsat = true; // conflicting units
                    return;
                }
                _ => {}
            }
            self.assign[ui] = 1;
            self.assign[ni] = 2;

            // Clauses containing u are satisfied — delete. In-place index
            // walk (nothing in the loop body pushes to `occ`), cleared
            // after so the storage is reused instead of dropped.
            for i in 0..self.occ[ui].len() {
                let ci = self.occ[ui][i];
                self.delete_clause(ci);
            }
            self.occ[ui].clear();
            // Clauses containing ¬u lose that literal.
            let nu = Lit(u.0 ^ 1);
            for i in 0..self.occ[ni].len() {
                let ci = self.occ[ni][i];
                let c = &self.clauses[ci as usize];
                if c.deleted {
                    continue;
                }
                let (off, len) = (c.off as usize, c.len as usize);
                let range = &mut self.data[off..off + len];
                if let Some(pos) = range.iter().position(|&l| l == nu) {
                    // In-place removal: shift the tail left, shrink len.
                    range.copy_within(pos + 1.., pos);
                    let c = &mut self.clauses[ci as usize];
                    c.len -= 1;
                    self.n_occ[ni] = self.n_occ[ni].saturating_sub(1);
                    let lits =
                        &self.data[c.off as usize..(c.off + c.len) as usize];
                    let sig = sig_of(lits);
                    let new_len = lits.len();
                    let first = lits.first().copied();
                    self.clauses[ci as usize].sig = sig;
                    match new_len {
                        0 => {
                            self.unsat = true;
                            return;
                        }
                        1 => {
                            self.unit_queue.push(first.unwrap());
                            // The clause itself dissolves into the unit.
                            self.delete_clause(ci);
                        }
                        _ => {}
                    }
                }
            }
            self.occ[ni].clear();
        }
    }

    // ---------- subsumption / strengthening ----------

    /// `subsumes(c, d)`: `Ok(None)` if c ⊆ d; `Ok(Some(l))` if c "almost"
    /// subsumes d — every literal of c appears in d except one literal `l`
    /// of c whose *negation* appears in d (self-subsuming resolution: d can
    /// be strengthened by removing `¬l`). `Err(())` otherwise. Both clause
    /// lit slices are sorted by `Lit.0`.
    fn subsumes(c_lits: &[Lit], c_sig: u64, d_lits: &[Lit], d_sig: u64) -> Result<Option<Lit>, ()> {
        if c_lits.len() > d_lits.len() || (c_sig & !d_sig) != 0 {
            return Err(());
        }
        let mut flipped: Option<Lit> = None;
        let mut di = 0usize;
        'outer: for &cl in c_lits {
            while di < d_lits.len() {
                let dl = d_lits[di];
                if dl == cl {
                    di += 1;
                    continue 'outer;
                }
                if dl.0 == cl.0 ^ 1 {
                    // Polarity flip — allowed once.
                    if flipped.is_some() {
                        return Err(());
                    }
                    flipped = Some(cl);
                    di += 1;
                    continue 'outer;
                }
                if dl.0 > cl.0 {
                    return Err(()); // cl missing from d
                }
                di += 1;
            }
            return Err(());
        }
        Ok(flipped)
    }

    /// Backward subsumption: use each clause (short ones first) to delete /
    /// strengthen the clauses sharing its least-occurring literal.
    fn subsumption_pass(&mut self) {
        // Process shorter clauses first — they subsume more. Order and
        // queued-flag storage are pooled; the deque is a head cursor over
        // the pooled order buffer plus pushes at its tail (a clause
        // re-enters at most a bounded number of times after
        // strengthening, so the buffer stays O(batch)).
        let mut order = std::mem::take(&mut self.sub_order);
        order.clear();
        order.extend(
            (0..self.clauses.len() as u32)
                .filter(|&i| !self.clauses[i as usize].deleted),
        );
        order.sort_by_key(|&i| self.clauses[i as usize].len);
        let mut queue = order;
        let mut head = 0usize;
        let mut queued = std::mem::take(&mut self.sub_queued);
        queued.clear();
        queued.resize(self.clauses.len(), true);

        while head < queue.len() {
            let ci = queue[head];
            head += 1;
            if (ci as usize) < queued.len() {
                queued[ci as usize] = false;
            }
            if self.clauses[ci as usize].deleted {
                continue;
            }
            if self.clauses[ci as usize].len == 0 {
                self.unsat = true;
                return;
            }
            // Pick the literal of ci whose VARIABLE occurs least, and scan
            // both polarities' occurrence lists — a self-subsumption
            // candidate contains the *negation* of the flipped literal, so
            // a single-polarity scan would miss it (this is why MiniSat's
            // SimpSolver keys occurrences per-variable).
            let (best_lit, best_occ) = {
                let lits = self.lits(ci);
                let mut bl = lits[0];
                let mut bo =
                    self.n_occ[bl.0 as usize] + self.n_occ[(bl.0 ^ 1) as usize];
                for &l in &lits[1..] {
                    let o = self.n_occ[l.0 as usize] + self.n_occ[(l.0 ^ 1) as usize];
                    if o < bo {
                        bo = o;
                        bl = l;
                    }
                }
                (bl, bo as usize)
            };
            if best_occ > SUB_OCC_LIM {
                continue;
            }
            // Reused scratch: both polarities' occurrence lists, copied so
            // the loop body can mutate `occ` freely. Taken/returned around
            // the loop — no per-clause allocation.
            let mut candidates = std::mem::take(&mut self.sub_scratch);
            candidates.clear();
            candidates.extend_from_slice(&self.occ[best_lit.0 as usize]);
            candidates.extend_from_slice(&self.occ[(best_lit.0 ^ 1) as usize]);
            for &di in &candidates {
                if di == ci || self.clauses[di as usize].deleted {
                    continue;
                }
                if self.clauses[ci as usize].deleted {
                    break;
                }
                let verdict = {
                    let c = &self.clauses[ci as usize];
                    let d = &self.clauses[di as usize];
                    let c_lits = &self.data[c.off as usize..(c.off + c.len) as usize];
                    let d_lits = &self.data[d.off as usize..(d.off + d.len) as usize];
                    // No stale-entry pre-check needed: `subsumes` verifies
                    // against d's current literals, so a stale occurrence
                    // can only produce a (correct) Err.
                    Self::subsumes(c_lits, c.sig, d_lits, d.sig)
                };
                match verdict {
                    Err(()) => {}
                    Ok(None) => {
                        self.delete_clause(di);
                        self.stats_subsumed += 1;
                    }
                    Ok(Some(flip_lit)) => {
                        // Strengthen d: remove ¬flip_lit.
                        let removed = Lit(flip_lit.0 ^ 1);
                        let d = &self.clauses[di as usize];
                        let (off, len) = (d.off as usize, d.len as usize);
                        let range = &mut self.data[off..off + len];
                        if let Some(pos) = range.iter().position(|&l| l == removed) {
                            range.copy_within(pos + 1.., pos);
                            let d = &mut self.clauses[di as usize];
                            d.len -= 1;
                            let lits =
                                &self.data[d.off as usize..(d.off + d.len) as usize];
                            let sig = sig_of(lits);
                            let new_len = lits.len();
                            let first = lits.first().copied();
                            self.clauses[di as usize].sig = sig;
                            self.n_occ[removed.0 as usize] =
                                self.n_occ[removed.0 as usize].saturating_sub(1);
                            self.stats_strengthened += 1;
                            match new_len {
                                0 => {
                                    self.unsat = true;
                                    return;
                                }
                                1 => {
                                    self.unit_queue.push(first.unwrap());
                                    self.delete_clause(di);
                                    self.propagate_units();
                                    if self.unsat {
                                        return;
                                    }
                                }
                                _ => {
                                    if !queued[di as usize] {
                                        queued[di as usize] = true;
                                        queue.push(di);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            self.sub_scratch = candidates;
        }
        queue.clear();
        self.sub_order = queue;
        self.sub_queued = queued;
    }

    // ---------- bounded variable elimination ----------

    fn eliminate_vars(&mut self) {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let num_vars = self.frozen.len();
        // Min-heap of (occurrence count, var) — cheapest first, var index
        // as deterministic tie-break. Vars touched by an elimination
        // re-enter the queue, so cascades (gate chains dissolving
        // end-to-end) are handled without global re-scans. Three guards
        // keep the worklist near-linear:
        //   - `queued` flag: at most one live heap entry per var.
        //   - touched set deduped per drain.
        //   - `fail_cost`: a var that failed elimination at cost k is only
        //     retried once its cost drops strictly below k (elimination
        //     can only become *easier* as the var's neighbourhood shrinks;
        //     retrying at unchanged cost re-runs the same doomed
        //     resolution enumeration).
        let cost = |p: &Self, v: u32| {
            p.n_occ[(2 * v) as usize] + p.n_occ[(2 * v + 1) as usize]
        };
        let mut queued = std::mem::take(&mut self.ve_queued);
        queued.clear();
        queued.resize(num_vars, false);
        let mut fail_cost = std::mem::take(&mut self.ve_fail);
        fail_cost.clear();
        fail_cost.resize(num_vars, u32::MAX);
        // Seed by bulk heapify — O(n) instead of n pushes' O(n log n).
        // Pop order is unaffected: keys are distinct (var index breaks
        // every tie), and a binary heap with a total order pops in
        // exactly sorted order whatever its internal layout.
        let mut seed = std::mem::take(&mut self.ve_seed);
        seed.clear();
        for v in 0..num_vars as u32 {
            if !self.frozen[v as usize] && !self.eliminated[v as usize] {
                let c = cost(self, v);
                if c > 0 {
                    seed.push(Reverse((c, v)));
                    queued[v as usize] = true;
                }
            }
        }
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::from(seed);

        self.touched.clear();
        while let Some(Reverse((c, v))) = heap.pop() {
            queued[v as usize] = false;
            if self.frozen[v as usize] || self.eliminated[v as usize] {
                continue;
            }
            let cur = cost(self, v);
            if cur == 0 {
                continue; // no occurrences left — nothing to eliminate
            }
            if cur != c {
                // Stale cost — reprioritize (single entry, flag re-set).
                if !queued[v as usize] {
                    heap.push(Reverse((cur, v)));
                    queued[v as usize] = true;
                }
                continue;
            }
            if cur >= fail_cost[v as usize] {
                continue; // nothing changed since the last failed attempt
            }
            if self.try_eliminate(v) {
                if self.unsat {
                    return;
                }
                let mut touched = std::mem::take(&mut self.touched);
                touched.sort_unstable();
                touched.dedup();
                for t in touched.drain(..) {
                    if self.frozen[t as usize]
                        || self.eliminated[t as usize]
                        || queued[t as usize]
                    {
                        continue;
                    }
                    let tc = cost(self, t);
                    if tc > 0 && tc < fail_cost[t as usize] {
                        heap.push(Reverse((tc, t)));
                        queued[t as usize] = true;
                    }
                }
                self.touched = touched;
            } else {
                fail_cost[v as usize] = cur;
                self.touched.clear();
            }
        }
        self.ve_seed = heap.into_vec();
        self.ve_queued = queued;
        self.ve_fail = fail_cost;
    }

    /// Attempt to eliminate variable `v` by resolution. Succeeds iff every
    /// resolvent is within `CLAUSE_LIM` and the number of non-tautological
    /// resolvents does not exceed the number of clauses removed.
    fn try_eliminate(&mut self, v: u32) -> bool {
        let pl = Lit(2 * v);
        let nl = Lit(2 * v + 1);

        // Collect live clause indices for each polarity, taking the
        // occurrence vectors out of `occ` (no clone). Failure paths put
        // them back — compacted, which is exactly what the historical
        // `live_occ` left behind; on success v has no live clauses, so
        // the lists stay empty.
        let pos = self.live_occ_take(pl);
        let neg = self.live_occ_take(nl);
        if pos.is_empty() && neg.is_empty() {
            self.occ_put_back(pl, pos, nl, neg);
            return false;
        }
        // Pure literal: no resolvents at all — every clause containing v
        // can be satisfied by picking v's polarity. Safe to remove for
        // non-frozen (invisible) vars.
        if pos.is_empty() || neg.is_empty() {
            for &ci in pos.iter().chain(neg.iter()) {
                self.delete_clause(ci);
            }
            self.eliminated[v as usize] = true;
            return true;
        }

        // Plan A — full VE, bit-identical to the historical behaviour:
        // every variable the old pass eliminated is still eliminated with
        // the exact same resolvent set (env × env included; those implied
        // resolvents are lean to drop but propagation-useful to keep —
        // dropping them for already-accepted pivots measurably hurt the
        // Sage2 family).
        // (Split diagnosis, 2026-08-05: restricting plan B to
        // product-limit failures only reproduced the pre-substitution
        // solver bit-for-bit — every substitution that fires, for better
        // AND worse, unlocks a pivot that failed the non-increasing
        // clause-count bound, not the product screen. Heavily-shared
        // gates essentially never satisfy the count bound even def×env.)
        let full_ok = pos.len() * neg.len() <= VE_PRODUCT_LIM
            && self.count_resolvents(&pos, &neg, v, |_, _| true);
        if full_ok {
            return self.commit_elimination(pos, neg, v, |_, _| true);
        }

        // Plan B — gate-driven substitution (Eén-Biere), only where full
        // VE failed its bounds: when v is a Tseitin gate output whose
        // definition clauses are all present (AND/OR: long clause +
        // matching binaries), the resolvents that carry information are
        // definition × environment; environment × environment resolvents
        // are implied by them. The reduced pair set both bypasses the
        // product blow-up on shared gate outputs and passes the
        // non-increasing bound far more often, so strictly more variables
        // get eliminated than before.
        //
        // Cheapest-possible budget screen before paying for detection:
        // any definition has ≥ 1 clause per side, so substitution always
        // enumerates ≥ pos + neg pairs.
        if !self.gate_subst || pos.len() + neg.len() > VE_SUB_PAIR_LIM {
            self.occ_put_back(pl, pos, nl, neg);
            return false;
        }
        let Some(gate) = self.find_gate(v, &pos, &neg) else {
            self.occ_put_back(pl, pos, nl, neg);
            return false;
        };
        let sub_pairs =
            gate.n_def_pos * neg.len() + pos.len() * gate.n_def_neg;
        if sub_pairs > VE_SUB_PAIR_LIM {
            self.occ_put_back(pl, pos, nl, neg);
            return false;
        }
        let allowed = |pi_idx: usize, ni_idx: usize| -> bool {
            gate.def_pos[pi_idx] || gate.def_neg[ni_idx]
        };
        if !self.count_resolvents(&pos, &neg, v, allowed) {
            self.occ_put_back(pl, pos, nl, neg);
            return false;
        }
        self.commit_elimination(pos, neg, v, allowed)
    }

    /// Detect an AND/OR-shaped Tseitin definition for pivot `v`: a
    /// clause `D` of length ≥ 3 containing `v` on one side such that
    /// every other literal `l ∈ D` has the binary `{¬v-side-lit, ¬l}` on
    /// the opposite side — `x = AND(a, b, …)` is `D = (x ∨ ā ∨ b̄ …)`
    /// plus `(x̄ ∨ a), (x̄ ∨ b), …`, and OR is the mirror image. The
    /// first qualifying `D` (deterministic scan order) wins.
    ///
    /// AND/OR only, on purpose. XOR and MUX definitions were both built
    /// and measured (2026-08-05): their substitutions have no absorbing
    /// polarity, so every def×env resolvent keeps both definition
    /// literals and the CNF strictly widens through adder chains
    /// (XOR: bench_6554 conflicts 20k → 41k) and barrel-shifter mux
    /// trees (MUX: bench_16728 2.1s → 19s clause-count-bounded, 54s
    /// even with a non-widening total-literal budget). AND/OR
    /// substitution replaces the pivot with one input literal per
    /// binary — width-neutral — and is the variant the corpus validated
    /// at −14.9% with bench_16728 newly solved.
    fn find_gate(&self, v: u32, pos: &[u32], neg: &[u32]) -> Option<GateDef> {
        if let Some(g) = self.find_and_gate(v, pos, neg, true) {
            return Some(g);
        }
        self.find_and_gate(v, neg, pos, false)
    }

    fn find_and_gate(
        &self,
        v: u32,
        with_x: &[u32],
        with_nx: &[u32],
        x_is_pos: bool,
    ) -> Option<GateDef> {
        // Index the opposite side's binaries by their non-pivot literal.
        // Lists are pre-screened against VE_SUB_PAIR_LIM, so a linear
        // probe per D literal is fine; build once per side.
        let mut bin_other: Vec<(u32, usize)> = Vec::new();
        for (i, &ci) in with_nx.iter().enumerate() {
            let lits = self.lits(ci);
            if lits.len() == 2 {
                let other = if lits[0].var_idx() == v as usize {
                    lits[1]
                } else {
                    lits[0]
                };
                bin_other.push((other.0, i));
            }
        }
        if bin_other.is_empty() {
            return None;
        }
        'candidate: for (di, &ci) in with_x.iter().enumerate() {
            let lits = self.lits(ci);
            if lits.len() < 3 {
                continue;
            }
            let mut bin_mask = vec![false; with_nx.len()];
            let mut bin_count = 0usize;
            for &l in lits {
                if l.var_idx() == v as usize {
                    continue;
                }
                let want = l.0 ^ 1;
                match bin_other.iter().find(|&&(o, _)| o == want) {
                    Some(&(_, bi)) => {
                        bin_mask[bi] = true;
                        bin_count += 1;
                    }
                    None => continue 'candidate,
                }
            }
            let mut d_mask = vec![false; with_x.len()];
            d_mask[di] = true;
            return Some(if x_is_pos {
                GateDef {
                    def_pos: d_mask,
                    def_neg: bin_mask,
                    n_def_pos: 1,
                    n_def_neg: bin_count,
                }
            } else {
                GateDef {
                    def_pos: bin_mask,
                    def_neg: d_mask,
                    n_def_pos: bin_count,
                    n_def_neg: 1,
                }
            });
        }
        None
    }

    /// Counting pre-pass shared by both elimination plans: same merge
    /// walk as `resolve_into`, computing only tautology + length,
    /// restricted to `allowed` pairs. True iff every resolvent is within
    /// `CLAUSE_LIM` and the count stays non-increasing. Allocates nothing.
    fn count_resolvents(
        &self,
        pos: &[u32],
        neg: &[u32],
        v: u32,
        allowed: impl Fn(usize, usize) -> bool,
    ) -> bool {
        let limit = pos.len() + neg.len();
        let mut count = 0usize;
        for (pi_idx, &pi) in pos.iter().enumerate() {
            for (ni_idx, &ni) in neg.iter().enumerate() {
                if !allowed(pi_idx, ni_idx) {
                    continue;
                }
                match Self::resolve_size(self.lits(pi), self.lits(ni), v) {
                    None => {}
                    Some(len) => {
                        if len > CLAUSE_LIM {
                            return false;
                        }
                        count += 1;
                        if count > limit {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Commit an accepted elimination: remove the originals, materialize
    /// and add each `allowed` resolvent (deletion only flags — the
    /// literal ranges survive for the resolution reads), then cascade
    /// units. Always returns true (the variable is gone).
    fn commit_elimination(
        &mut self,
        mut pos: Vec<u32>,
        mut neg: Vec<u32>,
        v: u32,
        allowed: impl Fn(usize, usize) -> bool,
    ) -> bool {
        for &ci in pos.iter().chain(neg.iter()) {
            self.delete_clause(ci);
        }
        self.eliminated[v as usize] = true;
        'outer: for (pi_idx, &pi) in pos.iter().enumerate() {
            for (ni_idx, &ni) in neg.iter().enumerate() {
                if !allowed(pi_idx, ni_idx) {
                    continue;
                }
                if self.resolve_into_scratch(pi, ni, v) {
                    self.add_clause_from_scratch();
                    if self.unsat {
                        break 'outer;
                    }
                }
            }
        }
        // Return the (now dead) occurrence buffers to their slots, empty:
        // semantically identical to leaving the taken-out slots' fresh
        // Vecs — v has no live clauses either way — but the capacity
        // survives for the pool's next flush instead of being freed once
        // per elimination (~2 × pp_elim allocator round-trips per batch).
        pos.clear();
        neg.clear();
        self.occ[(2 * v) as usize] = pos;
        self.occ[(2 * v + 1) as usize] = neg;
        if self.unsat {
            return true;
        }
        // Resolvent units cascade.
        self.propagate_units();
        true
    }

    /// Resolve clauses `pi` (contains v) and `ni` (contains ¬v) on `v`
    /// into the reused scratch buffer. Returns false for tautological
    /// resolvents (scratch contents then undefined).
    fn resolve_into_scratch(&mut self, pi: u32, ni: u32, v: u32) -> bool {
        let Preprocessor { data, clauses, res_scratch, .. } = self;
        let c = &clauses[pi as usize];
        let d = &clauses[ni as usize];
        let c_lits = &data[c.off as usize..(c.off + c.len) as usize];
        let d_lits = &data[d.off as usize..(d.off + d.len) as usize];
        res_scratch.clear();
        // Merge two sorted lists, dropping the pivot, rejecting tautologies.
        let (mut i, mut j) = (0usize, 0usize);
        while i < c_lits.len() || j < d_lits.len() {
            let next = match (c_lits.get(i), d_lits.get(j)) {
                (Some(&a), Some(&b)) => {
                    if a.0 <= b.0 {
                        i += 1;
                        a
                    } else {
                        j += 1;
                        b
                    }
                }
                (Some(&a), None) => {
                    i += 1;
                    a
                }
                (None, Some(&b)) => {
                    j += 1;
                    b
                }
                (None, None) => break,
            };
            if next.var_idx() == v as usize {
                continue; // pivot
            }
            match res_scratch.last() {
                Some(&prev) if prev == next => continue, // duplicate
                Some(&prev) if prev.var() == next.var() => return false, // taut
                _ => res_scratch.push(next),
            }
        }
        true
    }

    /// Size-only twin of [`Self::resolve_into_scratch`]: identical merge
    /// walk and tautology handling, but returns the resolvent length
    /// instead of building it. `None` for tautologies.
    fn resolve_size(c_lits: &[Lit], d_lits: &[Lit], v: u32) -> Option<usize> {
        let mut len = 0usize;
        let mut last: Option<Lit> = None;
        let (mut i, mut j) = (0usize, 0usize);
        while i < c_lits.len() || j < d_lits.len() {
            let next = match (c_lits.get(i), d_lits.get(j)) {
                (Some(&a), Some(&b)) => {
                    if a.0 <= b.0 {
                        i += 1;
                        a
                    } else {
                        j += 1;
                        b
                    }
                }
                (Some(&a), None) => {
                    i += 1;
                    a
                }
                (None, Some(&b)) => {
                    j += 1;
                    b
                }
                (None, None) => break,
            };
            if next.var_idx() == v as usize {
                continue; // pivot
            }
            match last {
                Some(prev) if prev == next => continue, // duplicate
                Some(prev) if prev.var() == next.var() => return None, // taut
                _ => {
                    last = Some(next);
                    len += 1;
                }
            }
        }
        Some(len)
    }

    // ---------- bookkeeping ----------

    /// Verified live occurrences of `l`, TAKEN out of the occurrence
    /// table: the stored list is compacted (stale entries from deletion /
    /// strengthening dropped) and moved to the caller; `occ[l]` is left
    /// empty. Callers on failure paths must return it via
    /// [`Self::occ_put_back`]; elimination success leaves the slot empty
    /// (correct — the variable has no live clauses).
    ///
    /// No dedup needed: each clause pushes to a literal's occurrence list
    /// exactly once at creation (its literals are distinct), so the list
    /// never holds a duplicate index.
    fn live_occ_take(&mut self, l: Lit) -> Vec<u32> {
        let li = l.0 as usize;
        let mut list = std::mem::take(&mut self.occ[li]);
        let Preprocessor { data, clauses, .. } = self;
        list.retain(|&ci| {
            let c = &clauses[ci as usize];
            if c.deleted {
                return false;
            }
            let lits = &data[c.off as usize..(c.off + c.len) as usize];
            lits.binary_search_by_key(&l.0, |x| x.0).is_ok()
        });
        list
    }

    /// Return both polarity lists taken by [`Self::live_occ_take`] after a
    /// failed elimination attempt — the variable's clauses are still live
    /// and later passes (unit propagation over the soup in particular)
    /// must find them through `occ`.
    fn occ_put_back(&mut self, pl: Lit, pos: Vec<u32>, nl: Lit, neg: Vec<u32>) {
        debug_assert!(self.occ[pl.0 as usize].is_empty());
        debug_assert!(self.occ[nl.0 as usize].is_empty());
        self.occ[pl.0 as usize] = pos;
        self.occ[nl.0 as usize] = neg;
    }

    fn delete_clause(&mut self, ci: u32) {
        let c = &mut self.clauses[ci as usize];
        if c.deleted {
            return;
        }
        c.deleted = true;
        let (off, len) = (c.off as usize, c.len as usize);
        for i in off..off + len {
            let l = self.data[i];
            self.n_occ[l.0 as usize] = self.n_occ[l.0 as usize].saturating_sub(1);
            self.touched.push(l.var_idx() as u32);
        }
    }

    /// Append the scratch resolvent (already sorted ascending and
    /// duplicate-free) to the arena as a fresh clause.
    fn add_clause_from_scratch(&mut self) {
        match self.res_scratch.len() {
            0 => {
                self.unsat = true;
                return;
            }
            1 => {
                self.unit_queue.push(self.res_scratch[0]);
                return;
            }
            _ => {}
        }
        debug_assert!(self.res_scratch.windows(2).all(|w| w[0].0 < w[1].0));
        let idx = self.clauses.len() as u32;
        let off = self.data.len() as u32;
        let len = self.res_scratch.len() as u32;
        let sig = sig_of(&self.res_scratch);
        self.data.extend_from_slice(&self.res_scratch);
        for i in 0..self.res_scratch.len() {
            let l = self.res_scratch[i];
            self.occ[l.0 as usize].push(idx);
            self.n_occ[l.0 as usize] += 1;
            self.touched.push(l.var_idx() as u32);
        }
        self.clauses.push(Clause { off, len, sig, deleted: false });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lit::Var;

    fn lit(v: u32, neg: bool) -> Lit {
        Lit::new(Var(v), neg)
    }

    fn run(clauses: Vec<Vec<Lit>>, num_vars: usize, frozen_vars: &[u32]) -> SimplifyResult {
        let mut frozen = vec![false; num_vars];
        for &v in frozen_vars {
            frozen[v as usize] = true;
        }
        Preprocessor::new(clauses, num_vars, frozen).run()
    }

    #[test]
    fn subsumption_removes_superset() {
        // {a} subsumes {a, b}; a frozen so VE doesn't fire on it.
        let r = run(
            vec![vec![lit(0, false)], vec![lit(0, false), lit(1, false)]],
            2,
            &[0, 1],
        );
        assert!(!r.unsat);
        // Only the unit {a} survives.
        assert_eq!(r.clause_vecs(), vec![vec![lit(0, false)]]);
    }

    #[test]
    fn self_subsumption_strengthens() {
        // {a, b} + {¬a, b, c} → strengthen to {b, c}.
        let r = run(
            vec![
                vec![lit(0, false), lit(1, false)],
                vec![lit(0, true), lit(1, false), lit(2, false)],
            ],
            3,
            &[0, 1, 2],
        );
        assert!(!r.unsat);
        assert!(r.clause_vecs().contains(&vec![lit(1, false), lit(2, false)]));
        assert_eq!(r.strengthened, 1);
    }

    #[test]
    fn gate_variable_is_eliminated() {
        // Tseitin AND: o ↔ a∧b, with o used once: (¬o ∨ x).
        // o is unfrozen — VE should resolve it away entirely.
        let o = 2u32;
        let clauses = vec![
            vec![lit(0, true), lit(1, true), lit(o, false)],
            vec![lit(0, false), lit(o, true)],
            vec![lit(1, false), lit(o, true)],
            vec![lit(o, true), lit(3, false)],
        ];
        let r = run(clauses, 4, &[0, 1, 3]);
        assert!(!r.unsat);
        assert!(r.eliminated.contains(&o));
        for c in &r.clause_vecs() {
            assert!(c.iter().all(|l| l.var_idx() != o as usize));
        }
    }

    #[test]
    fn unit_propagation_applies() {
        // {a}, {¬a, b} → both dissolve into units a, b.
        let r = run(
            vec![vec![lit(0, false)], vec![lit(0, true), lit(1, false)]],
            2,
            &[0, 1],
        );
        assert!(!r.unsat);
        let mut units: Vec<Vec<Lit>> = r.clause_vecs();
        units.sort();
        assert_eq!(units, vec![vec![lit(0, false)], vec![lit(1, false)]]);
    }

    #[test]
    fn conflicting_units_unsat() {
        let r = run(vec![vec![lit(0, false)], vec![lit(0, true)]], 1, &[0]);
        assert!(r.unsat);
        assert_eq!(r.clause_vecs(), vec![Vec::<Lit>::new()]);
    }

    #[test]
    fn frozen_vars_survive() {
        // Same gate shape but o frozen — nothing eliminated.
        let o = 2u32;
        let clauses = vec![
            vec![lit(0, true), lit(1, true), lit(o, false)],
            vec![lit(0, false), lit(o, true)],
            vec![lit(1, false), lit(o, true)],
        ];
        let r = run(clauses, 3, &[0, 1, 2]);
        assert!(r.eliminated.is_empty());
        assert_eq!(r.clause_vecs().len(), 3);
    }

    /// Helper: literals as (var, negated) pairs into clause vectors.
    fn cl(lits: &[(u32, bool)]) -> Vec<Lit> {
        lits.iter().map(|&(v, n)| lit(v, n)).collect()
    }

    #[test]
    fn and_gate_substitution_when_full_ve_fails() {
        // x(=v0) = AND(a(v1), b(v2)): D = (x ∨ ā ∨ b̄), binaries (x̄∨a)(x̄∨b).
        // Environment: 2 pos, 2 neg clauses over fresh frozen vars — full
        // VE counts f + 2e + ef = 10 > limit 7, substitution counts
        // f + 2e = 6 ≤ 7, so only the gate path eliminates x.
        let x = 0u32;
        let clauses = vec![
            cl(&[(x, false), (1, true), (2, true)]),  // D
            cl(&[(x, true), (1, false)]),             // bin a
            cl(&[(x, true), (2, false)]),             // bin b
            cl(&[(x, false), (3, false), (4, false)]),
            cl(&[(x, false), (5, false), (6, false)]),
            cl(&[(x, true), (7, false), (8, false)]),
            cl(&[(x, true), (9, false), (10, false)]),
        ];
        let r = run(clauses, 11, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert!(!r.unsat);
        assert!(r.eliminated.contains(&x), "gate path must eliminate x");
        // env×env resolvent {3,4,7,8} must NOT be present (implied).
        let envenv = cl(&[(3, false), (4, false), (7, false), (8, false)]);
        assert!(
            !r.clause_vecs().iter().any(|c| {
                let mut s = c.clone();
                s.sort_by_key(|l| l.0);
                s == envenv
            }),
            "environment×environment resolvent should be dropped"
        );
        // def×env resolvent (ā ∨ b̄ ∨ 7 ∨ 8) must be present.
        let defenv = {
            let mut s = cl(&[(1, true), (2, true), (7, false), (8, false)]);
            s.sort_by_key(|l| l.0);
            s
        };
        assert!(
            r.clause_vecs().iter().any(|c| {
                let mut s = c.clone();
                s.sort_by_key(|l| l.0);
                s == defenv
            }),
            "definition×environment resolvent must survive"
        );
    }

    #[test]
    fn pure_literal_clauses_dropped() {
        // v=2 occurs only positively and is unfrozen → its clauses vanish.
        let clauses = vec![
            vec![lit(0, false), lit(2, false)],
            vec![lit(1, false), lit(2, false)],
        ];
        let r = run(clauses, 3, &[0, 1]);
        assert!(r.eliminated.contains(&2));
        assert!(r.clause_vecs().is_empty());
    }
}
