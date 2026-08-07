//! Bounded FRAIG sweep: prove sim-equivalence candidates really equivalent
//! and merge them in the AIG before CNF emission.
//!
//! Pipeline position: runs at flush time, after the batch is bitblasted
//! into the AIG but before any of it is materialized to CNF (see
//! `SmtSolver::flush_pending`). Candidates come from multi-word random
//! simulation (same signatures as `Aig::sim_sweep`); each candidate pair is
//! confirmed by two conflict-budgeted SAT queries over a *scratch* solver
//! that encodes the nodes' cones with free inputs. Free inputs make the
//! proved equivalence unconditional — valid under any later assertions and
//! across push/pop scopes — so merging via `Aig::merge_equiv` is globally
//! sound.
//!
//! Deliberately bounded, prototype-grade:
//! - The sim-constant bucket (nodes all-0/all-1 over every sample) is NOT
//!   checked: on symbex workloads it is dominated by comparators that are
//!   almost-never-true under random inputs, i.e. satisfiable non-constants
//!   that would burn a SAT query each to disprove.
//! - Counterexample refinement is intra-class only: a disproof's SAT model
//!   is replayed over the remaining class members (free inputs completed
//!   with `false`) and everyone who visibly differs from the rep is dropped
//!   without spending queries. Dropped members are NOT re-classed among
//!   themselves (full FRAIG would), so some genuine merges are missed.
//! - Only nodes at index >= `start_idx` (new since the last sweep) are
//!   merge candidates, so incremental sessions pay per new batch, not per
//!   total accumulated AIG. Old nodes still serve as representatives —
//!   merging a new duplicate into an already-materialized old node is the
//!   best case (its SAT lit gets reused outright).
//!
//! ## Why this is off by default — and why optimizing it is not the answer
//!
//! Measured 2026-08-06. FRAIG's *relative* numbers are the best in the
//! codebase: on a matched subset of the symbex corpus it cuts original
//! clauses 8.7%, conflicts 33.9% and SAT-search time 45.9%, and on single
//! instances it removes ~88% of the CNF. Those percentages are a trap.
//! The absolute figures:
//!
//! | instance | full solve, no FRAIG | this sweep alone |
//! |---|---|---|
//! | bench_2467 | 0.02s | 17.0s |
//! | bench_4933 | 0.159s | 30.6s |
//! | bench_4351 | 0.319s | 23.8s |
//! | bench_13728 | 0.678s | 49.9s |
//! | bench_5906 | 1.635s | >200s (timeout) |
//!
//! The prize is a fraction of a SAT phase that lasts 0.0-1.2s, so the
//! whole theoretical saving is ~0.05-0.5s against a 17-50s sweep. That is
//! a 2-3 order of magnitude gap, and no constant-factor work closes it:
//!
//! - Removing the allocation-level waste (SipHash → FxHashMap,
//!   per-counterexample `HashMap` → epoch-stamped dense memo,
//!   `add_clause(vec![])` → `add_clause_from_slice`) bought 3-10%, with
//!   decisions bit-identical. Those fixes are kept.
//! - Replacing the bit-serial counterexample replay with a bit-parallel
//!   64-vector batch (`Aig::simulate`-style) was built and measured: 28%
//!   faster on bench_2467, 24% on bench_4351, but 32% SLOWER on
//!   bench_4933 (deferring pruning halved `cex_pruned`, 1949 → 949, and
//!   cost ~900 extra queries), and it proved slightly FEWER equivalences.
//!   Reverted — it reshuffles tens of seconds where sub-second is needed.
//!
//! The cost is dominated by the bounded SAT queries themselves (time
//! tracks query count closely across configurations), not by anything a
//! tighter inner loop fixes. FRAIG would need a fundamentally cheaper
//! equivalence oracle to matter here.
//!
//! Where it could still conceivably pay: an instance whose SAT phase is
//! tens of seconds, so that a 46% cut is worth more than the sweep. That
//! is the nobranch shape — but the sweep does not finish there either, so
//! it remains untested rather than promising.

use rustc_hash::FxHashMap as HashMap;

use crate::aig::{Aig, AigNode, AigRef};
use crate::lit::Lit;
use crate::solver::{SolveResult, Solver};

/// Signature width in 64-bit words (256 samples). Matches `Aig::sim_sweep`.
const SIM_WORDS: usize = 4;

#[derive(Copy, Clone, Debug, Default)]
pub struct FraigStats {
    /// Candidate (rep, member) pairs eligible for checking this sweep.
    pub candidates: u64,
    /// Pairs proven equivalent and merged.
    pub proven: u64,
    /// Pairs refuted by a SAT model (sim collision on unsampled inputs).
    pub disproven: u64,
    /// Pairs abandoned: per-query conflict budget or global query cap hit.
    pub skipped: u64,
    /// Members dropped for free by counterexample replay — inequivalent to
    /// their rep on a witnessed input, no SAT query spent.
    pub cex_pruned: u64,
    /// Total bounded SAT queries issued (up to 2 per pair).
    pub queries: u64,
}

impl FraigStats {
    pub fn accumulate(&mut self, o: FraigStats) {
        self.candidates += o.candidates;
        self.proven += o.proven;
        self.disproven += o.disproven;
        self.skipped += o.skipped;
        self.cex_pruned += o.cex_pruned;
        self.queries += o.queries;
    }
}

/// Run one bounded sweep over `aig`, merging proven-equivalent AND nodes.
/// `start_idx`: only nodes at or above this index are merged (see module
/// docs). `max_queries`: global SAT-query cap for this sweep.
/// `conflicts_per_query`: per-query conflict budget (>= 1; a query that
/// exhausts it counts as "skipped", never as an answer).
pub fn sweep(
    aig: &mut Aig,
    start_idx: u32,
    max_queries: u64,
    conflicts_per_query: u64,
    seed: u64,
) -> FraigStats {
    let mut stats = FraigStats::default();

    // --- 1. Simulate and bucket by canonical signature. ---
    let rounds: Vec<Vec<u64>> = (0..SIM_WORDS)
        .map(|w| aig.simulate(seed.wrapping_add(w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .collect();

    // node idx -> (canonical sig, phase). Phase = true means the node's raw
    // signature was complemented to canonicalize; two same-bucket nodes with
    // differing phases are candidates for `a ≡ ¬b`.
    let sig_of = |idx: usize| -> ([u64; SIM_WORDS], bool) {
        let mut sig = [0u64; SIM_WORDS];
        for w in 0..SIM_WORDS {
            sig[w] = rounds[w][idx];
        }
        let phase = sig[0] & 1 != 0;
        if phase {
            for s in sig.iter_mut() {
                *s = !*s;
            }
        }
        (sig, phase)
    };

    // Buckets hold (node_idx, phase) in ascending idx order (we iterate the
    // arena in order), so members[0] is always the oldest node — the rep.
    let mut buckets: HashMap<[u64; SIM_WORDS], Vec<(u32, bool)>> = HashMap::default();
    for idx in 1..aig.num_nodes() {
        if !matches!(aig.node(idx as u32), AigNode::And(..)) {
            continue;
        }
        let (sig, phase) = sig_of(idx);
        if sig.iter().all(|&s| s == 0) {
            continue; // sim-constant bucket — see module docs
        }
        buckets.entry(sig).or_default().push((idx as u32, phase));
    }

    // --- 2. Check members against their class rep, oldest classes first
    // (deterministic order; also biases budget toward shallow cones). ---
    let mut classes: Vec<Vec<(u32, bool)>> =
        buckets.into_values().filter(|v| v.len() >= 2).collect();
    classes.sort_by_key(|v| v[0].0);

    let mut enc = ScratchEncoder::new(aig.num_nodes());
    let mut memo = EvalMemo::new(aig.num_nodes());
    for class in classes {
        let (rep, rep_phase) = class[0];
        let mut members: Vec<(u32, bool)> = class[1..].to_vec();
        let mut i = 0;
        while i < members.len() {
            let (m, m_phase) = members[i];
            i += 1;
            if m < start_idx {
                continue; // pre-existing node; was a candidate in a past sweep
            }
            stats.candidates += 1;
            if stats.queries >= max_queries {
                stats.skipped += 1;
                continue;
            }
            let rep_lit = enc.lit_for(aig, rep);
            let m_lit = enc.lit_for(aig, m);
            // Same-phase members should be equal; cross-phase complementary.
            let m_lit = if rep_phase != m_phase { !m_lit } else { m_lit };

            // Equivalence = both difference directions UNSAT. Bounded: a
            // budget exhaustion (None) proves nothing — skip the pair.
            let mut counterexample = false;
            stats.queries += 1;
            match enc
                .solver
                .solve_under_assumptions_bounded(&[rep_lit, !m_lit], conflicts_per_query)
            {
                Some(SolveResult::Unsat) => {
                    stats.queries += 1;
                    match enc.solver.solve_under_assumptions_bounded(
                        &[!rep_lit, m_lit],
                        conflicts_per_query,
                    ) {
                        Some(SolveResult::Unsat) => {
                            let target = AigRef::from_parts(rep, rep_phase != m_phase);
                            aig.merge_equiv(m, target);
                            stats.proven += 1;
                            // merge_equiv rewrote `m`'s definition, but the
                            // scratch encoding (from the old structure)
                            // denotes the same — proven — function, so
                            // `enc` stays valid.
                        }
                        Some(SolveResult::Sat) => {
                            stats.disproven += 1;
                            counterexample = true;
                        }
                        None => stats.skipped += 1,
                    }
                }
                Some(SolveResult::Sat) => {
                    stats.disproven += 1;
                    counterexample = true;
                }
                None => stats.skipped += 1,
            }

            // Counterexample replay: the SAT model is a concrete input
            // vector on which rep and m differ. Evaluate the remaining
            // members on it (free/unencoded inputs completed with `false`)
            // and drop everyone who differs from rep — each is thereby
            // witnessed inequivalent, no query needed. This is what keeps
            // near-miss families (functions differing only on rare inputs)
            // from costing two queries per member.
            if counterexample && i < members.len() {
                memo.reset();
                let rv = eval_under_model(aig, rep, &enc, &mut memo);
                let mut w = i;
                for j in i..members.len() {
                    let (mj, pj) = members[j];
                    let mv = eval_under_model(aig, mj, &enc, &mut memo);
                    if mv == (rv ^ (pj != rep_phase)) {
                        members[w] = members[j];
                        w += 1;
                    } else {
                        stats.cex_pruned += 1;
                    }
                }
                members.truncate(w);
            }
        }
    }
    stats
}

/// Evaluate node `idx` on the input vector witnessed by the scratch
/// solver's current SAT model. Encoded inputs read their model value;
/// inputs the scratch solver never saw are completed with `false` (any
/// completion yields a legitimate concrete vector). Iterative; `memo` is
/// shared across members within one counterexample.
/// Per-counterexample evaluation memo over AIG node indices.
///
/// Dense and epoch-stamped rather than a hash map: node indices are a
/// dense space, the replay walks the same cones repeatedly, and a fresh
/// `HashMap` per counterexample meant re-growing a table from empty
/// thousands of times per sweep — profiling showed the sweep dominated
/// by `reserve_rehash` and SipHash, not by the SAT queries it exists to
/// avoid. Bumping the epoch resets the whole memo in O(1).
struct EvalMemo {
    val: Vec<bool>,
    stamp: Vec<u32>,
    epoch: u32,
    /// Reused DFS stack (was a fresh `Vec` per evaluated node).
    stack: Vec<u32>,
}

impl EvalMemo {
    fn new(num_nodes: usize) -> Self {
        EvalMemo {
            val: vec![false; num_nodes],
            stamp: vec![0; num_nodes],
            epoch: 0,
            stack: Vec::new(),
        }
    }
    /// Invalidate every entry. O(1); on wraparound, hard-clear.
    fn reset(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }
    #[inline]
    fn get(&self, idx: u32) -> Option<bool> {
        let i = idx as usize;
        (self.stamp[i] == self.epoch).then(|| self.val[i])
    }
    #[inline]
    fn set(&mut self, idx: u32, v: bool) {
        let i = idx as usize;
        self.val[i] = v;
        self.stamp[i] = self.epoch;
    }
}

fn eval_under_model(
    aig: &Aig,
    idx: u32,
    enc: &ScratchEncoder,
    memo: &mut EvalMemo,
) -> bool {
    let mut stack = std::mem::take(&mut memo.stack);
    stack.clear();
    stack.push(idx);
    while let Some(&top) = stack.last() {
        if memo.get(top).is_some() {
            stack.pop();
            continue;
        }
        let v = match aig.node(top) {
            AigNode::ConstTrue => Some(true),
            AigNode::Input(_) => Some(match enc.node_lit[top as usize] {
                Some(l) => enc.solver.value_of(l) == crate::lit::LBool::True,
                None => false,
            }),
            AigNode::And(a, b) => {
                match (memo.get(a.node_idx()), memo.get(b.node_idx())) {
                    (Some(av), Some(bv)) => {
                        Some((av ^ a.is_negated()) && (bv ^ b.is_negated()))
                    }
                    (None, _) => {
                        stack.push(a.node_idx());
                        None
                    }
                    (_, None) => {
                        stack.push(b.node_idx());
                        None
                    }
                }
            }
        };
        if let Some(v) = v {
            memo.set(top, v);
            stack.pop();
        }
    }
    memo.stack = stack;
    memo.get(idx).expect("root evaluated")
}

/// Lazy Tseitin encoder from AIG nodes into a private scratch `Solver`.
/// Inputs become fresh, unconstrained variables — deliberately decoupled
/// from the real solver's literals so proofs quantify over all inputs.
struct ScratchEncoder {
    solver: Solver,
    /// node idx -> scratch lit for the node's positive output.
    node_lit: Vec<Option<Lit>>,
    true_lit: Option<Lit>,
}

impl ScratchEncoder {
    fn new(num_nodes: usize) -> Self {
        ScratchEncoder {
            solver: Solver::new(),
            node_lit: vec![None; num_nodes],
            true_lit: None,
        }
    }

    fn true_lit(&mut self) -> Lit {
        if let Some(l) = self.true_lit {
            return l;
        }
        let v = self.solver.new_var();
        let l = Lit::new(v, false);
        self.solver.add_clause_from_slice(&[l]);
        self.true_lit = Some(l);
        l
    }

    #[inline]
    fn signed(&self, r: AigRef) -> Lit {
        let base = self.node_lit[r.node_idx() as usize].expect("cone encoded");
        if r.is_negated() { !base } else { base }
    }

    /// Encode `idx`'s cone (iteratively — cones can be millions deep) and
    /// return its positive-output lit.
    fn lit_for(&mut self, aig: &Aig, idx: u32) -> Lit {
        if let Some(l) = self.node_lit[idx as usize] {
            return l;
        }
        let mut worklist: Vec<u32> = vec![idx];
        while let Some(&top) = worklist.last() {
            if self.node_lit[top as usize].is_some() {
                worklist.pop();
                continue;
            }
            match aig.node(top) {
                AigNode::ConstTrue => {
                    let l = self.true_lit();
                    self.node_lit[top as usize] = Some(l);
                    worklist.pop();
                }
                AigNode::Input(_) => {
                    // Fresh free variable; the real SAT lit identity is
                    // irrelevant here (and must not leak in).
                    let v = self.solver.new_var();
                    self.node_lit[top as usize] = Some(Lit::new(v, false));
                    worklist.pop();
                }
                AigNode::And(a, b) if a == b => {
                    // Alias node from an earlier merge this session: reuse
                    // the target's lit, no gate clauses.
                    if self.node_lit[a.node_idx() as usize].is_some() {
                        self.node_lit[top as usize] = Some(self.signed(a));
                        worklist.pop();
                    } else {
                        worklist.push(a.node_idx());
                    }
                }
                AigNode::And(a, b) => {
                    let need_a = self.node_lit[a.node_idx() as usize].is_none();
                    let need_b = self.node_lit[b.node_idx() as usize].is_none();
                    if need_a {
                        worklist.push(a.node_idx());
                    }
                    if need_b {
                        worklist.push(b.node_idx());
                    }
                    if need_a || need_b {
                        continue;
                    }
                    let la = self.signed(a);
                    let lb = self.signed(b);
                    let v = Lit::new(self.solver.new_var(), false);
                    self.solver.add_clause_from_slice(&[!v, la]);
                    self.solver.add_clause_from_slice(&[!v, lb]);
                    self.solver.add_clause_from_slice(&[v, !la, !lb]);
                    self.node_lit[top as usize] = Some(v);
                    worklist.pop();
                }
            }
        }
        self.node_lit[idx as usize].expect("just encoded")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lit::Var;

    fn mk_input(aig: &mut Aig, var: u32) -> AigRef {
        aig.input(Lit::new(Var(var), false))
    }

    /// Same construction as `sim_sweep_finds_semantic_duplicates`: two
    /// structurally-distinct builds of `a ∨ b`. The sweep must prove and
    /// merge them.
    #[test]
    fn sweep_merges_semantic_or_duplicates() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let or1 = aig.or(a, b);
        let inner = aig.and(!a, b);
        let or2 = !aig.and(!a, !inner);
        assert_ne!(or1.node_idx(), or2.node_idx());

        let stats = sweep(&mut aig, 0, 1000, 1000, 42);
        assert_eq!(stats.proven, 1, "the duplicate pair must be proven");
        assert_eq!(stats.disproven, 0);
        // The younger node became an alias of the older one.
        let (dup, rep) = if or1.node_idx() > or2.node_idx() {
            (or1, or2)
        } else {
            (or2, or1)
        };
        match aig.node(dup.node_idx()) {
            AigNode::And(x, y) => {
                assert_eq!(x, y, "merged node must be an identity alias");
                // or1 and or2 are both negated refs to their AND nodes, so
                // the underlying nodes are same-phase equivalent.
                assert_eq!(x.node_idx(), rep.node_idx());
                assert!(!x.is_negated());
            }
            n => panic!("expected alias And node, got {:?}", n),
        }
    }

    /// `a ∧ b` vs `a ∧ c` collide on no signature (independent inputs), but
    /// `(a<b)`-style near-miss pairs do; emulate one with functions equal on
    /// most inputs: `and(a, b)` vs `and(a, and(b, c))` differ only when
    /// a=1, b=1, c=0 — random sim usually separates them, so instead build
    /// an exact sim collision manually is impractical here. What we CAN
    /// assert cheaply: a disprovable pair fed directly to the checker is
    /// refuted, via a class faked by identical signatures being unnecessary
    /// — the public entry only checks sim-equal pairs. So this test just
    /// documents that inequivalent nodes are never merged by a full sweep.
    #[test]
    fn sweep_never_merges_inequivalent_nodes() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let c = mk_input(&mut aig, 3);
        let n1 = aig.and(a, b);
        let n2 = aig.and(a, c);
        let n3 = aig.and(b, c);
        let _ = (n1, n2, n3);
        let before: Vec<String> = (0..aig.num_nodes())
            .map(|i| format!("{:?}", aig.node(i as u32)))
            .collect();
        let stats = sweep(&mut aig, 0, 1000, 1000, 7);
        assert_eq!(stats.proven, 0);
        let after: Vec<String> = (0..aig.num_nodes())
            .map(|i| format!("{:?}", aig.node(i as u32)))
            .collect();
        assert_eq!(before, after, "no node may be rewritten");
    }

    /// Cross-phase merge. The top AND node of `xor(a, b)` computes
    /// XNOR(a, b) (the xor ref is its negation); the top node of
    /// `xor(a, ¬b)` computes XOR(a, b). Complementary functions, distinct
    /// structure (different interior minterm nodes) — the sweep must
    /// detect the phase and merge them as complements. Interior minterm
    /// nodes are pairwise inequivalent, so exactly one merge happens.
    #[test]
    fn sweep_merges_complementary_pair() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let x1 = aig.xor(a, b);
        let x2 = aig.xor(a, !b);
        assert_ne!(x1.node_idx(), x2.node_idx());

        let stats = sweep(&mut aig, 0, 1000, 1000, 42);
        assert_eq!(stats.proven, 1);
        assert_eq!(stats.disproven, 0);
        let (dup, rep) = if x1.node_idx() > x2.node_idx() {
            (x1, x2)
        } else {
            (x2, x1)
        };
        match aig.node(dup.node_idx()) {
            AigNode::And(x, y) => {
                assert_eq!(x, y);
                assert_eq!(x.node_idx(), rep.node_idx());
                assert!(x.is_negated(), "complementary merge must carry phase");
            }
            n => panic!("expected alias, got {:?}", n),
        }
    }

    /// start_idx fencing: nodes older than the fence are never rewritten.
    #[test]
    fn sweep_respects_start_idx() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let or1 = aig.or(a, b);
        let inner = aig.and(!a, b);
        let or2 = !aig.and(!a, !inner);
        let fence = aig.num_nodes() as u32; // both duplicates predate this
        let _ = (or1, or2);
        let stats = sweep(&mut aig, fence, 1000, 1000, 42);
        assert_eq!(stats.proven, 0);
        assert_eq!(stats.candidates, 0);
    }
}
