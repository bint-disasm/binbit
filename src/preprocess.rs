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
//! Everything here is deterministic: candidate orders are index-based, and
//! all limits are fixed constants.

use crate::lit::Lit;

/// Upper bound on resolvent length. A candidate elimination that would
/// produce a clause longer than this is rejected (MiniSat's `cl-lim`).
const CLAUSE_LIM: usize = 20;

/// Skip variable elimination for variables whose positive×negative
/// occurrence product exceeds this — the pair enumeration alone would be
/// too expensive, and high-occurrence variables essentially never satisfy
/// the non-increasing bound anyway.
const VE_PRODUCT_LIM: usize = 100;

/// Don't use a clause as a backward-subsumption candidate if its
/// least-occurring literal still occurs more often than this.
const SUB_OCC_LIM: usize = 1_000;

/// Result of one preprocessing run.
pub struct SimplifyResult {
    /// Surviving clauses (units included). Order is deterministic.
    pub clauses: Vec<Vec<Lit>>,
    /// Variable indices eliminated by VE. The caller must ensure these can
    /// never be referenced by later clauses (see module docs).
    pub eliminated: Vec<u32>,
    /// Number of clauses removed by (self-)subsumption.
    pub subsumed: usize,
    /// Number of literals removed by strengthening.
    pub strengthened: usize,
    /// True if preprocessing derived the empty clause — formula is UNSAT.
    /// `clauses` then contains a single empty clause so the SAT core's
    /// `add_clause` records the dead state through its normal path.
    pub unsat: bool,
}

struct Clause {
    lits: Vec<Lit>,
    sig: u64,
    deleted: bool,
}

impl Clause {
    fn new(mut lits: Vec<Lit>) -> Self {
        lits.sort_by_key(|l| l.0);
        lits.dedup();
        let sig = lits.iter().fold(0u64, |s, l| s | 1u64 << (l.var_idx() & 63));
        Clause { lits, sig, deleted: false }
    }
}

pub struct Preprocessor {
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
    unsat: bool,
    stats_subsumed: usize,
    stats_strengthened: usize,
}

impl Preprocessor {
    /// `num_vars` bounds the variable indices appearing in `clauses`.
    /// `frozen[v]` marks variables that must survive.
    pub fn new(clauses: Vec<Vec<Lit>>, num_vars: usize, frozen: Vec<bool>) -> Self {
        assert_eq!(frozen.len(), num_vars);
        let mut p = Preprocessor {
            clauses: Vec::with_capacity(clauses.len()),
            occ: vec![Vec::new(); 2 * num_vars],
            n_occ: vec![0; 2 * num_vars],
            assign: vec![0; 2 * num_vars],
            frozen,
            eliminated: vec![false; num_vars],
            unit_queue: Vec::new(),
            touched: Vec::new(),
            unsat: false,
            stats_subsumed: 0,
            stats_strengthened: 0,
        };
        for lits in clauses {
            let c = Clause::new(lits);
            // Tautology: drop at intake.
            if c.lits.windows(2).any(|w| w[0].var() == w[1].var()) {
                continue;
            }
            if c.lits.is_empty() {
                p.unsat = true;
                continue;
            }
            if c.lits.len() == 1 {
                p.unit_queue.push(c.lits[0]);
            }
            let idx = p.clauses.len() as u32;
            for &l in &c.lits {
                p.occ[l.0 as usize].push(idx);
                p.n_occ[l.0 as usize] += 1;
            }
            p.clauses.push(c);
        }
        p
    }

    /// Run the full pipeline: unit propagation → subsumption fixpoint →
    /// bounded VE (with local subsumption on resolvents) → final sweep.
    pub fn run(mut self) -> SimplifyResult {
        self.propagate_units();
        if !self.unsat {
            self.subsumption_pass();
        }
        if !self.unsat {
            self.eliminate_vars();
        }
        let mut out: Vec<Vec<Lit>> = Vec::new();
        if self.unsat {
            out.push(Vec::new());
        } else {
            for c in &self.clauses {
                if !c.deleted {
                    out.push(c.lits.clone());
                }
            }
            // Re-emit discovered units (propagate_units removes them from
            // clause form once applied).
            for li in 0..self.assign.len() {
                if self.assign[li] == 1 {
                    out.push(vec![Lit(li as u32)]);
                }
            }
        }
        let eliminated = self
            .eliminated
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e)
            .map(|(v, _)| v as u32)
            .collect();
        SimplifyResult {
            clauses: out,
            eliminated,
            subsumed: self.stats_subsumed,
            strengthened: self.stats_strengthened,
            unsat: self.unsat,
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

            // Clauses containing u are satisfied — delete.
            let sat_list = std::mem::take(&mut self.occ[ui]);
            for ci in sat_list {
                self.delete_clause(ci);
            }
            // Clauses containing ¬u lose that literal.
            let shrink_list = std::mem::take(&mut self.occ[ni]);
            for ci in shrink_list {
                let c = &mut self.clauses[ci as usize];
                if c.deleted {
                    continue;
                }
                let nu = Lit(u.0 ^ 1);
                if let Some(pos) = c.lits.iter().position(|&l| l == nu) {
                    c.lits.remove(pos);
                    self.n_occ[ni] = self.n_occ[ni].saturating_sub(1);
                    c.sig = c
                        .lits
                        .iter()
                        .fold(0u64, |s, l| s | 1u64 << (l.var_idx() & 63));
                    match c.lits.len() {
                        0 => {
                            self.unsat = true;
                            return;
                        }
                        1 => {
                            let unit = c.lits[0];
                            self.unit_queue.push(unit);
                            // The clause itself dissolves into the unit.
                            self.delete_clause(ci);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // ---------- subsumption / strengthening ----------

    /// `subsumes(c, d)`: `Ok(None)` if c ⊆ d; `Ok(Some(l))` if c "almost"
    /// subsumes d — every literal of c appears in d except one literal `l`
    /// of c whose *negation* appears in d (self-subsuming resolution: d can
    /// be strengthened by removing `¬l`). `Err(())` otherwise. Both clause
    /// lit vectors are sorted by `Lit.0`.
    fn subsumes(c: &Clause, d: &Clause) -> Result<Option<Lit>, ()> {
        if c.lits.len() > d.lits.len() || (c.sig & !d.sig) != 0 {
            return Err(());
        }
        let mut flipped: Option<Lit> = None;
        let mut di = 0usize;
        'outer: for &cl in &c.lits {
            while di < d.lits.len() {
                let dl = d.lits[di];
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
        // Process shorter clauses first — they subsume more.
        let mut order: Vec<u32> = (0..self.clauses.len() as u32)
            .filter(|&i| !self.clauses[i as usize].deleted)
            .collect();
        order.sort_by_key(|&i| self.clauses[i as usize].lits.len());

        let mut queue: std::collections::VecDeque<u32> = order.into();
        let mut queued: Vec<bool> = vec![true; self.clauses.len()];

        while let Some(ci) = queue.pop_front() {
            if (ci as usize) < queued.len() {
                queued[ci as usize] = false;
            }
            if self.clauses[ci as usize].deleted {
                continue;
            }
            if self.clauses[ci as usize].lits.is_empty() {
                self.unsat = true;
                return;
            }
            // Pick the literal of ci whose VARIABLE occurs least, and scan
            // both polarities' occurrence lists — a self-subsumption
            // candidate contains the *negation* of the flipped literal, so
            // a single-polarity scan would miss it (this is why MiniSat's
            // SimpSolver keys occurrences per-variable).
            let (best_lit, best_occ) = {
                let c = &self.clauses[ci as usize];
                let mut bl = c.lits[0];
                let mut bo =
                    self.n_occ[bl.0 as usize] + self.n_occ[(bl.0 ^ 1) as usize];
                for &l in &c.lits[1..] {
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
            let mut candidates: Vec<u32> = self.occ[best_lit.0 as usize].clone();
            candidates.extend_from_slice(&self.occ[(best_lit.0 ^ 1) as usize]);
            for di in candidates {
                if di == ci || self.clauses[di as usize].deleted {
                    continue;
                }
                if self.clauses[ci as usize].deleted {
                    break;
                }
                let verdict = {
                    let c = &self.clauses[ci as usize];
                    let d = &self.clauses[di as usize];
                    // No stale-entry pre-check needed: `subsumes` verifies
                    // against d's current literals, so a stale occurrence
                    // can only produce a (correct) Err.
                    Self::subsumes(c, d)
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
                        let d = &mut self.clauses[di as usize];
                        if let Some(pos) = d.lits.iter().position(|&l| l == removed) {
                            d.lits.remove(pos);
                            d.sig = d
                                .lits
                                .iter()
                                .fold(0u64, |s, l| s | 1u64 << (l.var_idx() & 63));
                            self.n_occ[removed.0 as usize] =
                                self.n_occ[removed.0 as usize].saturating_sub(1);
                            self.stats_strengthened += 1;
                            match d.lits.len() {
                                0 => {
                                    self.unsat = true;
                                    return;
                                }
                                1 => {
                                    let unit = d.lits[0];
                                    self.unit_queue.push(unit);
                                    self.delete_clause(di);
                                    self.propagate_units();
                                    if self.unsat {
                                        return;
                                    }
                                }
                                _ => {
                                    if !queued[di as usize] {
                                        queued[di as usize] = true;
                                        queue.push_back(di);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
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
        let mut queued: Vec<bool> = vec![false; num_vars];
        let mut fail_cost: Vec<u32> = vec![u32::MAX; num_vars];
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        for v in 0..num_vars as u32 {
            if !self.frozen[v as usize] && !self.eliminated[v as usize] {
                let c = cost(self, v);
                if c > 0 {
                    heap.push(Reverse((c, v)));
                    queued[v as usize] = true;
                }
            }
        }

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
                for t in touched {
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
            } else {
                fail_cost[v as usize] = cur;
                self.touched.clear();
            }
        }
    }

    /// Attempt to eliminate variable `v` by resolution. Succeeds iff every
    /// resolvent is within `CLAUSE_LIM` and the number of non-tautological
    /// resolvents does not exceed the number of clauses removed.
    fn try_eliminate(&mut self, v: u32) -> bool {
        let pl = Lit(2 * v);
        let nl = Lit(2 * v + 1);

        // Collect live clause indices for each polarity (deduped, verified).
        let pos = self.live_occ(pl);
        let neg = self.live_occ(nl);
        if pos.is_empty() && neg.is_empty() {
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
        if pos.len() * neg.len() > VE_PRODUCT_LIM {
            return false;
        }

        let limit = pos.len() + neg.len();
        let mut resolvents: Vec<Vec<Lit>> = Vec::new();
        for &pi in &pos {
            for &ni in &neg {
                match Self::resolve(
                    &self.clauses[pi as usize],
                    &self.clauses[ni as usize],
                    v,
                ) {
                    None => {}
                    Some(r) => {
                        if r.len() > CLAUSE_LIM {
                            return false;
                        }
                        resolvents.push(r);
                        if resolvents.len() > limit {
                            return false;
                        }
                    }
                }
            }
        }

        // Commit: remove originals, add resolvents.
        for &ci in pos.iter().chain(neg.iter()) {
            self.delete_clause(ci);
        }
        self.eliminated[v as usize] = true;
        for r in resolvents {
            self.add_clause(r);
            if self.unsat {
                return true;
            }
        }
        // Resolvent units cascade.
        self.propagate_units();
        true
    }

    /// Resolve `c` (contains v) with `d` (contains ¬v) on `v`. Returns
    /// `None` for tautological resolvents.
    fn resolve(c: &Clause, d: &Clause, v: u32) -> Option<Vec<Lit>> {
        let mut out: Vec<Lit> = Vec::with_capacity(c.lits.len() + d.lits.len() - 2);
        // Merge two sorted lists, dropping the pivot, rejecting tautologies.
        let (mut i, mut j) = (0usize, 0usize);
        while i < c.lits.len() || j < d.lits.len() {
            let next = match (c.lits.get(i), d.lits.get(j)) {
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
            match out.last() {
                Some(&prev) if prev == next => continue, // duplicate
                Some(&prev) if prev.var() == next.var() => return None, // taut
                _ => out.push(next),
            }
        }
        Some(out)
    }

    // ---------- bookkeeping ----------

    /// Verified live occurrences of `l`: clauses that are not deleted and
    /// still contain `l`. Also compacts the occurrence list in passing.
    fn live_occ(&mut self, l: Lit) -> Vec<u32> {
        let li = l.0 as usize;
        let mut out = Vec::new();
        let list = std::mem::take(&mut self.occ[li]);
        for ci in list {
            let c = &self.clauses[ci as usize];
            if !c.deleted && c.lits.binary_search_by_key(&l.0, |x| x.0).is_ok() {
                if !out.contains(&ci) {
                    out.push(ci);
                }
            }
        }
        self.occ[li] = out.clone();
        out
    }

    fn delete_clause(&mut self, ci: u32) {
        let c = &mut self.clauses[ci as usize];
        if c.deleted {
            return;
        }
        c.deleted = true;
        for i in 0..self.clauses[ci as usize].lits.len() {
            let l = self.clauses[ci as usize].lits[i];
            self.n_occ[l.0 as usize] = self.n_occ[l.0 as usize].saturating_sub(1);
            self.touched.push(l.var_idx() as u32);
        }
    }

    fn add_clause(&mut self, lits: Vec<Lit>) {
        let c = Clause::new(lits);
        match c.lits.len() {
            0 => {
                self.unsat = true;
                return;
            }
            1 => {
                self.unit_queue.push(c.lits[0]);
                return;
            }
            _ => {}
        }
        let idx = self.clauses.len() as u32;
        for &l in &c.lits {
            self.occ[l.0 as usize].push(idx);
            self.n_occ[l.0 as usize] += 1;
            self.touched.push(l.var_idx() as u32);
        }
        self.clauses.push(c);
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
        assert_eq!(r.clauses, vec![vec![lit(0, false)]]);
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
        assert!(r.clauses.contains(&vec![lit(1, false), lit(2, false)]));
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
        for c in &r.clauses {
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
        let mut units: Vec<Vec<Lit>> = r.clauses.clone();
        units.sort();
        assert_eq!(units, vec![vec![lit(0, false)], vec![lit(1, false)]]);
    }

    #[test]
    fn conflicting_units_unsat() {
        let r = run(vec![vec![lit(0, false)], vec![lit(0, true)]], 1, &[0]);
        assert!(r.unsat);
        assert_eq!(r.clauses, vec![Vec::<Lit>::new()]);
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
        assert_eq!(r.clauses.len(), 3);
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
        assert!(r.clauses.is_empty());
    }
}
