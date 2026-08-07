//! Gaussian elimination over GF(2) on the formula's XOR skeleton.
//!
//! Why this exists: resolution is exponential on parity constraints
//! (Tseitin's classic result), and binbit's workload is parity-dense —
//! nobranch emits 156,628 XOR gates against 244,861 ANDs, because the
//! sum bit of every adder in its multiply-shift mixer is an XOR. CDCL
//! spends conflicts rediscovering, one resolution at a time, facts that
//! linear algebra reads off directly.
//!
//! The usual hard part of XOR reasoning is *finding* the XORs: a
//! CryptoMiniSat-style `XorFinder` has to recover parity constraints
//! from anonymous groups of CNF clauses. binbit skips that entirely — the
//! bitblaster already knows which gates are XORs (`GateKind::Xor`, see
//! `SmtSolver::emit_xor_gate`), so the constraints are handed over
//! exactly, for free, at emission time.
//!
//! What we do with them, in one offline pass before preprocessing:
//!
//! 1. **Forward elimination** to row echelon form over the sparse rows,
//!    which couples the circuit's definitional parities with the
//!    assertion's unit facts.
//! 2. **Unit propagation inside the linear system**: a solved row assigns
//!    a variable, which is substituted into every row mentioning it,
//!    potentially solving more rows. This is what carries an output
//!    constraint *backwards* through an XOR chain — the move that word-
//!    level inversion could not make, because at the bit level the
//!    adder's sum is linear even though its carry is not.
//!
//! Everything derived is implied by the formula, so the results are
//! emitted as ordinary clauses (units, and binaries for equivalences)
//! and cost nothing in soundness. Dropping a row is always safe: it can
//! only make us derive less.
//!
//! ## Verdict (2026-08-07): mechanism works, statically NEUTRAL — off
//!
//! On nobranch (156,628 XOR gates, the densest parity instance we have)
//! the system is **156,886 rows at rank 156,879** — essentially full
//! rank. A circuit's XOR skeleton is triangular by construction: every
//! gate output is a fresh variable defined by its inputs, so there is
//! almost nothing to derive until assertion constraints couple it.
//! Feeding in the assertion's parities (equality biconditionals are
//! 2-variable rows; see `assert_toplevel_direct`) yields 103 forced
//! variables and 225 equivalences — real, but too few to matter:
//! **paired +0.0% ± 3.3% conflicts over 8 seeds, better on 3/8.**
//!
//! Materializing longer derived rows as CNF is worse, and fails the same
//! way eager `pcaug` and `cnfmap` did — bulk clause addition:
//!
//! | rows emitted | added clauses | conflicts | SAT time |
//! |---|---|---|---|
//! | none | — | 572,694 | 24.3s |
//! | len 3 | +5,339 | 549,829 | 24.4s |
//! | len ≤4 | +367,062 | 661,628 | 47.3s |
//! | len ≤5 | +468,318 | 598,778 | 33.5s |
//!
//! ## Where the remaining value is, and what it would cost
//!
//! The echelon basis is NOT just the gate XORs re-expressed: 66,167 rows
//! (42%) are genuine combinations of length 4–15. A length-8 row goes
//! unit when 7 of its variables are assigned — variables scattered across
//! the circuit, where no single gate XOR is unit and CNF propagation sees
//! nothing. That inference is only reachable by propagating the rows
//! NATIVELY, since encoding one costs 2^(k-1) clauses (the table above).
//!
//! That means watched-variable XOR propagation inside `propagate`, plus a
//! `Reason::Xor` variant that `analyze` can expand into a reason clause
//! on demand — a change to the solver's hottest and most delicate code,
//! touching backtracking, clause GC and conflict analysis. The
//! measurement above says the information is there; it does not say the
//! propagation will pay for its overhead, and every bulk-information
//! experiment in this codebase so far has lost to its own cost.

use rustc_hash::FxHashMap as HashMap;

/// One parity constraint: `vars[0] ^ vars[1] ^ ... = rhs`, with `vars`
/// sorted and duplicate-free (a variable XORed with itself cancels).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub vars: Vec<u32>,
    pub rhs: bool,
}

impl Row {
    /// XOR two rows: symmetric difference of the variable sets, rhs
    /// XORed. Both inputs are sorted, so this is one merge pass.
    fn xor_into(&mut self, other: &Row, scratch: &mut Vec<u32>) {
        scratch.clear();
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.vars.len() && j < other.vars.len() {
            match self.vars[i].cmp(&other.vars[j]) {
                std::cmp::Ordering::Less => {
                    scratch.push(self.vars[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    scratch.push(other.vars[j]);
                    j += 1;
                }
                // Present in both: cancels.
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        scratch.extend_from_slice(&self.vars[i..]);
        scratch.extend_from_slice(&other.vars[j..]);
        std::mem::swap(&mut self.vars, scratch);
        self.rhs ^= other.rhs;
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct XorStats {
    /// Constraints handed in.
    pub rows_in: u64,
    /// Rows surviving forward elimination (the linear rank).
    pub rank: u64,
    /// Rows dropped for exceeding the length cap (fill-in control).
    pub dropped: u64,
    /// Variables assigned a constant by the system.
    pub units: u64,
    /// Variable pairs proven equal or complementary.
    pub equivs: u64,
    /// The system itself is contradictory (formula is UNSAT).
    pub conflict: bool,
    /// Echelon-row length histogram, bucketed [0]=len<=3, [1]=4..8,
    /// [2]=9..16, [3]=>16. Predicts whether IN-SEARCH propagation over
    /// these rows could infer anything the CNF encoding cannot: a basis
    /// that stays at length 3 is just the original gate XORs re-expressed
    /// and propagating it natively would add nothing.
    pub len_hist: [u64; 4],
    pub max_row: u64,
    /// Clauses emitted for materialized short rows.
    pub emitted: u64,
    /// Distinct variables appearing in the system. Together with `rank`
    /// this bounds what elimination can possibly determine: a system of
    /// `rank` independent equations over `vars` unknowns leaves
    /// `vars - rank` degrees of freedom, and no implementation quality
    /// changes that.
    pub vars: u64,
    /// Rows handed over for native propagation.
    pub native_rows: u64,
}

/// What the elimination proved.
#[derive(Default, Debug)]
pub struct XorFindings {
    /// `var` is forced to `value`.
    pub units: Vec<(u32, bool)>,
    /// `a ^ b = rhs` — equal when `rhs` is false, complementary when true.
    pub equivs: Vec<(u32, u32, bool)>,
    /// Derived echelon rows of length 3..=`emit_len`, i.e. parities that
    /// are genuine COMBINATIONS of gate XORs rather than any single gate.
    /// Encoding one costs 2^(k-1) clauses, so only short ones are worth
    /// materializing; the point of exposing them is to test whether the
    /// information helps CDCL before building native propagation for it.
    pub short_rows: Vec<(Vec<u32>, bool)>,
    /// Derived rows of length >= 3 handed to the SAT core for native
    /// propagation (see `Solver::set_xor_rows`). These are the
    /// combinations no CNF clause represents.
    pub long_rows: Vec<(Vec<u32>, bool)>,
    pub stats: XorStats,
}

/// The parity system. Rows are accumulated, then solved once.
pub struct XorSystem {
    rows: Vec<Row>,
    /// Shortest derived row worth propagating natively; see
    /// [`XorSystem::set_native_min`].
    native_min: usize,
}

impl Default for XorSystem {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            native_min: Self::NATIVE_MIN,
        }
    }
}

impl XorSystem {
    /// Default for [`XorSystem::set_native_min`]. Below this length the
    /// CNF encoding already does the job, so native propagation would
    /// only add watch traffic.
    pub const NATIVE_MIN: usize = 4;

    /// Shortest derived row handed to the SAT core for native
    /// propagation rather than materialized as clauses.
    pub fn set_native_min(&mut self, n: usize) {
        self.native_min = n;
    }

    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Add `vars[0] ^ ... = rhs`. Repeated variables cancel, so the
    /// caller may pass them in any order with duplicates.
    pub fn add(&mut self, vars: &[u32], rhs: bool) {
        let mut v: Vec<u32> = vars.to_vec();
        v.sort_unstable();
        let mut w = 0usize;
        let mut i = 0usize;
        while i < v.len() {
            if i + 1 < v.len() && v[i] == v[i + 1] {
                i += 2; // x ^ x = 0
                continue;
            }
            v[w] = v[i];
            w += 1;
            i += 1;
        }
        v.truncate(w);
        self.rows.push(Row { vars: v, rhs });
    }

    /// Eliminate, then propagate solved variables through the system.
    ///
    /// `max_len` caps row length during elimination: XORing sparse rows
    /// causes fill-in, and an uncapped run on a large circuit degenerates
    /// into dense linear algebra. Dropping an over-long row loses
    /// derivations but never produces a wrong one.
    pub fn solve(&mut self, max_len: usize) -> XorFindings {
        self.solve_emitting(max_len, 0)
    }

    /// As [`solve`], additionally returning derived rows up to
    /// `emit_len` variables for the caller to encode as CNF.
    pub fn solve_emitting(&mut self, max_len: usize, emit_len: usize) -> XorFindings {
        let mut f = XorFindings::default();
        f.stats.rows_in = self.rows.len() as u64;

        // ---- 1. Forward elimination to row echelon form. ----
        // `pivot[v]` = index in `reduced` of the row whose leading
        // variable is v. Each row is reduced against existing pivots
        // before taking its own.
        let mut reduced: Vec<Row> = Vec::new();
        let mut pivot: HashMap<u32, usize> = HashMap::default();
        let mut scratch: Vec<u32> = Vec::new();
        let rows = std::mem::take(&mut self.rows);
        for mut r in rows {
            while let Some(&p) = r.vars.first().and_then(|v| pivot.get(v)) {
                r.xor_into(&reduced[p], &mut scratch);
                if r.vars.len() > max_len {
                    break;
                }
            }
            if r.vars.is_empty() {
                if r.rhs {
                    // 0 = 1: the parity system alone refutes the formula.
                    f.stats.conflict = true;
                    return f;
                }
                continue; // 0 = 0, redundant
            }
            if r.vars.len() > max_len {
                f.stats.dropped += 1;
                continue;
            }
            pivot.insert(r.vars[0], reduced.len());
            reduced.push(r);
        }
        f.stats.rank = reduced.len() as u64;
        {
            let mut seen: std::collections::HashSet<u32> =
                std::collections::HashSet::default();
            for r in &reduced {
                seen.extend(r.vars.iter().copied());
            }
            f.stats.vars = seen.len() as u64;
        }
        for r in &reduced {
            let n = r.vars.len();
            let b = if n <= 3 { 0 } else if n <= 8 { 1 } else if n <= 16 { 2 } else { 3 };
            f.stats.len_hist[b] += 1;
            f.stats.max_row = f.stats.max_row.max(n as u64);
        }

        // ---- 2. Propagate solved rows through the system. ----
        // A length-1 row assigns its variable; substituting that into
        // every row mentioning it can solve further rows, and so on.
        // This is what walks an asserted output bit backwards along an
        // XOR chain.
        let mut occ: HashMap<u32, Vec<usize>> = HashMap::default();
        for (i, r) in reduced.iter().enumerate() {
            for &v in &r.vars {
                occ.entry(v).or_default().push(i);
            }
        }
        let mut assigned: HashMap<u32, bool> = HashMap::default();
        let mut queue: Vec<usize> = (0..reduced.len()).collect();
        let mut qi = 0usize;
        while let Some(&i) = queue.get(qi) {
            qi += 1;
            if reduced[i].vars.len() != 1 {
                continue;
            }
            let v = reduced[i].vars[0];
            let val = reduced[i].rhs;
            match assigned.get(&v) {
                Some(&prev) => {
                    if prev != val {
                        f.stats.conflict = true;
                        return f;
                    }
                    continue;
                }
                None => {
                    assigned.insert(v, val);
                    f.units.push((v, val));
                }
            }
            // Substitute into every other row holding v.
            let Some(rows_with_v) = occ.get(&v).cloned() else {
                continue;
            };
            for j in rows_with_v {
                if j == i {
                    continue;
                }
                if let Ok(k) = reduced[j].vars.binary_search(&v) {
                    reduced[j].vars.remove(k);
                    reduced[j].rhs ^= val;
                    // 0 = 1 is a refutation; a solved row joins the
                    // queue so its variable propagates in turn.
                    if reduced[j].vars.is_empty() && reduced[j].rhs {
                        f.stats.conflict = true;
                        return f;
                    }
                    if reduced[j].vars.len() == 1 {
                        queue.push(j);
                    }
                }
            }
        }

        // ---- 3. Read off two-variable rows as equivalences, and short
        // multi-variable rows for optional CNF materialization. ----
        for r in &reduced {
            if r.vars.len() == 2 {
                f.equivs.push((r.vars[0], r.vars[1], r.rhs));
            } else if r.vars.len() >= 3 {
                if r.vars.len() <= emit_len {
                    f.short_rows.push((r.vars.clone(), r.rhs));
                } else if r.vars.len() >= self.native_min {
                    // Only rows a CNF encoding cannot represent cheaply.
                    // A length-3 row is a gate XOR whose own 4 clauses
                    // already propagate it completely, so handing it to
                    // the parity engine is pure overhead — and every
                    // firing materializes a redundant reason clause.
                    f.long_rows.push((r.vars.clone(), r.rhs));
                }
            }
        }
        f.stats.units = f.units.len() as u64;
        f.stats.equivs = f.equivs.len() as u64;
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(rows: &[(&[u32], bool)]) -> XorSystem {
        let mut s = XorSystem::default();
        for (v, r) in rows {
            s.add(v, *r);
        }
        s
    }

    #[test]
    fn duplicate_vars_cancel() {
        let mut s = XorSystem::default();
        s.add(&[3, 1, 3], true); // x3 ^ x1 ^ x3 = x1
        let f = s.solve(64);
        assert_eq!(f.units, vec![(1, true)]);
    }

    /// The core move: a unit on a chain's output walks backwards.
    /// a^b=c, c^d=e, with e and a,b,d partially known.
    #[test]
    fn unit_propagates_backwards_through_a_chain() {
        // c = a ^ b   ->  a ^ b ^ c = 0
        // e = c ^ d   ->  c ^ d ^ e = 0
        // e = 1, a = 0, b = 0  =>  c = 0  =>  d = 1
        let mut s = sys(&[
            (&[1, 2, 3][..], false), // a^b^c = 0
            (&[3, 4, 5][..], false), // c^d^e = 0
            (&[5][..], true),        // e = 1
            (&[1][..], false),       // a = 0
            (&[2][..], false),       // b = 0
        ]);
        let f = s.solve(64);
        let m: std::collections::HashMap<u32, bool> = f.units.iter().copied().collect();
        assert_eq!(m.get(&3), Some(&false), "c must be derived 0");
        assert_eq!(m.get(&4), Some(&true), "d must be derived 1");
        assert!(!f.stats.conflict);
    }

    /// Elimination must combine rows that share no unit at all — this is
    /// the part plain unit propagation on the CNF cannot do.
    #[test]
    fn elimination_derives_equivalence_without_units() {
        // a^b^c = 0 and a^b^d = 1  =>  c^d = 1 (c != d), with nothing
        // assigned. No CNF unit propagation reaches this.
        let mut s = sys(&[(&[1, 2, 3][..], false), (&[1, 2, 4][..], true)]);
        let f = s.solve(64);
        assert!(f.units.is_empty());
        assert!(
            f.equivs.contains(&(3, 4, true)),
            "expected c^d=1, got {:?}",
            f.equivs
        );
    }

    #[test]
    fn detects_inconsistent_system() {
        // a^b = 0 and a^b = 1.
        let mut s = sys(&[(&[1, 2][..], false), (&[1, 2][..], true)]);
        assert!(s.solve(64).stats.conflict);
    }

    #[test]
    fn parity_of_a_full_cycle_is_derived() {
        // Classic Tseitin parity: a chain closed into a loop with odd
        // total parity is UNSAT, and resolution needs exponentially many
        // steps while elimination sees it at once.
        let n = 40u32;
        let mut rows: Vec<(Vec<u32>, bool)> = Vec::new();
        for i in 0..n {
            rows.push((vec![i, (i + 1) % n], false)); // x_i = x_{i+1}
        }
        // Force one link to be inequality: makes the cycle inconsistent.
        rows[0].1 = true;
        let mut s = XorSystem::default();
        for (v, r) in &rows {
            s.add(v, *r);
        }
        assert!(s.solve(64).stats.conflict, "odd cycle must be refuted");
    }

    #[test]
    fn long_rows_are_dropped_not_miscomputed() {
        // With a tight cap, elimination must simply derive less.
        let mut s = sys(&[
            (&[1, 2, 3, 4, 5][..], false),
            (&[1, 2, 3, 4, 6][..], true),
        ]);
        let f = s.solve(3);
        assert!(!f.stats.conflict);
        assert!(f.stats.dropped > 0);
    }
}

#[cfg(test)]
mod encoding_tests {

    /// The CNF materialization of a derived row (see
    /// `SmtSolver::solve_xor_system`) must accept exactly the assignments
    /// whose parity matches, and reject the others. Getting the polarity
    /// backwards inverts the constraint — it made nobranch report UNSAT.
    #[test]
    fn short_row_cnf_encoding_matches_parity() {
        for k in 2..=5usize {
            for rhs in [false, true] {
                // Build the clause set the emitter would produce.
                let mut clauses: Vec<Vec<(usize, bool)>> = Vec::new();
                for mask in 0..(1u32 << k) {
                    if (mask.count_ones() & 1 == 1) == rhs {
                        continue;
                    }
                    clauses.push(
                        (0..k).map(|i| (i, mask >> i & 1 == 1)).collect(),
                    );
                }
                assert_eq!(clauses.len(), 1 << (k - 1));
                // Check every assignment against both the parity
                // constraint and the clause set.
                for a in 0..(1u32 << k) {
                    let parity_ok = (a.count_ones() & 1 == 1) == rhs;
                    let cnf_ok = clauses.iter().all(|c| {
                        c.iter().any(|&(i, neg)| {
                            let bit = a >> i & 1 == 1;
                            bit != neg // literal true under `a`
                        })
                    });
                    assert_eq!(
                        parity_ok, cnf_ok,
                        "k={k} rhs={rhs} assignment {a:#b} disagrees"
                    );
                }
            }
        }
    }
}
