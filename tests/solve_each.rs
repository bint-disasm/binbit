//! Tests for `solve_each_under_assumptions`: batch feasibility with model
//! screening, warm start from a standing model, and residue-freedom.

use binbit::{BoolTerm, SmtResult, SmtSolver};

/// Tiny deterministic PRNG — a linear-congruential generator.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform_u32(&mut self, bound: u32) -> u32 {
        (self.next_u64() % bound as u64) as u32
    }
}

// ---------- Correctness vs per-candidate solves ----------

/// One randomized trial, as data: `x + y < bound` asserted, candidates of
/// the form `x == c` / `c < x + y`, assumptions of the form `y < c`.
struct TrialSpec {
    bound: u128,
    // (is_eq, constant): true → x == c, false → c < x + y
    cands: Vec<(bool, u128)>,
    asmps: Vec<u128>,
}

/// Deterministically rebuild the trial's solver and terms from the spec.
fn build(spec: &TrialSpec) -> (SmtSolver, Vec<BoolTerm>, Vec<BoolTerm>) {
    let mut s = SmtSolver::new();
    let x = s.bv_var(16);
    let y = s.bv_var(16);
    let sum = s.bv_add(x, y);
    let bound = s.bv_const(spec.bound, 16);
    let below = s.bv_ult(sum, bound);
    s.assert(below);
    let cands = spec
        .cands
        .iter()
        .map(|&(is_eq, cv)| {
            let c = s.bv_const(cv, 16);
            if is_eq {
                s.bv_eq(x, c)
            } else {
                s.bv_ult(c, sum)
            }
        })
        .collect();
    let asmps = spec
        .asmps
        .iter()
        .map(|&av| {
            let c = s.bv_const(av, 16);
            s.bv_ult(y, c)
        })
        .collect();
    (s, cands, asmps)
}

#[test]
fn matches_individual_solves_and_leaves_no_residue() {
    // Random mixes of feasible and infeasible candidates under random
    // assumption sets; batch results must equal what per-candidate
    // solve_under_assumptions reports on a freshly built identical
    // solver, and the batch must leave the solver clean enough that
    // re-solving each candidate individually AFTERWARDS (same solver)
    // still agrees.
    let mut rng = Rng::new(0xB1B1);
    for _trial in 0..50 {
        let spec = TrialSpec {
            bound: 200 + rng.uniform_u32(200) as u128,
            cands: (0..1 + rng.uniform_u32(8) as usize)
                .map(|_| (rng.uniform_u32(2) == 0, rng.uniform_u32(600) as u128))
                .collect(),
            asmps: (0..rng.uniform_u32(3) as usize)
                .map(|_| rng.uniform_u32(400) as u128)
                .collect(),
        };

        // Reference results, each from a FRESH solver — immune to any
        // state the batch might leak.
        let expected: Vec<SmtResult> = (0..spec.cands.len())
            .map(|i| {
                let (mut fresh, cands, asmps) = build(&spec);
                let mut a = asmps.clone();
                a.push(cands[i]);
                fresh.solve_under_assumptions(&a)
            })
            .collect();

        let (mut s, candidates, assumptions) = build(&spec);
        // Half the trials start warm: a prior solve leaves a standing
        // model for the batch to screen with.
        if rng.uniform_u32(2) == 0 {
            assert_eq!(s.solve_under_assumptions(&assumptions), {
                // Base feasibility must match what the references imply
                // only when some candidate was Sat; solve directly.
                let (mut fresh, _, asmps) = build(&spec);
                fresh.solve_under_assumptions(&asmps)
            });
        }
        let batch = s.solve_each_under_assumptions(&candidates, &assumptions);
        assert_eq!(batch, expected, "batch diverged from individual solves");

        // No residue: the same solver must still answer each candidate
        // identically after the batch.
        for (i, &c) in candidates.iter().enumerate() {
            let mut a = assumptions.clone();
            a.push(c);
            assert_eq!(
                s.solve_under_assumptions(&a),
                expected[i],
                "batch left residue affecting candidate {i}"
            );
        }
    }
}

// ---------- Warm start ----------

#[test]
fn warm_start_screens_with_zero_sat_work() {
    // After solving the path condition, a batch whose candidates the
    // standing model already satisfies must complete with NO SAT solve:
    // no new decisions, no new clauses, no new vars.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let y = s.bv_var(32);
    let hundred = s.bv_const(100, 32);
    let below = s.bv_ult(x, hundred);
    s.assert(below);

    let pc = s.bv_ult(y, hundred);
    assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);
    let xv = s.get_bv_value(x) as u128;
    let yv = s.get_bv_value(y) as u128;

    // Candidates the standing model satisfies by construction.
    let xc = s.bv_const(xv, 32);
    let yc = s.bv_const(yv, 32);
    let c1 = s.bv_eq(x, xc);
    let c2 = s.bv_eq(y, yc);
    let c3 = s.bool_and(c1, c2);

    let before = s.sat_stats();
    let res = s.solve_each_under_assumptions(&[c1, c2, c3], &[pc]);
    let after = s.sat_stats();

    assert_eq!(res, vec![SmtResult::Sat; 3]);
    assert_eq!(after.decisions, before.decisions, "warm screen ran a solve");
    assert_eq!(after.sat_clauses, before.sat_clauses, "warm screen emitted CNF");
    assert_eq!(after.sat_vars, before.sat_vars, "warm screen created SAT vars");

    // The model survived an all-screened batch: a second warm batch is
    // still free.
    let res2 = s.solve_each_under_assumptions(&[c3, c1], &[pc]);
    let after2 = s.sat_stats();
    assert_eq!(res2, vec![SmtResult::Sat; 2]);
    assert_eq!(after2.decisions, after.decisions);
}

#[test]
fn warm_start_branch_fanout_single_solve() {
    // The motivating shape: solve pc, then fan out on [cond, !cond]. The
    // standing model decides one side for free; only the other side may
    // need a real solve. Both answers must still be exact.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let ten = s.bv_const(10, 32);
    let pc = s.bv_ult(x, ten);
    assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);
    let xv = s.get_bv_value(x) as u128;

    // cond is true under the standing model, !cond needs a solve (and is
    // feasible: x < 10 has other values).
    let xc = s.bv_const(xv, 32);
    let cond = s.bv_eq(x, xc);
    let not_cond = s.bool_not(cond);
    let res = s.solve_each_under_assumptions(&[cond, not_cond], &[pc]);
    assert_eq!(res, vec![SmtResult::Sat, SmtResult::Sat]);

    // And an infeasible flip side: cond2 covers ALL of pc's models, so
    // !cond2 is Unsat under pc.
    let cond2 = s.bv_ult(x, ten);
    let not_cond2 = s.bool_not(cond2);
    let res2 = s.solve_each_under_assumptions(&[cond2, not_cond2], &[pc]);
    assert_eq!(res2, vec![SmtResult::Sat, SmtResult::Unsat]);
}

#[test]
fn warm_start_falls_back_when_model_violates_assumptions() {
    // The standing model (x = 3) falsifies the batch's assumptions, so
    // the warm path must NOT screen with it; results must be exact.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let ten = s.bv_const(10, 32);
    let below = s.bv_ult(x, ten);
    s.assert(below);

    let three = s.bv_const(3, 32);
    let eq3 = s.bv_eq(x, three);
    assert_eq!(s.solve_under_assumptions(&[eq3]), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 3);

    let ne3 = s.bv_ne(x, three);
    let four = s.bv_const(4, 32);
    let twenty = s.bv_const(20, 32);
    let eq4 = s.bv_eq(x, four);
    let eq20 = s.bv_eq(x, twenty);
    let res = s.solve_each_under_assumptions(&[eq3, eq4, eq20], &[ne3]);
    assert_eq!(
        res,
        vec![SmtResult::Unsat, SmtResult::Sat, SmtResult::Unsat]
    );
}

#[test]
fn warm_start_invalidated_by_new_assertion() {
    // An assertion added after the solve contradicts the standing model
    // (x = 3); the batch must not screen against the stale model.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let ten = s.bv_const(10, 32);
    let below = s.bv_ult(x, ten);
    s.assert(below);

    let three = s.bv_const(3, 32);
    let eq3 = s.bv_eq(x, three);
    assert_eq!(s.solve_under_assumptions(&[eq3]), SmtResult::Sat);

    let ne3 = s.bv_ne(x, three);
    s.assert(ne3); // pending until the batch flushes it

    let four = s.bv_const(4, 32);
    let eq4 = s.bv_eq(x, four);
    let res = s.solve_each_under_assumptions(&[eq3, eq4], &[]);
    assert_eq!(res, vec![SmtResult::Unsat, SmtResult::Sat]);
}

#[test]
fn cold_base_unsat_short_circuits() {
    // Contradictory assumptions: every candidate is Unsat regardless.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let one = s.bv_const(1, 8);
    let two = s.bv_const(2, 8);
    let a1 = s.bv_eq(x, one);
    let a2 = s.bv_eq(x, two);
    let anything = s.bv_ne(x, one);
    let res = s.solve_each_under_assumptions(&[anything, a1], &[a1, a2]);
    assert_eq!(res, vec![SmtResult::Unsat, SmtResult::Unsat]);

    // The solver is not poisoned: a sane follow-up solve works.
    assert_eq!(s.solve_under_assumptions(&[a1]), SmtResult::Sat);
}

/// Timing comparison on the symbex branch fan-out shape: per round, one
/// "reach the state" solve of the path condition, then decide feasibility
/// of both branch sides. Run with:
///   cargo test --release --test solve_each -- --ignored --nocapture
#[test]
#[ignore]
fn bench_branch_fanout_warm_vs_individual() {
    use std::time::Instant;

    const ROUNDS: u32 = 400;

    // A shared constraint with some real structure so solves aren't free.
    fn setup() -> (SmtSolver, binbit::BvTerm, binbit::BvTerm) {
        let mut s = SmtSolver::new();
        let x = s.bv_var(32);
        let y = s.bv_var(32);
        let prod = s.bv_mul(x, y);
        let big = s.bv_const(0x10000, 32);
        let below = s.bv_ult(prod, big);
        s.assert(below);
        (s, x, y)
    }

    // Individual: reach-solve + one solve per branch side (3 solves/round).
    let (mut s, x, y) = setup();
    let t0 = Instant::now();
    for r in 0..ROUNDS {
        let rc = s.bv_const(r as u128 + 2, 32);
        let pc = s.bv_eq(y, rc);
        assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);
        let k = s.bv_const((r as u128 % 200) + 1, 32);
        let cond = s.bv_ult(x, k);
        let not_cond = s.bool_not(cond);
        let _a = s.solve_under_assumptions(&[pc, cond]);
        let _b = s.solve_under_assumptions(&[pc, not_cond]);
    }
    let individual = t0.elapsed();

    // Batch warm: reach-solve, then solve_each screens with its model.
    let (mut s, x, y) = setup();
    let t0 = Instant::now();
    for r in 0..ROUNDS {
        let rc = s.bv_const(r as u128 + 2, 32);
        let pc = s.bv_eq(y, rc);
        assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);
        let k = s.bv_const((r as u128 % 200) + 1, 32);
        let cond = s.bv_ult(x, k);
        let not_cond = s.bool_not(cond);
        let _ = s.solve_each_under_assumptions(&[cond, not_cond], &[pc]);
    }
    let batch = t0.elapsed();

    println!(
        "branch fan-out, {ROUNDS} rounds: individual {individual:?}, \
         batch(warm) {batch:?} ({:.1}x)",
        individual.as_secs_f64() / batch.as_secs_f64()
    );
}

#[test]
fn empty_candidates_is_a_no_op() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let five = s.bv_const(5, 8);
    let eq5 = s.bv_eq(x, five);
    s.assert(eq5);
    let res = s.solve_each_under_assumptions(&[], &[]);
    assert!(res.is_empty());
    assert_eq!(s.solve(), SmtResult::Sat);
}
