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
fn warm_survives_known_bits_probe() {
    // A known-bits probe (the symbex fast-range pipeline) rewinds the SAT
    // trail but changes no semantics: the banked model must still screen
    // a following batch with zero SAT work.
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

    // Trail-destroying, semantics-preserving interlude.
    assert!(s.bv_known_bits_under_assumptions(x, &[pc]).is_some());

    let xc = s.bv_const(xv, 32);
    let yc = s.bv_const(yv, 32);
    let c1 = s.bv_eq(x, xc);
    let c2 = s.bv_eq(y, yc);
    let before = s.sat_stats();
    let res = s.solve_each_under_assumptions(&[c1, c2], &[pc]);
    let after = s.sat_stats();
    assert_eq!(res, vec![SmtResult::Sat; 2]);
    assert_eq!(after.decisions, before.decisions, "probe killed the warm start");
    assert_eq!(after.sat_clauses, before.sat_clauses);
    assert_eq!(after.sat_vars, before.sat_vars);
}

#[test]
fn warm_survives_unsat_side_query() {
    // An Unsat solve only adds implied (learned) clauses — every model of
    // the old formula still models the new one, so the banked model must
    // keep warming batches afterwards.
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

    let not_below = s.bool_not(below);
    assert_eq!(s.solve_under_assumptions(&[not_below]), SmtResult::Unsat);

    let xc = s.bv_const(xv, 32);
    let yc = s.bv_const(yv, 32);
    let c1 = s.bv_eq(x, xc);
    let c2 = s.bv_eq(y, yc);
    let before = s.sat_stats();
    let res = s.solve_each_under_assumptions(&[c1, c2], &[pc]);
    let after = s.sat_stats();
    assert_eq!(res, vec![SmtResult::Sat; 2]);
    assert_eq!(after.decisions, before.decisions, "unsat query killed the warm start");
    assert_eq!(after.sat_clauses, before.sat_clauses);
}

#[test]
fn warm_survives_pop() {
    // Popping a scope retracts assertions — weakening — so the model from
    // a solve taken inside the scope stays valid outside it.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let y = s.bv_var(32);
    let hundred = s.bv_const(100, 32);
    let five = s.bv_const(5, 32);
    let below = s.bv_ult(x, hundred);
    s.assert(below);

    s.push();
    let above5 = s.bv_ult(five, x);
    s.assert(above5);
    let pc = s.bv_ult(y, hundred);
    assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);
    let xv = s.get_bv_value(x) as u128;
    let yv = s.get_bv_value(y) as u128;
    s.pop();

    let xc = s.bv_const(xv, 32);
    let yc = s.bv_const(yv, 32);
    let c1 = s.bv_eq(x, xc);
    let c2 = s.bv_eq(y, yc);
    let before = s.sat_stats();
    let res = s.solve_each_under_assumptions(&[c1, c2], &[pc]);
    let after = s.sat_stats();
    assert_eq!(res, vec![SmtResult::Sat; 2]);
    assert_eq!(after.decisions, before.decisions, "pop killed the warm start");
    assert_eq!(after.sat_clauses, before.sat_clauses);
}

#[test]
fn warm_carries_across_batches_without_reach_solve() {
    // The user-shaped loop: no reach solve at all. Batch 1 runs cold, but
    // its final internal Sat solve leaves a model that must warm batch 2.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let hundred = s.bv_const(100, 32);
    let below = s.bv_ult(x, hundred);
    s.assert(below);

    let fifty = s.bv_const(50, 32);
    let cond = s.bv_ult(x, fifty);
    let not_cond = s.bool_not(cond);
    let res1 = s.solve_each_under_assumptions(&[cond, not_cond], &[]);
    assert_eq!(res1, vec![SmtResult::Sat; 2]);

    // Whatever the last internal solve's model said, a candidate built
    // from it must screen for free in the next batch.
    let xv = s.get_bv_value(x) as u128;
    let xc = s.bv_const(xv, 32);
    let c = s.bv_eq(x, xc);
    let before = s.sat_stats();
    let res2 = s.solve_each_under_assumptions(&[c], &[]);
    let after = s.sat_stats();
    assert_eq!(res2, vec![SmtResult::Sat]);
    assert_eq!(after.decisions, before.decisions, "batch model didn't carry over");
    assert_eq!(after.sat_clauses, before.sat_clauses);
    assert_eq!(after.sat_vars, before.sat_vars);
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

/// Timing in the fast-solve regime: constraints so easy every SAT check
/// is near-pure propagation. Here the batch's fixed overhead (structural
/// model screening) competes directly with just running both solves, so
/// this pins the worst case for `solve_each` vs two independent solves.
///   cargo test --release --test solve_each -- --ignored --nocapture
#[test]
#[ignore]
fn bench_branch_fanout_trivial() {
    use std::time::Instant;

    const ROUNDS: u32 = 4000;

    fn setup() -> (SmtSolver, binbit::BvTerm, binbit::BvTerm) {
        let mut s = SmtSolver::new();
        let x = s.bv_var(16);
        let y = s.bv_var(16);
        let bound = s.bv_const(1000, 16);
        let below = s.bv_ult(x, bound);
        s.assert(below);
        (s, x, y)
    }

    // Two independent solves per round, no reach solve.
    let (mut s, x, y) = setup();
    let t0 = Instant::now();
    for r in 0..ROUNDS {
        let rc = s.bv_const(r as u128 % 500, 16);
        let pc = s.bv_eq(y, rc);
        let k = s.bv_const((r as u128 % 999) + 1, 16);
        let cond = s.bv_ult(x, k);
        let not_cond = s.bool_not(cond);
        let a = s.solve_under_assumptions(&[pc, cond]);
        let b = s.solve_under_assumptions(&[pc, not_cond]);
        assert!(a == SmtResult::Sat || b == SmtResult::Sat);
    }
    let individual = t0.elapsed();

    // One batch per round, same queries.
    let (mut s, x, y) = setup();
    let t0 = Instant::now();
    for r in 0..ROUNDS {
        let rc = s.bv_const(r as u128 % 500, 16);
        let pc = s.bv_eq(y, rc);
        let k = s.bv_const((r as u128 % 999) + 1, 16);
        let cond = s.bv_ult(x, k);
        let not_cond = s.bool_not(cond);
        let res = s.solve_each_under_assumptions(&[cond, not_cond], &[pc]);
        assert!(res.contains(&SmtResult::Sat));
    }
    let batch = t0.elapsed();

    // The pair-specialized entry point, same queries.
    let (mut s, x, y) = setup();
    let t0 = Instant::now();
    for r in 0..ROUNDS {
        let rc = s.bv_const(r as u128 % 500, 16);
        let pc = s.bv_eq(y, rc);
        let k = s.bv_const((r as u128 % 999) + 1, 16);
        let cond = s.bv_ult(x, k);
        let res = s.solve_pair_under_assumptions(cond, &[pc]);
        assert!(res.0 == SmtResult::Sat || res.1 == SmtResult::Sat);
    }
    let pair = t0.elapsed();

    println!(
        "trivial fan-out, {ROUNDS} rounds: individual {individual:?}, \
         batch {batch:?} ({:.2}x), pair {pair:?} ({:.2}x)",
        individual.as_secs_f64() / batch.as_secs_f64(),
        individual.as_secs_f64() / pair.as_secs_f64()
    );
}

/// Timing on the real symbex shape: (almost) no assertions, ALL path
/// constraints carried as assumptions, constraints accumulating between
/// branch points. Two variants: with a reach-solve of the constraints
/// right before each fan-out (warm intended), and without (the standing
/// model is stale — from the previous round's candidate solve).
///   cargo test --release --test solve_each -- --ignored --nocapture
#[test]
#[ignore]
fn bench_symbex_assumption_shape() {
    use std::time::Instant;

    const ROUNDS: u32 = 200;

    // One symbolic step per round: x_{r+1} = x_r * 3 + r (fresh var each
    // round, linked by an assumption), then branch on x_{r+1} < 2^16.
    // Mul by an odd constant is a bijection mod 2^32, so both branch
    // sides stay feasible and every solve does real inversion work.
    // `probe` inserts a known-bits probe (the fast-range pipeline)
    // between the reach solve and the fan-out — it rewinds the SAT trail
    // and used to kill the warm start.
    fn run(reach_solve: bool, probe: bool) -> std::time::Duration {
        run_impl(reach_solve, probe, false)
    }

    fn run_impl(reach_solve: bool, probe: bool, use_pair: bool) -> std::time::Duration {
        let mut s = SmtSolver::new();
        let mut x = s.bv_var(32);
        let mut constraints: Vec<BoolTerm> = Vec::new();
        let k = s.bv_const(1 << 16, 32);
        let three = s.bv_const(3, 32);
        let t0 = Instant::now();
        for r in 0..ROUNDS {
            let x2 = s.bv_var(32);
            let m = s.bv_mul(x, three);
            let rc = s.bv_const(r as u128, 32);
            let sum = s.bv_add(m, rc);
            let link = s.bv_eq(x2, sum);
            constraints.push(link);
            x = x2;

            if reach_solve {
                assert_eq!(
                    s.solve_under_assumptions(&constraints),
                    SmtResult::Sat
                );
            }
            if probe {
                assert!(s
                    .bv_known_bits_under_assumptions(x2, &constraints)
                    .is_some());
            }
            let cond = s.bv_ult(x2, k);
            let not_cond = s.bool_not(cond);
            let res = if use_pair {
                let (a, b) = s.solve_pair_under_assumptions(cond, &constraints);
                vec![a, b]
            } else {
                s.solve_each_under_assumptions(&[cond, not_cond], &constraints)
            };
            assert!(res.contains(&SmtResult::Sat));
            // Take a feasible branch, symbex-style.
            constraints.push(if res[0] == SmtResult::Sat { cond } else { not_cond });
        }
        t0.elapsed()
    }

    let no_reach = run(false, false);
    let with_reach = run(true, false);
    let with_probe = run(true, true);
    let pair_no_reach = run_impl(false, false, true);
    println!(
        "symbex shape, {ROUNDS} rounds: solve_each only {no_reach:?}, \
         reach-solve + solve_each {with_reach:?}, \
         reach-solve + known-bits probe + solve_each {with_probe:?}, \
         solve_pair only {pair_no_reach:?}"
    );
}

// ---------- solve_pair ----------

#[test]
fn pair_matches_individual_solves() {
    // Random branch conditions under random assumption sets: the pair
    // result must equal per-side solve_under_assumptions on a fresh
    // solver, warm or cold, and leave no residue.
    let mut rng = Rng::new(0xFA1);
    for _trial in 0..50 {
        let spec = TrialSpec {
            bound: 200 + rng.uniform_u32(200) as u128,
            cands: vec![(rng.uniform_u32(2) == 0, rng.uniform_u32(600) as u128)],
            asmps: (0..rng.uniform_u32(3) as usize)
                .map(|_| rng.uniform_u32(400) as u128)
                .collect(),
        };
        let expected: Vec<SmtResult> = [false, true]
            .iter()
            .map(|&neg| {
                let (mut fresh, cands, asmps) = build(&spec);
                let side = if neg { fresh.bool_not(cands[0]) } else { cands[0] };
                let mut a = asmps.clone();
                a.push(side);
                fresh.solve_under_assumptions(&a)
            })
            .collect();

        let (mut s, cands, asmps) = build(&spec);
        if rng.uniform_u32(2) == 0 {
            // Warm the solver with a base solve first.
            let _ = s.solve_under_assumptions(&asmps);
        }
        let (rc, rn) = s.solve_pair_under_assumptions(cands[0], &asmps);
        assert_eq!(vec![rc, rn], expected, "pair diverged from individual solves");

        // No residue.
        let (rc2, rn2) = s.solve_pair_under_assumptions(cands[0], &asmps);
        assert_eq!((rc2, rn2), (rc, rn), "pair left residue");
    }
}

#[test]
fn pair_unsat_base_is_one_result() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let one = s.bv_const(1, 8);
    let two = s.bv_const(2, 8);
    let a1 = s.bv_eq(x, one);
    let a2 = s.bv_eq(x, two);
    let c = s.bv_ult(x, two);
    let res = s.solve_pair_under_assumptions(c, &[a1, a2]);
    assert_eq!(res, (SmtResult::Unsat, SmtResult::Unsat));
    // Not poisoned.
    assert_eq!(s.solve_under_assumptions(&[a1]), SmtResult::Sat);
}

#[test]
fn pair_forced_sides_are_exact() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let ten = s.bv_const(10, 32);
    let below = s.bv_ult(x, ten);
    s.assert(below);
    let pc = below;
    assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);

    // cond covers all of pc's models: (Sat, Unsat).
    let twenty = s.bv_const(20, 32);
    let tauto = s.bv_ult(x, twenty);
    assert_eq!(
        s.solve_pair_under_assumptions(tauto, &[pc]),
        (SmtResult::Sat, SmtResult::Unsat)
    );
    // cond infeasible under pc: (Unsat, Sat).
    let infeasible = s.bv_eq(x, twenty);
    assert_eq!(
        s.solve_pair_under_assumptions(infeasible, &[pc]),
        (SmtResult::Unsat, SmtResult::Sat)
    );
}

#[test]
fn pair_known_base_matches_solve_pair() {
    // Whenever the base really is Sat, solve_pair_assuming_base_sat must
    // return exactly what solve_pair returns — warm or cold.
    let mut rng = Rng::new(0xBA5E);
    for _trial in 0..50 {
        let spec = TrialSpec {
            bound: 200 + rng.uniform_u32(200) as u128,
            cands: vec![(rng.uniform_u32(2) == 0, rng.uniform_u32(600) as u128)],
            asmps: (0..rng.uniform_u32(3) as usize)
                .map(|_| rng.uniform_u32(400) as u128)
                .collect(),
        };
        // These bases are always satisfiable (x = y = 0 works).
        let (mut a, cands_a, asmps_a) = build(&spec);
        if rng.uniform_u32(2) == 0 {
            let _ = a.solve_under_assumptions(&asmps_a);
        }
        let ra = a.solve_pair_assuming_base_sat(cands_a[0], &asmps_a);

        let (mut b, cands_b, asmps_b) = build(&spec);
        let rb = b.solve_pair_under_assumptions(cands_b[0], &asmps_b);
        assert_eq!(ra, rb, "known-base pair diverged from solve_pair");
    }
}

#[test]
fn pair_known_base_forced_branches() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let ten = s.bv_const(10, 32);
    let below = s.bv_ult(x, ten);
    s.assert(below);
    let pc = below;

    // Forced-false: x == 20 is impossible under x < 10 — one solve.
    let twenty = s.bv_const(20, 32);
    let eq20 = s.bv_eq(x, twenty);
    assert_eq!(
        s.solve_pair_assuming_base_sat(eq20, &[pc]),
        (SmtResult::Unsat, SmtResult::Sat)
    );
    // Forced-true: x < 20 covers all of pc's models.
    let lt20 = s.bv_ult(x, twenty);
    assert_eq!(
        s.solve_pair_assuming_base_sat(lt20, &[pc]),
        (SmtResult::Sat, SmtResult::Unsat)
    );
    // Both feasible.
    let five = s.bv_const(5, 32);
    let lt5 = s.bv_ult(x, five);
    assert_eq!(
        s.solve_pair_assuming_base_sat(lt5, &[pc]),
        (SmtResult::Sat, SmtResult::Sat)
    );
}

#[test]
fn pair_known_base_contract_on_violated_guarantee() {
    // Documented contract: with an Unsat base the second component is
    // vacuous (Unsat, Sat). This test pins the behavior so a change is
    // deliberate, not accidental.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let one = s.bv_const(1, 8);
    let two = s.bv_const(2, 8);
    let a1 = s.bv_eq(x, one);
    let a2 = s.bv_eq(x, two);
    let c = s.bv_ult(x, two);
    assert_eq!(
        s.solve_pair_assuming_base_sat(c, &[a1, a2]),
        (SmtResult::Unsat, SmtResult::Sat)
    );
    // Solver not poisoned.
    assert_eq!(s.solve_under_assumptions(&[a1]), SmtResult::Sat);
}

#[test]
fn solve_many_enumerates_exactly() {
    // Blocking clauses now backtrack one level instead of rewinding the
    // trail — the enumerated VALUE SET must stay exact regardless.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let ten = s.bv_const(10, 8);
    let three = s.bv_const(3, 8);
    let below = s.bv_ult(x, ten);
    s.assert(below);
    let ge3 = s.bv_ule(three, x);
    let (mut vals, exhausted) = s.solve_many_u_under_assumptions(x, 100, &[ge3]);
    vals.sort();
    assert!(exhausted);
    assert_eq!(vals, vec![3, 4, 5, 6, 7, 8, 9]);
    // No residue: enumeration again gives the same set.
    let (mut vals2, ex2) = s.solve_many_u_under_assumptions(x, 100, &[ge3]);
    vals2.sort();
    assert!(ex2);
    assert_eq!(vals2, vec![3, 4, 5, 6, 7, 8, 9]);
    // And the solver still answers unrelated queries.
    let nine = s.bv_const(9, 8);
    let eq9 = s.bv_eq(x, nine);
    assert_eq!(s.solve_under_assumptions(&[eq9]), SmtResult::Sat);
}

/// Enumeration timing: values of a 12-bit var. Blocking clauses used to
/// rewind the whole trail per value; now each value costs a one-level
/// backtrack.
///   cargo test --release --test solve_each -- --ignored --nocapture
#[test]
#[ignore]
fn bench_solve_many_enumeration() {
    use std::time::Instant;
    let mut s = SmtSolver::new();
    let x = s.bv_var(12);
    let y = s.bv_var(12);
    let sum = s.bv_add(x, y);
    let bound = s.bv_const(3000, 12);
    let below = s.bv_ult(sum, bound);
    s.assert(below);
    let k = s.bv_const(2048, 12);
    let yk = s.bv_ult(y, k);
    let t0 = Instant::now();
    let (vals, _) = s.solve_many_u_under_assumptions(x, 3000, &[yk]);
    println!(
        "solve_many: {} values in {:?}",
        vals.len(),
        t0.elapsed()
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
