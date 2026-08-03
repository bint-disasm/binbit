//! Tests for `retire_dead_cones`: dropping the CNF of assumption cones no
//! live state can reach, so per-query cost tracks the live constraint set
//! instead of the whole session's history.

use binbit::{BoolTerm, SmtResult, SmtSolver};

/// Build `count` chained constraints `x_{i+1} == x_i * 3 + i` over fresh
/// vars, returning (solver, link terms, final var).
fn chain(count: u32) -> (SmtSolver, Vec<BoolTerm>, binbit::BvTerm) {
    let mut s = SmtSolver::new();
    let mut x = s.bv_var(32);
    let three = s.bv_const(3, 32);
    let mut links = Vec::new();
    for i in 0..count {
        let x2 = s.bv_var(32);
        let m = s.bv_mul(x, three);
        let ic = s.bv_const(i as u128, 32);
        let sum = s.bv_add(m, ic);
        let link = s.bv_eq(x2, sum);
        links.push(link);
        x = x2;
    }
    (s, links, x)
}

#[test]
fn retire_preserves_live_answers() {
    // Solve under all links, retire everything but the last few, and the
    // answers under the live suffix must match a fresh solver's.
    let (mut s, links, x) = chain(12);
    assert_eq!(s.solve_under_assumptions(&links), SmtResult::Sat);

    let live: Vec<BoolTerm> = links[8..].to_vec();
    let (retired, deleted) = s.retire_dead_cones(&live);
    assert!(retired > 0, "nothing retired");
    assert!(deleted > 0, "no clauses deleted");

    let k = s.bv_const(1 << 16, 32);
    let cond = s.bv_ult(x, k);
    let not_cond = s.bool_not(cond);
    let got = s.solve_each_under_assumptions(&[cond, not_cond], &live);

    let (mut fresh, flinks, fx) = chain(12);
    let flive: Vec<BoolTerm> = flinks[8..].to_vec();
    let fk = fresh.bv_const(1 << 16, 32);
    let fcond = fresh.bv_ult(fx, fk);
    let fnot = fresh.bool_not(fcond);
    let want = fresh.solve_each_under_assumptions(&[fcond, fnot], &flive);
    assert_eq!(got, want, "retirement changed live answers");
}

#[test]
fn retired_cone_rematerializes() {
    // A retired constraint used again must re-emit and answer correctly.
    let (mut s, links, _x) = chain(6);
    assert_eq!(s.solve_under_assumptions(&links), SmtResult::Sat);
    let (retired, _) = s.retire_dead_cones(&[]);
    assert!(retired > 0);

    // Every link individually is satisfiable again (fresh cones).
    for &l in &links {
        assert_eq!(s.solve_under_assumptions(&[l]), SmtResult::Sat);
    }
    // And the full conjunction still works.
    assert_eq!(s.solve_under_assumptions(&links), SmtResult::Sat);
}

#[test]
fn assertions_survive_retirement() {
    // Asserted cones are pinned: retiring with an empty live set must not
    // weaken the asserted formula.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let ten = s.bv_const(10, 32);
    let below = s.bv_ult(x, ten);
    s.assert(below);
    assert_eq!(s.solve(), SmtResult::Sat);

    // Build a fat dead cone.
    let y = s.bv_var(32);
    let prod = s.bv_mul(x, y);
    let big = s.bv_const(1 << 20, 32);
    let dead = s.bv_ult(prod, big);
    assert_eq!(s.solve_under_assumptions(&[dead]), SmtResult::Sat);

    let (retired, _) = s.retire_dead_cones(&[]);
    assert!(retired > 0, "dead mul cone not retired");

    // The assertion still constrains x.
    let twenty = s.bv_const(20, 32);
    let eq20 = s.bv_eq(x, twenty);
    assert_eq!(s.solve_under_assumptions(&[eq20]), SmtResult::Unsat);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert!(s.get_bv_value(x) < 10);
}

#[test]
fn banked_model_survives_retirement() {
    // Retirement only weakens the clause set, so a banked model must keep
    // warming batches whose assumptions stayed live.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let hundred = s.bv_const(100, 32);
    let below = s.bv_ult(x, hundred);
    s.assert(below);

    let y = s.bv_var(32);
    let pc = s.bv_ult(y, hundred);
    assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);
    let xv = s.get_bv_value(x) as u128;
    let yv = s.get_bv_value(y) as u128;

    // A dead cone to retire.
    let prod = s.bv_mul(x, y);
    let big = s.bv_const(1 << 20, 32);
    let dead = s.bv_ult(prod, big);
    let _ = s.bv_known_bits_under_assumptions(prod, &[dead]);

    let (retired, _) = s.retire_dead_cones(&[pc]);
    assert!(retired > 0);

    let xc = s.bv_const(xv, 32);
    let yc = s.bv_const(yv, 32);
    let c1 = s.bv_eq(x, xc);
    let c2 = s.bv_eq(y, yc);
    let before = s.sat_stats();
    let res = s.solve_each_under_assumptions(&[c1, c2], &[pc]);
    let after = s.sat_stats();
    assert_eq!(res, vec![SmtResult::Sat; 2]);
    assert_eq!(after.decisions, before.decisions, "retirement killed the warm start");
}

#[test]
fn retire_then_reuse_input_vars() {
    // Retiring a cone un-branches its input bits; a new cone over the
    // same BV variable must re-enable them and solve correctly.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let five = s.bv_const(5, 32);
    let eq5 = s.bv_eq(x, five);
    assert_eq!(s.solve_under_assumptions(&[eq5]), SmtResult::Sat);

    let (retired, _) = s.retire_dead_cones(&[]);
    assert!(retired > 0);

    let seven = s.bv_const(7, 32);
    let eq7 = s.bv_eq(x, seven);
    assert_eq!(s.solve_under_assumptions(&[eq7]), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 7);
    let three = s.bv_const(3, 32);
    let eq3 = s.bv_eq(x, three);
    let both = s.bool_and(eq7, eq3);
    assert_eq!(s.solve_under_assumptions(&[both]), SmtResult::Unsat);
}

#[test]
fn recycling_keeps_var_count_flat() {
    // Retire/re-materialize cycles must reuse SAT variable ids: the
    // per-var tables (and the banked-model snapshot) stay bounded by the
    // live set instead of growing with session length.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let hundred = s.bv_const(100, 32);
    let below = s.bv_ult(x, hundred);
    s.assert(below);

    let mut peak_after_first_cycle = 0;
    for round in 0..12u32 {
        let y = s.bv_var(32);
        let prod = s.bv_mul(x, y);
        let big = s.bv_const(1 << 20, 32);
        let c = s.bv_ult(prod, big);
        assert_eq!(s.solve_under_assumptions(&[c]), SmtResult::Sat);
        let (retired, _) = s.retire_dead_cones(&[]);
        assert!(retired > 0);
        let vars = s.sat_stats().sat_vars;
        if round == 0 {
            peak_after_first_cycle = vars;
        } else {
            // Fresh y-bits per round can't be recycled (32/round), but the
            // big mul cone's gate vars must be. Allow the input growth
            // plus slack, nothing near a full cone per round.
            let budget = peak_after_first_cycle + (round as usize + 1) * 40;
            assert!(
                vars <= budget,
                "vars grew past recycling budget: {vars} > {budget} at round {round}"
            );
        }
    }
}

#[test]
fn warm_screening_correct_after_recycling() {
    // A recycled variable's banked-model slots are poisoned, so warm
    // screening must recompute recycled cones structurally — answers
    // must match a fresh solver even when ids were reused.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let hundred = s.bv_const(100, 32);
    let below = s.bv_ult(x, hundred);
    s.assert(below);

    // Materialize and retire a fat cone to seed the free list.
    let y = s.bv_var(32);
    let prod = s.bv_mul(x, y);
    let big = s.bv_const(1 << 20, 32);
    let dead = s.bv_ult(prod, big);
    assert_eq!(s.solve_under_assumptions(&[dead]), SmtResult::Sat);
    let (retired, _) = s.retire_dead_cones(&[]);
    assert!(retired > 0);

    // Bank a model of pc, then build NEW cones over recycled ids and
    // screen them warm.
    let pc = s.bv_ult(y, hundred);
    assert_eq!(s.solve_under_assumptions(&[pc]), SmtResult::Sat);
    let xv = s.get_bv_value(x) as u128;
    let sum = s.bv_add(x, y);
    let k = s.bv_const(1 << 10, 32);
    let c1 = s.bv_ult(sum, k);
    let not_c1 = s.bool_not(c1);
    let got = s.solve_each_under_assumptions(&[c1, not_c1], &[pc]);

    let mut f = SmtSolver::new();
    let fx = f.bv_var(32);
    let fh = f.bv_const(100, 32);
    let fb = f.bv_ult(fx, fh);
    f.assert(fb);
    let fy = f.bv_var(32);
    let fpc = f.bv_ult(fy, fh);
    let fsum = f.bv_add(fx, fy);
    let fk = f.bv_const(1 << 10, 32);
    let fc1 = f.bv_ult(fsum, fk);
    let fnot = f.bool_not(fc1);
    let want = f.solve_each_under_assumptions(&[fc1, fnot], &[fpc]);
    assert_eq!(got, want, "recycled-id warm screening diverged");
    let _ = xv;
}

/// Accretion bench: a symbex-style session whose live window is a sliding
/// suffix of the constraint chain. Without retirement the solver keeps
/// paying for every cone ever built; with periodic retirement per-query
/// cost tracks the window.
///   cargo test --release --test retire -- --ignored --nocapture
#[test]
#[ignore]
fn bench_sliding_window_retirement() {
    use std::time::Instant;

    const ROUNDS: u32 = 1200;
    const WINDOW: usize = 24;
    const RETIRE_EVERY: u32 = 64;

    fn run(retire: bool) -> std::time::Duration {
        let mut s = SmtSolver::new();
        let mut x = s.bv_var(32);
        let three = s.bv_const(3, 32);
        let k = s.bv_const(1 << 16, 32);
        let mut constraints: Vec<BoolTerm> = Vec::new();
        let t0 = Instant::now();
        for r in 0..ROUNDS {
            let x2 = s.bv_var(32);
            let m = s.bv_mul(x, three);
            let rc = s.bv_const(r as u128, 32);
            let sum = s.bv_add(m, rc);
            let link = s.bv_eq(x2, sum);
            constraints.push(link);
            x = x2;
            // Live window: only the most recent constraints matter (the
            // symbex analog: shallow-suffix path constraints of the
            // states still queued).
            let live_from = constraints.len().saturating_sub(WINDOW);
            let live = &constraints[live_from..];

            let cond = s.bv_ult(x2, k);
            let (a, b) = s.solve_pair_under_assumptions(cond, live);
            assert!(a == SmtResult::Sat || b == SmtResult::Sat);

            if retire && r % RETIRE_EVERY == RETIRE_EVERY - 1 {
                let live_vec: Vec<BoolTerm> = live.to_vec();
                s.retire_dead_cones(&live_vec);
            }
        }
        t0.elapsed()
    }

    let no_retire = run(false);
    let with_retire = run(true);
    println!(
        "sliding window, {ROUNDS} rounds: no retirement {no_retire:?}, \
         retire every {RETIRE_EVERY} {with_retire:?} ({:.1}x)",
        no_retire.as_secs_f64() / with_retire.as_secs_f64()
    );
}
