//! Incremental queries against a LARGE formula — the shape the .smt2
//! dumps can't show (they are single-shot) and the synthetic symbex
//! driver can't show (its formulas are tiny).
//!
//! Builds a wide formula with many SAT variables, then runs many small
//! `solve_under_assumptions` queries over it. Per-query costs that scale
//! with TOTAL formula size (rather than with the query) show up here:
//! notably the banked-model snapshot, which copies the whole assignment
//! table on every solve that produces a new model.
//!
//!   ./target/release/examples/incr_bigformula [vars_k] [queries]

use binbit::{BoolTerm, SmtResult, SmtSolver};

fn main() {
    let mut a = std::env::args().skip(1);
    let vars_k: u32 = a.next().and_then(|v| v.parse().ok()).unwrap_or(40);
    let queries: u32 = a.next().and_then(|v| v.parse().ok()).unwrap_or(300);

    // NOTE: deliberately uses only API that exists in pre-session
    // builds, so the same driver compiles against both for A/B.
    let mut s = SmtSolver::new();

    // A wide, easy formula: many independent 32-bit variables each
    // constrained to a small range. Big variable count, trivial search.
    let mut xs = Vec::new();
    let lim = s.bv_const(1000, 32);
    for _ in 0..vars_k * 1000 / 32 {
        let x = s.bv_var(32);
        let c = s.bv_ult(x, lim);
        s.assert(c);
        xs.push(x);
    }
    let t_build = std::time::Instant::now();
    assert_eq!(s.solve(), SmtResult::Sat);
    let build = t_build.elapsed().as_secs_f64();

    // Many small queries over the existing formula.
    let t_q = std::time::Instant::now();
    for q in 0..queries {
        let i = (q as usize * 7919) % xs.len();
        let k = s.bv_const((q as u128 % 900) + 1, 32);
        let cond: BoolTerm = s.bv_ult(xs[i], k);
        assert_eq!(s.solve_under_assumptions(&[cond]), SmtResult::Sat);
    }
    let qt = t_q.elapsed().as_secs_f64();
    println!(
        "vars={} build={:.3}s  {} queries={:.3}s  ({:.3} ms/query)",
        s.sat_stats().sat_vars,
        build,
        queries,
        qt,
        qt * 1000.0 / queries as f64
    );
}
