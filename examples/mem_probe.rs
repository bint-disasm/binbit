//! Memory-scoping probe: a sliding-window symbex session with retirement
//! and var recycling active. With the SAT side capped, residual RSS
//! growth per round measures the AIG/term-DAG accretion — the input for
//! deciding whether term-graph reclamation is worth building.
//!   /usr/bin/time -l ./target/release/examples/mem_probe 4000

use binbit::{BoolTerm, SmtResult, SmtSolver};

fn main() {
    let rounds: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4000);
    const WINDOW: usize = 24;
    const RETIRE_EVERY: u32 = 64;

    let mut s = SmtSolver::new();
    s.set_core_tracking(false);
    let mut x = s.bv_var(32);
    let three = s.bv_const(3, 32);
    let k = s.bv_const(1 << 16, 32);
    let mut constraints: Vec<BoolTerm> = Vec::new();
    for r in 0..rounds {
        let x2 = s.bv_var(32);
        let m = s.bv_mul(x, three);
        let rc = s.bv_const(r as u128, 32);
        let sum = s.bv_add(m, rc);
        let link = s.bv_eq(x2, sum);
        constraints.push(link);
        x = x2;
        let live_from = constraints.len().saturating_sub(WINDOW);
        let cond = s.bv_ult(x2, k);
        let (a, b) = s.solve_pair_assuming_base_sat(cond, &constraints[live_from..]);
        assert!(a == SmtResult::Sat || b == SmtResult::Sat);
        if r % RETIRE_EVERY == RETIRE_EVERY - 1 {
            let live: Vec<BoolTerm> = constraints[live_from..].to_vec();
            s.retire_dead_cones(&live);
        }
    }
    println!(
        "{} rounds: sat_vars={} sat_clauses={}",
        rounds,
        s.sat_stats().sat_vars,
        s.sat_stats().sat_clauses
    );
}
