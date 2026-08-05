//! Is cut-mapping a one-time cost in incremental use?
//!
//! Splits an incremental session into two phases and times them apart:
//!   BUILD  — each round introduces a fresh cone (new AIG nodes), so the
//!            mapper plans new structure every round.
//!   REQUERY — the same terms are re-queried with no new structure, so
//!            every cone is already materialized and mapping should be
//!            entirely absent from the loop.
//!
//!   ./target/release/examples/cnfmap_amortize [rounds] [requeries]

use binbit::{BoolTerm, SmtResult, SmtSolver};

fn phase_times(mapped: bool, rounds: u32, requeries: u32) -> (f64, f64) {
    let mut s = SmtSolver::new();
    s.set_core_tracking(false);
    s.set_cnf_mapping(mapped);
    let mut x = s.bv_var(32);
    let three = s.bv_const(3, 32);
    let k = s.bv_const(1 << 16, 32);
    let mut constraints: Vec<BoolTerm> = Vec::new();
    let mut conds: Vec<BoolTerm> = Vec::new();

    let t0 = std::time::Instant::now();
    for r in 0..rounds {
        let x2 = s.bv_var(32);
        let m = s.bv_mul(x, three);
        let rc = s.bv_const(r as u128, 32);
        let sum = s.bv_add(m, rc);
        let link = s.bv_eq(x2, sum);
        constraints.push(link);
        x = x2;
        let cond = s.bv_ult(x2, k);
        conds.push(cond);
        let (a, b) = s.solve_pair_assuming_base_sat(cond, &constraints);
        assert!(a == SmtResult::Sat || b == SmtResult::Sat);
    }
    let build = t0.elapsed().as_secs_f64();

    // Re-query existing structure only: no new terms, so no new cones.
    let t1 = std::time::Instant::now();
    for q in 0..requeries {
        let i = (q as usize * 7919) % conds.len();
        let (a, b) = s.solve_pair_assuming_base_sat(conds[i], &constraints);
        assert!(a == SmtResult::Sat || b == SmtResult::Sat);
    }
    let requery = t1.elapsed().as_secs_f64();
    (build, requery)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let rounds: u32 = a.next().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let requeries: u32 = a.next().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let (cb, cq) = phase_times(false, rounds, requeries);
    let (mb, mq) = phase_times(true, rounds, requeries);
    println!(
        "build   ({rounds} new cones): classic {cb:.3}s  mapped {mb:.3}s  ({:.2}x)",
        mb / cb
    );
    println!(
        "requery ({requeries} old cones): classic {cq:.3}s  mapped {mq:.3}s  ({:.2}x)",
        mq / cq
    );
}
