//! Profiling driver: the symbex assumption shape (solve_pair per round,
//! growing constraint list, periodic retirement) run long enough to
//! sample. Not a benchmark — numbers here are for `sample`/Instruments.
//!   cargo build --release --example profile_symbex
//!   ./target/release/examples/profile_symbex [rounds] [retire_every]

use binbit::{BoolTerm, SmtResult, SmtSolver};

fn main() {
    let mut args = std::env::args().skip(1);
    let rounds: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(3000);
    let retire_every: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let cores: bool = args.next().map(|a| a == "cores").unwrap_or(false);

    let mut s = SmtSolver::new();
    s.set_core_tracking(cores);
    if std::env::var_os("BINBIT_CNFMAP").is_some() {
        s.set_cnf_mapping(true);
    }
    let mut x = s.bv_var(32);
    let three = s.bv_const(3, 32);
    let k = s.bv_const(1 << 16, 32);
    let mut constraints: Vec<BoolTerm> = Vec::new();
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
        // The taken-branch invariant makes the base always feasible —
        // the shape solve_pair_assuming_base_sat is for.
        let (a, b) = s.solve_pair_assuming_base_sat(cond, &constraints);
        assert!(a == SmtResult::Sat || b == SmtResult::Sat);
        constraints.push(if a == SmtResult::Sat { cond } else { s.bool_not(cond) });

        if retire_every > 0 && r % retire_every == retire_every - 1 {
            // Whole-history live set: retirement here only drops cones the
            // solver itself abandoned (screened-only sides), keeping the
            // profile honest for the growing-constraints shape.
            let live = constraints.clone();
            s.retire_dead_cones(&live);
        }
    }
    println!("{} rounds in {:?}", rounds, t0.elapsed());
}
