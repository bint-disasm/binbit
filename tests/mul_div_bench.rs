//! Observational benchmarks for the mul/div optimizations. Not asserted —
//! run with:
//!     cargo test --release --test mul_div_bench -- --ignored --nocapture

use binbit::{SmtResult, SmtSolver};
use std::time::Instant;

fn time<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t0 = Instant::now();
    let r = f();
    (r, t0.elapsed().as_secs_f64())
}

// Force a heavy multiply: find x, y < 2^K in width-N such that x * y == TARGET.
// Drives the solver through the Wallace-tree multiplier with truly symbolic
// operands on both sides.
fn mul_search_time(width: u32, k_upper_bits: u32, target: u128) -> f64 {
    let (_, t) = time(|| {
        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let y = s.bv_var(width);
        let bound = s.bv_const(1u128 << k_upper_bits, width);
        let lt1 = s.bv_ult(x, bound);
        let lt2 = s.bv_ult(y, bound);
        s.assert(lt1);
        s.assert(lt2);
        let prod = s.bv_mul(x, y);
        let target_c = s.bv_const(target, width);
        let eq = s.bv_eq(prod, target_c);
        s.assert(eq);
        let _ = s.solve();
    });
    t
}

// Force a constant-divisor divide through the magic-number path: find x
// with x / d == q for fixed constants d, q.
fn udiv_const_time(width: u32, d: u128, q: u128) -> f64 {
    let (_, t) = time(|| {
        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let d_const = s.bv_const(d, width);
        let q_const = s.bv_const(q, width);
        let quot = s.bv_udiv(x, d_const);
        let eq = s.bv_eq(quot, q_const);
        s.assert(eq);
        let res = s.solve();
        assert_eq!(res, SmtResult::Sat);
    });
    t
}

// Force a symbolic-divisor divide: both x and y unknown.
fn udiv_symbolic_time(width: u32, x_val: u128, y_val: u128) -> f64 {
    let expected = if y_val == 0 {
        (1u128 << width) - 1
    } else {
        x_val / y_val
    };
    let (_, t) = time(|| {
        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let y = s.bv_var(width);
        let cx = s.bv_const(x_val, width);
        let cy = s.bv_const(y_val, width);
        let eq_x = s.bv_eq(x, cx);
        let eq_y = s.bv_eq(y, cy);
        s.assert(eq_x);
        s.assert(eq_y);
        let q = s.bv_udiv(x, y);
        let ce = s.bv_const(expected, width);
        let eq = s.bv_eq(q, ce);
        s.assert(eq);
        let res = s.solve();
        assert_eq!(res, SmtResult::Sat);
    });
    t
}

#[test]
#[ignore]
fn bench_multiply() {
    println!("\n== multiply: variable × variable, solve time ==");
    for &w in &[16u32, 24, 32] {
        // Search for factors of a product. Low enough to complete fast but
        // non-trivial — at w=32 this still exercises the full tree.
        let t1 = mul_search_time(w, w - 2, 123456);
        let t2 = mul_search_time(w, w - 2, 987654);
        println!(
            "  w={:3}  target1={:>7.4}s  target2={:>7.4}s",
            w, t1, t2
        );
    }
}

// Same as udiv_const_time but routes the divisor through a Var so the
// builder's const-fold / magic path doesn't fire — this is the "before"
// measurement (full restoring-style bitblasted division).
fn udiv_disguised_const_time(width: u32, d: u128, q: u128) -> f64 {
    let (_, t) = time(|| {
        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let y = s.bv_var(width);
        let d_const = s.bv_const(d, width);
        let eq_y = s.bv_eq(y, d_const);
        s.assert(eq_y);
        let q_const = s.bv_const(q, width);
        let quot = s.bv_udiv(x, y); // y is a Var, magic skipped
        let eq = s.bv_eq(quot, q_const);
        s.assert(eq);
        let _ = s.solve();
    });
    t
}

#[test]
#[ignore]
fn bench_constant_divide() {
    println!("\n== udiv: symbolic dividend / constant divisor (magic vs bitblasted) ==");
    for &w in &[16u32, 32, 64] {
        let q: u128 = 5;
        for &d in &[3u128, 10, 100, 1000] {
            let t_bb = udiv_disguised_const_time(w, d, q);
            let t_mg = udiv_const_time(w, d, q);
            let speedup = if t_mg > 0.0 { t_bb / t_mg } else { 0.0 };
            println!(
                "  w={:>3} d={:>4}  bitblasted: {:7.4}s   magic: {:7.4}s   speedup: {:.1}x",
                w, d, t_bb, t_mg, speedup
            );
        }
    }
}

#[test]
#[ignore]
fn bench_symbolic_divide() {
    println!("\n== udiv: symbolic / symbolic (non-restoring path) ==");
    for &w in &[8u32, 16, 24, 32] {
        let t1 = udiv_symbolic_time(w, 100, 7);
        let t2 = udiv_symbolic_time(w, (1u128 << (w.min(32) - 1)) + 3, 13);
        println!("  w={:3}  case1={:>7.4}s  case2={:>7.4}s", w, t1, t2);
    }
}
