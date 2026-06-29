//! Verifies the constant-narrowing rewrites for bvult / bvule preserve
//! semantics across both LHS-constant and RHS-constant positions, edge
//! constants (0 / 1 / 2^k / mask), and a range of widths.

use binbit::{SmtResult, SmtSolver};

fn check_against_oracle<F, G>(
    width: u32,
    constants: &[u128],
    build: F,
    oracle: G,
) where
    F: Fn(&mut SmtSolver, binbit::BvTerm, binbit::BvTerm) -> binbit::BoolTerm,
    G: Fn(u128, u128, u32) -> bool,
{
    let m = if width >= 128 { u128::MAX } else { (1u128 << width) - 1 };
    let value_samples = [
        0u128,
        1,
        m / 2,
        m / 2 + 1,
        m - 1,
        m,
    ];
    for &c in constants {
        for &v in &value_samples {
            // RHS constant.
            {
                let mut s = SmtSolver::new();
                let x = s.bv_var(width);
                let cterm = s.bv_const(c, width);
                let p = build(&mut s, x, cterm);
                let xc = s.bv_const(v, width);
                let eq = s.bv_eq(x, xc);
                s.assert(eq);
                if oracle(v & m, c & m, width) {
                    s.assert(p);
                } else {
                    let np = s.bool_not(p);
                    s.assert(np);
                }
                if s.solve() != SmtResult::Sat {
                    panic!(
                        "RHS-const oracle disagrees: width={} v=0x{:x} c=0x{:x}",
                        width, v, c
                    );
                }
            }
            // LHS constant.
            {
                let mut s = SmtSolver::new();
                let x = s.bv_var(width);
                let cterm = s.bv_const(c, width);
                let p = build(&mut s, cterm, x);
                let xc = s.bv_const(v, width);
                let eq = s.bv_eq(x, xc);
                s.assert(eq);
                if oracle(c & m, v & m, width) {
                    s.assert(p);
                } else {
                    let np = s.bool_not(p);
                    s.assert(np);
                }
                if s.solve() != SmtResult::Sat {
                    panic!(
                        "LHS-const oracle disagrees: width={} v=0x{:x} c=0x{:x}",
                        width, v, c
                    );
                }
            }
        }
    }
}

#[test]
fn bvult_8bit_const_narrowing_matches_oracle() {
    check_against_oracle(
        8,
        &[0, 1, 2, 16, 0x40, 0x80, 0xff],
        |s, a, b| s.bv_ult(a, b),
        |a, b, _w| a < b,
    );
}

#[test]
fn bvult_16bit_const_narrowing_matches_oracle() {
    check_against_oracle(
        16,
        &[0, 1, 0x100, 0x1000, 0x7fff, 0x8000, 0xffff],
        |s, a, b| s.bv_ult(a, b),
        |a, b, _w| a < b,
    );
}

#[test]
fn bvult_32bit_const_narrowing_matches_oracle() {
    check_against_oracle(
        32,
        &[0, 0x100, 0x10000, 0x7fffffff, 0x80000000, 0xffffffff],
        |s, a, b| s.bv_ult(a, b),
        |a, b, _w| a < b,
    );
}

#[test]
fn bvule_8bit_const_narrowing_matches_oracle() {
    check_against_oracle(
        8,
        &[0, 1, 2, 16, 0x40, 0x80, 0xff],
        |s, a, b| s.bv_ule(a, b),
        |a, b, _w| a <= b,
    );
}

#[test]
fn bvule_16bit_const_narrowing_matches_oracle() {
    check_against_oracle(
        16,
        &[0, 1, 0x100, 0x1000, 0x7fff, 0x8000, 0xffff],
        |s, a, b| s.bv_ule(a, b),
        |a, b, _w| a <= b,
    );
}

#[test]
fn bvule_32bit_const_narrowing_matches_oracle() {
    check_against_oracle(
        32,
        &[0, 0x100, 0x10000, 0x7fffffff, 0x80000000, 0xffffffff],
        |s, a, b| s.bv_ule(a, b),
        |a, b, _w| a <= b,
    );
}

#[test]
fn bvult_edge_cases() {
    // x < 0 always false.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let zero = s.bv_const(0, 32);
    let p = s.bv_ult(x, zero);
    s.assert(p);
    assert_eq!(s.solve(), SmtResult::Unsat);

    // 0 < x iff x != 0.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let zero = s.bv_const(0, 32);
    let p = s.bv_ult(zero, x);
    let zero_const = s.bv_const(0, 32);
    let x_is_zero = s.bv_eq(x, zero_const);
    s.assert(p);
    s.assert(x_is_zero);
    assert_eq!(s.solve(), SmtResult::Unsat);

    // mask < x always false.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let mask = s.bv_const(0xffffffff, 32);
    let p = s.bv_ult(mask, x);
    s.assert(p);
    assert_eq!(s.solve(), SmtResult::Unsat);
}
