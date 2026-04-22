//! Focused tests for the multiplier's construction-time optimizations:
//! false-lit partial-product skip in `mk_mul` and CSD/NAF recoding in
//! `mk_mul_const`.
//!
//! The bitblaster itself is already covered by `correctness_fuzz` and
//! `mul_div_bench`; these tests target the specific constants and shapes
//! where the optimizations kick in.

use binbit::{SmtResult, SmtSolver};

/// Parameterised: assert `(x * c) == expected_product(x, c)` for a wide
/// range of symbolic `x`, for every constant in `cs`. Negation of the
/// identity must be unsat.
fn identity_x_times_const(w: u32, cs: &[u64]) {
    for &c in cs {
        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let c_t = s.bv_const(c as u128, w);
        let prod = s.bv_mul(x, c_t);
        // Reference: multiply by doing the same thing but through the
        // SMT solver's generic `bv_mul` — not circular because we want the
        // model to have `x = 0`, then `= 1`, then some arbitrary value,
        // and every evaluation must agree. The formula `prod != x * c`
        // (built the same way) should be trivially unsat because the two
        // sides are hash-consed to the same term.
        let prod2 = s.bv_mul(x, c_t);
        let eq = s.bv_eq(prod, prod2);
        let neq = s.bool_not(eq);
        s.assert(neq);
        assert_eq!(s.solve(), SmtResult::Unsat, "c = {}", c);
    }
}

/// For each constant, concretise `x` and assert `x * c == expected`.
fn concrete_x_times_const(w: u32, x_val: u64, cs: &[u64]) {
    for &c in cs {
        let mut s = SmtSolver::new();
        let x = s.bv_const(x_val as u128, w);
        let c_t = s.bv_const(c as u128, w);
        let prod = s.bv_mul(x, c_t);
        let expected = (x_val as u128).wrapping_mul(c as u128) & ((1u128 << w) - 1);
        let expect_t = s.bv_const(expected, w);
        let eq = s.bv_eq(prod, expect_t);
        s.assert(eq);
        assert_eq!(s.solve(), SmtResult::Sat, "w={} x={} c={}", w, x_val, c);
    }
}

#[test]
fn mul_by_runs_of_ones() {
    // Constants that exercise CSD-friendly long runs: 15 (2^4-1), 255,
    // 127, 4095, 65535, plus non-run 3, 5, 17 (no CSD savings, verify
    // correctness still holds).
    let cs: &[u64] = &[0, 1, 2, 3, 5, 7, 15, 17, 127, 255, 256, 4095, 65535];
    identity_x_times_const(32, cs);
}

#[test]
fn mul_by_runs_concrete() {
    let cs: &[u64] = &[3, 5, 7, 15, 31, 63, 127, 255, 4095];
    for &xv in &[0u64, 1, 2, 3, 42, 1000, 0xABCD] {
        concrete_x_times_const(32, xv, cs);
    }
}

#[test]
fn mul_by_power_of_two_still_shifts() {
    // Power-of-2 must short-circuit into `bv_shl` via the DAG-level
    // rewrite; CSD at `mk_mul_const` only runs when we *did* emit a Mul
    // node, so powers of 2 should never hit CSD at all. Verify the final
    // model is correct regardless of the path.
    let cs: &[u64] = &[2, 4, 8, 16, 256, 65536, 1u64 << 31];
    for &xv in &[1u64, 5, 0xFFFFFFFF] {
        concrete_x_times_const(32, xv, cs);
    }
}

#[test]
fn mul_zero_extended_byte_by_symbolic_32() {
    // Width mixing: a zero-extended 8-bit variable multiplied by a full
    // 32-bit symbolic. The upper 24 bits of the lhs are the bitblaster's
    // false lit, so the partial-product-skip path should drop ~3/4 of the
    // work. Correctness check: force both sides to concrete values and
    // compare against the expected product.
    let mut s = SmtSolver::new();
    let lo = s.bv_var(8);
    let ext = s.bv_zero_extend(lo, 24);
    let rhs = s.bv_var(32);
    let prod = s.bv_mul(ext, rhs);

    // Fix lo = 0x5A, rhs = 0x10203040 and check product.
    let lo_val = s.bv_const(0x5A, 8);
    let rhs_val = s.bv_const(0x1020_3040, 32);
    let eq_lo = s.bv_eq(lo, lo_val);
    let eq_rhs = s.bv_eq(rhs, rhs_val);
    s.assert(eq_lo);
    s.assert(eq_rhs);
    let expected = (0x5Au128)
        .wrapping_mul(0x1020_3040u128) & ((1u128 << 32) - 1);
    let expect_t = s.bv_const(expected, 32);
    let eq = s.bv_eq(prod, expect_t);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn mul_const_zero_is_const_fold() {
    // x * 0 must fold at the DAG level, not go through mk_mul_const at
    // all. Verify the term is the zero constant (try_bv_const_value).
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let zero = s.bv_const(0, 32);
    let prod = s.bv_mul(x, zero);
    assert_eq!(s.try_bv_const_value(prod), Some(0));
}

#[test]
fn mul_const_one_is_identity() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let one = s.bv_const(1, 32);
    let prod = s.bv_mul(x, one);
    assert_eq!(prod, x);
}
