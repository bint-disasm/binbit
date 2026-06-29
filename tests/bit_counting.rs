//! Tests for the bv_popcount / bv_clz / bv_ctz primitives.

use binbit::{BvContext, SmtResult, SmtSolver};

// ---------- constant folding ----------

#[test]
fn popcount_const_folds() {
    let mut ctx = BvContext::new();
    let c = ctx.bv_const(0b10110101, 8);
    let pc = ctx.bv_popcount(c);
    let folded = ctx
        .bv_known_bits(pc)
        .0;
    // popcount(0b10110101) = 5
    assert_eq!(folded, 5);
}

#[test]
fn clz_const_folds() {
    let mut ctx = BvContext::new();
    let c = ctx.bv_const(0b00001100, 8);
    let clz = ctx.bv_clz(c);
    // bit 3 is highest set → 4 leading zeros in 8-bit
    assert_eq!(ctx.bv_known_bits(clz).0, 4);
}

#[test]
fn clz_of_zero_is_width() {
    let mut ctx = BvContext::new();
    let c = ctx.bv_const(0, 16);
    let clz = ctx.bv_clz(c);
    assert_eq!(ctx.bv_known_bits(clz).0, 16);
}

#[test]
fn ctz_const_folds() {
    let mut ctx = BvContext::new();
    let c = ctx.bv_const(0b00011000, 8);
    let ctz = ctx.bv_ctz(c);
    // lowest set bit is bit 3 → 3 trailing zeros
    assert_eq!(ctx.bv_known_bits(ctz).0, 3);
}

#[test]
fn ctz_of_zero_is_width() {
    let mut ctx = BvContext::new();
    let c = ctx.bv_const(0, 16);
    let ctz = ctx.bv_ctz(c);
    assert_eq!(ctx.bv_known_bits(ctz).0, 16);
}

// ---------- symbolic agreement with concrete oracle ----------

fn check_op_against_oracle<F, G>(width: u32, build: F, oracle: G)
where
    F: Fn(&mut SmtSolver, binbit::BvTerm) -> binbit::BvTerm,
    G: Fn(u128) -> u128,
{
    // Sample a few representative values and assert the symbolic primitive
    // agrees with the concrete oracle on each.
    let samples = [0u128, 1, 2, 3, 0xff, 0x100, 0x80, 0xa5a5, u128::MAX];
    for &v in &samples {
        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let result = build(&mut s, x);
        let x_const = s.bv_const(v, width);
        let eq_x = s.bv_eq(x, x_const);
        s.assert(eq_x);
        let expected = s.bv_const(oracle(v), width);
        let eq_result = s.bv_eq(result, expected);
        s.assert(eq_result);
        match s.solve() {
            SmtResult::Sat => {} // expected
            SmtResult::Unsat => panic!(
                "symbolic op disagrees with oracle at width={}, v=0x{:x}",
                width, v,
            ),
        }
    }
}

#[test]
fn popcount_symbolic_matches_oracle_8bit() {
    check_op_against_oracle(
        8,
        |s, x| s.bv_popcount(x),
        |v| (v & 0xff).count_ones() as u128,
    );
}

#[test]
fn popcount_symbolic_matches_oracle_32bit() {
    check_op_against_oracle(
        32,
        |s, x| s.bv_popcount(x),
        |v| (v & 0xffffffff).count_ones() as u128,
    );
}

#[test]
fn clz_symbolic_matches_oracle_8bit() {
    check_op_against_oracle(
        8,
        |s, x| s.bv_clz(x),
        |v| {
            let masked = v & 0xff;
            if masked == 0 {
                8
            } else {
                (masked.leading_zeros() - 120) as u128
            }
        },
    );
}

#[test]
fn clz_symbolic_matches_oracle_16bit() {
    check_op_against_oracle(
        16,
        |s, x| s.bv_clz(x),
        |v| {
            let masked = v & 0xffff;
            if masked == 0 {
                16
            } else {
                (masked.leading_zeros() - 112) as u128
            }
        },
    );
}

#[test]
fn ctz_symbolic_matches_oracle_8bit() {
    check_op_against_oracle(
        8,
        |s, x| s.bv_ctz(x),
        |v| {
            let masked = v & 0xff;
            if masked == 0 { 8 } else { masked.trailing_zeros() as u128 }
        },
    );
}

#[test]
fn ctz_symbolic_matches_oracle_16bit() {
    check_op_against_oracle(
        16,
        |s, x| s.bv_ctz(x),
        |v| {
            let masked = v & 0xffff;
            if masked == 0 { 16 } else { masked.trailing_zeros() as u128 }
        },
    );
}

// ---------- known-bits propagation ----------

#[test]
fn popcount_result_has_high_bits_known_zero() {
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(32);
    let pc = ctx.bv_popcount(x);
    let (_, zeros) = ctx.bv_known_bits(pc);
    // popcount of a 32-bit value is in [0, 32], which fits in 6 bits.
    // So bits 6..31 should be known zero.
    let expected_zero_mask = !0x3fu128 & 0xffffffffu128;
    assert_eq!(zeros, expected_zero_mask, "expected high 26 bits zero");
}

// ---------- symbolic rotation ----------

fn check_rotate_against_oracle<F, G>(width: u32, build: F, oracle: G)
where
    F: Fn(&mut SmtSolver, binbit::BvTerm, binbit::BvTerm) -> binbit::BvTerm,
    G: Fn(u128, u32, u32) -> u128,
{
    let value_samples = [0u128, 1, 0xff, 0xa5a5, 0x80000001, 0xffffffff];
    let amount_samples = [0u32, 1, 3, 7, width - 1, width, width + 5];
    for &v in &value_samples {
        for &a in &amount_samples {
            let mut s = SmtSolver::new();
            let x = s.bv_var(width);
            let y = s.bv_var(width);
            let result = build(&mut s, x, y);
            let v_const = s.bv_const(v, width);
            let a_const = s.bv_const(a as u128, width);
            let eq_x = s.bv_eq(x, v_const);
            let eq_y = s.bv_eq(y, a_const);
            s.assert(eq_x);
            s.assert(eq_y);
            let expected = s.bv_const(oracle(v, a, width), width);
            let eq_r = s.bv_eq(result, expected);
            s.assert(eq_r);
            match s.solve() {
                SmtResult::Sat => {}
                SmtResult::Unsat => panic!(
                    "rotate disagrees with oracle: width={}, v=0x{:x}, amt={}",
                    width, v, a,
                ),
            }
        }
    }
}

fn rotl_oracle(v: u128, a: u32, w: u32) -> u128 {
    let a = a % w;
    let m = (1u128 << w) - 1;
    let v = v & m;
    if a == 0 { v } else { ((v << a) | (v >> (w - a))) & m }
}

fn rotr_oracle(v: u128, a: u32, w: u32) -> u128 {
    let a = a % w;
    let m = (1u128 << w) - 1;
    let v = v & m;
    if a == 0 { v } else { ((v >> a) | (v << (w - a))) & m }
}

#[test]
fn rotl_symbolic_matches_oracle_8bit() {
    check_rotate_against_oracle(8, |s, x, y| s.bv_rotate_left_dyn(x, y), rotl_oracle);
}

#[test]
fn rotl_symbolic_matches_oracle_32bit() {
    check_rotate_against_oracle(32, |s, x, y| s.bv_rotate_left_dyn(x, y), rotl_oracle);
}

#[test]
fn rotr_symbolic_matches_oracle_8bit() {
    check_rotate_against_oracle(8, |s, x, y| s.bv_rotate_right_dyn(x, y), rotr_oracle);
}

#[test]
fn rotr_symbolic_matches_oracle_32bit() {
    check_rotate_against_oracle(32, |s, x, y| s.bv_rotate_right_dyn(x, y), rotr_oracle);
}

#[test]
fn rotl_symbolic_non_power_of_two_width() {
    // Width 12 — exercises the urem-based fallback path.
    check_rotate_against_oracle(12, |s, x, y| s.bv_rotate_left_dyn(x, y), rotl_oracle);
}

#[test]
fn rotr_constant_amount_dispatches_to_const_builder() {
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(8);
    let amt = ctx.bv_const(3, 8);
    let rot_dyn = ctx.bv_rotate_right_dyn(x, amt);
    let rot_const = ctx.bv_rotate_right(x, 3);
    // Constant-amount path should hash-cons to the same term as the
    // direct constant builder.
    assert_eq!(rot_dyn, rot_const);
}

#[test]
fn rotl_const_folds() {
    let mut ctx = BvContext::new();
    let x = ctx.bv_const(0b00010001, 8);
    let amt = ctx.bv_const(3, 8);
    let r = ctx.bv_rotate_left_dyn(x, amt);
    assert_eq!(ctx.bv_known_bits(r).0, 0b10001000);
}
