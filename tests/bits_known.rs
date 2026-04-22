//! Tests for the bits-known abstract interpretation layer: verifies that
//! construction-time simplifications fold terms the SAT solver would
//! otherwise have to reason about.

use binbit::{BvContext, SmtResult, SmtSolver};

#[test]
fn zero_extend_high_bits_are_known_zero() {
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(8);
    let wide = ctx.bv_zero_extend(x, 24); // 8-bit x extended to 32
    let (ones, zeros) = ctx.bv_known_bits(wide);
    // High 24 bits must be known-zero.
    let hi_mask = ((1u128 << 24) - 1) << 8;
    assert_eq!(zeros & hi_mask, hi_mask, "high 24 bits should be known-zero");
    // Low 8 bits must remain unknown.
    assert_eq!(ones & 0xFF, 0);
    assert_eq!(zeros & 0xFF, 0);
}

#[test]
fn shl_low_bits_become_known_zero() {
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(32);
    let k = ctx.bv_const(8, 32);
    let shifted = ctx.bv_shl(x, k);
    let (_, zeros) = ctx.bv_known_bits(shifted);
    // Low 8 bits must be known-zero.
    assert_eq!(zeros & 0xFF, 0xFF);
}

#[test]
fn and_with_mask_propagates_known_zeros() {
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(32);
    let mask = ctx.bv_const(0xFF, 32);
    let masked = ctx.bv_and(x, mask);
    let (_, zeros) = ctx.bv_known_bits(masked);
    // High 24 bits must be known-zero after the AND.
    let hi_mask = (!0xFFu128) & ((1u128 << 32) - 1);
    assert_eq!(zeros & hi_mask, hi_mask);
}

#[test]
fn bv_eq_folds_false_on_bits_conflict() {
    // `bv_and(x, 0x00FF) == 0x100` has a forced-zero bit (bit 8) mismatched
    // with the constant's forced-one. Construction should fold to false.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let mask = s.bv_const(0xFF, 32);
    let masked = s.bv_and(x, mask);
    let target = s.bv_const(0x100, 32);
    let eq = s.bv_eq(masked, target);
    // The equality is unsat; assert it and check solver agrees.
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn bv_ult_folds_true_by_interval() {
    // `bv_and(x, 0xFF) < bv_or(y, 0x100_0000)` — LHS max = 0xFF, RHS min = 0x100_0000.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let y = s.bv_var(32);
    let lhs_mask = s.bv_const(0xFF, 32);
    let rhs_mask = s.bv_const(0x100_0000, 32);
    let lhs = s.bv_and(x, lhs_mask);
    let rhs = s.bv_or(y, rhs_mask);
    let lt = s.bv_ult(lhs, rhs);
    // Should fold to true at construction time; asserting its negation is unsat.
    let not_lt = s.bool_not(lt);
    s.assert(not_lt);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn extract_known_bits_preserved() {
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(32);
    let mask = ctx.bv_const(0xFF, 32);
    let masked = ctx.bv_and(x, mask);
    // Extract the high half — every bit must be known-zero.
    let high = ctx.bv_extract(masked, 31, 16);
    let (ones, zeros) = ctx.bv_known_bits(high);
    assert_eq!(ones, 0);
    assert_eq!(zeros, 0xFFFF); // all 16 bits known-zero
}

#[test]
fn ite_intersects_known_bits() {
    // If both branches have bit i forced to the same value, the ITE has it
    // too. The low byte of both branches is 0xAB.
    let mut ctx = BvContext::new();
    let c = ctx.bool_var();
    let x = ctx.bv_var(24);
    let low_byte = ctx.bv_const(0xAB, 8);
    let lhs = ctx.bv_concat(x, low_byte);
    let y = ctx.bv_var(24);
    let rhs = ctx.bv_concat(y, low_byte);
    let ite = ctx.bv_ite(c, lhs, rhs);
    let (ones, zeros) = ctx.bv_known_bits(ite);
    // Low byte forced: ones = 0xAB, zeros = ~0xAB & 0xFF = 0x54 in the low byte.
    assert_eq!(ones & 0xFF, 0xAB);
    assert_eq!(zeros & 0xFF, !0xAB & 0xFF);
}

#[test]
fn add_constant_propagates_known_low_zeros() {
    // `x << 8` has low 8 bits known-zero. Add a constant with low 8 bits zero —
    // the result's low 8 bits should stay known-zero.
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(32);
    let k = ctx.bv_const(8, 32);
    let shifted = ctx.bv_shl(x, k);
    let c = ctx.bv_const(0x10000, 32); // low 16 bits zero
    let sum = ctx.bv_add(shifted, c);
    let (_, zeros) = ctx.bv_known_bits(sum);
    // Low 8 bits should still be known-zero (both operands had them zero and
    // the carry ripple stays at zero).
    assert_eq!(zeros & 0xFF, 0xFF);
}

#[test]
fn fully_determined_and_folds_to_constant() {
    // `bv_and(const 0xF0, const 0x0F) = 0`. Bits-known says all 8 bits are
    // known-zero — construction should short-circuit to the constant.
    let mut ctx = BvContext::new();
    let a = ctx.bv_const(0xF0, 8);
    let b = ctx.bv_const(0x0F, 8);
    let and = ctx.bv_and(a, b);
    assert_eq!(ctx.bv_const_value(and), 0);
    // Built the same way by a different path should hash-cons to the same term.
    let zero = ctx.bv_const(0, 8);
    assert_eq!(and, zero);
}

#[test]
fn fully_determined_shift_folds() {
    // `(x << 32)` at width 32: entire value is forced to zero.
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(32);
    let k = ctx.bv_const(32, 32);
    let shifted = ctx.bv_shl(x, k);
    // Should fold to the 32-bit zero constant.
    let zero = ctx.bv_const(0, 32);
    assert_eq!(shifted, zero);
}

#[test]
fn known_bits_survive_hash_cons() {
    // Two structurally-identical terms must share their bits-known entry
    // through hash-consing.
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(16);
    let m = ctx.bv_const(0xFF, 16);
    let a = ctx.bv_and(x, m);
    let b = ctx.bv_and(x, m);
    assert_eq!(a, b);
    assert_eq!(ctx.bv_known_bits(a), ctx.bv_known_bits(b));
}
