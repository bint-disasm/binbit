//! Coverage tests for the extended SMT-BV operation set: shifts, multiply,
//! divide/remainder, extract/concat/extend, ITE, and signed comparisons.

use binbit::{SmtResult, SmtSolver};

// ---------- Extract / Concat / Extend ----------

#[test]
fn extract_low_bits() {
    // Extract bits 3..=0 of 0xAB = 10101011 → 0b1011 = 11.
    let mut s = SmtSolver::new();
    let v = s.bv_const(0xAB, 8);
    let low = s.bv_extract(v, 3, 0);
    let eleven = s.bv_const(11, 4);
    let eq = s.bv_eq(low, eleven);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn extract_high_bits() {
    // Extract bits 7..=4 of 0xAB = 10101011 → 0b1010 = 10.
    let mut s = SmtSolver::new();
    let v = s.bv_const(0xAB, 8);
    let high = s.bv_extract(v, 7, 4);
    let ten = s.bv_const(10, 4);
    let eq = s.bv_eq(high, ten);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn extract_round_trip_via_concat() {
    // concat(extract(x, 7, 4), extract(x, 3, 0)) == x for any 8-bit x.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let hi = s.bv_extract(x, 7, 4);
    let lo = s.bv_extract(x, 3, 0);
    let rebuilt = s.bv_concat(hi, lo);
    // Assert they differ → UNSAT.
    let diff = s.bv_ne(x, rebuilt);
    s.assert(diff);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn zero_extend_preserves_value() {
    // Zero-extending 0xFF (8 bits) by 8 → 0x00FF (16 bits).
    let mut s = SmtSolver::new();
    let x = s.bv_const(0xFF, 8);
    let ext = s.bv_zero_extend(x, 8);
    let expected = s.bv_const(0x00FF, 16);
    let eq = s.bv_eq(ext, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn sign_extend_negative_number() {
    // Sign-extending 0xFF (8-bit, = -1 signed) by 8 → 0xFFFF (16 bits).
    let mut s = SmtSolver::new();
    let x = s.bv_const(0xFF, 8);
    let ext = s.bv_sign_extend(x, 8);
    let expected = s.bv_const(0xFFFF, 16);
    let eq = s.bv_eq(ext, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn sign_extend_positive_number() {
    // Sign-extending 0x7F (positive) by 8 → 0x007F.
    let mut s = SmtSolver::new();
    let x = s.bv_const(0x7F, 8);
    let ext = s.bv_sign_extend(x, 8);
    let expected = s.bv_const(0x007F, 16);
    let eq = s.bv_eq(ext, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

// ---------- ITE ----------

#[test]
fn ite_selects_by_condition() {
    let mut s = SmtSolver::new();
    let five = s.bv_const(5, 8);
    let seven = s.bv_const(7, 8);
    let t = s.bool_true();
    let f = s.bool_false();

    let pick_t = s.bv_ite(t, five, seven);
    let pick_f = s.bv_ite(f, five, seven);
    let eq_t = s.bv_eq(pick_t, five);
    let eq_f = s.bv_eq(pick_f, seven);
    s.assert(eq_t);
    s.assert(eq_f);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn ite_symbolic_condition_forces_result() {
    // (c ? 10 : 20) == 10 forces c = true.
    let mut s = SmtSolver::new();
    let c = s.bool_var();
    let ten = s.bv_const(10, 8);
    let twenty = s.bv_const(20, 8);
    let picked = s.bv_ite(c, ten, twenty);
    let is_ten = s.bv_eq(picked, ten);
    s.assert(is_ten);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert!(s.get_bool_value(c));
}

// ---------- Shifts ----------

#[test]
fn shl_by_constant() {
    // 1 << 4 = 16.
    let mut s = SmtSolver::new();
    let one = s.bv_const(1, 8);
    let four = s.bv_const(4, 8);
    let shifted = s.bv_shl(one, four);
    let sixteen = s.bv_const(16, 8);
    let eq = s.bv_eq(shifted, sixteen);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn shl_overflow_zeros_result() {
    // 1 << 8 in 8-bit BV should be 0 (shift by >= width).
    let mut s = SmtSolver::new();
    let one = s.bv_const(1, 8);
    let eight = s.bv_const(8, 8);
    let shifted = s.bv_shl(one, eight);
    let zero = s.bv_const(0, 8);
    let eq = s.bv_eq(shifted, zero);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn shl_symbolic_amount() {
    // x << y = 8 with x = 1, 8-bit. y must be 3.
    let mut s = SmtSolver::new();
    let one = s.bv_const(1, 8);
    let y = s.bv_var(8);
    let shifted = s.bv_shl(one, y);
    let eight = s.bv_const(8, 8);
    let eq = s.bv_eq(shifted, eight);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(y), 3);
}

#[test]
fn lshr_fills_with_zero() {
    // 0x80 >> 4 (logical) = 0x08.
    let mut s = SmtSolver::new();
    let x = s.bv_const(0x80, 8);
    let four = s.bv_const(4, 8);
    let shifted = s.bv_lshr(x, four);
    let expected = s.bv_const(0x08, 8);
    let eq = s.bv_eq(shifted, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn ashr_replicates_sign_bit() {
    // 0x80 >>> 4 (arith, treating 0x80 as -128) = 0xF8.
    let mut s = SmtSolver::new();
    let x = s.bv_const(0x80, 8);
    let four = s.bv_const(4, 8);
    let shifted = s.bv_ashr(x, four);
    let expected = s.bv_const(0xF8, 8);
    let eq = s.bv_eq(shifted, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn ashr_positive_behaves_like_lshr() {
    // 0x40 >>> 2 = 0x10, same as lshr.
    let mut s = SmtSolver::new();
    let x = s.bv_const(0x40, 8);
    let two = s.bv_const(2, 8);
    let shifted = s.bv_ashr(x, two);
    let expected = s.bv_const(0x10, 8);
    let eq = s.bv_eq(shifted, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

// ---------- Signed comparisons ----------

#[test]
fn slt_orders_signed_values() {
    // -1 <s 1 in 8-bit: 0xFF vs 0x01.
    let mut s = SmtSolver::new();
    let neg_one = s.bv_const(0xFF, 8);
    let pos_one = s.bv_const(0x01, 8);
    let lt = s.bv_slt(neg_one, pos_one);
    s.assert(lt);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn slt_differs_from_ult_for_negatives() {
    // 0xFF <u 0x01 is false, but 0xFF <s 0x01 is true (-1 < 1).
    let mut s = SmtSolver::new();
    let big = s.bv_const(0xFF, 8);
    let one = s.bv_const(0x01, 8);
    // Assert ult: 0xFF <u 0x01 → UNSAT.
    let ult = s.bv_ult(big, one);
    s.assert(ult);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn sle_is_inclusive() {
    // -1 <=s -1 is SAT; -1 <=s -2 is UNSAT.
    let mut s = SmtSolver::new();
    let m1 = s.bv_const(0xFF, 8);
    let m1b = s.bv_const(0xFF, 8);
    let le = s.bv_sle(m1, m1b);
    s.assert(le);
    assert_eq!(s.solve(), SmtResult::Sat);

    let mut s = SmtSolver::new();
    let m1 = s.bv_const(0xFF, 8); // -1
    let m2 = s.bv_const(0xFE, 8); // -2
    let le = s.bv_sle(m1, m2);
    s.assert(le);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

// ---------- Multiply ----------

#[test]
fn multiply_constants() {
    // 6 * 7 = 42 in 8-bit.
    let mut s = SmtSolver::new();
    let a = s.bv_const(6, 8);
    let b = s.bv_const(7, 8);
    let prod = s.bv_mul(a, b);
    let forty_two = s.bv_const(42, 8);
    let eq = s.bv_eq(prod, forty_two);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn multiply_wraps_modulo_width() {
    // 16 * 16 = 256 in 8-bit → 0.
    let mut s = SmtSolver::new();
    let a = s.bv_const(16, 8);
    let b = s.bv_const(16, 8);
    let prod = s.bv_mul(a, b);
    let zero = s.bv_const(0, 8);
    let eq = s.bv_eq(prod, zero);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn factor_through_multiply() {
    // x * 7 = 42 with 8-bit arithmetic → x = 6 (or other values that wrap
    // around and also produce 42). Let's force unique: x < 10.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let seven = s.bv_const(7, 8);
    let prod = s.bv_mul(x, seven);
    let forty_two = s.bv_const(42, 8);
    let eq = s.bv_eq(prod, forty_two);
    let ten = s.bv_const(10, 8);
    let lt_ten = s.bv_ult(x, ten);
    s.assert(eq);
    s.assert(lt_ten);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 6);
}

// ---------- Divide / Remainder ----------

#[test]
fn udiv_exact() {
    // 20 / 4 = 5.
    let mut s = SmtSolver::new();
    let twenty = s.bv_const(20, 8);
    let four = s.bv_const(4, 8);
    let q = s.bv_udiv(twenty, four);
    let five = s.bv_const(5, 8);
    let eq = s.bv_eq(q, five);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn udiv_truncates_toward_zero() {
    // 23 / 4 = 5 (unsigned, truncating).
    let mut s = SmtSolver::new();
    let a = s.bv_const(23, 8);
    let b = s.bv_const(4, 8);
    let q = s.bv_udiv(a, b);
    let five = s.bv_const(5, 8);
    let eq = s.bv_eq(q, five);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn udiv_by_zero_is_all_ones() {
    // SMT-LIB: x / 0 = ~0 = 0xFF on 8-bit.
    let mut s = SmtSolver::new();
    let x = s.bv_const(42, 8);
    let zero = s.bv_const(0, 8);
    let q = s.bv_udiv(x, zero);
    let all_ones = s.bv_const(0xFF, 8);
    let eq = s.bv_eq(q, all_ones);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn urem_exact() {
    // 23 % 4 = 3.
    let mut s = SmtSolver::new();
    let a = s.bv_const(23, 8);
    let b = s.bv_const(4, 8);
    let r = s.bv_urem(a, b);
    let three = s.bv_const(3, 8);
    let eq = s.bv_eq(r, three);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn urem_by_zero_is_identity() {
    // SMT-LIB: x % 0 = x.
    let mut s = SmtSolver::new();
    let x = s.bv_const(42, 8);
    let zero = s.bv_const(0, 8);
    let r = s.bv_urem(x, zero);
    let expected = s.bv_const(42, 8);
    let eq = s.bv_eq(r, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn udiv_urem_preserve_dividend() {
    // Classic identity: a = (a / b) * b + (a % b) for b != 0. Prove it
    // for 8-bit by asserting the negation and expecting UNSAT.
    let mut s = SmtSolver::new();
    let a = s.bv_var(8);
    let b = s.bv_var(8);
    let q = s.bv_udiv(a, b);
    let r = s.bv_urem(a, b);
    let qb = s.bv_mul(q, b);
    let reconstructed = s.bv_add(qb, r);

    let zero = s.bv_const(0, 8);
    let b_nonzero = s.bv_ne(b, zero);
    let differs = s.bv_ne(a, reconstructed);
    let counterexample = s.bool_and(b_nonzero, differs);
    s.assert(counterexample);
    // No counterexample should exist → UNSAT.
    assert_eq!(s.solve(), SmtResult::Unsat);
}

// ---------- Composite / symex-flavoured ----------

#[test]
fn find_input_that_triggers_overflow() {
    // Classic symex bug finder: find x (8-bit) such that x * 2 < x.
    // In unsigned arithmetic, this happens iff x >= 128 (MSB set).
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let two = s.bv_const(2, 8);
    let doubled = s.bv_mul(x, two);
    let wraps = s.bv_ult(doubled, x);
    s.assert(wraps);
    assert_eq!(s.solve(), SmtResult::Sat);
    let v = s.get_bv_value(x);
    assert!(v >= 128, "got x={}, expected MSB set", v);
}

#[test]
fn hash_consing_dedupes_terms() {
    // Ensure the arena doesn't balloon when the same term is built twice.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let a1 = s.bv_add(x, y);
    let a2 = s.bv_add(x, y);
    // Same underlying handle.
    assert_eq!(a1, a2);
    // And works through solves.
    let ten = s.bv_const(10, 8);
    let eq = s.bv_eq(a1, ten);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(
        (s.get_bv_value(x).wrapping_add(s.get_bv_value(y))) & 0xFF,
        10
    );
}
