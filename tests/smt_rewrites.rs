//! Tests for the latest pass: word-level rewrites, UNSAT-state awareness,
//! and 128-bit BV support.

use binbit::{BvContext, SmtResult, SmtSolver};

// ---------- Associative rollup ----------

#[test]
fn add_constant_chain_collapses() {
    // `(x + 3) + 5 + 7` should reduce to a single `x + 15` node in the arena
    // because of associative rollup. We can't see the arena directly from
    // here, but we can verify the semantics are preserved.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let c3 = s.bv_const(3, 8);
    let c5 = s.bv_const(5, 8);
    let c7 = s.bv_const(7, 8);
    let t1 = s.bv_add(x, c3);
    let t2 = s.bv_add(t1, c5);
    let t3 = s.bv_add(t2, c7);

    // Also build `x + 15` directly; associative rollup should hash-cons both
    // to the same BvTerm.
    let c15 = s.bv_const(15, 8);
    let direct = s.bv_add(x, c15);
    assert_eq!(t3, direct);
}

#[test]
fn sub_normalizes_to_add() {
    // `(x - 3) + 5` should become `x + 2` after normalisation + rollup.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let c3 = s.bv_const(3, 8);
    let c5 = s.bv_const(5, 8);
    let t1 = s.bv_sub(x, c3);
    let t2 = s.bv_add(t1, c5);

    let c2 = s.bv_const(2, 8);
    let direct = s.bv_add(x, c2);
    assert_eq!(t2, direct);
}

#[test]
fn mul_chain_collapses_constants() {
    // `(x * 3) * 5` → `x * 15`.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let c3 = s.bv_const(3, 8);
    let c5 = s.bv_const(5, 8);
    let t1 = s.bv_mul(x, c3);
    let t2 = s.bv_mul(t1, c5);

    let c15 = s.bv_const(15, 8);
    let direct = s.bv_mul(x, c15);
    assert_eq!(t2, direct);
}

#[test]
fn and_chain_collapses_masks() {
    // `(x & 0xF0) & 0xFF` → `x & 0xF0`.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let m1 = s.bv_const(0xF0, 8);
    let m2 = s.bv_const(0xFF, 8);
    let t1 = s.bv_and(x, m1);
    let t2 = s.bv_and(t1, m2);
    let direct = s.bv_and(x, m1);
    assert_eq!(t2, direct);
}

#[test]
fn xor_chain_collapses() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let a = s.bv_const(0xA5, 8);
    let b = s.bv_const(0x5A, 8);
    let inner = s.bv_xor(x, a);
    let t = s.bv_xor(inner, b);
    // a XOR b = 0xFF.
    let direct_c = s.bv_const(0xFF, 8);
    let direct = s.bv_xor(x, direct_c);
    assert_eq!(t, direct);
}

#[test]
fn commutative_canonicalization_dedupes() {
    // `x + y` and `y + x` should hash-cons to the same term.
    let mut s = BvContext::new();
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let a = s.bv_add(x, y);
    let b = s.bv_add(y, x);
    assert_eq!(a, b);
}

// ---------- Structural rewrites ----------

#[test]
fn extract_of_extract_composes() {
    // extract(extract(x, 15, 4), 7, 2) should be extract(x, 11, 6).
    let mut s = BvContext::new();
    let x = s.bv_var(16);
    let mid = s.bv_extract(x, 15, 4); // width 12
    let outer = s.bv_extract(mid, 7, 2); // 6 bits from mid
    let direct = s.bv_extract(x, 11, 6);
    assert_eq!(outer, direct);
}

#[test]
fn extract_over_concat_routes_to_side() {
    // Extract from the low side of a concat hits `lo` only.
    let mut s = BvContext::new();
    let hi = s.bv_var(8);
    let lo = s.bv_var(8);
    let cat = s.bv_concat(hi, lo); // width 16
    // Bits [3..=0] are entirely in lo.
    let ex = s.bv_extract(cat, 3, 0);
    let direct = s.bv_extract(lo, 3, 0);
    assert_eq!(ex, direct);
}

#[test]
fn concat_of_adjacent_extracts_collapses() {
    // concat(extract(x, 7, 4), extract(x, 3, 0)) = x.
    let mut s = BvContext::new();
    let x = s.bv_var(8);
    let hi = s.bv_extract(x, 7, 4);
    let lo = s.bv_extract(x, 3, 0);
    let cat = s.bv_concat(hi, lo);
    assert_eq!(cat, x);
}

#[test]
fn double_shift_composition() {
    let mut s = BvContext::new();
    let x = s.bv_var(16);
    let c2 = s.bv_const(2, 16);
    let c3 = s.bv_const(3, 16);
    let inner = s.bv_shl(x, c2);
    let t = s.bv_shl(inner, c3);
    let c5 = s.bv_const(5, 16);
    let direct = s.bv_shl(x, c5);
    assert_eq!(t, direct);
}

#[test]
fn zero_extend_composition() {
    let mut s = BvContext::new();
    let x = s.bv_var(8);
    let e1 = s.bv_zero_extend(x, 4); // width 12
    let e2 = s.bv_zero_extend(e1, 4); // width 16
    let direct = s.bv_zero_extend(x, 8);
    assert_eq!(e2, direct);
}

// ---------- UNSAT-state awareness ----------

#[test]
fn has_model_tracks_solve_result() {
    let mut s = SmtSolver::new();
    assert!(!s.has_model()); // never solved

    let x = s.bv_var(8);
    let one = s.bv_const(1, 8);
    let eq = s.bv_eq(x, one);
    s.assert(eq);

    assert_eq!(s.solve(), SmtResult::Sat);
    assert!(s.has_model());

    // Further asserting should invalidate the model.
    let zero = s.bv_const(0, 8);
    let eq2 = s.bv_eq(x, zero);
    s.assert(eq2);
    assert!(!s.has_model(), "assert should invalidate prior SAT result");

    assert_eq!(s.solve(), SmtResult::Unsat);
    assert!(!s.has_model(), "UNSAT result leaves no model");
}

#[test]
fn smtlib_get_value_after_unsat_is_rejected() {
    // The runner emits an (error "…") line rather than bogus values.
    let script = r#"
        (declare-const x (_ BitVec 8))
        (assert (= x (_ bv1 8)))
        (assert (= x (_ bv2 8)))
        (check-sat)
        (get-value (x))
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("unsat"));
    assert!(out.contains("(error"), "output = {}", out);
}

// ---------- 128-bit BV support ----------

#[test]
fn bv_128_constant_roundtrip() {
    let mut s = SmtSolver::new();
    // Pick a value larger than u64::MAX to prove the 128-bit path works.
    let big = (u64::MAX as u128) + 12345; // > u64::MAX
    let width = 128u32;
    let x = s.bv_var(width);
    let c = s.bv_const(big, width);
    let eq = s.bv_eq(x, c);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value_u128(x), big);
}

#[test]
fn bv_128_add_wraps_correctly() {
    // 2^127 + 2^127 = 0 in 128-bit (high bit carries out).
    let mut s = SmtSolver::new();
    let half = s.bv_const(1u128 << 127, 128);
    let sum = s.bv_add(half, half);
    let zero = s.bv_const(0, 128);
    let eq = s.bv_eq(sum, zero);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn smtlib_hex_literal_wider_than_64() {
    // 128-bit hex literal: 32 hex digits.
    let script = r#"
        (declare-const x (_ BitVec 128))
        (assert (= x #x0000000000000001FFFFFFFFFFFFFFFF))
        (check-sat)
        (get-value (x))
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
    // Value: 2^64 * 1 + (2^64 - 1) = 2^64 + 2^64 - 1 = 2^65 - 1.
    let expected = (1u128 << 65) - 1;
    assert!(
        out.contains(&format!("(_ bv{} 128)", expected)),
        "output = {}",
        out
    );
}
