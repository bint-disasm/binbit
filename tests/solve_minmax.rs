//! Tests for the bit-hunt optimization primitives on `SmtSolver`:
//! `solve_min_u/s`, `solve_max_u/s`, and their `_limbs` wide variants.

use binbit::{SmtSolver, SmtResult};

fn fresh() -> SmtSolver {
    SmtSolver::new()
}

// ---------- unsigned ----------

#[test]
fn min_u_no_constraints_is_zero() {
    let mut s = fresh();
    let x = s.bv_var(8);
    assert_eq!(s.solve_min_u(x), Some(0));
}

#[test]
fn max_u_no_constraints_is_all_ones() {
    let mut s = fresh();
    let x = s.bv_var(8);
    assert_eq!(s.solve_max_u(x), Some(0xFF));
}

#[test]
fn min_u_with_lower_bound() {
    let mut s = fresh();
    let x = s.bv_var(8);
    let k = s.bv_const(42, 8);
    let ge = s.bv_uge(x, k);
    s.assert(ge);
    assert_eq!(s.solve_min_u(x), Some(42));
}

#[test]
fn max_u_with_upper_bound() {
    let mut s = fresh();
    let x = s.bv_var(8);
    let k = s.bv_const(200, 8);
    let le = s.bv_ule(x, k);
    s.assert(le);
    assert_eq!(s.solve_max_u(x), Some(200));
}

#[test]
fn min_u_with_sandwich_and_parity() {
    // 7 <= x <= 20 and x is odd (low bit set).
    let mut s = fresh();
    let x = s.bv_var(8);
    let k7 = s.bv_const(7, 8);
    let k20 = s.bv_const(20, 8);
    let one = s.bv_const(1, 8);
    let ge = s.bv_uge(x, k7);
    let le = s.bv_ule(x, k20);
    let parity_bit = s.bv_and(x, one);
    let odd = s.bv_eq(parity_bit, one);
    s.assert(ge);
    s.assert(le);
    s.assert(odd);
    assert_eq!(s.solve_min_u(x), Some(7)); // 7 is odd
    assert_eq!(s.solve_max_u(x), Some(19)); // 20 is even, so 19 wins
}

#[test]
fn min_u_on_unsat_is_none() {
    let mut s = fresh();
    let x = s.bv_var(8);
    let k1 = s.bv_const(10, 8);
    let k2 = s.bv_const(5, 8);
    let eq1 = s.bv_eq(x, k1);
    let eq2 = s.bv_eq(x, k2);
    s.assert(eq1);
    s.assert(eq2);
    assert_eq!(s.solve_min_u(x), None);
}

#[test]
fn max_u_on_unsat_is_none() {
    let mut s = fresh();
    let x = s.bv_var(4);
    let k = s.bv_const(3, 4);
    let eq = s.bv_eq(x, k);
    let ne = s.bv_ne(x, k);
    s.assert(eq);
    s.assert(ne);
    assert_eq!(s.solve_max_u(x), None);
}

// ---------- signed ----------

#[test]
fn min_s_no_constraints_is_int_min() {
    let mut s = fresh();
    let x = s.bv_var(8);
    assert_eq!(s.solve_min_s(x), Some(-128));
}

#[test]
fn max_s_no_constraints_is_int_max() {
    let mut s = fresh();
    let x = s.bv_var(8);
    assert_eq!(s.solve_max_s(x), Some(127));
}

#[test]
fn min_s_with_upper_bound() {
    // x <=_s -3 → minimum is still -128.
    let mut s = fresh();
    let x = s.bv_var(8);
    let k = s.bv_const(((-3i32) as u128) & 0xFF, 8);
    let le = s.bv_sle(x, k);
    s.assert(le);
    assert_eq!(s.solve_min_s(x), Some(-128));
    assert_eq!(s.solve_max_s(x), Some(-3));
}

#[test]
fn min_s_with_lower_bound() {
    // x >=_s 5 → signed min becomes 5.
    let mut s = fresh();
    let x = s.bv_var(8);
    let k = s.bv_const(5, 8);
    let ge = s.bv_sge(x, k);
    s.assert(ge);
    assert_eq!(s.solve_min_s(x), Some(5));
    assert_eq!(s.solve_max_s(x), Some(127));
}

#[test]
fn max_s_with_negative_only() {
    // x <=_s -1 (i.e. sign bit must be 1) → max is -1, min is -128.
    let mut s = fresh();
    let x = s.bv_var(8);
    let neg_one = s.bv_const(0xFF, 8);
    let le = s.bv_sle(x, neg_one);
    s.assert(le);
    assert_eq!(s.solve_max_s(x), Some(-1));
    assert_eq!(s.solve_min_s(x), Some(-128));
}

// ---------- model is consistent after opt ----------

#[test]
fn model_reflects_optimum_after_min() {
    let mut s = fresh();
    let x = s.bv_var(16);
    let y = s.bv_var(16);
    let sum = s.bv_add(x, y);
    let k = s.bv_const(100, 16);
    let eq = s.bv_eq(sum, k);
    s.assert(eq);

    let minx = s.solve_min_u(x).expect("sat");
    assert_eq!(minx, 0);
    // After the search, the solver's current model should show x=0, y=100.
    assert_eq!(s.get_bv_value(x), 0);
    assert_eq!(s.get_bv_value(y), 100);
}

#[test]
fn opt_survives_subsequent_solve_calls() {
    // After solve_min_u, a follow-up solve() should still be sat with the
    // original formula (not the opt assumptions — those were temporary).
    let mut s = fresh();
    let x = s.bv_var(8);
    let k = s.bv_const(10, 8);
    let ge = s.bv_uge(x, k);
    s.assert(ge);

    assert_eq!(s.solve_min_u(x), Some(10));
    // Re-solve without constraints on x should allow any x >= 10.
    assert_eq!(s.solve(), SmtResult::Sat);
    assert!(s.get_bv_value(x) >= 10);
}

// ---------- width = 1 edge case ----------

#[test]
fn min_max_width_one() {
    let mut s = fresh();
    let x = s.bv_var(1);
    assert_eq!(s.solve_min_u(x), Some(0));
    assert_eq!(s.solve_max_u(x), Some(1));
}

#[test]
fn signed_width_one() {
    // 1-bit signed is {-1, 0}: MSB 1 → -1, MSB 0 → 0.
    let mut s = fresh();
    let x = s.bv_var(1);
    assert_eq!(s.solve_min_s(x), Some(-1));
    assert_eq!(s.solve_max_s(x), Some(0));
}

// ---------- wide (> 128 bits) ----------

#[test]
fn min_u_limbs_wide_unconstrained() {
    let mut s = fresh();
    let x = s.bv_var(160);
    let limbs = s.solve_min_u_limbs(x).expect("sat");
    // 160 bits → 3 u64 limbs, all zero.
    assert_eq!(limbs, vec![0u64; 3]);
}

#[test]
fn max_u_limbs_wide_unconstrained() {
    let mut s = fresh();
    let x = s.bv_var(160);
    let limbs = s.solve_max_u_limbs(x).expect("sat");
    // High limb has only 32 active bits (160 - 2*64 = 32).
    assert_eq!(limbs[0], u64::MAX);
    assert_eq!(limbs[1], u64::MAX);
    assert_eq!(limbs[2], (1u64 << 32) - 1);
}

#[test]
fn min_u_limbs_wide_lower_bound() {
    // 200-bit variable constrained to >= specific value; the min should
    // match that value exactly.
    let mut s = fresh();
    let x = s.bv_var(200);
    // Target: a value whose bits are nonzero in both the middle and high
    // limbs so we exercise limb-crossing bit-hunt. 200 bits = 4 u64 limbs
    // (the top limb only uses its low 8 bits).
    let target_limbs = [0u64, 1u64 << 63, 0xFF, 0u64];
    let k = s.bv_const_wide(&target_limbs, 200);
    let ge = s.bv_uge(x, k);
    s.assert(ge);
    let got = s.solve_min_u_limbs(x).expect("sat");
    assert_eq!(got.as_slice(), target_limbs.as_slice());
}

// ---------- opt under push/pop ----------

#[test]
fn min_u_respects_scope() {
    let mut s = fresh();
    let x = s.bv_var(8);
    let ten = s.bv_const(10, 8);
    let ge = s.bv_uge(x, ten);
    s.assert(ge);

    s.push();
    let twenty = s.bv_const(20, 8);
    let ge20 = s.bv_uge(x, twenty);
    s.assert(ge20);
    assert_eq!(s.solve_min_u(x), Some(20));
    s.pop();

    // After pop, the x>=20 constraint is gone — min reverts to 10.
    assert_eq!(s.solve_min_u(x), Some(10));
}
