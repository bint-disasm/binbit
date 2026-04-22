//! End-to-end tests for the bitvector SMT layer. Each test builds a formula,
//! hands it to the solver, and verifies the model (or UNSAT status).

use binbit::{SmtResult, SmtSolver};

#[test]
fn constant_equality_is_sat_or_unsat() {
    // 5 == 5 should be SAT.
    let mut s = SmtSolver::new();
    let five_a = s.bv_const(5, 8);
    let five_b = s.bv_const(5, 8);
    let eq = s.bv_eq(five_a, five_b);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);

    // 5 == 6 is UNSAT.
    let mut s = SmtSolver::new();
    let five = s.bv_const(5, 8);
    let six = s.bv_const(6, 8);
    let eq = s.bv_eq(five, six);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn simple_addition_has_unique_solution() {
    // x + 3 = 10  (width 8)  =>  x = 7
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let three = s.bv_const(3, 8);
    let ten = s.bv_const(10, 8);
    let sum = s.bv_add(x, three);
    let eq = s.bv_eq(sum, ten);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 7);
}

#[test]
fn addition_wraps_modulo_width() {
    // 8-bit arithmetic: 200 + 100 mod 256 = 44.
    let mut s = SmtSolver::new();
    let a = s.bv_const(200, 8);
    let b = s.bv_const(100, 8);
    let expect = s.bv_const(44, 8);
    let sum = s.bv_add(a, b);
    let eq = s.bv_eq(sum, expect);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn subtraction() {
    // x - y = 7, with x = 20 => y = 13
    let mut s = SmtSolver::new();
    let twenty = s.bv_const(20, 8);
    let y = s.bv_var(8);
    let diff = s.bv_sub(twenty, y);
    let seven = s.bv_const(7, 8);
    let eq = s.bv_eq(diff, seven);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(y), 13);
}

#[test]
fn bitwise_ops_behave_as_expected() {
    // (x AND 0x0F) == 0x0A  (width 8)  =>  low nibble of x is 0xA.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let mask = s.bv_const(0x0F, 8);
    let ten = s.bv_const(0x0A, 8);
    let masked = s.bv_and(x, mask);
    let eq = s.bv_eq(masked, ten);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x) & 0x0F, 0x0A);
}

#[test]
fn xor_self_is_zero_assertion() {
    // For every x, x XOR x = 0. Asserting (x XOR x) != 0 is UNSAT.
    let mut s = SmtSolver::new();
    let x = s.bv_var(16);
    let zero = s.bv_const(0, 16);
    let xored = s.bv_xor(x, x);
    let ne = s.bv_ne(xored, zero);
    s.assert(ne);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn ult_ordering() {
    // Find x such that x < 10 AND x > 5.  x must be 6, 7, 8, or 9.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let five = s.bv_const(5, 8);
    let ten = s.bv_const(10, 8);
    let lt = s.bv_ult(x, ten);
    let gt = s.bv_ult(five, x);
    let both = s.bool_and(lt, gt);
    s.assert(both);
    assert_eq!(s.solve(), SmtResult::Sat);
    let v = s.get_bv_value(x);
    assert!(v > 5 && v < 10, "got {}", v);
}

#[test]
fn ule_is_inclusive() {
    // 5 <= 5 must be SAT.
    let mut s = SmtSolver::new();
    let a = s.bv_const(5, 8);
    let b = s.bv_const(5, 8);
    let le = s.bv_ule(a, b);
    s.assert(le);
    assert_eq!(s.solve(), SmtResult::Sat);

    // 6 <= 5 must be UNSAT.
    let mut s = SmtSolver::new();
    let a = s.bv_const(6, 8);
    let b = s.bv_const(5, 8);
    let le = s.bv_ule(a, b);
    s.assert(le);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn implies_encodes_correctly() {
    // p → q, p.  =>  q must be true.
    let mut s = SmtSolver::new();
    let p = s.bool_var();
    let q = s.bool_var();
    let imp = s.bool_implies(p, q);
    s.assert(imp);
    s.assert(p);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert!(s.get_bool_value(q));
}

#[test]
fn two_unknowns_multiple_constraints() {
    // x + y ≡ 42, x - y ≡ 10 (mod 256).  The uniqueness-over-ℤ solution is
    // (26, 16), but 8-bit arithmetic admits a second solution (154, 144)
    // reached via wrap-around. We just verify that whichever model the SAT
    // solver returns satisfies both constraints.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let c42 = s.bv_const(42, 8);
    let c10 = s.bv_const(10, 8);
    let sum = s.bv_add(x, y);
    let diff = s.bv_sub(x, y);
    let eq1 = s.bv_eq(sum, c42);
    let eq2 = s.bv_eq(diff, c10);
    s.assert(eq1);
    s.assert(eq2);
    assert_eq!(s.solve(), SmtResult::Sat);
    let xv = s.get_bv_value(x) as u16;
    let yv = s.get_bv_value(y) as u16;
    assert_eq!((xv + yv) & 0xFF, 42);
    assert_eq!((xv + 256 - yv) & 0xFF, 10);
}

#[test]
fn incremental_with_assumption_negates_model() {
    // Find *some* 8-bit x with x > 100; then under additional assumption
    // x != that value, find another.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let c100 = s.bv_const(100, 8);
    let gt = s.bv_ult(c100, x);
    s.assert(gt);

    assert_eq!(s.solve(), SmtResult::Sat);
    let v1 = s.get_bv_value(x);
    assert!(v1 > 100);

    // Now force x != v1 via an assumption and re-solve.
    let c_v1 = s.bv_const(v1 as u128, 8);
    let neq = s.bv_ne(x, c_v1);
    assert_eq!(s.solve_under_assumptions(&[neq]), SmtResult::Sat);
    let v2 = s.get_bv_value(x);
    assert_ne!(v1, v2);
    assert!(v2 > 100);
}

#[test]
fn unsat_overconstraint() {
    // x < 5 AND x > 10 is UNSAT.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let five = s.bv_const(5, 8);
    let ten = s.bv_const(10, 8);
    let lt = s.bv_ult(x, five);
    let gt = s.bv_ult(ten, x);
    let both = s.bool_and(lt, gt);
    s.assert(both);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn width_one_bv_behaves_like_bool() {
    // x: BV[1], x XOR 1 == 0  =>  x = 1.
    let mut s = SmtSolver::new();
    let x = s.bv_var(1);
    let one = s.bv_const(1, 1);
    let zero = s.bv_const(0, 1);
    let xored = s.bv_xor(x, one);
    let eq = s.bv_eq(xored, zero);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 1);
}

#[test]
fn full_64_bit_arithmetic() {
    // x + 1 == 2^32  on 64-bit =>  x = 2^32 - 1
    let mut s = SmtSolver::new();
    let x = s.bv_var(64);
    let one = s.bv_const(1, 64);
    let target = s.bv_const(1u128 << 32, 64);
    let sum = s.bv_add(x, one);
    let eq = s.bv_eq(sum, target);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), (1u64 << 32) - 1);
}
