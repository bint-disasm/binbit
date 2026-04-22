//! Tests for the SMT-level hooks that symbolic-execution integrations
//! depend on: constant-value probing, rotate ops, and conflict-budgeted
//! solving.

use binbit::{SmtResult, SmtSolver};

// ----- try_bv_const_value -----

#[test]
fn try_bv_const_value_on_literal() {
    let mut s = SmtSolver::new();
    let c = s.bv_const(0x1234, 32);
    assert_eq!(s.try_bv_const_value(c), Some(0x1234));
}

#[test]
fn try_bv_const_value_on_variable_is_none() {
    let mut s = SmtSolver::new();
    let v = s.bv_var(32);
    assert!(s.try_bv_const_value(v).is_none());
}

#[test]
fn try_bv_const_value_sees_folded_constant() {
    // `bv_and(x, 0)` folds to 0 via the existing const-fold path; the
    // accessor should report it as a constant.
    let mut s = SmtSolver::new();
    let x = s.bv_var(16);
    let zero = s.bv_const(0, 16);
    let and = s.bv_and(x, zero);
    assert_eq!(s.try_bv_const_value(and), Some(0));
}

#[test]
fn try_bv_const_value_sees_bits_known_fold() {
    // `bv_and(x, 0xFF) & 0xFF00` is forced to 0 by bits-known even though
    // neither operand is a raw constant — the fold at `push_bv` time
    // rewrites the term to the zero constant.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let low_mask = s.bv_const(0xFF, 32);
    let high_mask = s.bv_const(0xFF00, 32);
    let a = s.bv_and(x, low_mask);
    let b = s.bv_and(a, high_mask);
    assert_eq!(s.try_bv_const_value(b), Some(0));
}

#[test]
fn try_bv_const_value_none_for_wide_constants() {
    // Widths > 128 live in the limb pool; the u128-returning accessor
    // can't represent them and should return None (callers fall back to
    // `bv_const_value_limbs`).
    let mut s = SmtSolver::new();
    let c = s.bv_const_wide(&[1, 0, 0, 0], 256);
    assert!(s.try_bv_const_value(c).is_none());
}

// ----- bv_rotate_left / bv_rotate_right -----

#[test]
fn bv_rotate_left_constant() {
    let mut s = SmtSolver::new();
    let c = s.bv_const(0x8000_0001, 32);
    let rol = s.bv_rotate_left(c, 1);
    // 0x8000_0001 rol 1 = 0x0000_0003
    assert_eq!(s.try_bv_const_value(rol), Some(0x0000_0003));
}

#[test]
fn bv_rotate_right_constant() {
    let mut s = SmtSolver::new();
    let c = s.bv_const(0x0000_0003, 32);
    let ror = s.bv_rotate_right(c, 1);
    // 0x0000_0003 ror 1 = 0x8000_0001
    assert_eq!(s.try_bv_const_value(ror), Some(0x8000_0001));
}

#[test]
fn bv_rotate_zero_is_identity() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    assert_eq!(s.bv_rotate_left(x, 0), x);
    assert_eq!(s.bv_rotate_right(x, 0), x);
}

#[test]
fn bv_rotate_modulo_width() {
    // Rotating a 32-bit value by 33 bits == rotating by 1.
    let mut s = SmtSolver::new();
    let c = s.bv_const(0x8000_0000, 32);
    let a = s.bv_rotate_left(c, 33);
    let b = s.bv_rotate_left(c, 1);
    assert_eq!(a, b);
    assert_eq!(s.try_bv_const_value(a), Some(0x0000_0001));
}

#[test]
fn bv_rotate_round_trip_is_identity() {
    // Proof obligation: for any 32-bit x, `rol(ror(x, 7), 7) == x`. Assert
    // the negation is unsat.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let rot = s.bv_rotate_right(x, 7);
    let back = s.bv_rotate_left(rot, 7);
    let eq = s.bv_eq(x, back);
    let neq = s.bool_not(eq);
    s.assert(neq);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

// ----- solve_under_assumptions_bounded -----

#[test]
fn bounded_solve_trivial_sat() {
    // An easy SAT problem must complete well within 100 conflicts.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let c = s.bv_const(42, 8);
    let eq = s.bv_eq(x, c);
    s.assert(eq);
    let r = s.solve_under_assumptions_bounded(&[], 100);
    assert_eq!(r, Some(SmtResult::Sat));
}

#[test]
fn bounded_solve_trivial_unsat() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let c1 = s.bv_const(1, 8);
    let c2 = s.bv_const(2, 8);
    let eq1 = s.bv_eq(x, c1);
    let eq2 = s.bv_eq(x, c2);
    s.assert(eq1);
    s.assert(eq2);
    let r = s.solve_under_assumptions_bounded(&[], 100);
    assert_eq!(r, Some(SmtResult::Unsat));
}

#[test]
fn bounded_solve_budget_zero_is_unbounded() {
    // Budget 0 is the explicit "no limit" sentinel and must always resolve.
    let mut s = SmtSolver::new();
    let x = s.bv_var(16);
    let y = s.bv_var(16);
    let sum = s.bv_add(x, y);
    let c = s.bv_const(0x1234, 16);
    let eq = s.bv_eq(sum, c);
    s.assert(eq);
    let r = s.solve_under_assumptions_bounded(&[], 0);
    assert_eq!(r, Some(SmtResult::Sat));
}

#[test]
fn timed_solve_fast_sat_completes() {
    // Trivial SAT with a generous deadline — must resolve well inside the
    // budget. 250ms is plenty for a 32-bit equality.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let c = s.bv_const(0xDEAD_BEEF, 32);
    let eq = s.bv_eq(x, c);
    s.assert(eq);
    let r = s.solve_under_assumptions_timed(&[], std::time::Duration::from_millis(250));
    assert_eq!(r, Some(SmtResult::Sat));
}

#[test]
fn timed_solve_retry_after_none_works() {
    // After a timeout-driven `None`, a subsequent solve must succeed —
    // the solver is expected to stay in a consistent state.
    let mut s = SmtSolver::new();
    let x = s.bv_var(16);
    let c = s.bv_const(123, 16);
    let eq = s.bv_eq(x, c);
    s.assert(eq);
    // Impossibly-short budget: even 0ns probably won't trip before the
    // first conflict, but this test only cares that a retry works, not
    // that the first call actually timed out.
    let _ = s.solve_under_assumptions_timed(&[], std::time::Duration::from_nanos(1));
    let r2 = s.solve_under_assumptions_timed(&[], std::time::Duration::from_secs(5));
    assert_eq!(r2, Some(SmtResult::Sat));
}

#[test]
fn timed_solve_unsat_is_real_proof() {
    // UNSAT under a generous timeout is genuine — the timeout only ever
    // converts still-searching into None, never approximates.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let c1 = s.bv_const(5, 8);
    let c2 = s.bv_const(10, 8);
    let eq1 = s.bv_eq(x, c1);
    let eq2 = s.bv_eq(x, c2);
    s.assert(eq1);
    s.assert(eq2);
    let r = s.solve_under_assumptions_timed(&[], std::time::Duration::from_secs(1));
    assert_eq!(r, Some(SmtResult::Unsat));
}

#[test]
fn bounded_solve_with_assumptions_branches_feasibility() {
    // Symbex-style branch feasibility probe: a single shared formula, two
    // queries under opposite assumptions. Both must return a definite
    // answer under a generous budget.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let threshold = s.bv_const(100, 32);
    let above = s.bv_ult(threshold, x);
    // No top-level assertions yet — the formula is `true`.
    let above_taken = s.solve_under_assumptions_bounded(&[above], 10_000);
    let not_above = s.bool_not(above);
    let above_not_taken = s.solve_under_assumptions_bounded(&[not_above], 10_000);
    assert_eq!(above_taken, Some(SmtResult::Sat));
    assert_eq!(above_not_taken, Some(SmtResult::Sat));
}
