//! Tests for term-level Gaussian elimination over Z/2^w. The pass solves
//! coupled linear equality systems before bitblasting, installs the
//! solutions as variable substitutions, and detects inconsistent systems as
//! UNSAT. Model reads must stay consistent (a solved variable's value equals
//! its defining expression).

use binbit::{SmtResult, SmtSolver};

/// A uniquely-determined 2x2 system (`x + y = a, x + 2y = b`). Neither
/// equation is in `x = t` form, so plain substitution can't crack it — GE
/// must. The `x + 2y` shape is chosen so elimination yields an odd leading
/// coefficient (`y = b − a`), giving a unique solution mod 2^w.
#[test]
fn coupled_2x2_system_is_solved() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let y = s.bv_var(w);
    let a = s.bv_const(30, w);
    let b = s.bv_const(41, w);

    // x + y = 30
    let sum = s.bv_add(x, y);
    let e1 = s.bv_eq(sum, a);
    s.assert(e1);
    // x + 2y = 41. Subtracting the rows gives y = 11 (odd leading
    // coefficient) — a genuinely unique solution over Z/256, unlike
    // x - y = b, whose subtraction yields the zero-divisor 2y and two
    // models.
    let y2 = {
        let two = s.bv_const(2, w);
        s.bv_mul(y, two)
    };
    let x_plus_2y = s.bv_add(x, y2);
    let e2 = s.bv_eq(x_plus_2y, b);
    s.assert(e2);

    assert_eq!(s.solve(), SmtResult::Sat);
    // y = 41 - 30 = 11, x = 30 - 11 = 19.
    assert_eq!(s.get_bv_value(x), 19);
    assert_eq!(s.get_bv_value(y), 11);
}

/// Adding a third constraint consistent with the 2x2 solution stays SAT and
/// the values are unchanged.
#[test]
fn coupled_system_plus_consistent_constraint() {
    let mut s = SmtSolver::new();
    let w = 16;
    let x = s.bv_var(w);
    let y = s.bv_var(w);
    let a = s.bv_const(100, w);
    let b = s.bv_const(40, w);

    let sum = s.bv_add(x, y);
    let e1 = s.bv_eq(sum, a); // x + y = 100
    s.assert(e1);
    let diff = s.bv_sub(x, y);
    let e2 = s.bv_eq(diff, b); // x - y = 40
    s.assert(e2);
    // x = 70 consistent with the unique solution (70, 30).
    let seventy = s.bv_const(70, w);
    let e3 = s.bv_eq(x, seventy);
    s.assert(e3);

    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 70);
    assert_eq!(s.get_bv_value(y), 30);
}

/// A third constraint that contradicts the solved system → UNSAT.
#[test]
fn inconsistent_extra_constraint_is_unsat() {
    let mut s = SmtSolver::new();
    let w = 16;
    let x = s.bv_var(w);
    let y = s.bv_var(w);
    let a = s.bv_const(100, w);
    let b = s.bv_const(40, w);

    let sum = s.bv_add(x, y);
    let e1 = s.bv_eq(sum, a);
    s.assert(e1);
    let diff = s.bv_sub(x, y);
    let e2 = s.bv_eq(diff, b);
    s.assert(e2);
    // Unique solution is x = 70; assert x = 71.
    let wrong = s.bv_const(71, w);
    let e3 = s.bv_eq(x, wrong);
    s.assert(e3);

    assert_eq!(s.solve(), SmtResult::Unsat);
}

/// Directly inconsistent system: `x + y = 3, x + y = 5`. GE reduces the
/// second row to `0 = 2` → UNSAT, no search.
#[test]
fn directly_inconsistent_system_is_unsat() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let y = s.bv_var(w);
    let sum = s.bv_add(x, y);
    let three = s.bv_const(3, w);
    let five = s.bv_const(5, w);
    let e1 = s.bv_eq(sum, three);
    let e2 = s.bv_eq(sum, five);
    s.assert(e1);
    s.assert(e2);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

/// Coefficients that require a real modular inverse: `3x = 21 (mod 256)`.
/// 3 is odd (a unit mod 256), so x = 3^{-1} · 21 = 7. Confirms the
/// Hensel-lifted inverse path, not just coefficient-1 pivots.
#[test]
fn odd_coefficient_uses_modular_inverse() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let three_x = {
        let c = s.bv_const(3, w);
        s.bv_mul(x, c)
    };
    let c21 = s.bv_const(21, w);
    let e = s.bv_eq(three_x, c21);
    s.assert(e);

    // Another constraint referencing x so we can read its value back.
    let y = s.bv_var(w);
    let ey = s.bv_eq(y, x);
    s.assert(ey);

    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 7);
    assert_eq!(s.get_bv_value(y), 7);
}

/// `5x = 3 (mod 256)`: 5^{-1} mod 256 = 205, so x = 205·3 = 615 mod 256 =
/// 103. Verify the solver agrees with the hand computation.
#[test]
fn modular_inverse_nontrivial_solution() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let c5 = s.bv_const(5, w);
    let five_x = s.bv_mul(x, c5);
    let c3 = s.bv_const(3, w);
    let e = s.bv_eq(five_x, c3);
    s.assert(e);
    assert_eq!(s.solve(), SmtResult::Sat);
    // 5 * 103 = 515 = 2*256 + 3 ≡ 3 (mod 256). ✓
    assert_eq!(s.get_bv_value(x), 103);
}

/// Even-only coefficient: `2x = 6 (mod 256)` has TWO solutions (3 and 131),
/// so GE must NOT pivot on it (2 has no inverse mod 256). The equation
/// falls through to the SAT core, which still finds a valid model. This
/// guards against an unsound "solve" that would pick one solution and
/// wrongly drop the other.
#[test]
fn even_coefficient_not_pivoted_but_still_sat() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let c2 = s.bv_const(2, w);
    let two_x = s.bv_mul(x, c2);
    let c6 = s.bv_const(6, w);
    let e = s.bv_eq(two_x, c6);
    s.assert(e);
    assert_eq!(s.solve(), SmtResult::Sat);
    // Model must genuinely satisfy 2x ≡ 6.
    let xv = s.get_bv_value(x);
    assert_eq!(xv.wrapping_mul(2) & 0xFF, 6);
}

/// A 3-variable chain: x + y = 10, y + z = 7, z = 2. Back-substitution
/// gives z = 2, y = 5, x = 5.
#[test]
fn three_variable_chain() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let y = s.bv_var(w);
    let z = s.bv_var(w);

    let xy = s.bv_add(x, y);
    let c10 = s.bv_const(10, w);
    let e1 = s.bv_eq(xy, c10);
    s.assert(e1);

    let yz = s.bv_add(y, z);
    let c7 = s.bv_const(7, w);
    let e2 = s.bv_eq(yz, c7);
    s.assert(e2);

    let c2 = s.bv_const(2, w);
    let e3 = s.bv_eq(z, c2);
    s.assert(e3);

    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(z), 2);
    assert_eq!(s.get_bv_value(y), 5);
    assert_eq!(s.get_bv_value(x), 5);
}

/// GE must not disturb a satisfiable system with a free variable. `x + y =
/// 4` alone has many solutions; the model must satisfy the equation.
#[test]
fn underdetermined_system_stays_sat() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let y = s.bv_var(w);
    let sum = s.bv_add(x, y);
    let c4 = s.bv_const(4, w);
    let e = s.bv_eq(sum, c4);
    s.assert(e);
    assert_eq!(s.solve(), SmtResult::Sat);
    let xv = s.get_bv_value(x);
    let yv = s.get_bv_value(y);
    assert_eq!(xv.wrapping_add(yv) & 0xFF, 4);
}

/// Non-linear content must not be mis-parsed as a row. `x * y = 12` (two
/// variables multiplied) is not linear; the solver must still solve it by
/// search, not by a bogus GE pivot.
#[test]
fn nonlinear_equation_falls_through() {
    let mut s = SmtSolver::new();
    let w = 8;
    let x = s.bv_var(w);
    let y = s.bv_var(w);
    let prod = s.bv_mul(x, y);
    let c12 = s.bv_const(12, w);
    let e = s.bv_eq(prod, c12);
    s.assert(e);
    assert_eq!(s.solve(), SmtResult::Sat);
    let xv = s.get_bv_value(x);
    let yv = s.get_bv_value(y);
    assert_eq!(xv.wrapping_mul(yv) & 0xFF, 12);
}

/// Widths don't mix: a 2x2 system at width 8 and an independent one at
/// width 16 both get solved.
#[test]
fn separate_width_groups() {
    let mut s = SmtSolver::new();
    // width 8: x + y = 30, x + 2y = 41  → y=11, x=19
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let a = s.bv_const(30, 8);
    let b = s.bv_const(41, 8);
    let s1 = s.bv_add(x, y);
    let e1 = s.bv_eq(s1, a);
    s.assert(e1);
    let two8 = s.bv_const(2, 8);
    let y2 = s.bv_mul(y, two8);
    let d1 = s.bv_add(x, y2);
    let e2 = s.bv_eq(d1, b);
    s.assert(e2);

    // width 16: p + q = 100, p + 2q = 130  → q=30, p=70
    let p = s.bv_var(16);
    let q = s.bv_var(16);
    let c = s.bv_const(100, 16);
    let dd = s.bv_const(130, 16);
    let s2 = s.bv_add(p, q);
    let e3 = s.bv_eq(s2, c);
    s.assert(e3);
    let two16 = s.bv_const(2, 16);
    let q2 = s.bv_mul(q, two16);
    let d2 = s.bv_add(p, q2);
    let e4 = s.bv_eq(d2, dd);
    s.assert(e4);

    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 19);
    assert_eq!(s.get_bv_value(y), 11);
    assert_eq!(s.get_bv_value(p), 70);
    assert_eq!(s.get_bv_value(q), 30);
}
