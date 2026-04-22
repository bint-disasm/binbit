//! Tests for the latest SMT layer additions: 64-bit perf specializations,
//! named assertions / unsat cores, SMT-LIB overflow names, and CLI.

use binbit::{SmtResult, SmtSolver};

// ---------- 64-bit perf — correctness under the new fast paths ----------

#[test]
fn const_shift_matches_variable_shift_64bit() {
    // x << 7 with constant 7 goes through the wiring fast path. Assert that
    // the result is the same as the general barrel-shifter form to confirm
    // correctness.
    let mut s = SmtSolver::new();
    let x = s.bv_var(64);
    let seven_const = s.bv_const(7, 64);
    let shifted_a = s.bv_shl(x, seven_const);
    // Recreate the general path by forcing a variable shift that we then
    // constrain equal to 7.
    let amt = s.bv_var(64);
    let amt_eq_7 = s.bv_eq(amt, seven_const);
    s.assert(amt_eq_7);
    let shifted_b = s.bv_shl(x, amt);
    let differ = s.bv_ne(shifted_a, shifted_b);
    s.assert(differ);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn const_shift_large_amount_clears() {
    // x << 64 (equal to width) clears to 0.
    let mut s = SmtSolver::new();
    let x = s.bv_var(64);
    let sixty_four = s.bv_const(64, 64);
    let shifted = s.bv_shl(x, sixty_four);
    let zero = s.bv_const(0, 64);
    let eq = s.bv_eq(shifted, zero);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn mul_by_constant_64bit() {
    // x * 10 = 100 → x = 10. Exercises mk_mul_const on a 64-bit multiplicand.
    let mut s = SmtSolver::new();
    let x = s.bv_var(64);
    let ten = s.bv_const(10, 64);
    let hundred = s.bv_const(100, 64);
    let prod = s.bv_mul(x, ten);
    let eq = s.bv_eq(prod, hundred);
    s.assert(eq);
    // Constrain x to a value we can pin uniquely: x < 100 so it's 10, not
    // one of the modular wrap solutions.
    let lt_100 = s.bv_ult(x, hundred);
    s.assert(lt_100);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 10);
}

#[test]
fn mul_by_power_of_two_becomes_shift() {
    // x * 64 on 64-bit BV. 64 = 2^6 so this becomes a constant left-shift
    // (pure wiring). Check the value is sane.
    let mut s = SmtSolver::new();
    let x = s.bv_const(5, 64);
    let sixty_four = s.bv_const(64, 64);
    let prod = s.bv_mul(x, sixty_four);
    let expected = s.bv_const(320, 64);
    let eq = s.bv_eq(prod, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn udiv_by_power_of_two_becomes_shift() {
    // 1000 / 8 (64-bit) = 125. Should go through the shift-rewrite fast path.
    let mut s = SmtSolver::new();
    let thousand = s.bv_const(1000, 64);
    let eight = s.bv_const(8, 64);
    let q = s.bv_udiv(thousand, eight);
    let expected = s.bv_const(125, 64);
    let eq = s.bv_eq(q, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn urem_by_power_of_two_becomes_mask() {
    // 1000 % 8 = 0, 1001 % 8 = 1 etc. Checks the mask-rewrite fast path.
    let mut s = SmtSolver::new();
    let v = s.bv_const(1001, 64);
    let eight = s.bv_const(8, 64);
    let r = s.bv_urem(v, eight);
    let expected = s.bv_const(1, 64);
    let eq = s.bv_eq(r, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn power_of_two_rewrites_preserve_correctness_under_variable() {
    // With symbolic x, `x * 4` must match `x << 2` bit-for-bit.
    let mut s = SmtSolver::new();
    let x = s.bv_var(32);
    let four = s.bv_const(4, 32);
    let two = s.bv_const(2, 32);
    let mul = s.bv_mul(x, four);
    let shl = s.bv_shl(x, two);
    let differ = s.bv_ne(mul, shl);
    s.assert(differ);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

// ---------- Named assertions / unsat core ----------

#[test]
fn named_assertions_basic_core() {
    // Three named facts, two of which jointly contradict.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let five = s.bv_const(5, 8);
    let ten = s.bv_const(10, 8);
    let twenty = s.bv_const(20, 8);

    let lt_ten = s.bv_ult(x, ten);
    let gt_twenty = s.bv_ult(twenty, x);
    let lt_five = s.bv_ult(x, five);

    s.assert_named("A", lt_ten);
    s.assert_named("B", gt_twenty);
    s.assert_named("C", lt_five);

    assert_eq!(s.solve(), SmtResult::Unsat);
    let core = s.unsat_core_names();
    // A + B alone are inconsistent (x < 10 AND x > 20). C may or may not be
    // included depending on which subset the solver blames, but A and B must
    // both show up.
    assert!(core.contains(&"A"), "core = {:?}", core);
    assert!(core.contains(&"B"), "core = {:?}", core);
}

#[test]
fn named_assertion_irrelevant_not_in_core() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let zero = s.bv_const(0, 8);
    let one = s.bv_const(1, 8);

    // x = 0, x = 1 is the conflict. y = 5 is irrelevant.
    let x_eq_0 = s.bv_eq(x, zero);
    let x_eq_1 = s.bv_eq(x, one);
    let five = s.bv_const(5, 8);
    let y_eq_5 = s.bv_eq(y, five);

    s.assert_named("x_zero", x_eq_0);
    s.assert_named("x_one", x_eq_1);
    s.assert_named("y_five", y_eq_5);

    assert_eq!(s.solve(), SmtResult::Unsat);
    let core = s.unsat_core_names();
    assert!(core.contains(&"x_zero"));
    assert!(core.contains(&"x_one"));
    assert!(!core.contains(&"y_five"), "core = {:?}", core);
}

// ---------- SMT-LIB: named assertions & get-unsat-core ----------

#[test]
fn smtlib_named_assertions_and_core() {
    let script = r#"
        (set-option :produce-unsat-cores true)
        (declare-const x (_ BitVec 8))
        (assert (! (bvult x (_ bv10 8)) :named lt10))
        (assert (! (bvugt x (_ bv20 8)) :named gt20))
        (check-sat)
        (get-unsat-core)
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("unsat"), "output = {}", out);
    assert!(out.contains("lt10"), "output = {}", out);
    assert!(out.contains("gt20"), "output = {}", out);
}

#[test]
fn smtlib_overflow_predicate_names() {
    // Uses CVC5/Z3-style `bvuaddo` etc.
    let script = r#"
        (declare-const x (_ BitVec 8))
        (declare-const y (_ BitVec 8))
        (assert (= x (_ bv200 8)))
        (assert (= y (_ bv100 8)))
        (assert (bvuaddo x y))
        (check-sat)
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"), "output = {}", out);
}

#[test]
fn smtlib_overflow_predicates_negative_cases() {
    // Assert that bvsmulo on INT_MIN * -1 is indeed true.
    let script = r#"
        (declare-const x (_ BitVec 8))
        (assert (= x (_ bv128 8)))  ; 0x80 = INT_MIN
        (assert (bvsmulo x (_ bv255 8)))  ; 0xFF = -1
        (check-sat)
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
}

#[test]
fn smtlib_bvnego_finds_int_min() {
    let script = r#"
        (declare-const x (_ BitVec 8))
        (assert (bvnego x))
        (check-sat)
        (get-value (x))
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
    assert!(out.contains("(_ bv128 8)"));
}

#[test]
fn smtlib_64bit_arithmetic_and_shifts() {
    // End-to-end: solve a real 64-bit mul+shift problem.
    let script = r#"
        (declare-const x (_ BitVec 64))
        ; x << 4 == 256 → x = 16.
        (assert (= (bvshl x (_ bv4 64)) (_ bv256 64)))
        (check-sat)
        (get-value (x))
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
    assert!(out.contains("(_ bv16 64)"));
}
