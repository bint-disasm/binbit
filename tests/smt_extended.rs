//! Tests for the extended SMT features: signed div/mod, overflow predicates,
//! push/pop scoping, and constant folding.

use binbit::{SmtResult, SmtSolver};

// ---------- Signed div / rem / smod ----------

#[test]
fn sdiv_negative_by_positive() {
    // -10 / 3 (8-bit) = -3 (rounds toward zero).
    let mut s = SmtSolver::new();
    let neg10 = s.bv_const((-10i8) as u8 as u128, 8);
    let three = s.bv_const(3, 8);
    let q = s.bv_sdiv(neg10, three);
    let expected = s.bv_const((-3i8) as u8 as u128, 8);
    let eq = s.bv_eq(q, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn sdiv_negative_by_negative() {
    // -10 / -3 = 3.
    let mut s = SmtSolver::new();
    let neg10 = s.bv_const((-10i8) as u8 as u128, 8);
    let neg3 = s.bv_const((-3i8) as u8 as u128, 8);
    let q = s.bv_sdiv(neg10, neg3);
    let three = s.bv_const(3, 8);
    let eq = s.bv_eq(q, three);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn srem_follows_dividend_sign() {
    // -10 srem 3 = -1 (sign of -10).
    let mut s = SmtSolver::new();
    let neg10 = s.bv_const((-10i8) as u8 as u128, 8);
    let three = s.bv_const(3, 8);
    let r = s.bv_srem(neg10, three);
    let expected = s.bv_const((-1i8) as u8 as u128, 8);
    let eq = s.bv_eq(r, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn smod_follows_divisor_sign() {
    // -10 smod 3 = 2  (brings remainder into +3's sign half-plane).
    let mut s = SmtSolver::new();
    let neg10 = s.bv_const((-10i8) as u8 as u128, 8);
    let three = s.bv_const(3, 8);
    let m = s.bv_smod(neg10, three);
    let expected = s.bv_const(2, 8);
    let eq = s.bv_eq(m, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn bv_neg_round_trips() {
    // -(-x) = x for every 8-bit x.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let neg = s.bv_neg(x);
    let neg2 = s.bv_neg(neg);
    let differs = s.bv_ne(x, neg2);
    s.assert(differs);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

// ---------- Overflow predicates ----------

#[test]
fn uadd_overflow_on_wrap() {
    // 200 + 100 overflows unsigned 8-bit.
    let mut s = SmtSolver::new();
    let a = s.bv_const(200, 8);
    let b = s.bv_const(100, 8);
    let ovf = s.bv_uadd_overflow(a, b);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn uadd_overflow_clear_on_fits() {
    // 100 + 100 does not overflow unsigned 8-bit.
    let mut s = SmtSolver::new();
    let a = s.bv_const(100, 8);
    let b = s.bv_const(100, 8);
    let ovf = s.bv_uadd_overflow(a, b);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn sadd_overflow_on_positive_wrap() {
    // 100 + 100 signed 8-bit: 200 > 127 → overflows.
    let mut s = SmtSolver::new();
    let a = s.bv_const(100, 8);
    let b = s.bv_const(100, 8);
    let ovf = s.bv_sadd_overflow(a, b);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn sadd_no_overflow_on_mixed_signs() {
    // Mixed-sign additions never overflow.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    // Force sign(x) != sign(y): (x >> 7) != (y >> 7) via xor of MSBs.
    let msb_x = s.bv_extract(x, 7, 7);
    let msb_y = s.bv_extract(y, 7, 7);
    let diff_sign = s.bv_ne(msb_x, msb_y);
    s.assert(diff_sign);
    // And assert overflow — should be UNSAT.
    let ovf = s.bv_sadd_overflow(x, y);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn umul_overflow_on_small_inputs() {
    // 17 * 17 = 289 > 255 → overflow in 8-bit unsigned.
    let mut s = SmtSolver::new();
    let a = s.bv_const(17, 8);
    let b = s.bv_const(17, 8);
    let ovf = s.bv_umul_overflow(a, b);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn umul_no_overflow_when_product_fits() {
    let mut s = SmtSolver::new();
    let a = s.bv_const(5, 8);
    let b = s.bv_const(20, 8);
    let ovf = s.bv_umul_overflow(a, b);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn smul_overflow_on_int_min_times_neg_one() {
    // -128 * -1 overflows signed 8-bit (would be +128, not representable).
    let mut s = SmtSolver::new();
    let int_min = s.bv_const((-128i8) as u8 as u128, 8);
    let neg_one = s.bv_const((-1i8) as u8 as u128, 8);
    let ovf = s.bv_smul_overflow(int_min, neg_one);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn neg_overflow_only_on_int_min() {
    // Find the unique 8-bit x where -x overflows.
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let ovf = s.bv_neg_overflow(x);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(x), 0x80); // INT_MIN
}

#[test]
fn sdiv_overflow_only_on_int_min_over_neg_one() {
    // sdiv(a, b) overflows iff a = INT_MIN AND b = -1.
    let mut s = SmtSolver::new();
    let a = s.bv_var(8);
    let b = s.bv_var(8);
    let ovf = s.bv_sdiv_overflow(a, b);
    s.assert(ovf);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value(a), 0x80);
    assert_eq!(s.get_bv_value(b), 0xFF);
}

// ---------- Push / Pop scoping ----------

#[test]
fn push_pop_retracts_assertions() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let five = s.bv_const(5, 8);
    let ten = s.bv_const(10, 8);

    // Global assertion: x < 10.
    let lt10 = s.bv_ult(x, ten);
    s.assert(lt10);

    // Scope 1: add x > 5. Intersection 6..=9 is satisfiable.
    s.push();
    let gt5 = s.bv_ult(five, x);
    s.assert(gt5);
    assert_eq!(s.solve(), SmtResult::Sat);

    // Scope 2: add x == 100 (contradicts x < 10 → unsat under these scopes).
    s.push();
    let hundred = s.bv_const(100, 8);
    let eq100 = s.bv_eq(x, hundred);
    s.assert(eq100);
    assert_eq!(s.solve(), SmtResult::Unsat);

    // Pop scope 2: the contradictory eq100 is retracted. Still sat.
    s.pop();
    assert_eq!(s.solve(), SmtResult::Sat);

    // Pop scope 1: now only x < 10 remains.
    s.pop();
    assert_eq!(s.solve(), SmtResult::Sat);

    // Sanity: scope_depth bottoms out at 0.
    assert_eq!(s.scope_depth(), 0);
}

#[test]
fn push_pop_depth_tracking() {
    let mut s = SmtSolver::new();
    assert_eq!(s.scope_depth(), 0);
    s.push();
    s.push();
    s.push();
    assert_eq!(s.scope_depth(), 3);
    s.pop();
    assert_eq!(s.scope_depth(), 2);
    s.pop();
    s.pop();
    assert_eq!(s.scope_depth(), 0);
    // Extra pop is harmless.
    s.pop();
    assert_eq!(s.scope_depth(), 0);
}

// ---------- Constant folding ----------

#[test]
fn folding_collapses_constant_arithmetic() {
    let mut s = SmtSolver::new();
    // All-constant expression: no SAT vars ever materialize beyond the
    // dedicated true-lit. We just check behaviour.
    let a = s.bv_const(7, 8);
    let b = s.bv_const(3, 8);
    let c = s.bv_add(a, b);
    let d = s.bv_mul(c, b);
    // (7+3) * 3 = 30.
    let thirty = s.bv_const(30, 8);
    let eq = s.bv_eq(d, thirty);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn folding_identity_rewrites_return_input() {
    use binbit::BvContext;
    let mut ctx = BvContext::new();
    let x = ctx.bv_var(8);
    let zero = ctx.bv_const(0, 8);
    let one = ctx.bv_const(1, 8);
    // x + 0 should return x directly (no new node).
    assert_eq!(ctx.bv_add(x, zero), x);
    // 0 + x too.
    assert_eq!(ctx.bv_add(zero, x), x);
    // x * 1 == x.
    assert_eq!(ctx.bv_mul(x, one), x);
    // x * 0 == 0.
    assert_eq!(ctx.bv_mul(x, zero), zero);
    // x & x = x, x ^ x = 0.
    assert_eq!(ctx.bv_and(x, x), x);
    assert_eq!(ctx.bv_xor(x, x), zero);
    // not not x = x.
    let nx = ctx.bv_not(x);
    assert_eq!(ctx.bv_not(nx), x);
}

// ---------- SMT-LIB script ----------

#[test]
fn smtlib_simple_sat() {
    let script = r#"
        (set-logic QF_BV)
        (declare-const x (_ BitVec 8))
        (declare-const y (_ BitVec 8))
        (assert (= (bvadd x y) (_ bv42 8)))
        (assert (= (bvsub x y) (_ bv10 8)))
        (check-sat)
        (get-value (x y))
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    // Under 8-bit wrap-around, (26, 16) and (154, 144) both satisfy the
    // system. Accept either — the SAT solver's model search isn't canonical.
    assert!(out.contains("sat"));
    let ok_26_16 = out.contains("(_ bv26 8)") && out.contains("(_ bv16 8)");
    let ok_154_144 = out.contains("(_ bv154 8)") && out.contains("(_ bv144 8)");
    assert!(ok_26_16 || ok_154_144, "output = {}", out);
}

#[test]
fn smtlib_simple_unsat() {
    let script = r#"
        (declare-const x (_ BitVec 8))
        (assert (bvult x (_ bv5 8)))
        (assert (bvugt x (_ bv10 8)))
        (check-sat)
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.trim_end().ends_with("unsat"), "output = {}", out);
}

#[test]
fn smtlib_push_pop_script() {
    let script = r#"
        (declare-const x (_ BitVec 8))
        (assert (bvult x (_ bv10 8)))
        (push)
          (assert (= x (_ bv100 8)))
          (check-sat)
        (pop)
        (check-sat)
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    let results: Vec<&str> = out.lines().collect();
    assert_eq!(results, vec!["unsat", "sat"]);
}

#[test]
fn smtlib_signed_ops() {
    let script = r#"
        (declare-const x (_ BitVec 8))
        (declare-const y (_ BitVec 8))
        ; -10 / 3 (as 8-bit signed) should be representable.
        (assert (bvslt x (_ bv0 8)))
        (assert (= x (bvneg (_ bv10 8))))
        (assert (= y (bvsdiv x (_ bv3 8))))
        (check-sat)
        (get-value (y))
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
    // -3 as 8-bit = 0xFD = 253.
    assert!(out.contains("(_ bv253 8)"), "output = {}", out);
}

#[test]
fn smtlib_let_bindings_and_hex_literal() {
    let script = r#"
        (declare-const x (_ BitVec 16))
        (assert (let ((mask #xFF00))
                     (= (bvand x mask) #xAB00)))
        (check-sat)
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
}

#[test]
fn smtlib_extract_and_concat() {
    // x[15:8] is low byte of upper half; concat reassembles a 16-bit value.
    let script = r#"
        (declare-const x (_ BitVec 16))
        (assert (= x (concat ((_ extract 15 8) x) ((_ extract 7 0) x))))
        (check-sat)
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
}

#[test]
fn smtlib_ite_branches() {
    let script = r#"
        (declare-const p Bool)
        (declare-const x (_ BitVec 8))
        (assert (= x (ite p (_ bv10 8) (_ bv20 8))))
        (assert p)
        (check-sat)
        (get-value (x))
    "#;
    let out = binbit::run_script(script).expect("parse ok");
    assert!(out.contains("sat"));
    assert!(out.contains("(_ bv10 8)"));
}
