//! Tests for the state-merge primitives: `bv_select`, the `ite`-factoring
//! rewrite, and `assert_mutually_exclusive`.

use binbit::{BoolTerm, SmtResult, SmtSolver};

fn make_bool_var(s: &mut SmtSolver) -> BoolTerm {
    s.bool_var()
}

/// A Select with a single const-true selector short-circuits to that value.
#[test]
fn bv_select_first_true_wins() {
    let mut s = SmtSolver::new();
    let v0 = s.bv_const(10, 8);
    let v1 = s.bv_const(20, 8);
    let default = s.bv_const(99, 8);
    let sels = [s.bool_true(), s.bool_false()];
    let vals = [v0, v1];
    let out = s.bv_select(&sels, &vals, default);
    // Short-circuit should give us v0 itself — no new node, no constraint.
    assert_eq!(out, v0);
}

/// All-false selectors collapse to the default.
#[test]
fn bv_select_all_false_returns_default() {
    let mut s = SmtSolver::new();
    let default = s.bv_const(77, 8);
    let v0 = s.bv_const(10, 8);
    let v1 = s.bv_const(20, 8);
    let sels = [s.bool_false(), s.bool_false()];
    let vals = [v0, v1];
    let out = s.bv_select(&sels, &vals, default);
    assert_eq!(out, default);
}

/// Branches whose value equals the default drop out — they're indistinguishable
/// from falling through.
#[test]
fn bv_select_default_equal_branches_drop() {
    let mut s = SmtSolver::new();
    let default = s.bv_const(7, 8);
    let v_same = s.bv_const(7, 8); // same constant — dedupes via hash cons
    let v_diff = s.bv_const(11, 8);
    let s0 = s.bool_var();
    let s1 = s.bool_var();
    let sels = [s0, s1];
    let vals = [v_same, v_diff];
    // The s0 branch collapses; only s1 survives. Semantics:
    //   s0 ∧ ¬s1 → default   (s0 pair dropped, default picked when none match)
    //   ¬s0 ∧ s1 → v_diff
    //   s0 ∧ s1  → v_diff   (first-match originally v_same=default; drop → v_diff).
    // The last case changes "what value" but not "observed output" since default==v_same.
    let out = s.bv_select(&sels, &vals, default);
    // If s1 is forced true, the merged value must be v_diff (=11).
    s.assert(s1);
    let probe = s.bv_const(11, 8);
    let eq = s.bv_eq(out, probe);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

/// Build a 3-way merge with symbolic selectors + values, then check that
/// forcing one selector picks its value and all others end up unused.
#[test]
fn bv_select_three_way_merge_picks_right_value() {
    let mut s = SmtSolver::new();
    let default = s.bv_var(32);
    let v0 = s.bv_var(32);
    let v1 = s.bv_var(32);
    let v2 = s.bv_var(32);
    let s0 = s.bool_var();
    let s1 = s.bool_var();
    let s2 = s.bool_var();
    let sels = [s0, s1, s2];
    let vals = [v0, v1, v2];
    s.assert_mutually_exclusive(&sels);
    let out = s.bv_select(&sels, &vals, default);

    // Force s1 alone; require v1 = 0x1234_5678.
    s.assert(s1);
    let target = s.bv_const(0x1234_5678, 32);
    let eq_v1 = s.bv_eq(v1, target);
    s.assert(eq_v1);

    assert_eq!(s.solve(), SmtResult::Sat);
    // Since s1 is asserted and selectors are mutually exclusive, s0 and s2
    // must be false and the Select must match v1 = target.
    assert_eq!(s.get_bv_value_u128(out), 0x1234_5678);
    assert!(!s.get_bool_value(s0));
    assert!(!s.get_bool_value(s2));
}

/// A Select where no selector fires collapses to the default value.
#[test]
fn bv_select_default_path() {
    let mut s = SmtSolver::new();
    let default = s.bv_const(42, 16);
    let v0 = s.bv_var(16);
    let v1 = s.bv_var(16);
    let s0 = s.bool_var();
    let s1 = s.bool_var();
    s.assert_mutually_exclusive(&[s0, s1]);
    // Force every selector false so only the default path is live.
    let not_s0 = s.bool_not(s0);
    let not_s1 = s.bool_not(s1);
    s.assert(not_s0);
    s.assert(not_s1);
    let out = s.bv_select(&[s0, s1], &[v0, v1], default);
    let forty_two = s.bv_const(42, 16);
    let eq = s.bv_eq(out, forty_two);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert_eq!(s.get_bv_value_u128(out), 42);
}

/// Mutual exclusion forbids two selectors being simultaneously true.
#[test]
fn assert_mutually_exclusive_rejects_overlap() {
    let mut s = SmtSolver::new();
    let s0 = s.bool_var();
    let s1 = s.bool_var();
    let s2 = s.bool_var();
    s.assert_mutually_exclusive(&[s0, s1, s2]);
    // Try to force s0 ∧ s1 to both hold — should be unsat.
    s.assert(s0);
    s.assert(s1);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

/// ITE factoring: `ite(c, ite(d, x, y), ite(d, x, z))` → `ite(d, x, ite(c, y, z))`.
/// We don't probe the DAG shape directly — we verify the semantic output
/// matches the expected formula across symbolic inputs.
#[test]
fn ite_factoring_common_then_branch() {
    let mut s = SmtSolver::new();
    let c = make_bool_var(&mut s);
    let d = make_bool_var(&mut s);
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let z = s.bv_var(8);

    // Build both sides: the "factored-away" form (lhs) and the equivalent
    // flat form (rhs). They must agree on every assignment.
    let inner_t = s.bv_ite(d, x, y);
    let inner_e = s.bv_ite(d, x, z);
    let lhs = s.bv_ite(c, inner_t, inner_e);

    let inner_rhs = s.bv_ite(c, y, z);
    let rhs = s.bv_ite(d, x, inner_rhs);

    // Assert they differ — should be unsat (they're structurally identical
    // after factoring, and semantically equal regardless).
    let neq = s.bv_eq(lhs, rhs);
    let neq_not = s.bool_not(neq);
    s.assert(neq_not);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

/// A medium-sized state-merge stress test: 8 paths, 32-bit values, one per
/// path forced equal to a specific constant; verify the merge picks up the
/// right value for each fired selector.
#[test]
fn bv_select_eight_way_merge_stress() {
    let constants: [u128; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    for forced in 0..8 {
        let mut s = SmtSolver::new();
        let selectors: Vec<_> = (0..8).map(|_| s.bool_var()).collect();
        let values: Vec<_> = constants.iter().map(|&c| s.bv_const(c, 32)).collect();
        let default = s.bv_const(0xDEADBEEF, 32);
        s.assert_mutually_exclusive(&selectors);
        let out = s.bv_select(&selectors, &values, default);
        s.assert(selectors[forced]);

        assert_eq!(s.solve(), SmtResult::Sat, "forced={}", forced);
        assert_eq!(
            s.get_bv_value_u128(out),
            constants[forced],
            "forced={}",
            forced
        );
    }
}
