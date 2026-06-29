//! Sanity tests for bitblast cost attribution.
//!
//! Verifies that the per-term cost map reflects each term's *exclusive*
//! contribution to the SAT formula (subterm costs land on subterm rows,
//! not the parent's).

use binbit::{BvContext, SmtSolver};

#[test]
fn cost_tracking_is_off_by_default() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let z = s.bv_add(x, y);
    let zero = s.bv_const(0, 8);
    let eq = s.bv_eq(z, zero);
    s.assert(eq);
    let _ = s.solve();
    assert_eq!(s.bitblast_cost_report().len(), 0);
}

#[test]
fn cost_tracking_records_terms_after_solve() {
    let mut s = SmtSolver::new();
    s.enable_bitblast_cost_tracking();
    let x = s.bv_var(32);
    let y = s.bv_var(32);
    let sum = s.bv_add(x, y);
    let zero = s.bv_const(0, 32);
    let eq = s.bv_eq(sum, zero);
    s.assert(eq);
    let _ = s.solve();

    let report = s.bitblast_cost_report();
    // We should have at least one entry — the add term contributed real
    // clauses (ripple-carry adder).
    assert!(!report.is_empty(), "expected at least one cost entry");
    // Report must be sorted by clauses descending.
    for w in report.windows(2) {
        assert!(w[0].sat_clauses >= w[1].sat_clauses, "report not sorted");
    }
    // The add term should have width 32 (per the BvContext).
    let add_entry = report
        .iter()
        .find(|e| e.term == sum)
        .expect("add term should be in the report");
    assert_eq!(add_entry.width, 32);
    assert!(add_entry.sat_clauses > 0, "add term should have emitted clauses");
}

#[test]
fn cost_is_exclusive_of_subterms() {
    let mut s = SmtSolver::new();
    s.enable_bitblast_cost_tracking();
    // Build: nested = bvadd(bvadd(x, y), z). Outer add has its own work
    // (a second 32-bit adder); inner add has its own (the first).
    let x = s.bv_var(32);
    let y = s.bv_var(32);
    let z = s.bv_var(32);
    let inner = s.bv_add(x, y);
    let outer = s.bv_add(inner, z);
    let zero = s.bv_const(0, 32);
    let eq = s.bv_eq(outer, zero);
    s.assert(eq);
    let _ = s.solve();

    let report = s.bitblast_cost_report();
    let inner_e = report.iter().find(|e| e.term == inner).expect("inner missing");
    let outer_e = report.iter().find(|e| e.term == outer).expect("outer missing");

    // Both adds should emit roughly the same clauses (each is a 32-bit
    // ripple-carry on its own operands). If exclusive accounting were
    // broken and outer "ate" inner's cost, outer would be ~2× inner.
    let ratio = outer_e.sat_clauses as f64 / inner_e.sat_clauses as f64;
    assert!(
        (0.5..2.0).contains(&ratio),
        "outer ({} clauses) and inner ({} clauses) should be comparable; \
         far-off ratio implies cost is not exclusive",
        outer_e.sat_clauses,
        inner_e.sat_clauses,
    );
}

#[test]
fn hashconsed_subterm_charged_only_once() {
    let mut s = SmtSolver::new();
    s.enable_bitblast_cost_tracking();
    // Use the same add twice in different consumers. Hash-consing means
    // the second use returns the cached lits — no new clauses. The cost
    // should reflect ONE bitblast, not two.
    let x = s.bv_var(32);
    let y = s.bv_var(32);
    let sum = s.bv_add(x, y);
    let zero = s.bv_const(0, 32);
    let one = s.bv_const(1, 32);
    let eq_zero = s.bv_eq(sum, zero); // uses sum
    let eq_one = s.bv_eq(sum, one);   // uses sum again — cached
    s.assert(eq_zero);
    s.assert(eq_one);
    let _ = s.solve();

    let report = s.bitblast_cost_report();
    let sum_e = report.iter().find(|e| e.term == sum).expect("sum missing");
    // 32-bit ripple-carry: ~32 sum gates + ~32 carry-out gates ≈ 64
    // gate-output SAT vars per add. Definitively bounded well under 2×.
    assert!(
        sum_e.sat_vars < 200,
        "single 32-bit add should have <200 vars (got {}); \
         hash-cons might be double-counting",
        sum_e.sat_vars,
    );
}
