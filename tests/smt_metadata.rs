//! Tests for the bitblaster metadata layer: per-SAT-var origin tagging and
//! the ITE gate registry.

use binbit::{BvTerm, GateKind, SmtResult, SmtSolver, VarOrigin, Var};

/// Helper: drive the solver to SAT so all the lazy bitblasting happens and
/// we can then inspect the recorded metadata.
fn solve_sat(s: &mut SmtSolver) {
    assert_eq!(s.solve(), SmtResult::Sat);
}

// ---------- VarOrigin: input bits ----------

#[test]
fn input_bv_var_bits_are_tagged() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    // Force bitblasting by asserting something that references x.
    let c = s.bv_const(0, 8);
    let eq = s.bv_eq(x, c);
    s.assert(eq);
    solve_sat(&mut s);

    // Walk every SAT var and collect origins that are BvBit on `x`.
    let mut seen_bits: Vec<u32> = Vec::new();
    for i in 0..s.num_sat_vars() {
        if let VarOrigin::BvBit { term, bit } = s.var_origin(Var(i as u32)) {
            if term == x {
                seen_bits.push(bit);
            }
        }
    }
    seen_bits.sort();
    assert_eq!(seen_bits, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn bool_var_is_tagged() {
    let mut s = SmtSolver::new();
    let p = s.bool_var();
    s.assert(p);
    solve_sat(&mut s);

    let mut found_p = false;
    for i in 0..s.num_sat_vars() {
        if let VarOrigin::Bool { term } = s.var_origin(Var(i as u32)) {
            if term == p {
                found_p = true;
                break;
            }
        }
    }
    assert!(found_p, "expected a Bool-tagged SAT var for p");
}

#[test]
fn true_lit_is_tagged_exactly_once() {
    let mut s = SmtSolver::new();
    // Any constant use forces the true-lit to materialize.
    let c = s.bv_const(1, 4);
    let one = s.bv_const(1, 4);
    let eq = s.bv_eq(c, one);
    s.assert(eq);
    solve_sat(&mut s);

    let count_true = (0..s.num_sat_vars())
        .filter(|&i| matches!(s.var_origin(Var(i as u32)), VarOrigin::TrueLit))
        .count();
    assert_eq!(count_true, 1, "exactly one TrueLit should exist");
}

#[test]
fn gate_outputs_are_tagged_with_kind() {
    // x & y over 1-bit BVs → a single mk_and call → one And-tagged aux var.
    let mut s = SmtSolver::new();
    let x = s.bv_var(1);
    let y = s.bv_var(1);
    let and = s.bv_and(x, y);
    let zero = s.bv_const(0, 1);
    let eq = s.bv_eq(and, zero);
    s.assert(eq);
    solve_sat(&mut s);

    // Count AND gate outputs.
    let and_count = (0..s.num_sat_vars())
        .filter(|&i| {
            matches!(
                s.var_origin(Var(i as u32)),
                VarOrigin::GateOut { gate: GateKind::And, .. }
            )
        })
        .count();
    assert!(and_count >= 1, "expected at least one And-tagged gate output");
}

#[test]
fn gate_outputs_inherit_enclosing_term() {
    // Build a 4-bit XOR expression. The gate outputs should be tagged with
    // the XOR term as their enclosing BV term.
    let mut s = SmtSolver::new();
    let x = s.bv_var(4);
    let y = s.bv_var(4);
    let xor_term: BvTerm = s.bv_xor(x, y);
    let zero = s.bv_const(0, 4);
    let eq = s.bv_eq(xor_term, zero);
    s.assert(eq);
    solve_sat(&mut s);

    let xor_gates_tied_to_term = (0..s.num_sat_vars())
        .filter(|&i| match s.var_origin(Var(i as u32)) {
            VarOrigin::GateOut {
                gate: GateKind::Xor,
                term: Some(t),
            } => t == xor_term,
            _ => false,
        })
        .count();
    assert_eq!(
        xor_gates_tied_to_term, 4,
        "expected 4 xor gates (one per bit) tagged with the xor term"
    );
}

#[test]
fn activation_lits_are_tagged() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(8);
    let zero = s.bv_const(0, 8);
    let eq = s.bv_eq(x, zero);
    s.assert_named("main", eq);
    s.push();
    s.pop();

    // At least 2 activation literals exist — one for the named assertion's
    // control and one for the push scope we just opened + closed.
    let act_count = (0..s.num_sat_vars())
        .filter(|&i| matches!(s.var_origin(Var(i as u32)), VarOrigin::Activation))
        .count();
    assert!(
        act_count >= 2,
        "expected >= 2 activation lits, got {}",
        act_count
    );
}

// ---------- ITE registry ----------

#[test]
fn ite_gate_is_registered() {
    let mut s = SmtSolver::new();
    let c = s.bool_var();
    // Use symbolic branches so the ite isn't simplified away by the
    // "const vs ite-with-const-branches" eq rewrite.
    let a = s.bv_var(8);
    let b = s.bv_var(8);
    let ite = s.bv_ite(c, a, b);
    // Force bitblasting by asserting equality with another symbolic term.
    let probe = s.bv_var(8);
    let eq = s.bv_eq(ite, probe);
    s.assert(eq);
    solve_sat(&mut s);

    let gates = s.ite_gates();
    assert!(!gates.is_empty(), "expected some ITE gates to be registered");

    // Every gate should name our ITE as its source.
    for g in gates {
        assert_eq!(g.source_term, Some(ite));
    }
}

#[test]
fn ite_output_lookup_roundtrip() {
    let mut s = SmtSolver::new();
    let c = s.bool_var();
    let a = s.bv_var(4);
    let b = s.bv_var(4);
    let ite = s.bv_ite(c, a, b);
    let zero = s.bv_const(0, 4);
    let eq = s.bv_eq(ite, zero);
    s.assert(eq);
    solve_sat(&mut s);

    // Every registered gate's `o` literal should round-trip back through
    // ite_gate_for_output.
    let gates: Vec<_> = s.ite_gates().to_vec();
    for g in &gates {
        let found = s.ite_gate_for_output(g.o).expect("output lookup");
        assert_eq!(found.sel, g.sel);
        assert_eq!(found.t, g.t);
        assert_eq!(found.e, g.e);
    }
}

#[test]
fn symex_style_ite_chain_shares_sel_grouping() {
    // Build a small "symbolic read" — nested ITE over an address variable
    // selecting one of four values. This is the shape real symex produces.
    let mut s = SmtSolver::new();
    let addr = s.bv_var(2); // 2-bit address selector
    let v0 = s.bv_var(8);
    let v1 = s.bv_var(8);
    let v2 = s.bv_var(8);
    let v3 = s.bv_var(8);

    // Equalities against concrete addresses become our ITE conditions.
    let c0 = s.bv_const(0, 2);
    let c1 = s.bv_const(1, 2);
    let c2 = s.bv_const(2, 2);
    let is0 = s.bv_eq(addr, c0);
    let is1 = s.bv_eq(addr, c1);
    let is2 = s.bv_eq(addr, c2);

    // read = is0 ? v0 : (is1 ? v1 : (is2 ? v2 : v3))
    let inner2 = s.bv_ite(is2, v2, v3);
    let inner1 = s.bv_ite(is1, v1, inner2);
    let read = s.bv_ite(is0, v0, inner1);

    // Assert something so bitblasting runs.
    let expected = s.bv_const(42, 8);
    let eq = s.bv_eq(read, expected);
    s.assert(eq);
    solve_sat(&mut s);

    // We expect ITE gates to be grouped by `source_term` — three source
    // terms, one per ITE level. Verify the set has the right size.
    use std::collections::HashSet;
    let sources: HashSet<BvTerm> = s
        .ite_gates()
        .iter()
        .filter_map(|g| g.source_term)
        .collect();
    assert!(
        sources.len() >= 3,
        "expected >= 3 distinct source terms for nested ITE, got {}",
        sources.len()
    );
}
