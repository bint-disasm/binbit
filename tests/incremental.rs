use binbit::{LBool, Lit, SolveResult, Solver, Var};

fn lit(signed: i32) -> Lit {
    let v = Var((signed.unsigned_abs() - 1) as u32);
    Lit::new(v, signed < 0)
}

fn make_solver(nvars: u32) -> Solver {
    let mut s = Solver::new();
    for _ in 0..nvars {
        s.new_var();
    }
    s
}

#[test]
fn assumption_satisfied_by_formula() {
    // (x1 ∨ x2).  Assume x1=true — satisfiable.
    let mut s = make_solver(2);
    s.add_clause(vec![lit(1), lit(2)]);

    assert_eq!(s.solve_under_assumptions(&[lit(1)]), SolveResult::Sat);
    assert_eq!(s.value_of_var(Var(0)), LBool::True);
}

#[test]
fn assumption_contradicts_unit() {
    // Unit clause x1 forces x1=true. Assume x1=false → UNSAT with single-elt core.
    let mut s = make_solver(1);
    s.add_clause(vec![lit(1)]);

    assert_eq!(s.solve_under_assumptions(&[lit(-1)]), SolveResult::Unsat);
    let core = s.unsat_core();
    assert_eq!(core.len(), 1);
    assert_eq!(core[0], lit(-1)); // the assumption itself
}

#[test]
fn two_assumptions_clash_through_chain() {
    // x1 → x2, x3 → ¬x2. Assuming x1 AND x3 clashes through x2.
    // Encoded as clauses: (¬x1 ∨ x2), (¬x3 ∨ ¬x2).
    let mut s = make_solver(3);
    s.add_clause(vec![lit(-1), lit(2)]);
    s.add_clause(vec![lit(-3), lit(-2)]);

    // x1 alone: fine.
    assert_eq!(s.solve_under_assumptions(&[lit(1)]), SolveResult::Sat);
    // x3 alone: fine.
    assert_eq!(s.solve_under_assumptions(&[lit(3)]), SolveResult::Sat);
    // Both: UNSAT, and the core should name both.
    assert_eq!(
        s.solve_under_assumptions(&[lit(1), lit(3)]),
        SolveResult::Unsat
    );
    let core = s.unsat_core();
    // Core must at least mention x1 and x3.
    assert!(core.contains(&lit(1)));
    assert!(core.contains(&lit(3)));
}

#[test]
fn unsat_core_is_subset_of_assumptions() {
    // Irrelevant assumptions should NOT end up in the core.
    // Clauses: (¬x1 ∨ x2), (¬x3 ∨ ¬x2). Add an unrelated x4 / x5 clause.
    let mut s = make_solver(5);
    s.add_clause(vec![lit(-1), lit(2)]);
    s.add_clause(vec![lit(-3), lit(-2)]);
    s.add_clause(vec![lit(4), lit(5)]);

    // Assume x1, x3 (the clash), plus x4 (irrelevant).
    assert_eq!(
        s.solve_under_assumptions(&[lit(1), lit(4), lit(3)]),
        SolveResult::Unsat
    );
    let core = s.unsat_core();
    // x1 and x3 must be in the core; x4 should not.
    assert!(core.contains(&lit(1)));
    assert!(core.contains(&lit(3)));
    assert!(!core.contains(&lit(4)));
}

#[test]
fn repeated_solves_preserve_learned_clauses() {
    // The solver should cope with many back-to-back solve calls against the
    // same state. Also checks that switching assumption sets works.
    let mut s = make_solver(3);
    s.add_clause(vec![lit(1), lit(2), lit(3)]);
    s.add_clause(vec![lit(-1), lit(-2)]);
    s.add_clause(vec![lit(-1), lit(-3)]);

    // Without assumptions: SAT.
    assert_eq!(s.solve(), SolveResult::Sat);
    // Assuming x1 forces x2 and x3 to both be false, which is fine with clause 1 iff x1 is true, ok SAT.
    assert_eq!(s.solve_under_assumptions(&[lit(1)]), SolveResult::Sat);
    // Assume all three true: (¬x1 ∨ ¬x2) violates immediately → UNSAT.
    assert_eq!(
        s.solve_under_assumptions(&[lit(1), lit(2), lit(3)]),
        SolveResult::Unsat
    );
    // Back to a satisfiable assumption set.
    assert_eq!(s.solve_under_assumptions(&[lit(-1)]), SolveResult::Sat);
}

#[test]
fn add_clause_between_solves() {
    // Prove that add_clause works mid-session (incremental SMT pattern).
    let mut s = make_solver(3);
    s.add_clause(vec![lit(1), lit(2)]);

    // SAT.
    assert_eq!(s.solve(), SolveResult::Sat);

    // Add a new constraint that rules out the previous model region.
    s.add_clause(vec![lit(-1), lit(3)]);
    s.add_clause(vec![lit(-2), lit(3)]);

    // Now any model must have x3 true.
    assert_eq!(s.solve(), SolveResult::Sat);
    assert_eq!(s.value_of_var(Var(2)), LBool::True);

    // Force x3 false → UNSAT.
    assert_eq!(s.solve_under_assumptions(&[lit(-3)]), SolveResult::Unsat);
}
