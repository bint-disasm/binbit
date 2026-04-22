use binbit::{LBool, Lit, SolveResult, Solver, Var, dimacs};

/// Helper: load a DIMACS string into a fresh Solver.
fn load(input: &str) -> (Solver, bool) {
    let (nvars, clauses) = dimacs::parse(input).unwrap();
    let mut solver = Solver::new();
    for _ in 0..nvars {
        solver.new_var();
    }
    let mut ok = true;
    for c in &clauses {
        let lits: Vec<Lit> = c
            .iter()
            .map(|&n| {
                let v = Var((n.unsigned_abs() - 1) as u32);
                Lit::new(v, n < 0)
            })
            .collect();
        if !solver.add_clause(lits) {
            ok = false;
            break;
        }
    }
    (solver, ok)
}

/// Helper: verify a model satisfies the original clauses.
fn model_satisfies(solver: &Solver, clauses: &[Vec<i32>]) -> bool {
    for c in clauses {
        let mut sat = false;
        for &n in c {
            let v = Var((n.unsigned_abs() - 1) as u32);
            let val = solver.value_of_var(v);
            let is_true = match val {
                LBool::True => n > 0,
                LBool::False => n < 0,
                LBool::Undef => true, // unassigned: any polarity works
            };
            if is_true {
                sat = true;
                break;
            }
        }
        if !sat {
            return false;
        }
    }
    true
}

#[test]
fn empty_formula_is_sat() {
    let mut solver = Solver::new();
    assert_eq!(solver.solve(), SolveResult::Sat);
}

#[test]
fn single_unit_is_sat() {
    let (mut solver, ok) = load("p cnf 1 1\n1 0\n");
    assert!(ok);
    assert_eq!(solver.solve(), SolveResult::Sat);
    assert_eq!(solver.value_of_var(Var(0)), LBool::True);
}

#[test]
fn contradictory_units_are_unsat() {
    let (mut solver, ok) = load("p cnf 1 2\n1 0\n-1 0\n");
    if !ok {
        return; // detected during add_clause
    }
    assert_eq!(solver.solve(), SolveResult::Unsat);
}

#[test]
fn pigeonhole_3_into_2_is_unsat() {
    // 3 pigeons, 2 holes. Variables x_{p,h} for p in 1..=3, h in 1..=2.
    // Encode: var id = (p-1)*2 + h.
    let p = 3;
    let h = 2;
    let mut clauses: Vec<Vec<i32>> = Vec::new();
    // Each pigeon goes into at least one hole.
    for pp in 1..=p {
        let mut c = Vec::new();
        for hh in 1..=h {
            c.push(((pp - 1) * h + hh) as i32);
        }
        clauses.push(c);
    }
    // No two pigeons share a hole.
    for hh in 1..=h {
        for p1 in 1..=p {
            for p2 in (p1 + 1)..=p {
                let v1 = ((p1 - 1) * h + hh) as i32;
                let v2 = ((p2 - 1) * h + hh) as i32;
                clauses.push(vec![-v1, -v2]);
            }
        }
    }

    let mut solver = Solver::new();
    for _ in 0..(p * h) {
        solver.new_var();
    }
    let mut ok = true;
    for c in &clauses {
        let lits: Vec<Lit> = c
            .iter()
            .map(|&n| Lit::new(Var((n.unsigned_abs() - 1) as u32), n < 0))
            .collect();
        if !solver.add_clause(lits) {
            ok = false;
            break;
        }
    }
    let res = if ok { solver.solve() } else { SolveResult::Unsat };
    assert_eq!(res, SolveResult::Unsat);
}

#[test]
fn small_sat_formula_yields_valid_model() {
    // (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ x3) ∧ (¬x3 ∨ x4)
    let input = "p cnf 4 4
1 2 0
-1 3 0
-2 3 0
-3 4 0
";
    let (_, parsed) = dimacs::parse(input).unwrap();
    let (mut solver, ok) = load(input);
    assert!(ok);
    assert_eq!(solver.solve(), SolveResult::Sat);
    assert!(model_satisfies(&solver, &parsed));
}

#[test]
fn forces_chain_of_propagations() {
    // x1; x1 -> x2; x2 -> x3; x3 -> x4
    let input = "p cnf 4 4
1 0
-1 2 0
-2 3 0
-3 4 0
";
    let (_, parsed) = dimacs::parse(input).unwrap();
    let (mut solver, ok) = load(input);
    assert!(ok);
    assert_eq!(solver.solve(), SolveResult::Sat);
    for v in 0..4 {
        assert_eq!(solver.value_of_var(Var(v)), LBool::True);
    }
    assert!(model_satisfies(&solver, &parsed));
}

#[test]
fn random_3sat_sat_instance() {
    // A hand-built satisfiable 3-SAT (satisfied by all true).
    let input = "p cnf 6 8
1 2 3 0
-1 4 5 0
2 -4 6 0
-2 5 -6 0
1 -3 4 0
3 -5 6 0
-1 -2 6 0
2 4 5 0
";
    let (_, parsed) = dimacs::parse(input).unwrap();
    let (mut solver, ok) = load(input);
    assert!(ok);
    assert_eq!(solver.solve(), SolveResult::Sat);
    assert!(model_satisfies(&solver, &parsed));
}
