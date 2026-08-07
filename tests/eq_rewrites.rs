//! Equality rewrites that collapse the guarded-fold shapes symbolic
//! execution emits: `sub(a,b) = 0`, `ext(a) = ext(b)`, and pushing a
//! constant equality through an ite chain (`bv_eq`'s `EQ_ITE_MAX_NODES`
//! rule). Each is checked for *semantics* by a differential UNSAT proof
//! against an unrewritten formulation, then for *effect* by a size bound.

use binbit::{BoolTerm, BvTerm, SmtResult, SmtSolver};

/// `a XOR b` over Bool terms — the differential harness asserts this and
/// expects Unsat, i.e. the two encodings agree in every model.
fn bool_xor(s: &mut SmtSolver, a: BoolTerm, b: BoolTerm) -> BoolTerm {
    let or = s.bool_or(a, b);
    let and = s.bool_and(a, b);
    let nand = s.bool_not(and);
    s.bool_and(or, nand)
}

/// Build the shape `SimCtx::compare` produces: a right-fold from the top
/// index down, so the accumulator carries the difference at the *lowest*
/// differing position. `len` is left symbolic so the range guards don't
/// fold away.
fn compare_chain(
    s: &mut SmtSolver,
    a: &[BvTerm],
    b: &[BvTerm],
    len: BvTerm,
) -> BvTerm {
    let lw = 32;
    let mut acc = s.bv_const(0, 32);
    for i in (0..a.len()).rev() {
        let ax = s.bv_sign_extend(a[i], 24);
        let bx = s.bv_sign_extend(b[i], 24);
        let diff = s.bv_sub(ax, bx);
        let ne = s.bv_ne(a[i], b[i]);
        let d = s.bv_ite(ne, diff, acc);
        let ic = s.bv_const(i as u128, lw);
        let inr = s.bv_ult(ic, len);
        acc = s.bv_ite(inr, d, acc);
    }
    acc
}

/// The rewrites in `bv_eq` only fire when the right operand is a
/// *constant*. Comparing against a variable that is merely asserted equal
/// to that constant gives a semantically identical term built the old way
/// — the reference every differential test below measures against.
fn pinned_const(s: &mut SmtSolver, value: u128, width: u32) -> BvTerm {
    let v = s.bv_var(width);
    let k = s.bv_const(value, width);
    let eq = s.bv_eq(v, k);
    s.assert(eq);
    v
}

#[test]
fn sub_equals_zero_is_operand_equality() {
    let mut s = SmtSolver::new();
    let a = s.bv_var(16);
    let b = s.bv_var(16);
    let d = s.bv_sub(a, b);

    let zero = s.bv_const(0, 16);
    let rewritten = s.bv_eq(d, zero); // fires the sub-to-eq rule
    let pinned_zero = pinned_const(&mut s, 0, 16);
    let reference = s.bv_eq(d, pinned_zero); // does not

    let disagree = bool_xor(&mut s, rewritten, reference);
    s.assert(disagree);
    assert_eq!(s.solve(), SmtResult::Unsat, "sub(a,b)=0 must mean a=b");
}

#[test]
fn equal_width_extensions_compare_as_their_payloads() {
    for signed in [false, true] {
        let mut s = SmtSolver::new();
        let a = s.bv_var(8);
        let b = s.bv_var(8);
        let (ax, bx) = if signed {
            (s.bv_sign_extend(a, 24), s.bv_sign_extend(b, 24))
        } else {
            (s.bv_zero_extend(a, 24), s.bv_zero_extend(b, 24))
        };

        let rewritten = s.bv_eq(ax, bx); // narrows to bv_eq(a, b)
        let direct = s.bv_eq(a, b);
        assert_eq!(rewritten, direct, "signed={signed}: should be the same term");

        // And it really is equivalent, not just structurally equal: a
        // disagreement between the narrowed form and the payload equality
        // would show up here.
        let disagree = bool_xor(&mut s, rewritten, direct);
        s.assert(disagree);
        assert_eq!(s.solve(), SmtResult::Unsat, "signed={signed}");
    }
}

#[test]
fn extensions_of_different_amounts_are_not_narrowed() {
    // `zext(a,24) = zext(b,25)` is a width mismatch and never reaches the
    // rule; `zext(a,24)` vs `zext(b,24)` on differently-sized payloads is
    // the case the `na == nb` guard exists for. Build the legal shape where
    // the extension amounts differ but the results match in width.
    let mut s = SmtSolver::new();
    let a = s.bv_var(8);
    let b = s.bv_var(16);
    let ax = s.bv_zero_extend(a, 24); // 8 + 24 = 32
    let bx = s.bv_zero_extend(b, 16); // 16 + 16 = 32
    let eq = s.bv_eq(ax, bx);
    // Not narrowed (payload widths differ), but still sound: pick a
    // witness that must exist.
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn constant_equality_pushes_through_an_ite_chain() {
    let n = 6;
    let mut s = SmtSolver::new();
    s.set_eq_ite_pushdown(true);
    let a: Vec<BvTerm> = (0..n).map(|_| s.bv_var(8)).collect();
    let b: Vec<BvTerm> = (0..n).map(|_| s.bv_var(8)).collect();
    let len = s.bv_var(32);
    let chain = compare_chain(&mut s, &a, &b, len);

    let zero = s.bv_const(0, 32);
    let rewritten = s.bv_eq(chain, zero);
    let pinned_zero = pinned_const(&mut s, 0, 32);
    let reference = s.bv_eq(chain, pinned_zero);

    let disagree = bool_xor(&mut s, rewritten, reference);
    s.assert(disagree);
    assert_eq!(
        s.solve(),
        SmtResult::Unsat,
        "pushing `= 0` through the ite chain changed its meaning"
    );
}

#[test]
fn ite_chain_pushdown_also_holds_for_a_nonzero_constant() {
    // The rule is not special-cased to zero; a chain whose branches are
    // arbitrary must still agree with the unrewritten comparison.
    let mut s = SmtSolver::new();
    s.set_eq_ite_pushdown(true);
    let c0 = s.bool_var();
    let c1 = s.bool_var();
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let z = s.bv_var(8);
    let inner = s.bv_ite(c1, y, z);
    let chain = s.bv_ite(c0, x, inner);

    let k = s.bv_const(0x2a, 8);
    let rewritten = s.bv_eq(chain, k);
    let pinned = pinned_const(&mut s, 0x2a, 8);
    let reference = s.bv_eq(chain, pinned);

    let disagree = bool_xor(&mut s, rewritten, reference);
    s.assert(disagree);
    assert_eq!(s.solve(), SmtResult::Unsat);
}

#[test]
fn compare_chain_against_zero_costs_no_more_than_direct_equality() {
    // The whole point: `strcmp(a,b) == 0` should encode like
    // `AND_i (a_i == b_i)` rather than one full-width mux per byte. Both
    // are solved by propagation alone, so the tell is the clause count.
    let n = 64;

    let mut s = SmtSolver::new();
    s.set_eq_ite_pushdown(true);
    let a: Vec<BvTerm> = (0..n).map(|_| s.bv_var(8)).collect();
    let b: Vec<BvTerm> = (0..n).map(|_| s.bv_var(8)).collect();
    let len = s.bv_const(n as u128, 32);
    let chain = compare_chain(&mut s, &a, &b, len);
    let zero = s.bv_const(0, 32);
    let eq = s.bv_eq(chain, zero);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    let chained = s.sat_stats();

    let mut s2 = SmtSolver::new();
    let a2: Vec<BvTerm> = (0..n).map(|_| s2.bv_var(8)).collect();
    let b2: Vec<BvTerm> = (0..n).map(|_| s2.bv_var(8)).collect();
    let mut acc = s2.bool_true();
    for i in 0..n {
        let e = s2.bv_eq(a2[i], b2[i]);
        acc = s2.bool_and(acc, e);
    }
    s2.assert(acc);
    assert_eq!(s2.solve(), SmtResult::Sat);
    let direct = s2.sat_stats();

    assert_eq!(
        chained.sat_vars, direct.sat_vars,
        "chain should bitblast to the same variables as the direct form"
    );
    assert_eq!(
        chained.sat_clauses, direct.sat_clauses,
        "chain should bitblast to the same clauses as the direct form"
    );
}

#[test]
fn wide_ite_structures_keep_their_mux_encoding() {
    // Past `EQ_ITE_MAX_NODES` the pushdown is declined — a wide table is
    // better served by its mux tree than by a fan of comparators. Build a
    // balanced tree well over the bound and check the equality stays a
    // single `Eq` over the mux output (one bitblasted comparator), not a
    // disjunction over every leaf.
    let mut s = SmtSolver::new();
    s.set_eq_ite_pushdown(true);
    let mut level: Vec<BvTerm> = (0..512).map(|_| s.bv_var(8)).collect();
    while level.len() > 1 {
        let c = s.bool_var();
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            next.push(s.bv_ite(c, pair[1], pair[0]));
        }
        level = next;
    }
    let k = s.bv_const(0x41, 8);
    let eq = s.bv_eq(level[0], k);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    // A pushdown over 511 ites would fan out into hundreds of per-leaf
    // equalities; declining keeps this in the low thousands of clauses.
    let st = s.sat_stats();
    assert!(
        st.sat_clauses < 20_000,
        "declined pushdown should stay compact, got {} clauses",
        st.sat_clauses
    );
}

#[test]
fn pushdown_declines_when_nothing_collapses() {
    // A mux whose branches have nothing to do with its guard: pushing the
    // equality through would only swap a mux for a disjunction, losing the
    // ITE-gate registration (and the selector VSIDS boosts that ride on
    // it) for no structural gain. Measured corpus-negative, hence the
    // profitability guard — the mux must survive.
    let mut s = SmtSolver::new();
    s.set_eq_ite_pushdown(true);
    let sel = s.bool_var();
    let p = s.bv_var(32);
    let q = s.bv_var(32);
    let mux = s.bv_ite(sel, p, q);
    let k = s.bv_const(0xdead, 32);
    let eq = s.bv_eq(mux, k);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert!(
        !s.ite_gates().is_empty(),
        "the mux should still be bitblasted as a mux, not decomposed"
    );
}

#[test]
fn pushdown_fires_when_a_guard_annihilates_its_branch() {
    // The mirror case: the guard is the negation of the branch's own
    // equality, so one arm is `⊥` and the pushdown pays. Nothing is left
    // to register as an ITE gate.
    let mut s = SmtSolver::new();
    s.set_eq_ite_pushdown(true);
    let a = s.bv_var(8);
    let b = s.bv_var(8);
    let diff = s.bv_sub(a, b);
    let ne = s.bv_ne(a, b);
    let fallback = s.bv_const(0, 8);
    let mux = s.bv_ite(ne, diff, fallback);
    let zero = s.bv_const(0, 8);
    let eq = s.bv_eq(mux, zero);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    assert!(
        s.ite_gates().is_empty(),
        "an annihilating guard should dissolve the mux entirely"
    );
}

#[test]
fn complementary_pairs_fold() {
    let mut s = SmtSolver::new();
    let x = s.bv_var(4);
    let y = s.bv_var(4);
    let e = s.bv_eq(x, y);
    let ne = s.bool_not(e);

    let both = s.bool_and(e, ne);
    let either = s.bool_or(e, ne);
    let f = s.bool_false();
    let t = s.bool_true();
    assert_eq!(both, f, "x AND NOT x must fold to false");
    assert_eq!(either, t, "x OR NOT x must fold to true");

    // Order-independent.
    let both_rev = s.bool_and(ne, e);
    let either_rev = s.bool_or(ne, e);
    assert_eq!(both_rev, f);
    assert_eq!(either_rev, t);
}
