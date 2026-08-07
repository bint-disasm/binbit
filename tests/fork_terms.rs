//! `SmtSolver::fork_terms` — rebuilding a term DAG into a fresh solver so
//! a terminal solve gets one-batch CNF preprocessing.
//!
//! Two properties matter. Answer-equivalence: whatever the source solver
//! decides about a constraint set, the fork must decide the same, and any
//! model it produces must satisfy the original constraints. And
//! configuration fidelity: a fork that quietly reverted a tuning switch
//! would make every comparison against its source measure the missing
//! switch instead of the fork.

use binbit::{BoolTerm, BvTerm, SmtResult, SmtSolver, SolverConfig};

/// Assert `roots` into a fresh fork of `src` and solve it there. Returns
/// the fork's solver, the translated `bv_roots`, and the verdict.
fn solve_forked(
    src: &SmtSolver,
    roots: &[BoolTerm],
    bv_roots: &[BvTerm],
) -> (SmtSolver, Vec<BvTerm>, SmtResult) {
    let mut fork = src.fork_terms(roots, bv_roots);
    let mapped_bv: Vec<BvTerm> = bv_roots.iter().map(|&r| fork.bv(r).unwrap()).collect();
    assert_eq!(fork.assert_translated(roots), roots.len());
    let res = fork.solver().solve();
    (fork.into_solver(), mapped_bv, res)
}

#[test]
fn fork_agrees_on_sat_and_the_model_checks_out() {
    let mut src = SmtSolver::new();
    let x = src.bv_var(16);
    let y = src.bv_var(16);
    let sum = src.bv_add(x, y);
    let k = src.bv_const(1000, 16);
    let c1 = src.bv_eq(sum, k);
    let lo = src.bv_const(100, 16);
    let c2 = src.bv_ult(lo, x);

    let (mut f, vals, res) = solve_forked(&src, &[c1, c2], &[x, y]);
    assert_eq!(res, SmtResult::Sat);
    let xv = f.get_bv_value_u128(vals[0]) as u64;
    let yv = f.get_bv_value_u128(vals[1]) as u64;
    assert_eq!((xv.wrapping_add(yv)) & 0xffff, 1000, "x + y must be 1000");
    assert!(xv > 100, "x must exceed 100, got {xv}");
}

#[test]
fn fork_agrees_on_unsat() {
    let mut src = SmtSolver::new();
    let x = src.bv_var(8);
    let a = src.bv_const(10, 8);
    let b = src.bv_const(5, 8);
    let c1 = src.bv_ult(a, x); // x > 10
    let c2 = src.bv_ult(x, b); // x < 5

    let (_, _, res) = solve_forked(&src, &[c1, c2], &[x]);
    assert_eq!(res, SmtResult::Unsat);
}

#[test]
fn fork_covers_every_operator_family() {
    // One term per BvOp/BoolOp family that a real formula can carry, so a
    // newly added variant that `fork_build_*` forgets shows up as a panic
    // here rather than as a wrong answer in a client.
    let mut src = SmtSolver::new();
    let x = src.bv_var(16);
    let y = src.bv_var(16);
    let b = src.bool_var();

    let mut acc: Vec<BvTerm> = Vec::new();
    acc.push(src.bv_not(x));
    acc.push(src.bv_and(x, y));
    acc.push(src.bv_or(x, y));
    acc.push(src.bv_xor(x, y));
    acc.push(src.bv_add(x, y));
    acc.push(src.bv_sub(x, y));
    acc.push(src.bv_neg(x));
    acc.push(src.bv_mul(x, y));
    acc.push(src.bv_udiv(x, y));
    acc.push(src.bv_urem(x, y));
    acc.push(src.bv_sdiv(x, y));
    acc.push(src.bv_srem(x, y));
    acc.push(src.bv_smod(x, y));
    acc.push(src.bv_popcount(x));
    acc.push(src.bv_clz(x));
    acc.push(src.bv_ctz(x));
    acc.push(src.bv_rotate_left_dyn(x, y));
    acc.push(src.bv_rotate_right_dyn(x, y));
    acc.push(src.bv_shl(x, y));
    acc.push(src.bv_lshr(x, y));
    acc.push(src.bv_ashr(x, y));
    acc.push(src.bv_ite(b, x, y));
    let narrow = src.bv_extract(x, 7, 0);
    acc.push(src.bv_zero_extend(narrow, 8));
    acc.push(src.bv_sign_extend(narrow, 8));
    let hi = src.bv_extract(x, 15, 8);
    acc.push(src.bv_concat(hi, narrow));
    let s0 = src.bool_var();
    let s1 = src.bool_var();
    acc.push(src.bv_select(&[s0, s1], &[x, y], x));

    // Fold the lot into one constraint so everything is reachable.
    let mut folded = acc[0];
    for &t in &acc[1..] {
        folded = src.bv_xor(folded, t);
    }
    let zero = src.bv_const(0, 16);
    let mut preds: Vec<BoolTerm> = Vec::new();
    preds.push(src.bv_ult(zero, folded));
    preds.push(src.bv_ule(x, y));
    preds.push(src.bv_slt(x, y));
    preds.push(src.bv_sle(x, y));
    preds.push(src.bv_uadd_overflow(x, y));
    preds.push(src.bv_sadd_overflow(x, y));
    preds.push(src.bv_usub_overflow(x, y));
    preds.push(src.bv_ssub_overflow(x, y));
    preds.push(src.bv_umul_overflow(x, y));
    preds.push(src.bv_smul_overflow(x, y));
    preds.push(src.bv_neg_overflow(x));
    preds.push(src.bv_sdiv_overflow(x, y));
    let mut disj = preds[0];
    for &p in &preds[1..] {
        disj = src.bool_or(disj, p);
    }
    let t = src.bool_true();
    let f = src.bool_false();
    let imp = src.bool_implies(disj, t);
    let n = src.bool_not(f);
    let root = src.bool_and(imp, n);

    let (_, _, res) = solve_forked(&src, &[root], &[x, y]);
    assert_eq!(res, SmtResult::Sat, "fork of an all-operators formula");
}

#[test]
fn fork_carries_wide_constants() {
    let mut src = SmtSolver::new();
    let x = src.bv_var(192);
    let big = src.bv_const_wide(&[0xdead_beef_cafe_babe, 0x0123_4567_89ab_cdef, 7], 192);
    let eq = src.bv_eq(x, big);

    let (mut f, vals, res) = solve_forked(&src, &[eq], &[x]);
    assert_eq!(res, SmtResult::Sat);
    let limbs = f.get_bv_value_limbs(vals[0]);
    assert_eq!(&limbs[..3], &[0xdead_beef_cafe_babe, 0x0123_4567_89ab_cdef, 7]);
}

#[test]
fn unreachable_terms_do_not_translate() {
    let mut src = SmtSolver::new();
    let x = src.bv_var(8);
    let stranger = src.bv_var(8);
    let k = src.bv_const(3, 8);
    let c = src.bv_eq(x, k);

    let fork = src.fork_terms(&[c], &[]);
    assert!(fork.bv(x).is_some(), "x is reachable from the constraint");
    assert!(
        fork.bv(stranger).is_none(),
        "an unconstrained variable is not reachable and must not translate"
    );

    // ...unless it is named as a bv root, which is exactly what a caller
    // that wants its model value has to do.
    let fork2 = src.fork_terms(&[c], &[stranger]);
    assert!(fork2.bv(stranger).is_some());
}

#[test]
fn fork_restores_preprocessing_that_incremental_growth_forfeits() {
    // The motivating case. Grow a formula the way a symbolic executor does
    // — a probe solve after every constraint — and CNF preprocessing has
    // nothing left it is allowed to touch, because bounded variable
    // elimination may only eliminate gates allocated in the current batch.
    let mut src = SmtSolver::new();
    let inputs: Vec<BvTerm> = (0..6).map(|_| src.bv_var(16)).collect();
    let mut acc = src.bv_const(0x9e37, 16);
    let mut constraints: Vec<BoolTerm> = Vec::new();
    for &i in &inputs {
        acc = src.bv_add(acc, i);
        let k = src.bv_const(0x85eb, 16);
        acc = src.bv_mul(acc, k);
        let bound = src.bv_const(0x4000, 16);
        constraints.push(src.bv_ult(acc, bound));
    }
    for n in 1..=constraints.len() {
        // Probe as we go — this is what forfeits preprocessing.
        let _ = src.solve_under_assumptions(&constraints[..n]);
    }
    for &c in &constraints {
        src.assert(c);
    }
    let incremental = src.solve();
    let incremental_pp = src.sat_stats().pp_eliminated;

    let (f, _, forked) = solve_forked(&src, &constraints, &inputs);
    assert_eq!(incremental, forked, "fork must reach the same verdict");
    let forked_pp = f.sat_stats().pp_eliminated;

    assert_eq!(
        incremental_pp, 0,
        "an incrementally grown session should preprocess nothing — if this \
         ever becomes nonzero, binbit learned to unfreeze and fork_terms may \
         no longer be needed"
    );
    assert!(
        forked_pp > 0,
        "the fork should preprocess the whole formula in one batch, got {forked_pp}"
    );
}

/// Flip every tuning switch away from its default. Any setter that forgets
/// to record itself in `SolverConfig` leaves that field at the default,
/// which `config_records_every_setter` then catches.
fn flip_everything(s: &mut SmtSolver) {
    let d = SolverConfig::default();
    s.set_normalization(!d.normalization);
    s.set_substitution(!d.substitution);
    s.set_gaussian(!d.gaussian);
    s.set_bve(!d.bve);
    s.set_eq_ite_pushdown(!d.eq_ite_pushdown);
    s.set_core_tracking(!d.core_tracking);
    s.set_cone_retirement(!d.cone_retirement);
    s.set_cnf_mapping(!d.cnf_mapping);
    s.set_cnf_mapping_effort(!d.cnf_mapping_full);
    s.set_ite_branching_hints(!d.ite_branching_hints);
    s.set_aig_two_level(!d.aig_two_level);
    s.set_aig_two_level_subst(!d.aig_two_level_subst);
    s.set_fraig(!d.fraig);
    s.set_ve_gate_substitution(Some(true));
    s.set_input_branching(2);
    s.set_phase_seed(0xabcd);
    s.set_aig_rewrite(!d.aig_rewrite);
    s.set_vivification(!d.vivification);
    s.set_target_phases(!d.target_phases);
    s.set_xor_reasoning(!d.xor_reasoning);
    s.set_pcaug_lazy(!d.pcaug_lazy); // implies set_pcaug(true)
    s.set_pcaug_capacity(1234);
    s.set_pcaug_interval(4321);
    s.set_xor_native(!d.xor_native);
    s.set_xor_emit_len(3);
    s.set_xor_native_min(d.xor_native_min + 1);
    s.set_pcaug_budget(d.pcaug_budget + 1);
    s.set_aig_subst_share_limit(1);
    s.set_cnf_prime_emission(!d.cnf_prime_emission);
    s.set_augmentation_recycle(!d.aug_recycle);
    s.set_augmentation_hot_fraction(d.aug_hot_frac * 2.0);
    // Set last: it forces `two_level` on and `two_level_subst` off, and the
    // replay in `apply_config` has to reproduce that ordering.
    s.set_aig_two_level_post(!d.aig_two_level_post);
}

#[test]
fn config_records_every_setter() {
    let mut s = SmtSolver::new();
    assert_eq!(s.config(), SolverConfig::default(), "a fresh solver is default");
    flip_everything(&mut s);
    let c = s.config();
    let d = SolverConfig::default();

    // Every field must have moved. A setter that failed to record would
    // leave its field at the default and trip exactly one of these.
    assert_ne!(c.normalization, d.normalization);
    assert_ne!(c.substitution, d.substitution);
    assert_ne!(c.gaussian, d.gaussian);
    assert_ne!(c.bve, d.bve);
    assert_ne!(c.eq_ite_pushdown, d.eq_ite_pushdown);
    assert_ne!(c.core_tracking, d.core_tracking);
    assert_ne!(c.cone_retirement, d.cone_retirement);
    assert_ne!(c.cnf_mapping, d.cnf_mapping);
    assert_ne!(c.cnf_mapping_full, d.cnf_mapping_full);
    assert_ne!(c.ite_branching_hints, d.ite_branching_hints);
    assert_ne!(c.aig_two_level, d.aig_two_level);
    assert_ne!(c.aig_two_level_post, d.aig_two_level_post);
    assert_ne!(c.fraig, d.fraig);
    assert_ne!(c.ve_gate_subst, d.ve_gate_subst);
    assert_ne!(c.input_branching, d.input_branching);
    assert_ne!(c.phase_seed, d.phase_seed);
    assert_ne!(c.aig_rewrite, d.aig_rewrite);
    assert_ne!(c.vivification, d.vivification);
    assert_ne!(c.target_phases, d.target_phases);
    assert_ne!(c.xor_reasoning, d.xor_reasoning);
    assert_ne!(c.pcaug, d.pcaug);
    assert_ne!(c.pcaug_lazy, d.pcaug_lazy);
    assert_ne!(c.pcaug_capacity, d.pcaug_capacity);
    assert_ne!(c.pcaug_interval, d.pcaug_interval);
    assert_ne!(c.xor_native, d.xor_native);
    assert_ne!(c.xor_emit_len, d.xor_emit_len);
    assert_ne!(c.xor_native_min, d.xor_native_min);
    assert_ne!(c.pcaug_budget, d.pcaug_budget);
    assert_ne!(c.aig_subst_share_limit, d.aig_subst_share_limit);
    assert_ne!(c.cnf_prime_emission, d.cnf_prime_emission);
    assert_ne!(c.aug_recycle, d.aug_recycle);
    assert_ne!(c.aug_hot_frac, d.aug_hot_frac);
}

#[test]
fn fork_inherits_the_sources_configuration() {
    // The bug this guards against: a fork on library defaults silently
    // drops whatever the caller tuned, and any A/B against the source then
    // measures the missing switches rather than the fork.
    let mut src = SmtSolver::new();
    flip_everything(&mut src);
    let x = src.bv_var(8);
    let k = src.bv_const(7, 8);
    let c = src.bv_eq(x, k);

    let mut fork = src.fork_terms(&[c], &[x]);
    assert_eq!(
        fork.solver().config(),
        src.config(),
        "fork must arrive configured exactly like its source"
    );

    // And a deliberate difference stays a difference.
    fork.solver().set_bve(true);
    assert!(fork.solver().config().bve);
    assert_ne!(fork.solver().config(), src.config());
}
