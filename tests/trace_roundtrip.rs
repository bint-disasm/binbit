//! Round-trip tests for the SMT-LIB 2 emitter: build a formula via the
//! `SmtSolver` API, solve it to record the ground-truth result, then dump to
//! SMT-LIB and re-solve the parsed script — the result must match.

use binbit::{SmtResult, SmtSolver, dump_smtlib, run_script};

fn solve_direct(build: impl FnOnce(&mut SmtSolver) -> Vec<binbit::BoolTerm>) -> (SmtResult, String) {
    let mut s = SmtSolver::new();
    let asserts = build(&mut s);
    for &a in &asserts {
        s.assert(a);
    }
    let r = s.solve();
    let dump = dump_smtlib(&s, &asserts);
    (r, dump)
}

fn replay(script: &str) -> SmtResult {
    let out = run_script(script).expect("parse/run");
    if out.contains("unsat") {
        SmtResult::Unsat
    } else if out.contains("sat") {
        SmtResult::Sat
    } else {
        panic!("unexpected run_script output: {:?}", out);
    }
}

fn assert_roundtrip(expected: SmtResult, script: &str) {
    assert_eq!(replay(script), expected, "script:\n{}", script);
}

#[test]
fn roundtrip_simple_bv_eq_sat() {
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(8);
        let k = s.bv_const(42, 8);
        let eq = s.bv_eq(x, k);
        vec![eq]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_unsat_contradiction() {
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(8);
        let a = s.bv_const(1, 8);
        let b = s.bv_const(2, 8);
        let eq1 = s.bv_eq(x, a);
        let eq2 = s.bv_eq(x, b);
        vec![eq1, eq2]
    });
    assert_eq!(r, SmtResult::Unsat);
    assert_roundtrip(SmtResult::Unsat, &dump);
}

#[test]
fn roundtrip_mixed_ops() {
    // Exercise most BvOp variants in a single formula.
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(16);
        let y = s.bv_var(16);
        let z = s.bv_var(16);
        let a = s.bv_add(x, y);
        let m = s.bv_mul(a, z);
        let two = s.bv_const(2, 16);
        let ax = s.bv_ashr(m, two);
        let sx = s.bv_sub(ax, y);
        let hi = s.bv_extract(sx, 15, 8);
        let lo = s.bv_extract(sx, 7, 0);
        let back = s.bv_concat(hi, lo);
        let eq = s.bv_eq(back, sx);
        vec![eq] // tautology-shaped but symbolic → sat
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_bool_combinators() {
    let (r, dump) = solve_direct(|s| {
        let p = s.bool_var();
        let q = s.bool_var();
        let r = s.bool_var();
        let a = s.bool_and(p, q);
        let b = s.bool_or(a, r);
        let imp = s.bool_implies(p, b);
        vec![imp]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_comparisons_all_kinds() {
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(8);
        let y = s.bv_var(8);
        let lt_u = s.bv_ult(x, y);
        let le_u = s.bv_ule(x, y);
        let lt_s = s.bv_slt(x, y);
        let le_s = s.bv_sle(x, y);
        let all = {
            let a = s.bool_and(lt_u, le_u);
            let b = s.bool_and(lt_s, le_s);
            s.bool_and(a, b)
        };
        vec![all]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_ite_and_extend() {
    let (r, dump) = solve_direct(|s| {
        let c = s.bool_var();
        let x = s.bv_var(8);
        let y = s.bv_var(8);
        let ite = s.bv_ite(c, x, y);
        let zx = s.bv_zero_extend(ite, 8); // now 16-bit
        let sx = s.bv_sign_extend(y, 8);
        let eq = s.bv_eq(zx, sx);
        vec![eq]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_overflow_predicates() {
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(8);
        let y = s.bv_var(8);
        let uadd_ov = s.bv_uadd_overflow(x, y);
        let sadd_ov = s.bv_sadd_overflow(x, y);
        let usub_ov = s.bv_usub_overflow(x, y);
        let umul_ov = s.bv_umul_overflow(x, y);
        let any = {
            let a = s.bool_or(uadd_ov, sadd_ov);
            let b = s.bool_or(usub_ov, umul_ov);
            s.bool_or(a, b)
        };
        vec![any]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_division_ops() {
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(8);
        let y = s.bv_var(8);
        let ud = s.bv_udiv(x, y);
        let ur = s.bv_urem(x, y);
        let sd = s.bv_sdiv(x, y);
        let sr = s.bv_srem(x, y);
        let sm = s.bv_smod(x, y);
        // Combine into a formula that exercises all four division ops.
        let a1 = s.bv_add(ud, ur);
        let a2 = s.bv_add(sd, sr);
        let a3 = s.bv_add(a1, sm);
        let eq = s.bv_eq(a2, a3);
        vec![eq]
    });
    // Result may be sat or unsat depending on solver; we just want the
    // roundtrip to agree.
    assert_roundtrip(r, &dump);
}

#[test]
fn roundtrip_select() {
    // Build a 4-way `bv_select`: x0 if s0, else x1 if s1, else default.
    let (r, dump) = solve_direct(|s| {
        let s0 = s.bool_var();
        let s1 = s.bool_var();
        let s2 = s.bool_var();
        let v0 = s.bv_var(8);
        let v1 = s.bv_var(8);
        let v2 = s.bv_var(8);
        let def = s.bv_var(8);
        let sel = s.bv_select(&[s0, s1, s2], &[v0, v1, v2], def);
        let probe = s.bv_const(42, 8);
        let eq = s.bv_eq(sel, probe);
        vec![eq]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_wide_constant() {
    // 256-bit constant exercises the limb-based wide-const path.
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(256);
        let limbs: [u64; 4] = [0x1122334455667788, 0xaabbccddeeff0011, 1, 2];
        let k = s.bv_const_wide(&limbs, 256);
        let eq = s.bv_eq(x, k);
        vec![eq]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
}

#[test]
fn roundtrip_shared_subterm_dag() {
    // One subterm referenced many times — the emitter must deduplicate via
    // define-fun rather than inline-expand exponentially.
    let (r, dump) = solve_direct(|s| {
        let x = s.bv_var(32);
        let y = s.bv_var(32);
        let shared = s.bv_add(x, y);
        // Fan-out of 8 uses of `shared`.
        let a = s.bv_add(shared, shared);
        let b = s.bv_add(a, shared);
        let c = s.bv_add(b, shared);
        let d = s.bv_add(c, shared);
        let e = s.bv_mul(d, shared);
        let eq = s.bv_eq(e, shared);
        vec![eq]
    });
    assert_eq!(r, SmtResult::Sat);
    assert_roundtrip(SmtResult::Sat, &dump);
    // Output size should be linear in DAG size — sanity-check by capping at
    // something much smaller than the 8× exponential-inlining lower bound.
    assert!(
        dump.len() < 4096,
        "dump too large ({} bytes); shared subterm was inlined?",
        dump.len()
    );
}
