//! Propagation augmentation (`set_pcaug` / `set_pcaug_lazy`) adds only
//! IMPLIED clauses, so it must be answer-equivalent to the plain
//! pipeline on every query type — eagerly or on demand — and any model
//! it returns must still satisfy the assertion.

use binbit::{BoolTerm, SmtResult, SmtSolver};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, bound: u32) -> u32 {
        (self.next() % bound as u64) as u32
    }
}

/// Arithmetic-heavy random term: adders and muxes are exactly the
/// shapes whose gate encoding has propagation holes, so this is where
/// the augmenter actually fires.
fn random_query(s: &mut SmtSolver, rng: &mut Rng) -> BoolTerm {
    let x = s.bv_var(16);
    let y = s.bv_var(16);
    let mut t = x;
    for _ in 0..(2 + rng.below(4)) {
        let c = s.bv_const(rng.next() as u128 & 0xFFFF, 16);
        t = match rng.below(6) {
            0 => s.bv_add(t, y),
            1 => s.bv_mul(t, c),
            2 => s.bv_xor(t, y),
            3 => s.bv_and(t, c),
            4 => s.bv_sub(c, t),
            _ => {
                let cond = s.bv_ult(t, c);
                s.bv_ite(cond, y, t)
            }
        };
    }
    let k = s.bv_const(rng.next() as u128 & 0xFFFF, 16);
    match rng.below(3) {
        0 => s.bv_ult(t, k),
        1 => s.bv_eq(t, k),
        _ => {
            let a = s.bv_ult(k, t);
            let b = s.bv_eq(y, k);
            s.bool_and(a, b)
        }
    }
}

/// Modes: (eager augmentation, on-demand augmentation, two-level AIG
/// rewriting). aig2 is included because it is the pairing the bint
/// workload runs, and it changes the circuit pcaug derives from — the
/// derived gate graph, the surviving BVE gate variables, and therefore
/// every clause the pass emits.
const MODES: [(bool, bool, bool); 4] = [
    (true, false, false),
    (false, true, false),
    (false, true, true),
    (true, false, true),
];

fn configure(s: &mut SmtSolver, m: (bool, bool, bool)) {
    let (eager, lazy, aig2) = m;
    s.set_aig_two_level(aig2);
    s.set_pcaug(eager);
    s.set_pcaug_lazy(lazy);
}

#[test]
fn augmented_answers_match_plain() {
    let mut rng = Rng(0xA06_1234);
    for trial in 0..60 {
        let seed = rng.next();
        let run = |m: (bool, bool, bool)| -> (SmtResult, SmtResult, SmtResult) {
            let mut s = SmtSolver::new();
            configure(&mut s, m);
            let mut r = Rng(seed);
            let q = random_query(&mut s, &mut r);
            let assumed = s.solve_under_assumptions(&[q]);
            let nq = s.bool_not(q);
            let opposite = s.solve_under_assumptions(&[nq]);
            s.assert(q);
            let asserted = s.solve();
            (assumed, opposite, asserted)
        };
        let base = run((false, false, false));
        for m in MODES {
            assert_eq!(base, run(m), "trial {trial} diverged (mode {m:?})");
        }
    }
}

#[test]
fn augmented_incremental_matches_plain() {
    // Incremental sessions augment per flush, so later batches must not
    // be disturbed by clauses banked or added for earlier ones.
    let mut rng = Rng(0x1CE_BEEF);
    for trial in 0..25 {
        let seed = rng.next();
        let run = |m: (bool, bool, bool)| {
            let mut s = SmtSolver::new();
            configure(&mut s, m);
            let mut r = Rng(seed);
            let mut out = Vec::new();
            for _ in 0..3 {
                let q = random_query(&mut s, &mut r);
                out.push(s.solve_under_assumptions(&[q]));
                s.assert(q);
                out.push(s.solve());
            }
            out
        };
        let base = run((false, false, false));
        for m in MODES {
            assert_eq!(base, run(m), "trial {trial} diverged (mode {m:?})");
        }
    }
}

#[test]
fn augmented_models_satisfy_the_assertion() {
    for m in MODES {
        let mut s = SmtSolver::new();
        configure(&mut s, m);
        let x = s.bv_var(16);
        let y = s.bv_var(16);
        let sum = s.bv_add(x, y);
        let k = s.bv_const(500, 16);
        let eq = s.bv_eq(sum, k);
        s.assert(eq);
        assert_eq!(s.solve(), SmtResult::Sat);
        let xv = s.get_bv_value(x);
        let yv = s.get_bv_value(y);
        assert_eq!((xv + yv) & 0xFFFF, 500, "model violates x+y=500");
    }
}

/// The augmenter must actually fire on chained symbolic adders — the
/// carry composition is its canonical target — and on the lazy path
/// every derived clause must be banked rather than added. Without this
/// the equivalence tests above could be passing vacuously.
#[test]
fn lazy_path_banks_what_it_derives() {
    // Chained symbolic adds: carry-out of one adder feeds the next, so
    // cuts span several gates and the propagation holes are real. (A
    // single adder does not fire — its gate vars are consumed by the
    // equality's own cone before any multi-gate cut survives.)
    let mut s = SmtSolver::new();
    s.set_pcaug_lazy(true);
    let x = s.bv_var(32);
    let y = s.bv_var(32);
    let a = s.bv_add(x, y);
    let b = s.bv_add(a, x);
    let t = s.bv_add(b, y);
    let k = s.bv_const(0x1234_5678, 32);
    let eq = s.bv_eq(t, k);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);

    let (roots, _, derived, _) = s.pcaug_report();
    let (banked, injected, _) = s.pcaug_lazy_report();
    assert!(roots > 0, "no gate roots examined");
    assert!(derived > 0, "augmenter derived nothing on chained adders");
    assert_eq!(banked, derived, "lazy mode must bank, not add");
    assert!(injected <= banked, "injected more than was banked");

    // The working set is bounded by construction and never exceeds the
    // number injected.
    let (_, _, live) = s.pcaug_set_report();
    assert!(live <= injected as usize, "working set exceeds injections");

    // The eager path on the same instance adds them straight to the DB
    // and banks nothing.
    let mut s2 = SmtSolver::new();
    s2.set_pcaug(true);
    let x = s2.bv_var(32);
    let y = s2.bv_var(32);
    let a = s2.bv_add(x, y);
    let b = s2.bv_add(a, x);
    let t = s2.bv_add(b, y);
    let k = s2.bv_const(0x1234_5678, 32);
    let eq = s2.bv_eq(t, k);
    s2.assert(eq);
    assert_eq!(s2.solve(), SmtResult::Sat);
    assert_eq!(s2.pcaug_lazy_report().0, 0, "eager mode must not bank");
}

/// The working set must stay bounded and actually evict on an instance
/// that runs long enough to sweep repeatedly — that bound is the whole
/// point of the on-demand design (monotonic injection degenerates into
/// adding the entire bank).
#[test]
fn working_set_stays_bounded_and_evicts() {
    let mut s = SmtSolver::new();
    s.set_pcaug_lazy(true);
    // Small ceiling and a short interval so the bound and the eviction
    // path are both exercised without needing a long solve.
    s.set_pcaug_capacity(8);
    s.set_pcaug_interval(25);
    // A miniature mixer: adder chains (what the augmenter banks) stirred
    // by odd-constant multiplies (what makes it hard enough to restart,
    // and so to sweep). Sized to run thousands of conflicts in
    // milliseconds.
    let x = s.bv_var(16);
    let y = s.bv_var(16);
    let mut t = s.bv_add(x, y);
    for i in 0..4u128 {
        let odd = s.bv_const(0x9E37 + i * 2, 16);
        t = s.bv_mul(t, odd);
        t = s.bv_add(t, x);
        t = s.bv_xor(t, y);
        t = s.bv_add(t, y);
    }
    let k = s.bv_const(0xBEEF, 16);
    let eq = s.bv_eq(t, k);
    s.assert(eq);
    let res = s.solve();
    assert!(res == SmtResult::Sat || res == SmtResult::Unsat);

    let (banked, injected, sweeps) = s.pcaug_lazy_report();
    let (evicted, _units, live) = s.pcaug_set_report();
    assert!(banked > 0, "nothing banked");
    assert!(sweeps > 0, "no sweep ran — instance too easy to exercise eviction");
    assert!(
        live <= 8,
        "working set {live} exceeded its ceiling of 8"
    );
    // Every injected clause is either still live or was evicted (units
    // and root-satisfied clauses never enter the set).
    assert!(
        live as u64 + evicted <= injected,
        "live {live} + evicted {evicted} exceeds injected {injected}"
    );
}
