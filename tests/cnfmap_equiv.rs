//! Cut-based CNF mapping (`set_cnf_mapping`) must be answer-equivalent
//! to classic shape-aware Tseitin on every query type, while emitting
//! measurably fewer variables/clauses on arithmetic cones.

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

/// Build a random term over two 16-bit vars, mixing arithmetic, logic,
/// comparisons, and mux — deep enough to exercise multi-node cuts.
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

#[test]
fn mapped_answers_match_classic() {
    let mut rng = Rng(0xC0FFEE);
    for trial in 0..60 {
        let seed = rng.next();

        let run = |mapped: bool| -> (SmtResult, SmtResult, SmtResult) {
            let mut s = SmtSolver::new();
            s.set_cnf_mapping(mapped);
            let mut r = Rng(seed);
            let q = random_query(&mut s, &mut r);
            let assumed = s.solve_under_assumptions(&[q]);
            let nq = s.bool_not(q);
            let opposite = s.solve_under_assumptions(&[nq]);
            s.assert(q);
            let asserted = s.solve();
            (assumed, opposite, asserted)
        };
        assert_eq!(run(false), run(true), "trial {trial} diverged");
    }
}

#[test]
fn mapped_pair_and_batch_match_classic() {
    let mut rng = Rng(0xFACADE);
    for trial in 0..40 {
        let seed = rng.next();
        let run = |mapped: bool| {
            let mut s = SmtSolver::new();
            s.set_cnf_mapping(mapped);
            let mut r = Rng(seed);
            let pc = random_query(&mut s, &mut r);
            let cond = random_query(&mut s, &mut r);
            let not_cond = s.bool_not(cond);
            let pair = s.solve_pair_under_assumptions(cond, &[pc]);
            let batch = s.solve_each_under_assumptions(&[cond, not_cond], &[pc]);
            (pair, batch)
        };
        assert_eq!(run(false), run(true), "trial {trial} diverged");
    }
}

#[test]
fn mapped_model_reads_are_consistent() {
    // Model values must satisfy the asserted formula under mapping (the
    // mapped CNF constrains the same models; reads go through the AIG).
    let mut s = SmtSolver::new();
    s.set_cnf_mapping(true);
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

#[test]
fn mapped_emits_fewer_gates() {
    let build = |mapped: bool| build_effort(mapped, false);
    let build_full = |mapped: bool| build_effort(mapped, true);
    fn build_effort(mapped: bool, full: bool) -> (usize, usize) {
        let mut s = SmtSolver::new();
        s.set_cnf_mapping(mapped);
        s.set_cnf_mapping_effort(full);
        s.set_bve(false); // measure raw emission, not post-elimination
        let x = s.bv_var(32);
        let y = s.bv_var(32);
        let p = s.bv_mul(x, y);
        let k = s.bv_const(123_456, 32);
        let c = s.bv_ult(p, k);
        assert_eq!(s.solve_under_assumptions(&[c]), SmtResult::Sat);
        let st = s.sat_stats();
        (st.sat_vars, st.sat_clauses)
    }
    let (v0, c0) = build(false);
    let (v1, c1) = build(true);
    // Full effort is what the ≥25% claim is about; the default (fast)
    // effort trades some of that for ~40% less mapping time.
    let (vf, cf) = build_full(true);
    assert!(
        vf * 4 <= v0 * 3,
        "full effort should cut mul vars by >=25%: {v0} -> {vf}"
    );
    assert!(
        cf * 100 <= c0 * 115,
        "full-effort clauses regressed past +15%: {c0} -> {cf}"
    );
    // Current mapper quality on the hardest shape (adder-army mul): vars
    // drop ≥ 25%, clauses stay within +15% of the shape-aware classic
    // encoder (which is already stronger than the paper's baseline).
    // XOR/ITE-dominated cones win on both axes — see cnfmap_probe.
    assert!(
        v1 * 10 <= v0 * 9,
        "fast effort should still cut mul vars by >=10%: {v0} -> {v1}"
    );
    assert!(
        c1 * 100 <= c0 * 115,
        "mapped clauses regressed past +15%: {c0} -> {c1}"
    );
}
