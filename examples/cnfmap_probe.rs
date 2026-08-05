use binbit::{SmtResult, SmtSolver};
fn probe(name: &str, f: impl Fn(&mut SmtSolver) -> binbit::BoolTerm) {
    let run = |mapped: bool| {
        let mut s = SmtSolver::new();
        s.set_cnf_mapping(mapped);
        s.set_bve(false);
        let c = f(&mut s);
        assert_ne!(s.solve_under_assumptions(&[c]), SmtResult::Unsat);
        let st = s.sat_stats();
        (st.sat_vars, st.sat_clauses)
    };
    let (v0, c0) = run(false);
    let (v1, c1) = run(true);
    println!("{name:10} classic {v0:6}v {c0:7}c   mapped {v1:6}v {c1:7}c");
}
fn main() {
    probe("xor-chain", |s| {
        let x = s.bv_var(32); let y = s.bv_var(32); let z = s.bv_var(32);
        let a = s.bv_xor(x, y); let b = s.bv_xor(a, z);
        let k = s.bv_const(12345, 32); s.bv_eq(b, k)
    });
    probe("add", |s| {
        let x = s.bv_var(32); let y = s.bv_var(32);
        let a = s.bv_add(x, y);
        let k = s.bv_const(12345, 32); s.bv_ult(a, k)
    });
    probe("mul", |s| {
        let x = s.bv_var(32); let y = s.bv_var(32);
        let p = s.bv_mul(x, y);
        let k = s.bv_const(123456, 32); s.bv_ult(p, k)
    });
    probe("ite-chain", |s| {
        let x = s.bv_var(16); let mut t = s.bv_var(16);
        for i in 0..8 {
            let k = s.bv_const(i * 37, 16);
            let c = s.bv_eq(x, k);
            let v = s.bv_const(i * 91 + 3, 16);
            t = s.bv_ite(c, v, t);
        }
        let k = s.bv_const(500, 16); s.bv_ult(t, k)
    });
}
