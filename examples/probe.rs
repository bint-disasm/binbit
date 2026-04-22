use binbit::{SmtResult, SmtSolver};
fn main() {
    for w in [16u32, 32, 64] {
        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let three = s.bv_const(3, w);
        let five = s.bv_const(5, w);
        let q = s.bv_udiv(x, three);
        let eq = s.bv_eq(q, five);
        s.assert(eq);
        let res = s.solve();
        print!("w={}: {:?}", w, res);
        if res == SmtResult::Sat {
            println!(" x = {}", s.get_bv_value(x));
        } else {
            println!();
        }
    }
}
