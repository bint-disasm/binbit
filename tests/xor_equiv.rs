//! GF(2) elimination (`set_xor_reasoning`) only ever adds clauses that
//! are implied by the formula, so it must be answer-equivalent to the
//! plain pipeline — including when derived rows are materialized as CNF.

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
    fn below(&mut self, b: u32) -> u32 {
        (self.next() % b as u64) as u32
    }
}

/// XOR/add-heavy term: the parity skeleton is what elimination works on.
fn random_query(s: &mut SmtSolver, rng: &mut Rng) -> BoolTerm {
    let x = s.bv_var(8);
    let y = s.bv_var(8);
    let mut t = x;
    for _ in 0..(2 + rng.below(4)) {
        let c = s.bv_const(rng.next() as u128 & 0xFF, 8);
        t = match rng.below(5) {
            0 => s.bv_xor(t, y),
            1 => s.bv_add(t, y),
            2 => s.bv_xor(t, c),
            3 => s.bv_add(t, c),
            _ => s.bv_and(t, c),
        };
    }
    let k = s.bv_const(rng.next() as u128 & 0xFF, 8);
    if rng.below(2) == 0 {
        s.bv_eq(t, k)
    } else {
        s.bv_ult(t, k)
    }
}

#[test]
fn xor_reasoning_matches_plain() {
    let mut rng = Rng(0x9E37_79B9);
    for trial in 0..80 {
        let seed = rng.next();
        let run = |xor: bool| -> (SmtResult, SmtResult) {
            let mut s = SmtSolver::new();
            s.set_xor_reasoning(xor);
            let mut r = Rng(seed);
            let q = random_query(&mut s, &mut r);
            let a = s.solve_under_assumptions(&[q]);
            s.assert(q);
            (a, s.solve())
        };
        assert_eq!(run(false), run(true), "trial {trial} diverged");
    }
}

/// A model returned under XOR reasoning must still satisfy the
/// assertion — catches a derived row that is not actually implied.
#[test]
fn xor_models_satisfy_the_assertion() {
    let mut s = SmtSolver::new();
    s.set_xor_reasoning(true);
    let x = s.bv_var(16);
    let y = s.bv_var(16);
    let a = s.bv_xor(x, y);
    let b = s.bv_add(a, x);
    let k = s.bv_const(0x1234, 16);
    let eq = s.bv_eq(b, k);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
    let xv = s.get_bv_value(x);
    let yv = s.get_bv_value(y);
    assert_eq!(((xv ^ yv) + xv) & 0xFFFF, 0x1234, "model violates assertion");
}
