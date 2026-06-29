//! SMT-LIB 2 emitter for `SmtSolver` state — counterpart to the `smtlib`
//! reader. Walks the BV / Bool term DAGs reachable from a given set of
//! assertions and prints a self-contained QF_BV script replayable via
//! `binbit --smt foo.smt2` (or z3 / bitwuzla / cvc5).
//!
//! Every non-leaf term is emitted once as `(define-fun ...)` in topological
//! order, so output size is linear in DAG size — no blowup from inlining
//! shared subterms into a tree.

use std::collections::HashMap;
use std::fmt::Write;

use crate::bv::{BoolOp, BoolTerm, BvOp, BvTerm};
use crate::smt::SmtSolver;

/// Emit an SMT-LIB 2 script asserting each `BoolTerm` in `assertions`.
///
/// The caller supplies the assertion list — `SmtSolver` doesn't retain the
/// un-bitblasted terms (only the resulting SAT clauses), so consumers are
/// expected to keep a parallel `Vec<BoolTerm>` of everything they've passed
/// to `assert` / `assert_named`. A tiny wrapper does the job:
///
/// ```no_run
/// use binbit::{BoolTerm, SmtSolver, dump_smtlib};
///
/// struct TracedSolver {
///     solver: SmtSolver,
///     assertions: Vec<BoolTerm>,
/// }
/// impl TracedSolver {
///     fn assert(&mut self, t: BoolTerm) {
///         self.solver.assert(t);
///         self.assertions.push(t);
///     }
///     fn dump(&self) -> String {
///         dump_smtlib(&self.solver, &self.assertions)
///     }
/// }
/// ```
///
/// Caveats:
/// - `push` / `pop` state is not preserved — output is a flat snapshot of
///   the supplied assertions.
/// - `alias_bv_vars` and `assert_mutually_exclusive` aren't reflected.
///   Record them as explicit assertions if you need them in the output.
/// - `BvOp::Select` lowers to nested ITE (semantically equivalent; loses
///   the pairwise-exclusion optimization on replay).
pub fn dump_smtlib(solver: &SmtSolver, assertions: &[BoolTerm]) -> String {
    let mut p = Printer {
        s: solver,
        bv_label: HashMap::new(),
        bool_label: HashMap::new(),
        bv_order: Vec::new(),
        bool_order: Vec::new(),
        bv_vars: Vec::new(),
        bool_vars: Vec::new(),
        counter: 0,
    };
    for &a in assertions {
        p.visit_bool(a);
    }
    p.emit(assertions)
}

struct Printer<'a> {
    s: &'a SmtSolver,
    bv_label: HashMap<BvTerm, String>,
    bool_label: HashMap<BoolTerm, String>,
    bv_order: Vec<BvTerm>,
    bool_order: Vec<BoolTerm>,
    bv_vars: Vec<(u32, u32)>, // (var_id, width)
    bool_vars: Vec<u32>,
    counter: u32,
}

impl<'a> Printer<'a> {
    fn fresh(&mut self, prefix: &str) -> String {
        let n = self.counter;
        self.counter += 1;
        format!("{}{}", prefix, n)
    }

    fn visit_bv(&mut self, t: BvTerm) {
        if self.bv_label.contains_key(&t) {
            return;
        }
        match self.s.ctx.bv_op(t) {
            BvOp::Var(id) => {
                let w = self.s.ctx.width_of(t);
                self.bv_vars.push((id, w));
                self.bv_label.insert(t, format!("v{}", id));
                return;
            }
            BvOp::Const => {
                // Empty label ⇒ render inline at each use-site.
                self.bv_label.insert(t, String::new());
                return;
            }
            BvOp::Not(a) | BvOp::Neg(a) | BvOp::Extract(a, _, _)
            | BvOp::ZeroExtend(a, _) | BvOp::SignExtend(a, _)
            | BvOp::Popcount(a) | BvOp::Clz(a) | BvOp::Ctz(a) => {
                self.visit_bv(a);
            }
            BvOp::And(a, b) | BvOp::Or(a, b) | BvOp::Xor(a, b)
            | BvOp::Add(a, b) | BvOp::Sub(a, b) | BvOp::Mul(a, b)
            | BvOp::Udiv(a, b) | BvOp::Urem(a, b)
            | BvOp::Sdiv(a, b) | BvOp::Srem(a, b) | BvOp::Smod(a, b)
            | BvOp::Shl(a, b) | BvOp::Lshr(a, b) | BvOp::Ashr(a, b)
            | BvOp::RotateLeft(a, b) | BvOp::RotateRight(a, b)
            | BvOp::Concat(a, b) => {
                self.visit_bv(a);
                self.visit_bv(b);
            }
            BvOp::Ite(c, tt, ee) => {
                self.visit_bool(c);
                self.visit_bv(tt);
                self.visit_bv(ee);
            }
            BvOp::Select(idx) => {
                let tbl = &self.s.ctx.select_tables[idx as usize];
                for &sel in tbl.selectors.iter() { self.visit_bool(sel); }
                for &v in tbl.values.iter()      { self.visit_bv(v); }
                self.visit_bv(tbl.default);
            }
        }
        let lbl = self.fresh("bv");
        self.bv_label.insert(t, lbl);
        self.bv_order.push(t);
    }

    fn visit_bool(&mut self, t: BoolTerm) {
        if self.bool_label.contains_key(&t) {
            return;
        }
        match self.s.ctx.bool_op(t) {
            BoolOp::True | BoolOp::False => {
                self.bool_label.insert(t, String::new());
                return;
            }
            BoolOp::Var(id) => {
                self.bool_vars.push(id);
                self.bool_label.insert(t, format!("p{}", id));
                return;
            }
            BoolOp::Not(a) => self.visit_bool(a),
            BoolOp::And(a, b) | BoolOp::Or(a, b) | BoolOp::Implies(a, b) => {
                self.visit_bool(a);
                self.visit_bool(b);
            }
            BoolOp::Eq(a, b) | BoolOp::Ult(a, b) | BoolOp::Ule(a, b)
            | BoolOp::Slt(a, b) | BoolOp::Sle(a, b)
            | BoolOp::UaddOverflow(a, b) | BoolOp::SaddOverflow(a, b)
            | BoolOp::UsubOverflow(a, b) | BoolOp::SsubOverflow(a, b)
            | BoolOp::UmulOverflow(a, b) | BoolOp::SmulOverflow(a, b)
            | BoolOp::SdivOverflow(a, b) => {
                self.visit_bv(a);
                self.visit_bv(b);
            }
            BoolOp::NegOverflow(a) => self.visit_bv(a),
        }
        let lbl = self.fresh("p");
        self.bool_label.insert(t, lbl);
        self.bool_order.push(t);
    }

    fn render_bv(&self, t: BvTerm) -> String {
        match self.bv_label.get(&t) {
            Some(l) if !l.is_empty() => return l.clone(),
            Some(_) => {} // empty label ⇒ inline constant
            None => unreachable!("unvisited BvTerm: {:?}", t),
        }
        let w = self.s.ctx.width_of(t);
        if w <= 128 {
            format!("(_ bv{} {})", self.s.ctx.bv_const_value_u128(t), w)
        } else {
            render_wide_const(&self.s.ctx.bv_const_value_limbs(t), w)
        }
    }

    fn render_bool(&self, t: BoolTerm) -> String {
        match self.bool_label.get(&t) {
            Some(l) if !l.is_empty() => return l.clone(),
            _ => {}
        }
        match self.s.ctx.bool_op(t) {
            BoolOp::True => "true".into(),
            BoolOp::False => "false".into(),
            _ => unreachable!(),
        }
    }

    fn render_bv_rhs(&self, t: BvTerm) -> String {
        match self.s.ctx.bv_op(t) {
            BvOp::Var(_) | BvOp::Const => unreachable!("no define-fun for leaves"),
            BvOp::Not(a)        => format!("(bvnot {})", self.render_bv(a)),
            BvOp::Neg(a)        => format!("(bvneg {})", self.render_bv(a)),
            BvOp::And(a, b)     => format!("(bvand {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Or(a, b)      => format!("(bvor {} {})",  self.render_bv(a), self.render_bv(b)),
            BvOp::Xor(a, b)     => format!("(bvxor {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Add(a, b)     => format!("(bvadd {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Sub(a, b)     => format!("(bvsub {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Mul(a, b)     => format!("(bvmul {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Udiv(a, b)    => format!("(bvudiv {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Urem(a, b)    => format!("(bvurem {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Sdiv(a, b)    => format!("(bvsdiv {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Srem(a, b)    => format!("(bvsrem {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Smod(a, b)    => format!("(bvsmod {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Shl(a, b)     => format!("(bvshl {} {})",  self.render_bv(a), self.render_bv(b)),
            BvOp::Lshr(a, b)    => format!("(bvlshr {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::Ashr(a, b)    => format!("(bvashr {} {})", self.render_bv(a), self.render_bv(b)),
            // SMT-LIB has no rotate-by-symbolic; inline as `(x << m) | (x >> (w - m))`
            // where `m = amt mod w`. The `urem` is there to handle out-of-range
            // amounts robustly.
            BvOp::RotateLeft(a, b) => {
                let xs = self.render_bv(a);
                let amt = self.render_bv(b);
                let w = self.s.ctx.width_of(a);
                let w_const = format!("(_ bv{} {})", w, w);
                let m = format!("(bvurem {} {})", amt, w_const);
                format!(
                    "(bvor (bvshl {} {}) (bvlshr {} (bvsub {} {})))",
                    xs, m, xs, w_const, m,
                )
            }
            BvOp::RotateRight(a, b) => {
                let xs = self.render_bv(a);
                let amt = self.render_bv(b);
                let w = self.s.ctx.width_of(a);
                let w_const = format!("(_ bv{} {})", w, w);
                let m = format!("(bvurem {} {})", amt, w_const);
                format!(
                    "(bvor (bvlshr {} {}) (bvshl {} (bvsub {} {})))",
                    xs, m, xs, w_const, m,
                )
            }
            BvOp::Extract(a, hi, lo) => format!("((_ extract {} {}) {})", hi, lo, self.render_bv(a)),
            BvOp::Concat(a, b)  => format!("(concat {} {})", self.render_bv(a), self.render_bv(b)),
            BvOp::ZeroExtend(a, n) => format!("((_ zero_extend {}) {})", n, self.render_bv(a)),
            BvOp::SignExtend(a, n) => format!("((_ sign_extend {}) {})", n, self.render_bv(a)),
            // SMT-LIB has no standard popcount / clz / ctz, so inline the
            // expansions in primitive ops. Output is verbose but parseable
            // by z3 / bitwuzla / cvc5 and round-trips through binbit's
            // own parser.
            BvOp::Popcount(a) => {
                let xs = self.render_bv(a);
                Self::render_popcount(&xs, self.s.ctx.width_of(a))
            }
            BvOp::Clz(a) => {
                let xs = self.render_bv(a);
                Self::render_clz(&xs, self.s.ctx.width_of(a))
            }
            BvOp::Ctz(a) => {
                let xs = self.render_bv(a);
                Self::render_ctz(&xs, self.s.ctx.width_of(a))
            }
            BvOp::Ite(c, tt, ee) => format!(
                "(ite {} {} {})",
                self.render_bool(c), self.render_bv(tt), self.render_bv(ee),
            ),
            BvOp::Select(idx) => {
                // First-match semantics: earliest true selector wins;
                // fall through to `default` if none match.
                let tbl = &self.s.ctx.select_tables[idx as usize];
                let mut out = self.render_bv(tbl.default);
                for i in (0..tbl.selectors.len()).rev() {
                    out = format!(
                        "(ite {} {} {})",
                        self.render_bool(tbl.selectors[i]),
                        self.render_bv(tbl.values[i]),
                        out,
                    );
                }
                out
            }
        }
    }

    fn render_bool_rhs(&self, t: BoolTerm) -> String {
        match self.s.ctx.bool_op(t) {
            BoolOp::True | BoolOp::False | BoolOp::Var(_) => unreachable!(),
            BoolOp::Not(a)         => format!("(not {})", self.render_bool(a)),
            BoolOp::And(a, b)      => format!("(and {} {})", self.render_bool(a), self.render_bool(b)),
            BoolOp::Or(a, b)       => format!("(or {} {})",  self.render_bool(a), self.render_bool(b)),
            BoolOp::Implies(a, b)  => format!("(=> {} {})",  self.render_bool(a), self.render_bool(b)),
            BoolOp::Eq(a, b)       => format!("(= {} {})",    self.render_bv(a), self.render_bv(b)),
            BoolOp::Ult(a, b)      => format!("(bvult {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::Ule(a, b)      => format!("(bvule {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::Slt(a, b)      => format!("(bvslt {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::Sle(a, b)      => format!("(bvsle {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::UaddOverflow(a, b) => format!("(bvuaddo {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::SaddOverflow(a, b) => format!("(bvsaddo {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::UsubOverflow(a, b) => format!("(bvusubo {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::SsubOverflow(a, b) => format!("(bvssubo {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::UmulOverflow(a, b) => format!("(bvumulo {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::SmulOverflow(a, b) => format!("(bvsmulo {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::SdivOverflow(a, b) => format!("(bvsdivo {} {})", self.render_bv(a), self.render_bv(b)),
            BoolOp::NegOverflow(a)     => format!("(bvnego {})",     self.render_bv(a)),
        }
    }

    /// Inline popcount as a chain of zero-extended single-bit adds. Each
    /// `((_ extract i i) xs)` produces a 1-bit term that we zero-extend to
    /// the full width and sum. Verbose but uses only standard SMT-LIB.
    fn render_popcount(xs: &str, w: u32) -> String {
        if w == 1 {
            return xs.to_string();
        }
        let mut acc = format!("((_ zero_extend {}) ((_ extract 0 0) {}))", w - 1, xs);
        for i in 1..w {
            acc = format!(
                "(bvadd {} ((_ zero_extend {}) ((_ extract {} {}) {})))",
                acc, w - 1, i, i, xs,
            );
        }
        acc
    }

    /// Inline CLZ as `popcount(~(x | x>>1 | x>>2 | ... | x>>(W/2)))`. The
    /// OR-fold makes every bit at-or-below the highest set bit a 1, then
    /// popcount-of-not counts the cleared (leading-zero) bits.
    fn render_clz(xs: &str, w: u32) -> String {
        if w == 1 {
            return format!("(bvnot {})", xs);
        }
        let mut y = xs.to_string();
        let mut k = 1u32;
        while k < w {
            y = format!("(bvor {} (bvlshr {} (_ bv{} {})))", y, y, k, w);
            k <<= 1;
        }
        let ny = format!("(bvnot {})", y);
        Self::render_popcount(&ny, w)
    }

    /// Inline CTZ via `popcount(~x & (x - 1))`. For `x == 0` the
    /// expression collapses to all-ones whose popcount is `w` (the
    /// SMT-LIB convention).
    fn render_ctz(xs: &str, w: u32) -> String {
        if w == 1 {
            return format!("(bvnot {})", xs);
        }
        let m = format!(
            "(bvand (bvnot {}) (bvsub {} (_ bv1 {})))",
            xs, xs, w,
        );
        Self::render_popcount(&m, w)
    }

    fn emit(&self, assertions: &[BoolTerm]) -> String {
        let mut out = String::new();
        out.push_str("(set-logic QF_BV)\n");
        for &(id, w) in &self.bv_vars {
            let _ = writeln!(out, "(declare-fun v{} () (_ BitVec {}))", id, w);
        }
        for &id in &self.bool_vars {
            let _ = writeln!(out, "(declare-fun p{} () Bool)", id);
        }
        for &t in &self.bv_order {
            let w = self.s.ctx.width_of(t);
            let _ = writeln!(
                out,
                "(define-fun {} () (_ BitVec {}) {})",
                self.bv_label[&t], w, self.render_bv_rhs(t),
            );
        }
        for &t in &self.bool_order {
            let _ = writeln!(
                out,
                "(define-fun {} () Bool {})",
                self.bool_label[&t], self.render_bool_rhs(t),
            );
        }
        for &a in assertions {
            let _ = writeln!(out, "(assert {})", self.render_bool(a));
        }
        out.push_str("(check-sat)\n");
        out.push_str("(exit)\n");
        out
    }
}

/// Render a wide (>128-bit) constant as `#b<bits>` MSB-first. `limbs` is
/// LSB-first (limb[0] holds the low 64 bits).
fn render_wide_const(limbs: &[u64], width: u32) -> String {
    let mut s = String::with_capacity(3 + width as usize);
    s.push_str("#b");
    for i in (0..width).rev() {
        let limb = (i / 64) as usize;
        let bit = (i % 64) as u32;
        let v = (limbs.get(limb).copied().unwrap_or(0) >> bit) & 1;
        s.push(if v == 1 { '1' } else { '0' });
    }
    s
}
