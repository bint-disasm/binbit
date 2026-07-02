//! SMT-LIB 2 frontend for the BV solver. Parses a script, dispatches commands
//! to an `SmtSolver`, and returns the collected output.
//!
//! Supported subset (QF_BV):
//! - Commands: `set-logic`, `set-option`, `set-info`, `declare-const`,
//!   `declare-fun`, `define-const`, `define-fun`, `assert`, `check-sat`,
//!   `check-sat-assuming`, `push`, `pop`, `get-value`, `get-model`, `exit`.
//! - Sorts: `Bool`, `(_ BitVec n)`.
//! - Constants: `true`/`false`; `#bXXX`, `#xXXX`, `(_ bvN W)`.
//! - Bool ops: `and`, `or`, `not`, `xor`, `=>`, `ite` (on Bool).
//! - Comparisons: `=`, `distinct`.
//! - Arithmetic ops: bvadd, bvsub, bvneg, bvmul, bvudiv, bvurem, bvsdiv,
//!   bvsrem, bvsmod.
//! - Shifts: bvshl, bvlshr, bvashr.
//! - Bitwise: bvand, bvor, bvxor, bvnot.
//! - Comparisons: bv[u/s][lt/le/gt/ge].
//! - Structural: `concat`, `((_ extract h l) x)`, `((_ zero_extend n) x)`,
//!   `((_ sign_extend n) x)`.
//! - Conditional: `ite` (on BV too).
//! - `let` bindings.
//!
//! Intentionally deferred: quantifiers, uninterpreted functions, arrays,
//! overflow predicates (we have them in the Rust API but not yet wired to
//! SMT-LIB syntax — the standard doesn't mandate names for these).

use std::collections::HashMap;

use crate::bv::{self, BoolTerm, BvTerm};
use crate::smt::{SmtResult, SmtSolver};

/// Sort tag. SMT-LIB has richer sorts; we only care about Bool vs BV[n].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Sort {
    Bool,
    Bv(u32),
}

/// A term returned by expression parsing, tagged with its sort.
#[derive(Copy, Clone, Debug)]
enum TaggedTerm {
    Bool(BoolTerm),
    Bv(BvTerm, u32),
}

impl TaggedTerm {
    fn sort(&self) -> Sort {
        match self {
            TaggedTerm::Bool(_) => Sort::Bool,
            TaggedTerm::Bv(_, w) => Sort::Bv(*w),
        }
    }
    fn as_bool(&self) -> Result<BoolTerm, String> {
        match self {
            TaggedTerm::Bool(t) => Ok(*t),
            _ => Err(format!("expected Bool, got {:?}", self.sort())),
        }
    }
    fn as_bv(&self) -> Result<(BvTerm, u32), String> {
        match self {
            TaggedTerm::Bv(t, w) => Ok((*t, *w)),
            _ => Err(format!("expected BV, got {:?}", self.sort())),
        }
    }
}

/// Run a complete SMT-LIB script against a fresh solver. Collects the output
/// lines — `sat` / `unsat` / `unknown` results, plus anything produced by
/// `get-value` / `get-model`.
pub fn run_script(script: &str) -> Result<String, String> {
    let mut solver = SmtSolver::new();
    run_script_with(&mut solver, script)
}

/// Same as [`run_script`], but runs against an existing solver so callers
/// can inspect state after the script finishes.
pub fn run_script_with(solver: &mut SmtSolver, script: &str) -> Result<String, String> {
    let mut runner = Runner::new(solver);
    runner.run(script)?;
    Ok(runner.output)
}

struct Runner<'a> {
    solver: &'a mut SmtSolver,
    // Global symbol table (populated by declare/define).
    symbols: HashMap<String, TaggedTerm>,
    // Names whose binding is still the fresh var created by declare-fun /
    // declare-const and which have NOT yet been resolved by any evaluation.
    // Such a name is eligible for `(assert (= X expr))` substitution — we
    // can safely rebind the symbol to `expr` and drop the assertion, because
    // nothing has captured the old fresh-var term yet. First lookup through
    // `eval_atom` evicts the name from this set.
    declared: std::collections::HashSet<String>,
    // Let bindings: per-name shadow stacks + per-scope frames recording
    // which names to unbind on pop. O(1) lookup regardless of nesting
    // depth — Sage2-style dumps nest `let` thousands deep, and a
    // scope-list scan per symbol reference is quadratic over the file
    // (bench_3335: 5.7s of pure lookup, zero SAT conflicts).
    let_bindings: HashMap<String, Vec<TaggedTerm>>,
    let_frames: Vec<Vec<String>>,
    output: String,
}

impl<'a> Runner<'a> {
    fn new(solver: &'a mut SmtSolver) -> Self {
        Runner {
            solver,
            symbols: HashMap::new(),
            declared: std::collections::HashSet::new(),
            let_bindings: HashMap::new(),
            let_frames: Vec::new(),
            output: String::new(),
        }
    }

    fn run(&mut self, script: &str) -> Result<(), String> {
        let mut parser = Parser::new(script);
        while let Some(expr) = parser.next_sexpr()? {
            self.run_command(&expr)?;
        }
        Ok(())
    }

    fn run_command(&mut self, expr: &SExpr) -> Result<(), String> {
        let list = match expr {
            SExpr::List(xs) => xs,
            SExpr::Atom(_) => {
                return Err(format!("expected a command S-expression, got atom"));
            }
        };
        let head = match list.first() {
            Some(SExpr::Atom(s)) => s.as_str(),
            _ => return Err(format!("empty command")),
        };

        match head {
            // No-ops we parse but don't need to do anything with.
            "set-logic" | "set-option" | "set-info" => Ok(()),

            "declare-const" => {
                // (declare-const <name> <sort>)
                let name = atom(list.get(1))?;
                let sort = parse_sort(list.get(2).ok_or("declare-const: missing sort")?)?;
                let term = match sort {
                    Sort::Bool => TaggedTerm::Bool(self.solver.bool_var()),
                    Sort::Bv(w) => TaggedTerm::Bv(self.solver.bv_var(w), w),
                };
                self.symbols.insert(name.to_string(), term);
                self.declared.insert(name.to_string());
                Ok(())
            }

            "declare-fun" => {
                // (declare-fun <name> () <sort>) — 0-arity only (our subset).
                let name = atom(list.get(1))?;
                match list.get(2) {
                    Some(SExpr::List(params)) if params.is_empty() => {}
                    _ => return Err("declare-fun: only 0-arity supported".into()),
                }
                let sort = parse_sort(list.get(3).ok_or("declare-fun: missing sort")?)?;
                let term = match sort {
                    Sort::Bool => TaggedTerm::Bool(self.solver.bool_var()),
                    Sort::Bv(w) => TaggedTerm::Bv(self.solver.bv_var(w), w),
                };
                self.symbols.insert(name.to_string(), term);
                self.declared.insert(name.to_string());
                Ok(())
            }

            "define-const" => {
                // (define-const <name> <sort> <expr>)
                let name = atom(list.get(1))?;
                let expected = parse_sort(list.get(2).ok_or("define-const: missing sort")?)?;
                let body = self.eval_expr(list.get(3).ok_or("define-const: missing body")?)?;
                if body.sort() != expected {
                    return Err(format!(
                        "define-const {}: expected {:?}, got {:?}",
                        name,
                        expected,
                        body.sort()
                    ));
                }
                self.symbols.insert(name.to_string(), body);
                Ok(())
            }

            "define-fun" => {
                // (define-fun <name> () <sort> <expr>) — 0-arity.
                let name = atom(list.get(1))?;
                match list.get(2) {
                    Some(SExpr::List(params)) if params.is_empty() => {}
                    _ => return Err("define-fun: only 0-arity supported".into()),
                }
                let expected = parse_sort(list.get(3).ok_or("define-fun: missing sort")?)?;
                let body = self.eval_expr(list.get(4).ok_or("define-fun: missing body")?)?;
                if body.sort() != expected {
                    return Err(format!(
                        "define-fun {}: expected {:?}, got {:?}",
                        name,
                        expected,
                        body.sort()
                    ));
                }
                self.symbols.insert(name.to_string(), body);
                Ok(())
            }

            "assert" => {
                // Handle `(assert (! phi :named name))` by pulling the name
                // out and calling `assert_named`. Otherwise plain `assert`.
                let body_expr = list.get(1).ok_or("assert: missing body")?;
                if let Some((inner, name)) = extract_named(body_expr) {
                    let body = self.eval_expr(inner)?;
                    let b = body.as_bool()?;
                    self.solver.assert_named(name, b);
                    return Ok(());
                }
                self.assert_body(body_expr)
            }

            "get-unsat-core" => {
                // SMT-LIB prints the core as a parenthesized list of names.
                let names = self.solver.unsat_core_names();
                self.output.push('(');
                for (i, name) in names.iter().enumerate() {
                    if i > 0 {
                        self.output.push(' ');
                    }
                    self.output.push_str(name);
                }
                self.output.push_str(")\n");
                Ok(())
            }

            "check-sat" => {
                let r = self.solver.solve();
                self.output.push_str(match r {
                    SmtResult::Sat => "sat\n",
                    SmtResult::Unsat => "unsat\n",
                });
                Ok(())
            }

            "check-sat-assuming" => {
                let asmps_list = match list.get(1) {
                    Some(SExpr::List(xs)) => xs,
                    _ => return Err("check-sat-assuming: expected a list".into()),
                };
                let mut asmps = Vec::with_capacity(asmps_list.len());
                for a in asmps_list {
                    let t = self.eval_expr(a)?;
                    asmps.push(t.as_bool()?);
                }
                let r = self.solver.solve_under_assumptions(&asmps);
                self.output.push_str(match r {
                    SmtResult::Sat => "sat\n",
                    SmtResult::Unsat => "unsat\n",
                });
                Ok(())
            }

            "push" => {
                let n = match list.get(1) {
                    Some(SExpr::Atom(s)) => s.parse::<usize>().unwrap_or(1),
                    _ => 1,
                };
                for _ in 0..n {
                    self.solver.push();
                }
                Ok(())
            }

            "pop" => {
                let n = match list.get(1) {
                    Some(SExpr::Atom(s)) => s.parse::<usize>().unwrap_or(1),
                    _ => 1,
                };
                for _ in 0..n {
                    self.solver.pop();
                }
                Ok(())
            }

            "get-value" => {
                if !self.solver.has_model() {
                    // Per SMT-LIB, get-value is only valid when the solver
                    // holds a satisfying model. Emit an error response
                    // rather than silently returning stale bits.
                    self.output
                        .push_str("(error \"get-value called without a satisfying model\")\n");
                    return Ok(());
                }
                let terms_list = match list.get(1) {
                    Some(SExpr::List(xs)) => xs,
                    _ => return Err("get-value: expected a list".into()),
                };
                self.output.push('(');
                for (i, expr) in terms_list.iter().enumerate() {
                    if i > 0 {
                        self.output.push(' ');
                    }
                    let printed = format_sexpr(expr);
                    let tt = self.eval_expr(expr)?;
                    let val_str = match tt {
                        TaggedTerm::Bool(t) => {
                            let v = self.solver.get_bool_value(t);
                            if v { "true".to_string() } else { "false".to_string() }
                        }
                        TaggedTerm::Bv(t, w) => format_bv_value(self.solver.get_bv_value_limbs(t).as_slice(), w),
                    };
                    self.output.push('(');
                    self.output.push_str(&printed);
                    self.output.push(' ');
                    self.output.push_str(&val_str);
                    self.output.push(')');
                }
                self.output.push_str(")\n");
                Ok(())
            }

            "get-model" => {
                if !self.solver.has_model() {
                    self.output
                        .push_str("(error \"get-model called without a satisfying model\")\n");
                    return Ok(());
                }
                // Emit values for every declared symbol.
                self.output.push_str("(model\n");
                // Iterate in insertion order-ish; HashMap isn't ordered, but
                // for debugging it's fine.
                let names: Vec<String> = self.symbols.keys().cloned().collect();
                for name in names {
                    let tt = *self.symbols.get(&name).unwrap();
                    let (val_str, sort_str) = match tt {
                        TaggedTerm::Bool(t) => {
                            let v = self.solver.get_bool_value(t);
                            (
                                if v { "true".to_string() } else { "false".to_string() },
                                "Bool".to_string(),
                            )
                        }
                        TaggedTerm::Bv(t, w) => (
                            format_bv_value(self.solver.get_bv_value_limbs(t).as_slice(), w),
                            format!("(_ BitVec {})", w),
                        ),
                    };
                    self.output.push_str(&format!(
                        "  (define-fun {} () {} {})\n",
                        name, sort_str, val_str
                    ));
                }
                self.output.push_str(")\n");
                Ok(())
            }

            "exit" => {
                // End of script — stop processing further commands.
                Ok(())
            }

            other => Err(format!("unsupported command: {}", other)),
        }
    }

    /// Convert an S-expression into a `TaggedTerm` in the solver's term DAG.
    fn eval_expr(&mut self, expr: &SExpr) -> Result<TaggedTerm, String> {
        match expr {
            SExpr::Atom(s) => self.eval_atom(s),
            SExpr::List(xs) => self.eval_list(xs),
        }
    }

    fn eval_atom(&mut self, s: &str) -> Result<TaggedTerm, String> {
        // Literals.
        if s == "true" {
            return Ok(TaggedTerm::Bool(self.solver.bool_true()));
        }
        if s == "false" {
            return Ok(TaggedTerm::Bool(self.solver.bool_false()));
        }
        if let Some(rest) = s.strip_prefix("#b") {
            let w = rest.len() as u32;
            if w == 0 {
                return Err(format!("empty binary literal: {}", s));
            }
            if w <= 128 {
                let v = u128::from_str_radix(rest, 2)
                    .map_err(|e| format!("bad binary literal {}: {}", s, e))?;
                return Ok(TaggedTerm::Bv(self.solver.bv_const(v, w), w));
            }
            // Wide: parse into limbs, bit-by-bit (LSB at position 0 of limb 0).
            let limbs = parse_binary_to_limbs(rest)?;
            return Ok(TaggedTerm::Bv(self.solver.bv_const_wide(&limbs, w), w));
        }
        if let Some(rest) = s.strip_prefix("#x") {
            let w = (rest.len() as u32) * 4;
            if w == 0 {
                return Err(format!("empty hex literal: {}", s));
            }
            if w <= 128 {
                let v = u128::from_str_radix(rest, 16)
                    .map_err(|e| format!("bad hex literal {}: {}", s, e))?;
                return Ok(TaggedTerm::Bv(self.solver.bv_const(v, w), w));
            }
            let limbs = parse_hex_to_limbs(rest)?;
            return Ok(TaggedTerm::Bv(self.solver.bv_const_wide(&limbs, w), w));
        }

        // Named symbol lookup: first in local let bindings, then globals.
        if let Some(stack) = self.let_bindings.get(s) {
            if let Some(&t) = stack.last() {
                return Ok(t);
            }
        }
        if let Some(&t) = self.symbols.get(s) {
            // First resolution commits this name to its current binding —
            // subsequent (assert (= s expr)) cannot safely substitute any more,
            // so drop s from the substitutable set.
            self.declared.remove(s);
            return Ok(t);
        }
        Err(format!("unknown symbol: {}", s))
    }

    fn eval_list(&mut self, xs: &[SExpr]) -> Result<TaggedTerm, String> {
        let head = xs.first().ok_or("empty expression list")?;

        // Indexed identifiers: `(_ <name> <args...>)`.
        if let SExpr::Atom(a) = head {
            if a == "_" {
                return self.eval_indexed(&xs[1..]);
            }
            // Attribute wrapper `(! expr :attr ...)` — attributes are
            // discarded in non-assert contexts (they're metadata only).
            // For `(assert (! phi :named foo))` the assert handler pulls
            // out the name before reaching here.
            if a == "!" {
                let inner = xs.get(1).ok_or("!: missing inner expr")?;
                return self.eval_expr(inner);
            }
        }
        // `((_ extract h l) x)` pattern — head is itself a list.
        if let SExpr::List(inner) = head {
            return self.eval_indexed_apply(inner, &xs[1..]);
        }

        let op = atom(Some(head))?;
        let args = &xs[1..];

        match op {
            // `let` bindings.
            "let" => {
                let bindings_list = match args.first() {
                    Some(SExpr::List(xs)) => xs,
                    _ => return Err("let: expected bindings list".into()),
                };
                let body = args
                    .get(1)
                    .ok_or("let: missing body")?;

                // SMT-LIB `let` is parallel: evaluate every bound value
                // in the OUTER scope before any binding takes effect.
                let mut pairs: Vec<(String, TaggedTerm)> =
                    Vec::with_capacity(bindings_list.len());
                for b in bindings_list {
                    let pair = match b {
                        SExpr::List(xs) if xs.len() == 2 => xs,
                        _ => return Err("let binding: expected (name expr)".into()),
                    };
                    let name = atom(pair.first())?;
                    let value = self.eval_expr(&pair[1])?;
                    pairs.push((name.to_string(), value));
                }
                let mut frame = Vec::with_capacity(pairs.len());
                for (name, value) in pairs {
                    self.let_bindings
                        .entry(name.clone())
                        .or_default()
                        .push(value);
                    frame.push(name);
                }
                self.let_frames.push(frame);
                let result = self.eval_expr(body);
                let frame = self.let_frames.pop().expect("balanced let frames");
                for name in frame {
                    if let Some(stack) = self.let_bindings.get_mut(&name) {
                        stack.pop();
                        if stack.is_empty() {
                            self.let_bindings.remove(&name);
                        }
                    }
                }
                result
            }

            // Boolean ops (chainable).
            "and" => self.bool_fold(args, |s, a, b| s.solver.bool_and(a, b), true),
            "or" => self.bool_fold(args, |s, a, b| s.solver.bool_or(a, b), false),
            "xor" => {
                let args = self.eval_bool_args(args)?;
                let mut it = args.into_iter();
                let first = it.next().ok_or("xor: no args")?;
                let mut acc = first;
                for next in it {
                    let nota = self.solver.bool_not(acc);
                    let notb = self.solver.bool_not(next);
                    let a_and_notb = self.solver.bool_and(acc, notb);
                    let nota_and_b = self.solver.bool_and(nota, next);
                    acc = self.solver.bool_or(a_and_notb, nota_and_b);
                }
                Ok(TaggedTerm::Bool(acc))
            }
            "not" => {
                let a = self.eval_bool(args.first())?;
                Ok(TaggedTerm::Bool(self.solver.bool_not(a)))
            }
            "=>" => {
                let args = self.eval_bool_args(args)?;
                if args.len() < 2 {
                    return Err("=>: requires at least 2 args".into());
                }
                // Right-associative: (=> a b c) = a => (b => c).
                let mut it = args.into_iter().rev();
                let mut acc = it.next().unwrap();
                for a in it {
                    acc = self.solver.bool_implies(a, acc);
                }
                Ok(TaggedTerm::Bool(acc))
            }

            // ite — may be Bool or Bv depending on branches.
            "ite" => {
                let c = self.eval_bool(args.first())?;
                let t = self.eval_expr(args.get(1).ok_or("ite: missing then")?)?;
                let e = self.eval_expr(args.get(2).ok_or("ite: missing else")?)?;
                match (t, e) {
                    (TaggedTerm::Bool(tt), TaggedTerm::Bool(ee)) => {
                        // Bool ITE: (c ∧ t) ∨ (¬c ∧ e).
                        let ca = self.solver.bool_and(c, tt);
                        let nc = self.solver.bool_not(c);
                        let cb = self.solver.bool_and(nc, ee);
                        Ok(TaggedTerm::Bool(self.solver.bool_or(ca, cb)))
                    }
                    (TaggedTerm::Bv(tt, wt), TaggedTerm::Bv(ee, we)) if wt == we => {
                        Ok(TaggedTerm::Bv(self.solver.bv_ite(c, tt, ee), wt))
                    }
                    _ => Err("ite: branches have mismatched sorts".into()),
                }
            }

            // Equality / distinctness work on any sort, but we only need BV
            // and Bool here.
            "=" => {
                if args.len() < 2 {
                    return Err("=: need at least 2 args".into());
                }
                let first = self.eval_expr(&args[0])?;
                let mut acc = self.solver.bool_true();
                for a in &args[1..] {
                    let next = self.eval_expr(a)?;
                    let this_eq = match (first, next) {
                        (TaggedTerm::Bool(a), TaggedTerm::Bool(b)) => {
                            // Bool equality = XNOR.
                            let na = self.solver.bool_not(a);
                            let nb = self.solver.bool_not(b);
                            let both_t = self.solver.bool_and(a, b);
                            let both_f = self.solver.bool_and(na, nb);
                            self.solver.bool_or(both_t, both_f)
                        }
                        (TaggedTerm::Bv(a, wa), TaggedTerm::Bv(b, wb)) if wa == wb => {
                            self.solver.bv_eq(a, b)
                        }
                        _ => return Err("=: operands must share a sort".into()),
                    };
                    acc = self.solver.bool_and(acc, this_eq);
                }
                Ok(TaggedTerm::Bool(acc))
            }

            "distinct" => {
                // Pairwise disequality.
                if args.len() < 2 {
                    return Err("distinct: need at least 2 args".into());
                }
                let terms: Vec<TaggedTerm> =
                    args.iter().map(|a| self.eval_expr(a)).collect::<Result<_, _>>()?;
                let mut acc = self.solver.bool_true();
                for i in 0..terms.len() {
                    for j in (i + 1)..terms.len() {
                        let ne = match (terms[i], terms[j]) {
                            (TaggedTerm::Bv(a, wa), TaggedTerm::Bv(b, wb)) if wa == wb => {
                                self.solver.bv_ne(a, b)
                            }
                            (TaggedTerm::Bool(a), TaggedTerm::Bool(b)) => {
                                let na = self.solver.bool_not(a);
                                let nb = self.solver.bool_not(b);
                                let both_t = self.solver.bool_and(a, b);
                                let both_f = self.solver.bool_and(na, nb);
                                let eq = self.solver.bool_or(both_t, both_f);
                                self.solver.bool_not(eq)
                            }
                            _ => return Err("distinct: operand sort mismatch".into()),
                        };
                        acc = self.solver.bool_and(acc, ne);
                    }
                }
                Ok(TaggedTerm::Bool(acc))
            }

            // ---- BV bitwise ----
            "bvnot" => self.bv_unary(args, |s, a, _| s.solver.bv_not(a)),
            "bvneg" => self.bv_unary(args, |s, a, _| s.solver.bv_neg(a)),
            "bvand" => self.bv_binary_fold(args, |s, a, b| s.solver.bv_and(a, b)),
            "bvor" => self.bv_binary_fold(args, |s, a, b| s.solver.bv_or(a, b)),
            "bvxor" => self.bv_binary_fold(args, |s, a, b| s.solver.bv_xor(a, b)),
            "bvnand" => self.bv_binary(args, |s, a, b| {
                let t = s.solver.bv_and(a, b);
                s.solver.bv_not(t)
            }),
            "bvnor" => self.bv_binary(args, |s, a, b| {
                let t = s.solver.bv_or(a, b);
                s.solver.bv_not(t)
            }),
            "bvxnor" => self.bv_binary(args, |s, a, b| {
                let t = s.solver.bv_xor(a, b);
                s.solver.bv_not(t)
            }),
            // `bvcomp`: 1-bit BV, equal-or-not. Standard SMT-LIB abbreviation
            // defined as `ite(a = b, #b1, #b0)`.
            "bvcomp" => {
                if args.len() != 2 {
                    return Err("bvcomp: expected 2 args".into());
                }
                let (a, wa) = self.eval_expr(&args[0])?.as_bv()?;
                let (b, wb) = self.eval_expr(&args[1])?.as_bv()?;
                if wa != wb {
                    return Err(format!("bvcomp width mismatch: {} vs {}", wa, wb));
                }
                let eq = self.solver.bv_eq(a, b);
                let one = self.solver.bv_const(1, 1);
                let zero = self.solver.bv_const(0, 1);
                Ok(TaggedTerm::Bv(self.solver.bv_ite(eq, one, zero), 1))
            }
            // `bvredand`: 1-bit BV, 1 iff all bits of `a` are 1.
            "bvredand" => {
                if args.len() != 1 {
                    return Err("bvredand: expected 1 arg".into());
                }
                let (a, w) = self.eval_expr(&args[0])?.as_bv()?;
                let all_ones = self.solver.bv_const(bv::mask(w), w);
                let eq = self.solver.bv_eq(a, all_ones);
                let one = self.solver.bv_const(1, 1);
                let zero = self.solver.bv_const(0, 1);
                Ok(TaggedTerm::Bv(self.solver.bv_ite(eq, one, zero), 1))
            }
            // `bvredor`: 1-bit BV, 1 iff any bit of `a` is 1.
            "bvredor" => {
                if args.len() != 1 {
                    return Err("bvredor: expected 1 arg".into());
                }
                let (a, w) = self.eval_expr(&args[0])?.as_bv()?;
                let zero_bv = self.solver.bv_const(0, w);
                let is_zero = self.solver.bv_eq(a, zero_bv);
                let zero1 = self.solver.bv_const(0, 1);
                let one1 = self.solver.bv_const(1, 1);
                Ok(TaggedTerm::Bv(self.solver.bv_ite(is_zero, zero1, one1), 1))
            }
            // CVC5 uses `bvite` as a synonym for `ite` when the branches are
            // bitvectors. Dispatch back to the regular `ite` handler.
            "bvite" => {
                if args.len() != 3 {
                    return Err("bvite: expected 3 args".into());
                }
                let c = self.eval_bool(Some(&args[0]))?;
                let t = self.eval_expr(&args[1])?;
                let e = self.eval_expr(&args[2])?;
                match (t, e) {
                    (TaggedTerm::Bv(tt, wt), TaggedTerm::Bv(ee, we)) if wt == we => {
                        Ok(TaggedTerm::Bv(self.solver.bv_ite(c, tt, ee), wt))
                    }
                    _ => Err("bvite: branches must be BV of same width".into()),
                }
            }

            // ---- BV arithmetic ----
            "bvadd" => self.bv_binary_fold(args, |s, a, b| s.solver.bv_add(a, b)),
            "bvsub" => self.bv_binary(args, |s, a, b| s.solver.bv_sub(a, b)),
            "bvmul" => self.bv_binary_fold(args, |s, a, b| s.solver.bv_mul(a, b)),
            "bvudiv" => self.bv_binary(args, |s, a, b| s.solver.bv_udiv(a, b)),
            "bvurem" => self.bv_binary(args, |s, a, b| s.solver.bv_urem(a, b)),
            "bvsdiv" => self.bv_binary(args, |s, a, b| s.solver.bv_sdiv(a, b)),
            "bvsrem" => self.bv_binary(args, |s, a, b| s.solver.bv_srem(a, b)),
            "bvsmod" => self.bv_binary(args, |s, a, b| s.solver.bv_smod(a, b)),

            // ---- BV shifts ----
            "bvshl" => self.bv_binary(args, |s, a, b| s.solver.bv_shl(a, b)),
            "bvlshr" => self.bv_binary(args, |s, a, b| s.solver.bv_lshr(a, b)),
            "bvashr" => self.bv_binary(args, |s, a, b| s.solver.bv_ashr(a, b)),

            // ---- BV comparisons ----
            "bvult" => self.bv_compare(args, |s, a, b| s.solver.bv_ult(a, b)),
            "bvule" => self.bv_compare(args, |s, a, b| s.solver.bv_ule(a, b)),
            "bvugt" => self.bv_compare(args, |s, a, b| s.solver.bv_ugt(a, b)),
            "bvuge" => self.bv_compare(args, |s, a, b| s.solver.bv_uge(a, b)),
            "bvslt" => self.bv_compare(args, |s, a, b| s.solver.bv_slt(a, b)),
            "bvsle" => self.bv_compare(args, |s, a, b| s.solver.bv_sle(a, b)),
            "bvsgt" => self.bv_compare(args, |s, a, b| s.solver.bv_sgt(a, b)),
            "bvsge" => self.bv_compare(args, |s, a, b| s.solver.bv_sge(a, b)),

            // ---- Overflow predicates (CVC5/Z3-style names) ----
            "bvuaddo" => self.bv_compare(args, |s, a, b| s.solver.bv_uadd_overflow(a, b)),
            "bvsaddo" => self.bv_compare(args, |s, a, b| s.solver.bv_sadd_overflow(a, b)),
            "bvusubo" => self.bv_compare(args, |s, a, b| s.solver.bv_usub_overflow(a, b)),
            "bvssubo" => self.bv_compare(args, |s, a, b| s.solver.bv_ssub_overflow(a, b)),
            "bvumulo" => self.bv_compare(args, |s, a, b| s.solver.bv_umul_overflow(a, b)),
            "bvsmulo" => self.bv_compare(args, |s, a, b| s.solver.bv_smul_overflow(a, b)),
            "bvsdivo" => self.bv_compare(args, |s, a, b| s.solver.bv_sdiv_overflow(a, b)),
            "bvnego" => {
                if args.len() != 1 {
                    return Err("bvnego: expected 1 arg".into());
                }
                let (a, _w) = self.eval_expr(&args[0])?.as_bv()?;
                Ok(TaggedTerm::Bool(self.solver.bv_neg_overflow(a)))
            }

            // ---- Structural ----
            "concat" => {
                if args.len() < 2 {
                    return Err("concat: need at least 2 args".into());
                }
                let (mut term, mut width) = {
                    let (t, w) = self.eval_expr(&args[0])?.as_bv()?;
                    (t, w)
                };
                for a in &args[1..] {
                    let (t2, w2) = self.eval_expr(a)?.as_bv()?;
                    term = self.solver.bv_concat(term, t2);
                    width += w2;
                }
                Ok(TaggedTerm::Bv(term, width))
            }

            other => Err(format!("unsupported operator: {}", other)),
        }
    }

    /// Parse `(_ <name> <args...>)` — either a BV constant literal or a
    /// pre-composed indexed operator that happens to appear bare.
    fn eval_indexed(&mut self, rest: &[SExpr]) -> Result<TaggedTerm, String> {
        // Literal form: `(_ bvN WIDTH)`.
        let name = atom(rest.first())?;
        if let Some(num_str) = name.strip_prefix("bv") {
            let w: u32 = atom(rest.get(1))?
                .parse()
                .map_err(|e| format!("bad width: {}", e))?;
            if w < 1 || w > bv::MAX_BV_WIDTH {
                return Err(format!("width {} out of range 1..={}", w, bv::MAX_BV_WIDTH));
            }
            if w <= 128 {
                let v: u128 = num_str
                    .parse()
                    .map_err(|e| format!("bad bv<n>: {}", e))?;
                return Ok(TaggedTerm::Bv(self.solver.bv_const(v, w), w));
            }
            // For wide `(_ bvN W)`, parse the decimal number into limbs.
            let limbs = parse_decimal_to_limbs(num_str, w)?;
            return Ok(TaggedTerm::Bv(self.solver.bv_const_wide(&limbs, w), w));
        }
        Err(format!("unsupported indexed form: {}", name))
    }

    /// `((_ extract h l) x)`, `((_ zero_extend n) x)`, `((_ sign_extend n) x)`.
    fn eval_indexed_apply(
        &mut self,
        head: &[SExpr],
        args: &[SExpr],
    ) -> Result<TaggedTerm, String> {
        if atom(head.first())? != "_" {
            return Err("expected `_` prefix in indexed application".into());
        }
        let name = atom(head.get(1))?;
        match name {
            "extract" => {
                let high: u32 = atom(head.get(2))?
                    .parse()
                    .map_err(|e| format!("extract high: {}", e))?;
                let low: u32 = atom(head.get(3))?
                    .parse()
                    .map_err(|e| format!("extract low: {}", e))?;
                let (x, _w) = self.eval_expr(args.first().ok_or("extract: no arg")?)?.as_bv()?;
                let r = self.solver.bv_extract(x, high, low);
                Ok(TaggedTerm::Bv(r, high - low + 1))
            }
            "zero_extend" => {
                let n: u32 = atom(head.get(2))?
                    .parse()
                    .map_err(|e| format!("zero_extend n: {}", e))?;
                let (x, w) = self.eval_expr(args.first().ok_or("zero_extend: no arg")?)?.as_bv()?;
                let r = self.solver.bv_zero_extend(x, n);
                Ok(TaggedTerm::Bv(r, w + n))
            }
            "sign_extend" => {
                let n: u32 = atom(head.get(2))?
                    .parse()
                    .map_err(|e| format!("sign_extend n: {}", e))?;
                let (x, w) = self.eval_expr(args.first().ok_or("sign_extend: no arg")?)?.as_bv()?;
                let r = self.solver.bv_sign_extend(x, n);
                Ok(TaggedTerm::Bv(r, w + n))
            }
            // `(_ repeat n) x` — concat `x` with itself n times.
            "repeat" => {
                let n: u32 = atom(head.get(2))?
                    .parse()
                    .map_err(|e| format!("repeat n: {}", e))?;
                if n == 0 {
                    return Err("repeat n must be ≥ 1".into());
                }
                let (x, w) = self.eval_expr(args.first().ok_or("repeat: no arg")?)?.as_bv()?;
                let mut result = x;
                for _ in 1..n {
                    result = self.solver.bv_concat(result, x);
                }
                Ok(TaggedTerm::Bv(result, w * n))
            }
            // `(_ rotate_left k) x` — circular shift left by k.
            // Encoded as `(x << k) | (x >>L (w - k))`, with k reduced mod w.
            "rotate_left" => {
                let k: u32 = atom(head.get(2))?
                    .parse()
                    .map_err(|e| format!("rotate_left k: {}", e))?;
                let (x, w) = self.eval_expr(args.first().ok_or("rotate_left: no arg")?)?.as_bv()?;
                let k_mod = k % w;
                if k_mod == 0 {
                    return Ok(TaggedTerm::Bv(x, w));
                }
                let shl_amt = self.solver.bv_const(k_mod as u128, w);
                let shr_amt = self.solver.bv_const((w - k_mod) as u128, w);
                let left = self.solver.bv_shl(x, shl_amt);
                let right = self.solver.bv_lshr(x, shr_amt);
                Ok(TaggedTerm::Bv(self.solver.bv_or(left, right), w))
            }
            // `(_ rotate_right k) x` — circular shift right by k.
            "rotate_right" => {
                let k: u32 = atom(head.get(2))?
                    .parse()
                    .map_err(|e| format!("rotate_right k: {}", e))?;
                let (x, w) = self.eval_expr(args.first().ok_or("rotate_right: no arg")?)?.as_bv()?;
                let k_mod = k % w;
                if k_mod == 0 {
                    return Ok(TaggedTerm::Bv(x, w));
                }
                let shr_amt = self.solver.bv_const(k_mod as u128, w);
                let shl_amt = self.solver.bv_const((w - k_mod) as u128, w);
                let left = self.solver.bv_lshr(x, shr_amt);
                let right = self.solver.bv_shl(x, shl_amt);
                Ok(TaggedTerm::Bv(self.solver.bv_or(left, right), w))
            }
            other => Err(format!("unsupported indexed op: {}", other)),
        }
    }

    // ---------- small helpers ----------

    fn eval_bool(&mut self, expr: Option<&SExpr>) -> Result<BoolTerm, String> {
        let e = expr.ok_or("missing argument")?;
        self.eval_expr(e)?.as_bool()
    }

    fn eval_bool_args(&mut self, args: &[SExpr]) -> Result<Vec<BoolTerm>, String> {
        args.iter().map(|a| self.eval_expr(a)?.as_bool()).collect()
    }

    fn bool_fold<F>(
        &mut self,
        args: &[SExpr],
        mut f: F,
        identity: bool,
    ) -> Result<TaggedTerm, String>
    where
        F: FnMut(&mut Self, BoolTerm, BoolTerm) -> BoolTerm,
    {
        let vs = self.eval_bool_args(args)?;
        let mut it = vs.into_iter();
        let start = it.next().unwrap_or_else(|| {
            if identity {
                self.solver.bool_true()
            } else {
                self.solver.bool_false()
            }
        });
        let mut acc = start;
        for v in it {
            acc = f(self, acc, v);
        }
        Ok(TaggedTerm::Bool(acc))
    }

    fn bv_binary_fold<F>(
        &mut self,
        args: &[SExpr],
        mut f: F,
    ) -> Result<TaggedTerm, String>
    where
        F: FnMut(&mut Self, BvTerm, BvTerm) -> BvTerm,
    {
        if args.is_empty() {
            return Err("bv op: no args".into());
        }
        let (first, w) = self.eval_expr(&args[0])?.as_bv()?;
        let mut acc = first;
        for a in &args[1..] {
            let (t, w2) = self.eval_expr(a)?.as_bv()?;
            if w != w2 {
                return Err(format!("width mismatch: {} vs {}", w, w2));
            }
            acc = f(self, acc, t);
        }
        Ok(TaggedTerm::Bv(acc, w))
    }

    fn bv_binary<F>(&mut self, args: &[SExpr], mut f: F) -> Result<TaggedTerm, String>
    where
        F: FnMut(&mut Self, BvTerm, BvTerm) -> BvTerm,
    {
        if args.len() != 2 {
            return Err(format!("binary op: expected 2 args, got {}", args.len()));
        }
        let (a, wa) = self.eval_expr(&args[0])?.as_bv()?;
        let (b, wb) = self.eval_expr(&args[1])?.as_bv()?;
        if wa != wb {
            return Err(format!("width mismatch: {} vs {}", wa, wb));
        }
        Ok(TaggedTerm::Bv(f(self, a, b), wa))
    }

    fn bv_unary<F>(&mut self, args: &[SExpr], mut f: F) -> Result<TaggedTerm, String>
    where
        F: FnMut(&mut Self, BvTerm, u32) -> BvTerm,
    {
        if args.len() != 1 {
            return Err(format!("unary op: expected 1 arg, got {}", args.len()));
        }
        let (a, w) = self.eval_expr(&args[0])?.as_bv()?;
        Ok(TaggedTerm::Bv(f(self, a, w), w))
    }

    fn bv_compare<F>(&mut self, args: &[SExpr], mut f: F) -> Result<TaggedTerm, String>
    where
        F: FnMut(&mut Self, BvTerm, BvTerm) -> BoolTerm,
    {
        if args.len() != 2 {
            return Err(format!("compare op: expected 2 args, got {}", args.len()));
        }
        let (a, wa) = self.eval_expr(&args[0])?.as_bv()?;
        let (b, wb) = self.eval_expr(&args[1])?.as_bv()?;
        if wa != wb {
            return Err(format!("width mismatch: {} vs {}", wa, wb));
        }
        Ok(TaggedTerm::Bool(f(self, a, b)))
    }

    /// Core assert logic: handle one body expression, recursing into `(and ...)`
    /// so nested conjuncts each get their own shot at substitution / aliasing.
    /// A lot of symbolic-execution output bundles hundreds of independent
    /// conditions under one big `(assert (and ...))` — decomposing exposes
    /// substitutable equalities that would otherwise be buried.
    fn assert_body(&mut self, body_expr: &SExpr) -> Result<(), String> {
        // `(and P1 P2 ... Pn)` at the top level: recurse on each conjunct.
        // `(and)` = true, `(and P)` = P — both handled by the general loop.
        if let SExpr::List(xs) = body_expr {
            if let Some(SExpr::Atom(head)) = xs.first() {
                if head == "and" {
                    for arg in &xs[1..] {
                        self.assert_body(arg)?;
                    }
                    return Ok(());
                }
                // `(not (or P1 P2 ...))` = `(and (not P1) (not P2) ...)` —
                // De Morgan gives us another decomposition opportunity.
                if head == "not" && xs.len() == 2 {
                    if let SExpr::List(inner) = &xs[1] {
                        if let Some(SExpr::Atom(ih)) = inner.first() {
                            if ih == "or" {
                                for arg in &inner[1..] {
                                    let negated = SExpr::List(vec![
                                        SExpr::Atom("not".to_string()),
                                        arg.clone(),
                                    ]);
                                    self.assert_body(&negated)?;
                                }
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
        // Top-level equality substitution.
        if let Some((name, rhs_expr)) = self.match_substitution(body_expr) {
            let name = name.to_string();
            let rhs = self.eval_expr(rhs_expr)?;
            self.symbols.insert(name.clone(), rhs);
            self.declared.remove(&name);
            return Ok(());
        }
        // BV1-as-Bool substitution: `(assert (= (= X 1bv1) rhs_bool))` or
        // `(= (= X 0bv1) rhs_bool)` (either order). When X is a fresh BV1
        // atom, the assertion says `(X == k) iff rhs_bool`. We can rebind
        // X := ite(rhs_bool, k, ¬k) in the symbol table: any future use of
        // X as a BV bitblasts the ite cleanly, and any future `(= X 1bv1)`
        // collapses to `rhs_bool` (or `¬rhs_bool`) through the existing
        // `bv_eq(const, ite(c, k, ¬k))` fold. Common in Spear / catchconv
        // traces where 1-bit BV vars are used as Bool flags.
        if let Some((name, k, rhs_expr)) = self.match_bv1_as_bool_subst(body_expr) {
            let name = name.to_string();
            let rhs_bool = self.eval_expr(rhs_expr)?.as_bool()?;
            let (t_br, e_br) = if k {
                (
                    self.solver.bv_const(1, 1),
                    self.solver.bv_const(0, 1),
                )
            } else {
                (
                    self.solver.bv_const(0, 1),
                    self.solver.bv_const(1, 1),
                )
            };
            let ite_term = self.solver.bv_ite(rhs_bool, t_br, e_br);
            self.symbols.insert(name.clone(), TaggedTerm::Bv(ite_term, 1));
            self.declared.remove(&name);
            return Ok(());
        }
        // Union-find equality over atom-vs-atom pairs.
        if let Some((lt, rt)) = self.match_var_var_equality(body_expr) {
            let aliased = match (lt, rt) {
                (TaggedTerm::Bv(a, _), TaggedTerm::Bv(b, _)) => {
                    self.solver.alias_bv_vars(a, b)
                }
                (TaggedTerm::Bool(a), TaggedTerm::Bool(b)) => {
                    self.solver.alias_bool_vars(a, b)
                }
                _ => false,
            };
            if aliased {
                return Ok(());
            }
        }
        let body = self.eval_expr(body_expr)?;
        let b = body.as_bool()?;
        self.solver.assert(b);
        Ok(())
    }

    /// Match `(= (= X k_bv1) rhs)` or `(= rhs (= X k_bv1))` where X is a
    /// declare-fun BV1 atom that hasn't been resolved yet, and k_bv1 is
    /// either `(_ bv0 1)` or `(_ bv1 1)`. Returns `(X_name, k_is_one, rhs_expr)`.
    fn match_bv1_as_bool_subst<'e>(
        &self,
        expr: &'e SExpr,
    ) -> Option<(&'e str, bool, &'e SExpr)> {
        let xs = match expr {
            SExpr::List(xs) => xs,
            _ => return None,
        };
        if xs.len() != 3 {
            return None;
        }
        match &xs[0] {
            SExpr::Atom(s) if s == "=" => {}
            _ => return None,
        }
        // Try both (lhs = inner-eq, rhs = rhs_bool) and the reversed order.
        self.try_bv1_subst_side(&xs[1], &xs[2])
            .or_else(|| self.try_bv1_subst_side(&xs[2], &xs[1]))
    }

    fn try_bv1_subst_side<'e>(
        &self,
        inner_eq: &'e SExpr,
        rhs: &'e SExpr,
    ) -> Option<(&'e str, bool, &'e SExpr)> {
        // `inner_eq` must be `(= X <bv1 const>)` or `(= <bv1 const> X)`.
        let inner_xs = match inner_eq {
            SExpr::List(xs) => xs,
            _ => return None,
        };
        if inner_xs.len() != 3 {
            return None;
        }
        match &inner_xs[0] {
            SExpr::Atom(s) if s == "=" => {}
            _ => return None,
        }
        let match_const = |e: &SExpr| -> Option<bool> {
            if let SExpr::Atom(s) = e {
                if s == "#b0" {
                    return Some(false);
                }
                if s == "#b1" {
                    return Some(true);
                }
            }
            None
        };
        // Try atom-on-left, const-on-right.
        let (name_expr, const_expr) =
            match (match_const(&inner_xs[2]), match_const(&inner_xs[1])) {
                (Some(_), _) => (&inner_xs[1], &inner_xs[2]),
                (_, Some(_)) => (&inner_xs[2], &inner_xs[1]),
                _ => {
                    // Check the `(_ bv1 1)` / `(_ bv0 1)` form too.
                    let is_bv1_const = |e: &SExpr| -> Option<bool> {
                        if let SExpr::List(xs) = e {
                            if xs.len() == 3 {
                                if let (
                                    Some(SExpr::Atom(underscore)),
                                    Some(SExpr::Atom(numname)),
                                    Some(SExpr::Atom(w)),
                                ) = (xs.first(), xs.get(1), xs.get(2))
                                {
                                    if underscore == "_" && w == "1" {
                                        if let Some(num) = numname.strip_prefix("bv") {
                                            if num == "0" {
                                                return Some(false);
                                            }
                                            if num == "1" {
                                                return Some(true);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        None
                    };
                    let r1 = is_bv1_const(&inner_xs[2]);
                    let r2 = is_bv1_const(&inner_xs[1]);
                    match (r1, r2) {
                        (Some(_), _) => (&inner_xs[1], &inner_xs[2]),
                        (_, Some(_)) => (&inner_xs[2], &inner_xs[1]),
                        _ => return None,
                    }
                }
            };
        let name = match name_expr {
            SExpr::Atom(s) => s.as_str(),
            _ => return None,
        };
        // Shadowing check — let binds must not capture `name`.
        if self.let_bindings.get(name).map_or(false, |s| !s.is_empty()) {
            return None;
        }
        if !self.declared.contains(name) {
            return None;
        }
        // Must be a BV1 atom.
        let tt = *self.symbols.get(name)?;
        let w = match tt {
            TaggedTerm::Bv(_, w) => w,
            _ => return None,
        };
        if w != 1 {
            return None;
        }
        // Determine whether the const is 1 or 0.
        let k_is_one = {
            if let SExpr::Atom(s) = const_expr {
                if s == "#b0" {
                    false
                } else if s == "#b1" {
                    true
                } else {
                    return None;
                }
            } else if let SExpr::List(xs) = const_expr {
                let numname = match xs.get(1) {
                    Some(SExpr::Atom(s)) => s.as_str(),
                    _ => return None,
                };
                let num = numname.strip_prefix("bv")?;
                match num {
                    "0" => false,
                    "1" => true,
                    _ => return None,
                }
            } else {
                return None;
            }
        };
        // rhs must be Bool-sorted; we defer the actual sort check to
        // `eval_expr().as_bool()?` at the call site.
        Some((name, k_is_one, rhs))
    }

    /// If `expr` is `(= X Y)` with X and Y both atoms pointing at globals
    /// whose bindings are bare `BvVar` / `BoolVar` (not expressions), return
    /// those two TaggedTerms for union-find aliasing. Otherwise None.
    fn match_var_var_equality(&self, expr: &SExpr) -> Option<(TaggedTerm, TaggedTerm)> {
        let xs = match expr {
            SExpr::List(xs) => xs,
            _ => return None,
        };
        if xs.len() != 3 {
            return None;
        }
        match &xs[0] {
            SExpr::Atom(s) if s == "=" => {}
            _ => return None,
        }
        let name_of = |e: &SExpr| -> Option<String> {
            match e {
                SExpr::Atom(s) => Some(s.clone()),
                _ => None,
            }
        };
        let lname = name_of(&xs[1])?;
        let rname = name_of(&xs[2])?;
        // Names shadowed by a `let` binding aren't eligible.
        let shadowed = |m: &HashMap<String, Vec<TaggedTerm>>, n: &str| {
            m.get(n).map_or(false, |s| !s.is_empty())
        };
        if shadowed(&self.let_bindings, &lname) || shadowed(&self.let_bindings, &rname) {
            return None;
        }
        let &lt = self.symbols.get(&lname)?;
        let &rt = self.symbols.get(&rname)?;
        // Only bare var-vs-var pairs qualify. Expression-backed bindings
        // (from earlier substitutions) have already been rewritten; unioning
        // would lose that context.
        let is_leaf_var = |t: TaggedTerm| -> bool {
            match t {
                TaggedTerm::Bv(bt, _) => matches!(
                    self.solver.bv_op_of(bt),
                    crate::bv::BvOp::Var(_)
                ),
                TaggedTerm::Bool(bt) => matches!(
                    self.solver.bool_op_of(bt),
                    crate::bv::BoolOp::Var(_)
                ),
            }
        };
        if !is_leaf_var(lt) || !is_leaf_var(rt) {
            return None;
        }
        // Sorts must match. Our eval_expr would catch mismatched `=` anyway,
        // but matching on TaggedTerm keeps the alias call well-typed.
        if lt.sort() != rt.sort() {
            return None;
        }
        Some((lt, rt))
    }

    /// If `expr` is `(= X rhs)` or `(= rhs X)` with X a declare-fun atom
    /// whose fresh-var binding has not been resolved anywhere yet, return
    /// (X, rhs). Otherwise None.
    fn match_substitution<'e>(&self, expr: &'e SExpr) -> Option<(&'e str, &'e SExpr)> {
        let xs = match expr {
            SExpr::List(xs) => xs,
            _ => return None,
        };
        if xs.len() != 3 {
            return None;
        }
        let head = match &xs[0] {
            SExpr::Atom(s) if s == "=" => s,
            _ => return None,
        };
        let _ = head;
        let try_side = |name_expr: &'e SExpr, other: &'e SExpr| -> Option<(&'e str, &'e SExpr)> {
            let name = match name_expr {
                SExpr::Atom(s) => s.as_str(),
                _ => return None,
            };
            // Skip built-in literals and names bound by an outer `let` —
            // let-scoped names never enter `self.declared`, but a name might
            // coincidentally shadow a declared symbol; in that case we must
            // not substitute, because the let wins.
            if self.let_bindings.get(name).map_or(false, |s| !s.is_empty()) {
                return None;
            }
            if !self.declared.contains(name) {
                return None;
            }
            Some((name, other))
        };
        try_side(&xs[1], &xs[2]).or_else(|| try_side(&xs[2], &xs[1]))
    }
}

fn atom(e: Option<&SExpr>) -> Result<&str, String> {
    match e {
        Some(SExpr::Atom(s)) => Ok(s),
        _ => Err("expected atom".into()),
    }
}

/// Recognise the `(! <expr> :named <sym> ...)` wrapper used to attach a
/// name to an assertion for UNSAT core extraction. Returns the inner
/// expression and the name when the pattern matches.
fn extract_named(expr: &SExpr) -> Option<(&SExpr, &str)> {
    let xs = match expr {
        SExpr::List(xs) => xs,
        _ => return None,
    };
    match xs.first()? {
        SExpr::Atom(s) if s == "!" => {}
        _ => return None,
    }
    let inner = xs.get(1)?;
    let mut i = 2;
    while i < xs.len() {
        if let SExpr::Atom(k) = &xs[i] {
            if k == ":named" {
                if let SExpr::Atom(name) = xs.get(i + 1)? {
                    return Some((inner, name.as_str()));
                }
            }
        }
        // Skip over any other `:key value` attribute pair.
        i += 2;
    }
    None
}

fn parse_sort(e: &SExpr) -> Result<Sort, String> {
    match e {
        SExpr::Atom(s) if s == "Bool" => Ok(Sort::Bool),
        SExpr::List(xs) if xs.len() == 3 => {
            // (_ BitVec n)
            if atom(xs.first())? == "_" && atom(xs.get(1))? == "BitVec" {
                let n: u32 = atom(xs.get(2))?
                    .parse()
                    .map_err(|e| format!("bad BitVec width: {}", e))?;
                Ok(Sort::Bv(n))
            } else {
                Err(format!("unknown sort: {:?}", e))
            }
        }
        _ => Err(format!("unknown sort: {:?}", e)),
    }
}

fn format_sexpr(e: &SExpr) -> String {
    match e {
        SExpr::Atom(s) => s.clone(),
        SExpr::List(xs) => {
            let inner: Vec<String> = xs.iter().map(format_sexpr).collect();
            format!("({})", inner.join(" "))
        }
    }
}

// ==================== S-expression parser ====================

#[derive(Debug, Clone)]
enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Parser { src, pos: 0 }
    }

    fn next_sexpr(&mut self) -> Result<Option<SExpr>, String> {
        self.skip_ws_and_comments();
        if self.pos >= self.src.len() {
            return Ok(None);
        }
        Ok(Some(self.parse_one()?))
    }

    fn parse_one(&mut self) -> Result<SExpr, String> {
        self.skip_ws_and_comments();
        let b = self.peek_byte().ok_or("unexpected end of input")?;
        if b == b'(' {
            self.pos += 1;
            let mut items = Vec::new();
            loop {
                self.skip_ws_and_comments();
                match self.peek_byte() {
                    None => return Err("unclosed list".into()),
                    Some(b')') => {
                        self.pos += 1;
                        break;
                    }
                    _ => items.push(self.parse_one()?),
                }
            }
            Ok(SExpr::List(items))
        } else if b == b'|' {
            // Quoted symbol: |foo bar|.
            self.pos += 1;
            let start = self.pos;
            while let Some(c) = self.peek_byte() {
                if c == b'|' {
                    let s = self.src[start..self.pos].to_string();
                    self.pos += 1;
                    return Ok(SExpr::Atom(s));
                }
                self.pos += 1;
            }
            Err("unclosed |symbol|".into())
        } else if b == b'"' {
            // String literal: not used by our grammar but skip it gracefully.
            self.pos += 1;
            let start = self.pos;
            while let Some(c) = self.peek_byte() {
                self.pos += 1;
                if c == b'"' {
                    return Ok(SExpr::Atom(self.src[start..self.pos - 1].to_string()));
                }
            }
            Err("unclosed string literal".into())
        } else {
            // Atom: consume until whitespace, paren, or comment.
            let start = self.pos;
            while let Some(c) = self.peek_byte() {
                if c.is_ascii_whitespace() || c == b'(' || c == b')' || c == b';' {
                    break;
                }
                self.pos += 1;
            }
            if start == self.pos {
                return Err(format!("unexpected byte: {:?}", b as char));
            }
            Ok(SExpr::Atom(self.src[start..self.pos].to_string()))
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            match self.peek_byte() {
                Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                Some(b';') => {
                    // Comment to end of line.
                    while let Some(c) = self.peek_byte() {
                        self.pos += 1;
                        if c == b'\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }
}

/// Render a BV model value. For widths ≤ 128 we emit the familiar decimal
/// `(_ bvN W)` form; for wider values we emit a hex `#x…` literal padded to
/// `ceil(w/4)` nibbles so the whole value round-trips through the parser.
fn format_bv_value(limbs: &[u64], w: u32) -> String {
    if w <= 128 {
        let mut v: u128 = 0;
        if !limbs.is_empty() {
            v |= limbs[0] as u128;
        }
        if limbs.len() >= 2 {
            v |= (limbs[1] as u128) << 64;
        }
        if w < 128 {
            v &= (1u128 << w) - 1;
        }
        return format!("(_ bv{} {})", v, w);
    }
    // Hex rendering, MSB-first, padded to (w + 3) / 4 nibbles.
    let total_nibs = ((w + 3) / 4) as usize;
    let mut s = String::with_capacity(total_nibs + 2);
    s.push_str("#x");
    for i in (0..total_nibs).rev() {
        let bit = i * 4;
        let limb = limbs.get(bit / 64).copied().unwrap_or(0);
        let nib = (limb >> (bit % 64)) & 0xF;
        s.push(match nib {
            0..=9 => (b'0' + nib as u8) as char,
            10..=15 => (b'a' + (nib as u8 - 10)) as char,
            _ => unreachable!(),
        });
    }
    s
}

// ==================== Wide-literal parsing helpers ====================

fn limbs_for_width(w: u32) -> usize {
    ((w as usize) + 63) / 64
}

/// Parse a binary string (MSB-first, no `#b` prefix) into little-endian limbs.
fn parse_binary_to_limbs(s: &str) -> Result<Vec<u64>, String> {
    let w = s.len();
    if w == 0 {
        return Err("empty binary literal".into());
    }
    let mut limbs = vec![0u64; limbs_for_width(w as u32)];
    for (i, c) in s.bytes().enumerate() {
        let bit = match c {
            b'0' => 0u64,
            b'1' => 1u64,
            _ => return Err(format!("bad binary digit: {:?}", c as char)),
        };
        // MSB is s[0]; bit position in value is (w - 1 - i).
        let pos = w - 1 - i;
        limbs[pos / 64] |= bit << (pos % 64);
    }
    Ok(limbs)
}

/// Parse a hex string (MSB-first, no `#x` prefix) into little-endian limbs.
fn parse_hex_to_limbs(s: &str) -> Result<Vec<u64>, String> {
    let nibs = s.len();
    if nibs == 0 {
        return Err("empty hex literal".into());
    }
    let w = (nibs as u32) * 4;
    let mut limbs = vec![0u64; limbs_for_width(w)];
    for (i, c) in s.bytes().enumerate() {
        let nib = match c {
            b'0'..=b'9' => (c - b'0') as u64,
            b'a'..=b'f' => (c - b'a' + 10) as u64,
            b'A'..=b'F' => (c - b'A' + 10) as u64,
            _ => return Err(format!("bad hex digit: {:?}", c as char)),
        };
        // MSB nibble is s[0]; nibble index from LSB is (nibs - 1 - i).
        let pos = (nibs - 1 - i) * 4;
        limbs[pos / 64] |= nib << (pos % 64);
    }
    Ok(limbs)
}

/// Parse a decimal string into little-endian u64 limbs sized for `w` bits.
/// The result is masked to `w` bits.
fn parse_decimal_to_limbs(s: &str, w: u32) -> Result<Vec<u64>, String> {
    if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
        return Err(format!("bad decimal literal: {}", s));
    }
    let nlimbs = limbs_for_width(w);
    let mut limbs = vec![0u64; nlimbs];
    // Schoolbook multiply-add: acc = acc * 10 + d, over little-endian limbs.
    for c in s.bytes() {
        let d = (c - b'0') as u64;
        let mut carry: u128 = d as u128;
        for limb in limbs.iter_mut() {
            let v = (*limb as u128) * 10u128 + carry;
            *limb = v as u64;
            carry = v >> 64;
        }
        if carry != 0 {
            return Err(format!("decimal literal {} overflows {} bits", s, w));
        }
    }
    // Mask top limb to the target width.
    if w % 64 != 0 {
        let top = nlimbs - 1;
        let mask = (1u64 << (w % 64)) - 1;
        let kept = limbs[top] & mask;
        if kept != limbs[top] {
            return Err(format!("decimal literal {} overflows {} bits", s, w));
        }
        limbs[top] = kept;
    }
    Ok(limbs)
}
