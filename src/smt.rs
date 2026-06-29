//! SMT solver for quantifier-free bitvector logic (QF_BV).
//!
//! Strategy: eager bitblasting. Every BV term of width N becomes N SAT
//! literals (one per bit, LSB-first). Every Boolean term becomes one SAT
//! literal. Gates are encoded via the standard Tseitin translation, which
//! produces mostly 2- and 3-literal clauses — exactly what the SAT core is
//! optimized for.
//!
//! Supported BV ops: not, and, or, xor, add, sub, mul, udiv, urem, shl,
//! lshr, ashr, extract, concat, zero-extend, sign-extend, ite.
//! Comparisons: eq/ne, ult/ule/ugt/uge (unsigned), slt/sle/sgt/sge (signed).

use rustc_hash::FxHashMap as HashMap;

use crate::bv::{BoolOp, BoolTerm, BvContext, BvOp, BvTerm, mask};
use crate::lit::{LBool, Lit, Var};
use crate::solver::{SolveResult, Solver};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SmtResult {
    Sat,
    Unsat,
}

/// What a freshly-allocated SAT variable represents in the BV layer. Recorded
/// at allocation time by the bitblaster so downstream consumers (e.g. a
/// future word-level branching heuristic or an ITE-aware propagator) can
/// reason about SAT variables in BV terms.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum VarOrigin {
    /// Bit `bit` of bitvector term `term` (an input variable or a result
    /// slice handed out by `bitblast_bv`).
    BvBit { term: BvTerm, bit: u32 },
    /// The sole SAT literal representing a Bool-sorted term.
    Bool { term: BoolTerm },
    /// The pinned always-true SAT literal.
    TrueLit,
    /// Output of a Tseitin-encoded gate. `term` is the BV term being
    /// bitblasted at the time, if any — useful for grouping aux bits back
    /// to their source expression.
    GateOut { gate: GateKind, term: Option<BvTerm> },
    /// Activation literal for a `push` scope or a `:named` assertion.
    Activation,
    /// Unclassified fallback — shouldn't appear in finished bitblast output.
    Unknown,
}

/// Which gate produced a SAT variable. Kept deliberately small so that
/// downstream code can `match` exhaustively.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum GateKind {
    And,
    Or,
    Xor,
    Ite,
    /// Sum output of a full adder. Used inside ripple-carry chains.
    FaSum,
    /// Carry-out of a full adder.
    FaCarry,
}

/// Recorded ITE gate: semantically `o ↔ (sel ∧ t) ∨ (¬sel ∧ e)`. Emitted by
/// every `mk_mux` invocation; stored in `SmtSolver::ite_gates` so that
/// future ITE-aware propagation / branching can look them up cheaply.
/// Post-solve statistics, comparable to what mature solvers print via their
/// `:statistics` interface. Returned by [`SmtSolver::sat_stats`].
#[derive(Copy, Clone, Debug)]
pub struct SmtSolverStats {
    /// Total SAT variables allocated (inputs + Tseitin gate outputs).
    pub sat_vars: usize,
    /// Total clauses in the DB, including learned.
    pub sat_clauses: usize,
    /// Cumulative conflicts across all `solve*` calls this session.
    pub conflicts: u64,
    /// Cumulative decisions across all `solve*` calls this session.
    pub decisions: u64,
    pub restarts: u64,
    pub learned: u64,
    pub propagations: u64,
    pub bv_aliased: usize,
    pub bool_aliased: usize,
    pub bv_var_total: usize,
    pub bv_nodes_total: usize,
    pub bv_vars_bitblasted: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct IteGate {
    pub sel: Lit,
    pub t: Lit,
    pub e: Lit,
    pub o: Lit,
    /// Source BV term being bitblasted when this gate was emitted, if we
    /// were inside `bitblast_bv`. Lets callers group the N per-bit ITE gates
    /// of a width-N BV ITE back to the single source `BvOp::Ite` node.
    pub source_term: Option<BvTerm>,
}

pub struct SmtSolver {
    pub ctx: BvContext,
    sat: Solver,

    // Bitblast caches: each BV/Bool term is translated exactly once and the
    // result reused on subsequent references. Critical for shared DAGs —
    // without this, a shared subterm could be re-encoded combinatorially
    // many times.
    bv_cache: HashMap<BvTerm, Vec<Lit>>,
    bool_cache: HashMap<BoolTerm, Lit>,

    // SAT-literal encoding for symbolic variables. Populated lazily on first
    // use so we don't allocate SAT vars for unused symbols.
    bv_var_lits: HashMap<u32, Vec<Lit>>,
    bool_var_lits: HashMap<u32, Lit>,

    // Union-find over BV / Bool variable ids. After `alias_bv_vars(x, y)`,
    // both BvVar(x_id) and BvVar(y_id) resolve to the same SAT literals, so
    // `(= x y)` becomes a free no-op. Populated lazily from the SMT-LIB
    // layer when `(assert (= X Y))` is seen with X and Y both declared vars.
    bv_var_parent: Vec<u32>,
    bool_var_parent: Vec<u32>,

    // AIG-style gate caches: structurally identical gates share a single
    // output literal and one set of SAT clauses. On a symbex workload this
    // roughly halves the SAT-variable count — our bitblaster previously
    // emitted a fresh gate for every call regardless of whether the same
    // Tseitin fan-in had been encoded before.
    // Keys are canonicalized: (min, max) for commutative ops.
    and_cache: HashMap<(Lit, Lit), Lit>,
    or_cache: HashMap<(Lit, Lit), Lit>,
    xor_cache: HashMap<(Lit, Lit), Lit>,
    mux_cache: HashMap<(Lit, Lit, Lit), Lit>,

    // Reusable single SAT lit pinned to true — used as the `true` constant,
    // and its negation is the `false` constant.
    true_lit: Option<Lit>,

    // Stack of "activation literals" — one per open `push` scope. Every
    // assertion made inside scope `k` is added as `(¬act_k ∨ clause-for-
    // assertion)`. On `pop(k)` we force `act_k = false` via a unit clause,
    // which makes those guarded clauses vacuously satisfied forever. Level 0
    // (no push) uses unguarded clauses, same as before.
    activation_stack: Vec<Lit>,

    // Deferred-assertion queues, one per scope level. Index 0 is the
    // outermost (unguarded) level; index k ≥ 1 matches `activation_stack[k-1]`.
    // Assertions are stashed here rather than being bitblasted eagerly — this
    // lets preprocessing passes (variable aliasing, rewrite propagation) run
    // over the full assertion set before any SAT encoding is committed.
    // Flushed in `flush_pending()`, called at every `solve*`.
    pending: Vec<Vec<BoolTerm>>,

    // Named assertions: `(name, control_lit)` pairs in insertion order.
    // The assertion lives as a clause `(¬control ∨ phi)` so the SAT solver
    // can blame the control if the overall formula is UNSAT. We assume
    // every control is true during solve; after UNSAT, the unsat core
    // points at which names participated.
    named_controls: Vec<(String, Lit)>,

    // Last solve result if the formula hasn't been modified since. Reading
    // model values before any `solve*` or after a state-changing command
    // (assert, push, pop) is meaningless per SMT-LIB — `has_model()` lets
    // callers check first.
    last_result: Option<SmtResult>,

    // --- Metadata layer -----------------------------------------------------
    // Parallel to the SAT variable table: `var_origin[i]` records what SAT
    // variable `i` is for — a BV input bit, a gate output, an activation
    // literal, etc. Populated at allocation time.
    var_origin: Vec<VarOrigin>,
    // While bitblasting a given BV term, this holds its handle so that aux
    // vars allocated during the translation can be tagged with the enclosing
    // term. Push/pop via save-and-restore in `bitblast_bv`.
    current_bv_ctx: Option<BvTerm>,

    // Every ITE gate emitted by `mk_mux`, in insertion order.
    ite_gates: Vec<IteGate>,
    // Reverse index: for each SAT literal that's an ITE output, which gate
    // (by index into `ite_gates`) produced it.
    ite_out_to_gate: HashMap<Lit, usize>,

    // When true (the default), each ITE gate emitted bumps its `sel`
    // variable's VSIDS activity. This steers the SAT solver toward
    // branching on selectors first — a single decision on `sel` resolves
    // the whole ITE subtree, which is a huge win on symex memory reads
    // that are encoded as deep ITE chains over the address variable.
    ite_branching_hints: bool,
}

impl SmtSolver {
    pub fn new() -> Self {
        SmtSolver {
            ctx: BvContext::new(),
            sat: Solver::new(),
            bv_cache: HashMap::default(),
            bool_cache: HashMap::default(),
            bv_var_lits: HashMap::default(),
            bool_var_lits: HashMap::default(),
            bv_var_parent: Vec::new(),
            bool_var_parent: Vec::new(),
            and_cache: HashMap::default(),
            or_cache: HashMap::default(),
            xor_cache: HashMap::default(),
            mux_cache: HashMap::default(),
            true_lit: None,
            activation_stack: Vec::new(),
            pending: vec![Vec::new()],
            named_controls: Vec::new(),
            last_result: None,
            var_origin: Vec::new(),
            current_bv_ctx: None,
            ite_gates: Vec::new(),
            ite_out_to_gate: HashMap::default(),
            ite_branching_hints: true,
        }
    }

    /// Enable or disable the ITE-aware branching hint. On (the default)
    /// means every `mk_mux` call boosts its selector's VSIDS activity;
    /// off disables that boost entirely. Useful to benchmark the impact
    /// of the heuristic on a given workload.
    pub fn set_ite_branching_hints(&mut self, on: bool) {
        self.ite_branching_hints = on;
    }

    // ---------- Delegating term builders ----------

    pub fn bv_var(&mut self, width: u32) -> BvTerm { self.ctx.bv_var(width) }
    pub fn bv_op_of(&self, t: BvTerm) -> BvOp { self.ctx.bv_nodes[t.0 as usize].op }
    pub fn bool_op_of(&self, t: BoolTerm) -> BoolOp { self.ctx.bool_nodes[t.0 as usize] }
    pub fn bv_const(&mut self, value: u128, width: u32) -> BvTerm { self.ctx.bv_const(value, width) }
    pub fn bv_const_wide(&mut self, limbs: &[u64], width: u32) -> BvTerm {
        self.ctx.bv_const_wide(limbs, width)
    }
    /// Returns the inline-stored constant value if `t` is a folded constant
    /// of width ≤ 128, else `None`. Preferred over the panicking
    /// `bv_const_value*` family when the caller doesn't already know `t`
    /// is constant — e.g. symbolic-execution `to_u64`-style concretization
    /// checks ("did this term fold?"). Constants wider than 128 bits
    /// return `None`; reach for `bv_const_value_limbs` if you need them.
    pub fn try_bv_const_value(&self, t: BvTerm) -> Option<u128> {
        self.ctx.try_bv_const_value(t)
    }
    pub fn bv_width(&self, t: BvTerm) -> u32 { self.ctx.width_of(t) }

    pub fn bv_not(&mut self, x: BvTerm) -> BvTerm { self.ctx.bv_not(x) }
    pub fn bv_and(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_and(x, y) }
    pub fn bv_or(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_or(x, y) }
    pub fn bv_xor(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_xor(x, y) }

    pub fn bv_add(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_add(x, y) }
    pub fn bv_sub(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_sub(x, y) }
    pub fn bv_neg(&mut self, x: BvTerm) -> BvTerm { self.ctx.bv_neg(x) }
    /// Population count of `x` (number of 1 bits). Result width = input width.
    pub fn bv_popcount(&mut self, x: BvTerm) -> BvTerm { self.ctx.bv_popcount(x) }
    /// Count leading zeros — `clz(0) = width`. Result width = input width.
    pub fn bv_clz(&mut self, x: BvTerm) -> BvTerm { self.ctx.bv_clz(x) }
    /// Count trailing zeros — `ctz(0) = width`. Result width = input width.
    pub fn bv_ctz(&mut self, x: BvTerm) -> BvTerm { self.ctx.bv_ctz(x) }
    /// Rotate `x` left by a symbolic `amount` (modulo width). Both operands
    /// must have the same width. Falls through to the constant builder when
    /// `amount` is a constant.
    pub fn bv_rotate_left_dyn(&mut self, x: BvTerm, amount: BvTerm) -> BvTerm {
        self.ctx.bv_rotate_left_dyn(x, amount)
    }
    /// Mirror of [`Self::bv_rotate_left_dyn`].
    pub fn bv_rotate_right_dyn(&mut self, x: BvTerm, amount: BvTerm) -> BvTerm {
        self.ctx.bv_rotate_right_dyn(x, amount)
    }
    pub fn bv_mul(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_mul(x, y) }
    pub fn bv_udiv(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_udiv(x, y) }
    pub fn bv_urem(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_urem(x, y) }
    pub fn bv_sdiv(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_sdiv(x, y) }
    pub fn bv_srem(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_srem(x, y) }
    pub fn bv_smod(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_smod(x, y) }

    pub fn bv_shl(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_shl(x, y) }
    pub fn bv_lshr(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_lshr(x, y) }
    pub fn bv_ashr(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_ashr(x, y) }
    pub fn bv_rotate_left(&mut self, x: BvTerm, shift: u32) -> BvTerm {
        self.ctx.bv_rotate_left(x, shift)
    }
    pub fn bv_rotate_right(&mut self, x: BvTerm, shift: u32) -> BvTerm {
        self.ctx.bv_rotate_right(x, shift)
    }

    pub fn bv_extract(&mut self, x: BvTerm, high: u32, low: u32) -> BvTerm {
        self.ctx.bv_extract(x, high, low)
    }
    pub fn bv_concat(&mut self, x: BvTerm, y: BvTerm) -> BvTerm { self.ctx.bv_concat(x, y) }
    pub fn bv_zero_extend(&mut self, x: BvTerm, n: u32) -> BvTerm { self.ctx.bv_zero_extend(x, n) }
    pub fn bv_sign_extend(&mut self, x: BvTerm, n: u32) -> BvTerm { self.ctx.bv_sign_extend(x, n) }

    pub fn bv_ite(&mut self, c: BoolTerm, t: BvTerm, e: BvTerm) -> BvTerm {
        self.ctx.bv_ite(c, t, e)
    }

    /// N-way first-match select (state-merge φ-node). See [`BvContext::bv_select`].
    pub fn bv_select(
        &mut self,
        selectors: &[BoolTerm],
        values: &[BvTerm],
        default: BvTerm,
    ) -> BvTerm {
        self.ctx.bv_select(selectors, values, default)
    }

    /// Assert that at most one of `selectors` can be true in any model.
    /// Emits the pairwise exclusion clauses `¬s_i ∨ ¬s_j` (O(N²)). Combine
    /// with [`bv_select`] when merging program states: the Select nodes
    /// bitblast to mux chains that the SAT solver would otherwise have to
    /// explore as independent decisions; these clauses let unit propagation
    /// collapse the chain the moment one selector is known.
    ///
    /// Completeness (`∨ s_i = ⊤`) is *not* asserted — callers who know the
    /// selectors also cover the state space should follow up with an
    /// additional `assert(bool_or_of_all(selectors))`.
    pub fn assert_mutually_exclusive(&mut self, selectors: &[BoolTerm]) {
        // Push a chain of pairwise negations as Bool terms through the
        // normal `assert` path so they participate in pending-queue flush
        // and scope activation. Conceptually we're asserting `¬(s_i ∧ s_j)`
        // for every pair — the cheapest form the `assert_toplevel_direct`
        // path produces for these is a 3-lit clause per pair, which is
        // what we'd want anyway.
        for i in 0..selectors.len() {
            for j in (i + 1)..selectors.len() {
                let a = self.ctx.bool_and(selectors[i], selectors[j]);
                let not_both = self.ctx.bool_not(a);
                self.assert(not_both);
            }
        }
    }

    pub fn bool_true(&mut self) -> BoolTerm { self.ctx.bool_true() }
    pub fn bool_false(&mut self) -> BoolTerm { self.ctx.bool_false() }
    pub fn bool_var(&mut self) -> BoolTerm { self.ctx.bool_var() }
    pub fn bool_not(&mut self, x: BoolTerm) -> BoolTerm { self.ctx.bool_not(x) }
    pub fn bool_and(&mut self, x: BoolTerm, y: BoolTerm) -> BoolTerm { self.ctx.bool_and(x, y) }
    pub fn bool_or(&mut self, x: BoolTerm, y: BoolTerm) -> BoolTerm { self.ctx.bool_or(x, y) }
    pub fn bool_implies(&mut self, x: BoolTerm, y: BoolTerm) -> BoolTerm {
        self.ctx.bool_implies(x, y)
    }

    pub fn bv_eq(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_eq(x, y) }
    pub fn bv_ne(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_ne(x, y) }
    pub fn bv_ult(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_ult(x, y) }
    pub fn bv_ule(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_ule(x, y) }
    pub fn bv_ugt(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_ugt(x, y) }
    pub fn bv_uge(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_uge(x, y) }
    pub fn bv_slt(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_slt(x, y) }
    pub fn bv_sle(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_sle(x, y) }
    pub fn bv_sgt(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_sgt(x, y) }
    pub fn bv_sge(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_sge(x, y) }

    pub fn bv_uadd_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_uadd_overflow(x, y) }
    pub fn bv_sadd_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_sadd_overflow(x, y) }
    pub fn bv_usub_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_usub_overflow(x, y) }
    pub fn bv_ssub_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_ssub_overflow(x, y) }
    pub fn bv_umul_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_umul_overflow(x, y) }
    pub fn bv_smul_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_smul_overflow(x, y) }
    pub fn bv_neg_overflow(&mut self, x: BvTerm) -> BoolTerm { self.ctx.bv_neg_overflow(x) }
    pub fn bv_sdiv_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm { self.ctx.bv_sdiv_overflow(x, y) }

    // ---------- Variable aliasing (union-find) ----------

    /// If `x` and `y` are both bare `BvVar` nodes of equal width, union them
    /// so any future bitblast of either returns the same SAT literals.
    /// Returns `true` on success (alias installed), `false` when the terms
    /// aren't both BvVars (the caller should emit the equality as a clause
    /// in that case). Must be called *before* either variable has been
    /// bitblasted — otherwise there are already distinct SAT vars allocated
    /// and the alias would only affect future fresh lookups.
    pub fn alias_bv_vars(&mut self, x: BvTerm, y: BvTerm) -> bool {
        let (BvOp::Var(xid), BvOp::Var(yid)) = (self.ctx.bv_op(x), self.ctx.bv_op(y)) else {
            return false;
        };
        if self.ctx.width_of(x) != self.ctx.width_of(y) {
            return false;
        }
        if self.bv_var_lits.contains_key(&xid) || self.bv_var_lits.contains_key(&yid) {
            return false;
        }
        self.union_bv_var_ids(xid, yid);
        true
    }

    /// Same as [`alias_bv_vars`] but for Bool vars.
    pub fn alias_bool_vars(&mut self, x: BoolTerm, y: BoolTerm) -> bool {
        let (BoolOp::Var(xid), BoolOp::Var(yid)) =
            (self.ctx.bool_nodes[x.0 as usize], self.ctx.bool_nodes[y.0 as usize])
        else {
            return false;
        };
        if self.bool_var_lits.contains_key(&xid) || self.bool_var_lits.contains_key(&yid) {
            return false;
        }
        self.union_bool_var_ids(xid, yid);
        true
    }

    fn ensure_bv_parent(&mut self, id: u32) {
        while (self.bv_var_parent.len() as u32) <= id {
            let next = self.bv_var_parent.len() as u32;
            self.bv_var_parent.push(next); // self-parent = root
        }
    }
    fn ensure_bool_parent(&mut self, id: u32) {
        while (self.bool_var_parent.len() as u32) <= id {
            let next = self.bool_var_parent.len() as u32;
            self.bool_var_parent.push(next);
        }
    }

    /// Resolve a BV var id to the root of its union-find class, with path
    /// compression. Ids that were never aliased are their own roots.
    fn find_bv_var_root(&mut self, id: u32) -> u32 {
        self.ensure_bv_parent(id);
        let mut cur = id;
        loop {
            let p = self.bv_var_parent[cur as usize];
            if p == cur {
                break;
            }
            let gp = self.bv_var_parent[p as usize];
            self.bv_var_parent[cur as usize] = gp; // half-path compression
            cur = gp;
        }
        cur
    }
    fn find_bool_var_root(&mut self, id: u32) -> u32 {
        self.ensure_bool_parent(id);
        let mut cur = id;
        loop {
            let p = self.bool_var_parent[cur as usize];
            if p == cur {
                break;
            }
            let gp = self.bool_var_parent[p as usize];
            self.bool_var_parent[cur as usize] = gp;
            cur = gp;
        }
        cur
    }

    fn union_bv_var_ids(&mut self, a: u32, b: u32) {
        let ra = self.find_bv_var_root(a);
        let rb = self.find_bv_var_root(b);
        if ra == rb {
            return;
        }
        // Pick smaller id as the root — deterministic and keeps the cache
        // keyed at the earliest-allocated var.
        let (root, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.bv_var_parent[child as usize] = root;
    }
    fn union_bool_var_ids(&mut self, a: u32, b: u32) {
        let ra = self.find_bool_var_root(a);
        let rb = self.find_bool_var_root(b);
        if ra == rb {
            return;
        }
        let (root, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.bool_var_parent[child as usize] = root;
    }

    // ---------- Solver control ----------

    /// Assert that `t` must hold in any model. If called inside a push
    /// scope, the assertion is guarded by that scope's activation literal
    /// and will be retracted on the matching pop. Bitblasting is deferred
    /// until `solve*` — preprocessing passes (variable aliasing etc.) run
    /// between `assert` and `solve`.
    pub fn assert(&mut self, t: BoolTerm) {
        self.last_result = None; // state change invalidates the model
        let depth = self.activation_stack.len();
        self.pending[depth].push(t);
    }

    /// Assert `t` with a name so it can participate in an UNSAT core.
    /// Each named assertion is guarded by a fresh SAT literal that's
    /// assumed true at solve-time; when the formula is UNSAT, the core
    /// identifies which names are needed.
    pub fn assert_named(&mut self, name: impl Into<String>, t: BoolTerm) {
        self.last_result = None;
        let phi = self.bitblast_bool(t);
        let control = self.new_sat_lit_tagged(VarOrigin::Activation);
        // Clause: `(¬control ∨ phi)` — with any push-scope activation
        // folded in so named assertions respect scoping too.
        match self.activation_stack.last() {
            None => self.sat.add_clause(vec![!control, phi]),
            Some(&act) => self.sat.add_clause(vec![!control, !act, phi]),
        };
        self.named_controls.push((name.into(), control));
    }

    /// After a UNSAT result, returns the names of named assertions that
    /// appear in the SAT-level unsat core. Order matches insertion order.
    pub fn unsat_core_names(&self) -> Vec<&str> {
        let core = self.sat.unsat_core();
        let core_set: std::collections::HashSet<Lit> = core.iter().copied().collect();
        self.named_controls
            .iter()
            .filter(|(_, l)| core_set.contains(l))
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// Open a new scope. Every subsequent `assert` is retractable via `pop`.
    pub fn push(&mut self) {
        self.last_result = None;
        let act = self.new_sat_lit_tagged(VarOrigin::Activation);
        self.activation_stack.push(act);
        self.pending.push(Vec::new());
    }

    /// Close the most recently-opened scope. All assertions made inside it
    /// become vacuous. Ignored if no scope is open.
    pub fn pop(&mut self) {
        self.last_result = None;
        if let Some(act) = self.activation_stack.pop() {
            // Any pending (un-flushed) assertions in this scope are simply
            // dropped — they never reached the SAT solver. Flushed assertions
            // are already guarded by `act` and become vacuous once `act=false`.
            self.pending.pop();
            self.sat.add_clause(vec![!act]);
        }
    }

    /// Current number of open push scopes.
    pub fn scope_depth(&self) -> usize {
        self.activation_stack.len()
    }

    pub fn solve(&mut self) -> SmtResult {
        self.flush_pending();
        let asmps = self.built_assumptions(&[]);
        let result = match self.sat.solve_under_assumptions(&asmps) {
            SolveResult::Sat => SmtResult::Sat,
            SolveResult::Unsat => SmtResult::Unsat,
        };
        self.last_result = Some(result);
        result
    }

    pub fn solve_under_assumptions(&mut self, assumptions: &[BoolTerm]) -> SmtResult {
        self.flush_pending();
        let mut extras = Vec::with_capacity(assumptions.len());
        for &t in assumptions {
            extras.push(self.bitblast_bool(t));
        }
        let asmps = self.built_assumptions(&extras);
        let result = match self.sat.solve_under_assumptions(&asmps) {
            SolveResult::Sat => SmtResult::Sat,
            SolveResult::Unsat => SmtResult::Unsat,
        };
        self.last_result = Some(result);
        result
    }

    /// Bounded variant of [`solve_under_assumptions`]: returns `None` once
    /// `max_conflicts` SAT conflicts have accumulated during this call (and
    /// leaves the solver in a consistent state for a subsequent retry with
    /// a larger budget or different assumptions). A budget of `0` means
    /// unbounded. Useful for symbolic-execution branch feasibility probes
    /// that want "fast yes / fast no / give up" semantics rather than an
    /// indefinite wait.
    ///
    /// A `Some(SmtResult::Unsat)` return is a genuine UNSAT proof over the
    /// formula + assumptions, not a budget-driven approximation — the
    /// budget only converts still-searching states into `None`.
    ///
    /// Do not call this with `max_conflicts = 0` when the SAT problem is
    /// known-intractable; prefer [`solve_under_assumptions`] which panics
    /// would not be raised, rather than the `.expect()` in the unbounded
    /// path. In practice the two are identical at budget 0.
    pub fn solve_under_assumptions_bounded(
        &mut self,
        assumptions: &[BoolTerm],
        max_conflicts: u64,
    ) -> Option<SmtResult> {
        self.flush_pending();
        let mut extras = Vec::with_capacity(assumptions.len());
        for &t in assumptions {
            extras.push(self.bitblast_bool(t));
        }
        let asmps = self.built_assumptions(&extras);
        match self
            .sat
            .solve_under_assumptions_bounded(&asmps, max_conflicts)?
        {
            SolveResult::Sat => {
                self.last_result = Some(SmtResult::Sat);
                Some(SmtResult::Sat)
            }
            SolveResult::Unsat => {
                self.last_result = Some(SmtResult::Unsat);
                Some(SmtResult::Unsat)
            }
        }
    }

    /// Wall-clock-bounded variant of [`solve_under_assumptions`]. Returns
    /// `None` when `timeout` elapses before the search completes. Semantics
    /// match [`solve_under_assumptions_bounded`] otherwise — `Some(Unsat)`
    /// is a real proof, the solver is left consistent after `None`, and a
    /// retry with a longer deadline or different assumptions works.
    ///
    /// Use this when a symbex runner wants a real-time ceiling on per-query
    /// cost (e.g. `Duration::from_millis(250)` for branch-feasibility
    /// probes). The deadline is checked on every conflict — a few
    /// nanoseconds of `Instant::now()` overhead per conflict, negligible
    /// for all but the hottest conflict-per-microsecond workloads, where
    /// [`solve_under_assumptions_bounded`] is the alternative.
    pub fn solve_under_assumptions_timed(
        &mut self,
        assumptions: &[BoolTerm],
        timeout: std::time::Duration,
    ) -> Option<SmtResult> {
        self.flush_pending();
        let mut extras = Vec::with_capacity(assumptions.len());
        for &t in assumptions {
            extras.push(self.bitblast_bool(t));
        }
        let asmps = self.built_assumptions(&extras);
        match self.sat.solve_under_assumptions_timed(&asmps, timeout)? {
            SolveResult::Sat => {
                self.last_result = Some(SmtResult::Sat);
                Some(SmtResult::Sat)
            }
            SolveResult::Unsat => {
                self.last_result = Some(SmtResult::Unsat);
                Some(SmtResult::Unsat)
            }
        }
    }

    // ---------- Optimization: solve_min / solve_max ----------
    //
    // "Bit-hunt" search: walk the target term's bitblasted SAT lits from
    // MSB down to LSB and, at each bit, try forcing it to its preferred
    // polarity (0 for min, 1 for max) via a single-literal assumption. A
    // sat response locks that bit in; an unsat response flips the choice
    // and moves on. Exactly `width` solve calls, each adding one unit
    // assumption to the accumulated prefix — strictly cheaper than
    // bitblasting an O(W)-wide comparator for every iteration of a
    // caller-side binary search. The SAT solver's learned clauses carry
    // across iterations since all state is preserved.
    //
    // After a successful search, the solver is left in a sat state whose
    // model realizes the returned optimum, so `get_bv_value_*` on other
    // terms reflects values consistent with the optimal assignment.

    /// Minimum unsigned value of `x` satisfying all active assertions.
    /// Returns `None` if the formula is unsat. Panics if `x`'s width > 128
    /// — use [`solve_min_u_limbs`] for wider terms.
    pub fn solve_min_u(&mut self, x: BvTerm) -> Option<u128> {
        assert!(self.ctx.width_of(x) <= 128, "solve_min_u: width > 128");
        self.solve_min_u_limbs(x).map(|l| limbs_to_u128(&l))
    }

    /// Maximum unsigned value of `x` satisfying all active assertions.
    pub fn solve_max_u(&mut self, x: BvTerm) -> Option<u128> {
        assert!(self.ctx.width_of(x) <= 128, "solve_max_u: width > 128");
        self.solve_max_u_limbs(x).map(|l| limbs_to_u128(&l))
    }

    /// Minimum signed (two's complement) value of `x` satisfying all
    /// active assertions, returned as `i128` with sign extension from
    /// `x`'s width.
    pub fn solve_min_s(&mut self, x: BvTerm) -> Option<i128> {
        let w = self.ctx.width_of(x);
        assert!(w <= 128, "solve_min_s: width > 128");
        self.solve_min_s_limbs(x)
            .map(|l| sign_extend_limbs_i128(&l, w))
    }

    /// Maximum signed (two's complement) value of `x` satisfying all
    /// active assertions.
    pub fn solve_max_s(&mut self, x: BvTerm) -> Option<i128> {
        let w = self.ctx.width_of(x);
        assert!(w <= 128, "solve_max_s: width > 128");
        self.solve_max_s_limbs(x)
            .map(|l| sign_extend_limbs_i128(&l, w))
    }

    /// Arbitrary-width variant of [`solve_min_u`]. Returns the minimum as
    /// little-endian u64 limbs (LSB-first, same layout as
    /// [`get_bv_value_limbs`]).
    pub fn solve_min_u_limbs(&mut self, x: BvTerm) -> Option<Vec<u64>> {
        let bits = self.opt_prologue(x)?;
        Some(self.bit_hunt(&bits, |_| false))
    }

    /// Arbitrary-width variant of [`solve_max_u`].
    pub fn solve_max_u_limbs(&mut self, x: BvTerm) -> Option<Vec<u64>> {
        let bits = self.opt_prologue(x)?;
        Some(self.bit_hunt(&bits, |_| true))
    }

    /// Arbitrary-width signed-min. Signed order differs from unsigned only
    /// at the sign bit: for minimum, we prefer sign-bit 1 (most negative),
    /// then zero elsewhere.
    pub fn solve_min_s_limbs(&mut self, x: BvTerm) -> Option<Vec<u64>> {
        let bits = self.opt_prologue(x)?;
        let msb = bits.len() - 1;
        Some(self.bit_hunt(&bits, |i| i == msb))
    }

    /// Arbitrary-width signed-max. Prefer sign-bit 0 (non-negative), then
    /// ones elsewhere.
    pub fn solve_max_s_limbs(&mut self, x: BvTerm) -> Option<Vec<u64>> {
        let bits = self.opt_prologue(x)?;
        let msb = bits.len() - 1;
        Some(self.bit_hunt(&bits, |i| i != msb))
    }

    /// Shared opt-query setup: flush, bitblast the target term (so its
    /// SAT lits exist and the formula incorporates all its clauses), and
    /// do the initial feasibility check. Returns `None` when unsat (and
    /// updates `last_result` accordingly); returns the LSB-first SAT lits
    /// of `x` when sat, with `last_result = Sat`.
    fn opt_prologue(&mut self, x: BvTerm) -> Option<Vec<Lit>> {
        self.flush_pending();
        // bitblast BEFORE the initial solve so the feasibility check sees
        // any clauses the bitblaster adds for `x` — otherwise a sat model
        // from the smaller formula could be falsified by the extra gates.
        let bits = self.bitblast_bv(x);
        let asmps = self.built_assumptions(&[]);
        match self.sat.solve_under_assumptions(&asmps) {
            SolveResult::Sat => {
                self.last_result = Some(SmtResult::Sat);
                Some(bits)
            }
            SolveResult::Unsat => {
                self.last_result = Some(SmtResult::Unsat);
                None
            }
        }
    }

    /// Core bit-hunt: given LSB-first SAT lits and a policy function
    /// describing which value each bit prefers (`true` = prefer 1,
    /// `false` = prefer 0), return the optimal bit pattern as u64 limbs.
    /// Caller guarantees the formula is sat before invocation (via
    /// [`opt_prologue`]).
    fn bit_hunt(&mut self, bits: &[Lit], want_one: impl Fn(usize) -> bool) -> Vec<u64> {
        let w = bits.len();
        let nlimbs = (w + 63) / 64;
        let mut limbs = vec![0u64; nlimbs];
        let mut fixed: Vec<Lit> = Vec::with_capacity(w);
        for i in (0..w).rev() {
            let b = bits[i];
            let prefer_one = want_one(i);
            let first_try = if prefer_one { b } else { !b };
            fixed.push(first_try);
            let asmps = self.built_assumptions(&fixed);
            let sat = matches!(
                self.sat.solve_under_assumptions(&asmps),
                SolveResult::Sat
            );
            if sat {
                if prefer_one {
                    limbs[i / 64] |= 1u64 << (i % 64);
                }
            } else {
                // The opposite polarity must be sat under `fixed[..-1]`,
                // by exhaustion of the two possibilities.
                fixed.pop();
                fixed.push(!first_try);
                if !prefer_one {
                    limbs[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        // Leave the SAT solver in a state whose current model realizes
        // the returned optimum, so the caller can read other terms'
        // values via `get_bv_value_*` afterward.
        let asmps = self.built_assumptions(&fixed);
        let _ = self.sat.solve_under_assumptions(&asmps);
        self.last_result = Some(SmtResult::Sat);
        limbs
    }

    /// Top-level assertion emit, specialized on the outermost Bool shape to
    /// avoid synthesizing gates whose output is immediately forced. For a BV
    /// equality root `(assert (= x y))`, directly emit 2N guarded bit-
    /// biconditionals instead of the generic Tseitin chain (which would cost
    /// 2N-1 gate vars and ≈7N clauses per equality) — this saves SAT vars on
    /// workloads dominated by `(assert (= reg-i expr))` SSA-style equalities.
    /// For a negated equality, emit a single N-lit disjunction of per-bit
    /// XOR gates. Other assertion shapes fall back to the general bitblast.
    fn assert_toplevel_direct(&mut self, t: BoolTerm, act_lit: Option<Lit>) {
        let op = self.ctx.bool_nodes[t.0 as usize];
        if let BoolOp::Eq(a, b) = op {
            // Skip width-1: mk_xor already short-circuits to the other lit
            // when one side is a constant, and the general path emits a
            // single biconditional anyway. Only the wide case wins here.
            let w = self.ctx.width_of(a);
            if w >= 2 {
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                for i in 0..ab.len() {
                    match act_lit {
                        None => {
                            self.sat.add_clause(vec![!ab[i], bb[i]]);
                            self.sat.add_clause(vec![ab[i], !bb[i]]);
                        }
                        Some(act) => {
                            self.sat.add_clause(vec![!act, !ab[i], bb[i]]);
                            self.sat.add_clause(vec![!act, ab[i], !bb[i]]);
                        }
                    }
                }
                return;
            }
        }
        if let BoolOp::Not(inner) = op {
            if let BoolOp::Eq(a, b) = self.ctx.bool_nodes[inner.0 as usize] {
                let w = self.ctx.width_of(a);
                if w >= 2 {
                    // `¬(x = y)` = some bit differs. Build per-bit XOR lits
                    // and OR them in one clause. Gate vars still needed, but
                    // we skip the AND chain on top.
                    let ab = self.bitblast_bv(a);
                    let bb = self.bitblast_bv(b);
                    let mut clause = Vec::with_capacity(ab.len() + 1);
                    if let Some(act) = act_lit {
                        clause.push(!act);
                    }
                    for i in 0..ab.len() {
                        let x = self.mk_xor(ab[i], bb[i]);
                        clause.push(x);
                    }
                    self.sat.add_clause(clause);
                    return;
                }
            }
        }
        // General path: one Tseitin lit for the whole assertion.
        let lit = self.bitblast_bool(t);
        match act_lit {
            None => {
                self.sat.add_clause(vec![lit]);
            }
            Some(act) => {
                self.sat.add_clause(vec![!act, lit]);
            }
        }
    }

    /// Bitblast every pending assertion, emitting SAT clauses. After flush,
    /// the pending queues for every scope are empty and all those assertions
    /// live in the SAT core (guarded by activation literals for scopes ≥ 1).
    fn flush_pending(&mut self) {
        for depth in 0..self.pending.len() {
            let terms = std::mem::take(&mut self.pending[depth]);
            if terms.is_empty() {
                continue;
            }
            let act_lit = if depth == 0 {
                None
            } else {
                Some(self.activation_stack[depth - 1])
            };
            for t in terms {
                self.assert_toplevel_direct(t, act_lit);
            }
        }
    }


    /// Returns true iff the solver currently holds a valid SAT model —
    /// i.e. the most recent operation was a `solve*` that returned SAT and
    /// nothing has changed the assertion state since. Safe to call before
    /// `get_bv_value` / `get_bool_value`.
    pub fn has_model(&self) -> bool {
        self.last_result == Some(SmtResult::Sat)
    }

    /// Post-flush SAT statistics — useful for profiling. Only meaningful
    /// after `solve*` has flushed the pending queue; before that, the
    /// numbers reflect only clauses emitted by prior solves.
    pub fn sat_stats(&self) -> SmtSolverStats {
        // Count BV/Bool vars that got merged into another root by alias_*.
        let bv_aliased = self
            .bv_var_parent
            .iter()
            .enumerate()
            .filter(|(i, p)| **p as usize != *i)
            .count();
        let bool_aliased = self
            .bool_var_parent
            .iter()
            .enumerate()
            .filter(|(i, p)| **p as usize != *i)
            .count();
        SmtSolverStats {
            sat_vars: self.sat.num_vars(),
            sat_clauses: self.sat.num_clauses(),
            conflicts: self.sat.stats_conflicts,
            decisions: self.sat.stats_decisions,
            restarts: self.sat.stats_restarts,
            learned: self.sat.stats_learned,
            propagations: self.sat.stats_propagations,
            bv_aliased,
            bool_aliased,
            bv_var_total: self.ctx.bv_var_widths.len(),
            bv_nodes_total: self.ctx.bv_nodes.len(),
            bv_vars_bitblasted: self.bv_var_lits.len(),
        }
    }

    // ---------- Metadata accessors ----------

    /// What does this SAT variable represent? Returns `VarOrigin::Unknown`
    /// for any variable the bitblaster didn't explicitly tag (including
    /// out-of-range indices).
    pub fn var_origin(&self, v: Var) -> VarOrigin {
        self.var_origin
            .get(v.idx())
            .copied()
            .unwrap_or(VarOrigin::Unknown)
    }

    /// Number of SAT variables that have been allocated + tagged. Equal to
    /// the underlying SAT solver's var count after the first bitblast.
    pub fn num_sat_vars(&self) -> usize {
        self.var_origin.len()
    }

    /// If `l` is the output literal of a recorded ITE gate, return it.
    /// Hash lookup; safe to call on any literal.
    pub fn ite_gate_for_output(&self, l: Lit) -> Option<IteGate> {
        self.ite_out_to_gate.get(&l).map(|&i| self.ite_gates[i])
    }

    /// Iterator over every ITE gate emitted so far, in insertion order.
    pub fn ite_gates(&self) -> &[IteGate] {
        &self.ite_gates
    }

    /// Assemble the SAT-level assumption list for a solve: push-scope
    /// activations (so their guarded clauses stay live), plus named-assertion
    /// controls (so the SAT core can blame them), plus any user-supplied
    /// extras that got passed through `solve_under_assumptions`.
    fn built_assumptions(&self, extras: &[Lit]) -> Vec<Lit> {
        let mut a = Vec::with_capacity(
            self.activation_stack.len() + self.named_controls.len() + extras.len(),
        );
        a.extend_from_slice(&self.activation_stack);
        a.extend(self.named_controls.iter().map(|(_, l)| *l));
        a.extend_from_slice(extras);
        a
    }

    /// Read a BV value out of the current SAT model. Widths up to 64 are
    /// safe; for wider BVs the upper bits are truncated — use
    /// [`get_bv_value_u128`] for the full range.
    pub fn get_bv_value(&mut self, t: BvTerm) -> u64 {
        self.get_bv_value_u128(t) as u64
    }

    /// Full-precision model read: supports widths up to 128.
    pub fn get_bv_value_u128(&mut self, t: BvTerm) -> u128 {
        let bits = self.bitblast_bv(t);
        let mut v = 0u128;
        for (i, &lit) in bits.iter().enumerate() {
            if self.sat.value_of(lit) == LBool::True {
                v |= 1u128 << i;
            }
        }
        v
    }

    /// Arbitrary-width model read: returns little-endian u64 limbs. Works
    /// for any BV width including those exceeding 128 bits.
    pub fn get_bv_value_limbs(&mut self, t: BvTerm) -> Vec<u64> {
        let bits = self.bitblast_bv(t);
        let nlimbs = (bits.len() + 63) / 64;
        let mut limbs = vec![0u64; nlimbs];
        for (i, &lit) in bits.iter().enumerate() {
            if self.sat.value_of(lit) == LBool::True {
                limbs[i / 64] |= 1u64 << (i % 64);
            }
        }
        limbs
    }

    pub fn get_bool_value(&mut self, t: BoolTerm) -> bool {
        let lit = self.bitblast_bool(t);
        self.sat.value_of(lit) != LBool::False
    }

    // ---------- Bitblasting ----------

    /// Produce N SAT literals (LSB-first) representing the bits of `t`.
    fn bitblast_bv(&mut self, t: BvTerm) -> Vec<Lit> {
        if let Some(cached) = self.bv_cache.get(&t) {
            return cached.clone();
        }
        // Record the current BV term so aux vars created inside gate helpers
        // (mk_and, mk_mux, ripple_carry_add, …) inherit it as their origin
        // context. Save / restore so recursive bitblast calls see their own
        // enclosing term.
        let prev_ctx = self.current_bv_ctx;
        self.current_bv_ctx = Some(t);
        let node = self.ctx.bv_nodes[t.0 as usize];
        let bits = match node.op {
            BvOp::Var(id) => self.get_or_make_bv_var(id, node.width),
            BvOp::Const => {
                let tl = self.get_true_lit();
                if node.wide == crate::bv::WIDE_NONE {
                    // Fast path: value lives inline as a u128.
                    let value = node.value;
                    (0..node.width)
                        .map(|i| if (value >> i) & 1 == 1 { tl } else { !tl })
                        .collect()
                } else {
                    // Wide path: read bit-i from the context's limb pool.
                    let limbs = self.ctx.wide_limbs(node.wide).to_vec();
                    (0..node.width)
                        .map(|i| {
                            let li = (i as usize) / 64;
                            let bi = i % 64;
                            if (limbs[li] >> bi) & 1 == 1 { tl } else { !tl }
                        })
                        .collect()
                }
            }
            BvOp::Not(x) => {
                let xb = self.bitblast_bv(x);
                xb.iter().map(|&l| !l).collect()
            }
            BvOp::And(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                self.zipwith(&xb, &yb, |s, a, b| s.mk_and(a, b))
            }
            BvOp::Or(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                self.zipwith(&xb, &yb, |s, a, b| s.mk_or(a, b))
            }
            BvOp::Xor(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                self.zipwith(&xb, &yb, |s, a, b| s.mk_xor(a, b))
            }
            BvOp::Add(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                let cin = !self.get_true_lit();
                self.ripple_carry_add(&xb, &yb, cin).0
            }
            BvOp::Sub(x, y) => {
                // a - b = a + ~b + 1
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                let y_neg: Vec<Lit> = yb.iter().map(|&l| !l).collect();
                let cin = self.get_true_lit();
                self.ripple_carry_add(&xb, &y_neg, cin).0
            }
            BvOp::Neg(x) => {
                let xb = self.bitblast_bv(x);
                self.mk_neg(&xb)
            }
            BvOp::Mul(x, y) => {
                // If either operand is constant, use the sparse fast path:
                // only emit adders for the 1-bits of the constant, instead
                // of a full N×N shift-and-add. For 64-bit mul-by-small-const
                // this collapses ~24k gates into a handful.
                let x_const = self.const_bv_value(x);
                let y_const = self.const_bv_value(y);
                let w = self.ctx.width_of(x) as usize;
                match (x_const, y_const) {
                    (Some(c), None) => {
                        let yb = self.bitblast_bv(y);
                        self.mk_mul_const(&yb, c, w)
                    }
                    (None, Some(c)) => {
                        let xb = self.bitblast_bv(x);
                        self.mk_mul_const(&xb, c, w)
                    }
                    _ => {
                        let xb = self.bitblast_bv(x);
                        let yb = self.bitblast_bv(y);
                        self.mk_mul(&xb, &yb)
                    }
                }
            }
            BvOp::Udiv(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                let (q, _r) = self.mk_udivmod(&xb, &yb);
                // bvudiv(x, 0) = all ones (SMT-LIB).
                let yb2 = self.bitblast_bv(y);
                let y_is_zero = self.mk_all_zero(&yb2);
                let ones = vec![self.get_true_lit(); q.len()];
                self.mux_vec(y_is_zero, &ones, &q)
            }
            BvOp::Urem(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                let (_q, r) = self.mk_udivmod(&xb, &yb);
                // bvurem(x, 0) = x.
                let yb2 = self.bitblast_bv(y);
                let xb2 = self.bitblast_bv(x);
                let y_is_zero = self.mk_all_zero(&yb2);
                self.mux_vec(y_is_zero, &xb2, &r)
            }
            BvOp::Sdiv(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                self.mk_sdiv(&xb, &yb)
            }
            BvOp::Srem(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                self.mk_srem(&xb, &yb)
            }
            BvOp::Smod(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                self.mk_smod(&xb, &yb)
            }
            BvOp::Shl(x, y) => {
                let xb = self.bitblast_bv(x);
                // Fast path: const shift amount → pure re-wiring, zero gates.
                if let Some(amt) = self.const_shift_amt(y) {
                    self.mk_shl_const(&xb, amt)
                } else {
                    let yb = self.bitblast_bv(y);
                    self.mk_shl(&xb, &yb)
                }
            }
            BvOp::Lshr(x, y) => {
                let xb = self.bitblast_bv(x);
                let zero_fill = !self.get_true_lit();
                if let Some(amt) = self.const_shift_amt(y) {
                    self.mk_shr_const(&xb, amt, zero_fill)
                } else {
                    let yb = self.bitblast_bv(y);
                    self.mk_shr(&xb, &yb, zero_fill)
                }
            }
            BvOp::Ashr(x, y) => {
                let xb = self.bitblast_bv(x);
                let sign = xb[xb.len() - 1];
                if let Some(amt) = self.const_shift_amt(y) {
                    self.mk_shr_const(&xb, amt, sign)
                } else {
                    let yb = self.bitblast_bv(y);
                    self.mk_shr(&xb, &yb, sign)
                }
            }
            BvOp::Extract(x, high, low) => {
                let xb = self.bitblast_bv(x);
                xb[low as usize..=high as usize].to_vec()
            }
            BvOp::Concat(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                // y occupies the low bits, x the high bits.
                let mut result = yb;
                result.extend(xb);
                result
            }
            BvOp::ZeroExtend(x, n) => {
                let xb = self.bitblast_bv(x);
                let mut result = xb;
                let zero = !self.get_true_lit();
                for _ in 0..n {
                    result.push(zero);
                }
                result
            }
            BvOp::SignExtend(x, n) => {
                let xb = self.bitblast_bv(x);
                let sign = xb[xb.len() - 1];
                let mut result = xb;
                for _ in 0..n {
                    result.push(sign);
                }
                result
            }
            BvOp::Ite(c, t_term, e_term) => {
                let cl = self.bitblast_bool(c);
                let tb = self.bitblast_bv(t_term);
                let eb = self.bitblast_bv(e_term);
                self.mux_vec(cl, &tb, &eb)
            }
            BvOp::Select(idx) => {
                // Bitblast the Select as a right-to-left chain of muxes:
                // `out = mux(sel_0, val_0, mux(sel_1, val_1, … mux(sel_N,
                // val_N, default)))`. This preserves first-match semantics
                // (earlier selectors shadow later ones) and bit-level fold
                // is automatic — `mk_mux(s, x, x)` collapses and
                // `mk_mux(T/F, …)` short-circuits, so bits where every
                // branch agrees don't spawn a gate.
                //
                // If exclusion clauses for the selectors are installed via
                // `assert_mutually_exclusive`, SAT propagation collapses
                // each cascade in O(1): the one true selector forces the
                // chosen branch and every other selector is forced false
                // at the same decision level.
                let table = self.ctx.select_tables[idx as usize].clone();
                let default_bits = self.bitblast_bv(table.default);
                let sel_lits: Vec<Lit> = table
                    .selectors
                    .iter()
                    .map(|&s| self.bitblast_bool(s))
                    .collect();
                let value_bit_vecs: Vec<Vec<Lit>> = table
                    .values
                    .iter()
                    .map(|&v| self.bitblast_bv(v))
                    .collect();
                let n_bits = default_bits.len();
                let mut output: Vec<Lit> = default_bits.clone();
                // Walk right-to-left so the FIRST (outermost) selector ends
                // up taking priority — mux(sel_i, val_i, acc) means "if
                // sel_i, val_i, else whatever the tail produced".
                for i in (0..sel_lits.len()).rev() {
                    let sel = sel_lits[i];
                    let value = &value_bit_vecs[i];
                    for bit in 0..n_bits {
                        output[bit] = self.mk_mux(sel, value[bit], output[bit]);
                    }
                }
                output
            }
            BvOp::Popcount(x) => {
                let expanded = self.build_popcount_expansion(x);
                self.bitblast_bv(expanded)
            }
            BvOp::Clz(x) => {
                let expanded = self.build_clz_expansion(x);
                self.bitblast_bv(expanded)
            }
            BvOp::Ctz(x) => {
                let expanded = self.build_ctz_expansion(x);
                self.bitblast_bv(expanded)
            }
            BvOp::RotateLeft(x, amount) => {
                let expanded = self.build_rotate_dyn_expansion(x, amount, true);
                self.bitblast_bv(expanded)
            }
            BvOp::RotateRight(x, amount) => {
                let expanded = self.build_rotate_dyn_expansion(x, amount, false);
                self.bitblast_bv(expanded)
            }
        };
        self.current_bv_ctx = prev_ctx;
        self.bv_cache.insert(t, bits.clone());
        bits
    }

    /// Produce a single SAT literal for `t`.
    fn bitblast_bool(&mut self, t: BoolTerm) -> Lit {
        if let Some(&cached) = self.bool_cache.get(&t) {
            return cached;
        }
        let op = self.ctx.bool_nodes[t.0 as usize];
        let lit = match op {
            BoolOp::True => self.get_true_lit(),
            BoolOp::False => !self.get_true_lit(),
            BoolOp::Var(id) => {
                let id = self.find_bool_var_root(id);
                if let Some(&cached) = self.bool_var_lits.get(&id) {
                    cached
                } else {
                    let l = self.new_sat_lit_tagged(VarOrigin::Bool { term: t });
                    self.bool_var_lits.insert(id, l);
                    l
                }
            }
            BoolOp::Not(x) => {
                let xl = self.bitblast_bool(x);
                !xl
            }
            BoolOp::And(x, y) => {
                let xl = self.bitblast_bool(x);
                let yl = self.bitblast_bool(y);
                self.mk_and(xl, yl)
            }
            BoolOp::Or(x, y) => {
                let xl = self.bitblast_bool(x);
                let yl = self.bitblast_bool(y);
                self.mk_or(xl, yl)
            }
            BoolOp::Implies(x, y) => {
                let xl = self.bitblast_bool(x);
                let yl = self.bitblast_bool(y);
                self.mk_or(!xl, yl)
            }
            BoolOp::Eq(a, b) => {
                // Width-1 fast path: equality between two 1-bit values is
                // either a direct literal reuse or a single XNOR — skip the
                // O(w) bit-for-bit loop and its temporary vector alloc.
                // In particular `(= x (_ bv1 1))` where x is BV1 is a pure
                // lift of x's single bit to Bool: no new gate, no clause.
                if self.ctx.width_of(a) == 1 {
                    let a_lit = self.bitblast_bv(a)[0];
                    let b_lit = self.bitblast_bv(b)[0];
                    let tl = self.get_true_lit();
                    if b_lit == tl {
                        return a_lit;
                    }
                    if b_lit == !tl {
                        return !a_lit;
                    }
                    if a_lit == tl {
                        return b_lit;
                    }
                    if a_lit == !tl {
                        return !b_lit;
                    }
                    // Two non-trivial 1-bit vars: single XNOR gate is cheaper
                    // than the generic mk_bitwise_eq path.
                    let n_and = self.mk_and(!a_lit, !b_lit);
                    let both = self.mk_and(a_lit, b_lit);
                    return self.mk_or(n_and, both);
                }
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                self.mk_bitwise_eq(&ab, &bb)
            }
            BoolOp::Ult(a, b) => {
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                self.mk_ult(&ab, &bb)
            }
            BoolOp::Ule(a, b) => {
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let blt_a = self.mk_ult(&bb, &ab);
                !blt_a
            }
            BoolOp::Slt(a, b) => {
                // Signed less-than reduces to unsigned less-than with the
                // sign bits flipped: flipping moves negative numbers below
                // positive ones under unsigned ordering.
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let a_flip = flip_msb(&ab);
                let b_flip = flip_msb(&bb);
                self.mk_ult(&a_flip, &b_flip)
            }
            BoolOp::Sle(a, b) => {
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let a_flip = flip_msb(&ab);
                let b_flip = flip_msb(&bb);
                let blt_a = self.mk_ult(&b_flip, &a_flip);
                !blt_a
            }
            BoolOp::UaddOverflow(a, b) => {
                // Overflow bit = final carry-out of plain ripple-carry add.
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let cin = !self.get_true_lit();
                let (_sum, cout) = self.ripple_carry_add(&ab, &bb, cin);
                cout
            }
            BoolOp::SaddOverflow(a, b) => {
                // Signed add overflows iff: sign(a) == sign(b) && sign(sum) != sign(a).
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let cin = !self.get_true_lit();
                let (sum, _) = self.ripple_carry_add(&ab, &bb, cin);
                let a_sign = ab[ab.len() - 1];
                let b_sign = bb[bb.len() - 1];
                let s_sign = sum[sum.len() - 1];
                let same_sign = !self.mk_xor(a_sign, b_sign);
                let flipped = self.mk_xor(a_sign, s_sign);
                self.mk_and(same_sign, flipped)
            }
            BoolOp::UsubOverflow(a, b) => {
                // a - b borrows iff a <u b.
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                self.mk_ult(&ab, &bb)
            }
            BoolOp::SsubOverflow(a, b) => {
                // Signed sub overflows iff: sign(a) != sign(b) && sign(a-b) != sign(a).
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let b_neg: Vec<Lit> = bb.iter().map(|&l| !l).collect();
                let cin = self.get_true_lit();
                let (diff, _) = self.ripple_carry_add(&ab, &b_neg, cin);
                let a_sign = ab[ab.len() - 1];
                let b_sign = bb[bb.len() - 1];
                let d_sign = diff[diff.len() - 1];
                let diff_sign_ops = self.mk_xor(a_sign, b_sign);
                let flipped = self.mk_xor(a_sign, d_sign);
                self.mk_and(diff_sign_ops, flipped)
            }
            BoolOp::UmulOverflow(a, b) => {
                // Compute the full 2N-bit unsigned product, then OR-reduce the
                // high N bits.
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let hi = self.mk_umul_hi(&ab, &bb);
                self.mk_any_set(&hi)
            }
            BoolOp::SmulOverflow(a, b) => {
                // Compute full 2N-bit signed product; overflow iff the high
                // N bits are not all equal to the sign bit of the low N bits.
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let (lo, hi) = self.mk_smul_full(&ab, &bb);
                // Expected high bits: all replicas of lo's MSB (sign of the
                // truncated product). Overflow iff any differ.
                let sign_of_lo = lo[lo.len() - 1];
                let diffs: Vec<Lit> =
                    hi.iter().map(|&h| self.mk_xor(h, sign_of_lo)).collect();
                self.mk_any_set(&diffs)
            }
            BoolOp::NegOverflow(a) => {
                // -x overflows iff x = INT_MIN = sign-bit-set, all-others-zero.
                let ab = self.bitblast_bv(a);
                let n = ab.len();
                let mut conds = Vec::with_capacity(n);
                conds.push(ab[n - 1]); // MSB must be 1
                for i in 0..n - 1 {
                    conds.push(!ab[i]); // others must be 0
                }
                let mut acc = conds[0];
                for c in conds.iter().skip(1) {
                    acc = self.mk_and(acc, *c);
                }
                acc
            }
            BoolOp::SdivOverflow(a, b) => {
                // Overflows iff a = INT_MIN AND b = -1.
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let n = ab.len();
                // a = INT_MIN: MSB(a) = 1, rest = 0.
                let mut a_is_min = ab[n - 1];
                for i in 0..n - 1 {
                    a_is_min = self.mk_and(a_is_min, !ab[i]);
                }
                // b = -1: all bits = 1.
                let mut b_is_minus_one = bb[0];
                for i in 1..n {
                    b_is_minus_one = self.mk_and(b_is_minus_one, bb[i]);
                }
                self.mk_and(a_is_min, b_is_minus_one)
            }
        };
        self.bool_cache.insert(t, lit);
        lit
    }

    // ---------- Low-level SAT helpers ----------

    /// Allocate a fresh SAT literal tagged with the given metadata.
    fn new_sat_lit_tagged(&mut self, origin: VarOrigin) -> Lit {
        let v = self.sat.new_var();
        // Keep var_origin aligned 1-to-1 with SAT variables.
        debug_assert_eq!(self.var_origin.len(), v.idx());
        self.var_origin.push(origin);
        Lit::new(v, false)
    }


    fn get_or_make_bv_var(&mut self, id: u32, width: u32) -> Vec<Lit> {
        // Route through the union-find root: aliased vars share SAT literals.
        let id = self.find_bv_var_root(id);
        if let Some(cached) = self.bv_var_lits.get(&id) {
            return cached.clone();
        }
        // We need the BvTerm handle to tag each bit. Look it up by scanning
        // the context — BV variables are leaves with `BvOp::Var(id)`.
        let term = {
            let mut found = None;
            for (idx, node) in self.ctx.bv_nodes.iter().enumerate() {
                if let BvOp::Var(vid) = node.op {
                    if vid == id {
                        found = Some(BvTerm(idx as u32));
                        break;
                    }
                }
            }
            found
        };
        let bits: Vec<Lit> = (0..width)
            .map(|bit| {
                let origin = match term {
                    Some(t) => VarOrigin::BvBit { term: t, bit },
                    None => VarOrigin::Unknown,
                };
                self.new_sat_lit_tagged(origin)
            })
            .collect();
        self.bv_var_lits.insert(id, bits.clone());
        bits
    }

    /// A literal pinned to true. Allocated once on first use, backed by a
    /// unit clause; every reference to "false" uses its negation.
    fn get_true_lit(&mut self) -> Lit {
        if let Some(l) = self.true_lit {
            return l;
        }
        let l = self.new_sat_lit_tagged(VarOrigin::TrueLit);
        self.sat.add_clause(vec![l]);
        self.true_lit = Some(l);
        l
    }

    fn zipwith<F>(&mut self, a: &[Lit], b: &[Lit], mut f: F) -> Vec<Lit>
    where
        F: FnMut(&mut Self, Lit, Lit) -> Lit,
    {
        assert_eq!(a.len(), b.len());
        (0..a.len()).map(|i| f(self, a[i], b[i])).collect()
    }

    /// Allocate an output literal for a freshly-emitted gate, tagging it
    /// with metadata for later introspection. Uses the currently-enclosing
    /// BV term (if any) to group gate outputs back to their source.
    fn new_gate_lit(&mut self, kind: GateKind) -> Lit {
        let origin = VarOrigin::GateOut {
            gate: kind,
            term: self.current_bv_ctx,
        };
        self.new_sat_lit_tagged(origin)
    }

    /// AND gate.  o ↔ (a ∧ b).  Encodes the biconditional with 3 clauses.
    /// Fast paths: trivial identities, and the pinned-`true_lit` constants
    /// (critical — bitblasting BV constants generates a flood of `mk_and(x, T)`
    /// and `mk_and(x, F)` calls that would otherwise each allocate a gate).
    fn mk_and(&mut self, a: Lit, b: Lit) -> Lit {
        if a == b { return a; }
        if a == !b { return !self.get_true_lit(); }
        if let Some(tl) = self.true_lit {
            if a == tl { return b; }            // T ∧ b = b
            if b == tl { return a; }            // a ∧ T = a
            if a == !tl { return !tl; }         // F ∧ b = F
            if b == !tl { return !tl; }         // a ∧ F = F
        }
        let key = if a.0 <= b.0 { (a, b) } else { (b, a) };
        if let Some(&cached) = self.and_cache.get(&key) {
            return cached;
        }
        // De Morgan cross-cache: `and(a, b) = ¬or(¬a, ¬b)`. If we already
        // emitted the corresponding OR gate, reuse its negated output and
        // avoid a duplicate AND gate + 3 clauses.
        let neg_key = {
            let na = !a;
            let nb = !b;
            if na.0 <= nb.0 { (na, nb) } else { (nb, na) }
        };
        if let Some(&cached) = self.or_cache.get(&neg_key) {
            let o = !cached;
            self.and_cache.insert(key, o);
            return o;
        }
        let o = self.new_gate_lit(GateKind::And);
        self.sat.add_clause(vec![!a, !b, o]);
        self.sat.add_clause(vec![a, !o]);
        self.sat.add_clause(vec![b, !o]);
        self.and_cache.insert(key, o);
        self.or_cache.insert(neg_key, !o);
        o
    }

    /// OR gate.  o ↔ (a ∨ b).
    fn mk_or(&mut self, a: Lit, b: Lit) -> Lit {
        if a == b { return a; }
        if a == !b { return self.get_true_lit(); }
        if let Some(tl) = self.true_lit {
            if a == tl { return tl; }           // T ∨ b = T
            if b == tl { return tl; }           // a ∨ T = T
            if a == !tl { return b; }           // F ∨ b = b
            if b == !tl { return a; }           // a ∨ F = a
        }
        let key = if a.0 <= b.0 { (a, b) } else { (b, a) };
        if let Some(&cached) = self.or_cache.get(&key) {
            return cached;
        }
        // De Morgan cross-cache: `or(a, b) = ¬and(¬a, ¬b)`.
        let neg_key = {
            let na = !a;
            let nb = !b;
            if na.0 <= nb.0 { (na, nb) } else { (nb, na) }
        };
        if let Some(&cached) = self.and_cache.get(&neg_key) {
            let o = !cached;
            self.or_cache.insert(key, o);
            return o;
        }
        let o = self.new_gate_lit(GateKind::Or);
        self.sat.add_clause(vec![a, b, !o]);
        self.sat.add_clause(vec![!a, o]);
        self.sat.add_clause(vec![!b, o]);
        self.or_cache.insert(key, o);
        self.and_cache.insert(neg_key, !o);
        o
    }

    /// XOR gate.  o ↔ (a ⊕ b).
    fn mk_xor(&mut self, a: Lit, b: Lit) -> Lit {
        if a == b { return !self.get_true_lit(); }
        if a == !b { return self.get_true_lit(); }
        if let Some(tl) = self.true_lit {
            if a == tl { return !b; }           // T ⊕ b = ¬b
            if b == tl { return !a; }           // a ⊕ T = ¬a
            if a == !tl { return b; }           // F ⊕ b = b
            if b == !tl { return a; }           // a ⊕ F = a
        }
        // XOR under polarity: xor(a,b) = xor(¬a,¬b), and xor(a,b) = ¬xor(¬a,b).
        // Canonicalize so the first arg is always positive polarity; if we
        // had to flip exactly one, negate the cached output.
        let (ca, cb, flip) = {
            let a_pos = (a.0 & 1) == 0;
            let b_pos = (b.0 & 1) == 0;
            match (a_pos, b_pos) {
                (true, true) | (false, false) => {
                    // Both same polarity: strip to positives (or keep as-is)
                    let pa = Lit(a.0 & !1);
                    let pb = Lit(b.0 & !1);
                    (pa, pb, false)
                }
                (true, false) => (a, !b, true),
                (false, true) => (!a, b, true),
            }
        };
        let key = if ca.0 <= cb.0 { (ca, cb) } else { (cb, ca) };
        if let Some(&cached) = self.xor_cache.get(&key) {
            return if flip { !cached } else { cached };
        }
        let o = self.new_gate_lit(GateKind::Xor);
        self.sat.add_clause(vec![!a, !b, !o]);
        self.sat.add_clause(vec![a, b, !o]);
        self.sat.add_clause(vec![a, !b, o]);
        self.sat.add_clause(vec![!a, b, o]);
        // Cache the "positive form" output: with ca, cb both positive,
        // xor(ca, cb) would be an o' whose clauses are the positive form.
        // But the `o` we just emitted uses the original (a, b). They're
        // structurally the same gate. The trick: for the canonical key, we
        // want xor(ca, cb) = o when flip=false, and !o when flip=true.
        let canonical_out = if flip { !o } else { o };
        self.xor_cache.insert(key, canonical_out);
        o
    }

    /// 2:1 MUX.  o ↔ (sel ∧ t) ∨ (¬sel ∧ e).
    /// Registers the gate in `ite_gates` so future ITE-aware passes can
    /// find it without scanning clauses.
    fn mk_mux(&mut self, sel: Lit, t: Lit, e: Lit) -> Lit {
        if t == e { return t; }
        if let Some(tl) = self.true_lit {
            if sel == tl { return t; }          // const-true sel
            if sel == !tl { return e; }         // const-false sel
        }
        // Also handle mux degenerates involving constants on branches.
        if let Some(tl) = self.true_lit {
            if t == tl && e == !tl { return sel; }       // mux(s, T, F) = s
            if t == !tl && e == tl { return !sel; }      // mux(s, F, T) = ¬s
            if t == sel { /* mux(s, s, e) = s ∨ e */
                return self.mk_or(sel, e);
            }
            if e == !sel { /* mux(s, t, ¬s) = s ∧ t */
                return self.mk_and(sel, t);
            }
            let _ = tl;
        }
        // Canonicalize sel to positive polarity: mux(¬s, t, e) = mux(s, e, t).
        let (csel, ct, ce) = if (sel.0 & 1) == 1 {
            (!sel, e, t)
        } else {
            (sel, t, e)
        };
        let key = (csel, ct, ce);
        if let Some(&cached) = self.mux_cache.get(&key) {
            return cached;
        }
        let o = self.new_gate_lit(GateKind::Ite);
        self.sat.add_clause(vec![!sel, !t, o]);
        self.sat.add_clause(vec![!sel, t, !o]);
        self.sat.add_clause(vec![sel, !e, o]);
        self.sat.add_clause(vec![sel, e, !o]);
        // Record the structural gate.
        let idx = self.ite_gates.len();
        self.ite_gates.push(IteGate {
            sel,
            t,
            e,
            o,
            source_term: self.current_bv_ctx,
        });
        self.ite_out_to_gate.insert(o, idx);
        self.mux_cache.insert(key, o);
        // Branching hint: deciding `sel` resolves the whole ITE subtree,
        // so bump its VSIDS activity once per gate. Width-N ITEs naturally
        // stack N bumps onto the same selector, giving deep / wide ITE
        // fan-outs a proportionally strong priority.
        if self.ite_branching_hints {
            self.sat.boost_var_activity(sel.var());
        }
        o
    }

    /// Bit-parallel MUX: pick `t[i]` when `sel` is true, else `e[i]`.
    fn mux_vec(&mut self, sel: Lit, t: &[Lit], e: &[Lit]) -> Vec<Lit> {
        assert_eq!(t.len(), e.len());
        (0..t.len()).map(|i| self.mk_mux(sel, t[i], e[i])).collect()
    }

    /// Full adder for one bit. Returns (sum, cout).
    ///   sum = a ⊕ b ⊕ cin
    ///   cout = majority(a, b, cin)
    fn mk_full_adder(&mut self, a: Lit, b: Lit, cin: Lit) -> (Lit, Lit) {
        let a_xor_b = self.mk_xor(a, b);
        let sum = self.mk_xor(a_xor_b, cin);
        let a_and_b = self.mk_and(a, b);
        let cin_and_xor = self.mk_and(cin, a_xor_b);
        let cout = self.mk_or(a_and_b, cin_and_xor);
        (sum, cout)
    }

    /// Build a divide-and-conquer popcount tree for `x` as BV terms. The
    /// resulting term is then bitblasted by the caller via the normal
    /// dispatch path. Each level pairs adjacent partial sums, zero-extends
    /// by one bit, and adds — total gate count is O(W log W) instead of
    /// the chained-add form's O(W²).
    fn build_popcount_expansion(&mut self, x: BvTerm) -> BvTerm {
        let w = self.ctx.width_of(x);
        debug_assert!(w >= 2, "single-bit popcount short-circuited in bv_popcount");
        let mut layer: Vec<BvTerm> = (0..w)
            .map(|i| self.ctx.bv_extract(x, i, i))
            .collect();
        let mut layer_w: u32 = 1;
        while layer.len() > 1 {
            let next_w = layer_w + 1;
            let mut next: Vec<BvTerm> = Vec::with_capacity(layer.len().div_ceil(2));
            let mut it = layer.into_iter();
            while let Some(a) = it.next() {
                let a_e = self.ctx.bv_zero_extend(a, next_w - layer_w);
                match it.next() {
                    Some(b) => {
                        let b_e = self.ctx.bv_zero_extend(b, next_w - layer_w);
                        next.push(self.ctx.bv_add(a_e, b_e));
                    }
                    None => next.push(a_e),
                }
            }
            layer = next;
            layer_w = next_w;
        }
        let result = layer[0];
        if layer_w < w {
            self.ctx.bv_zero_extend(result, w - layer_w)
        } else {
            result
        }
    }

    /// Build the CLZ expansion as `popcount(~(x | x>>1 | x>>2 | ...))`.
    /// The OR-fold drags the highest set bit downward so every bit at-or-
    /// below it becomes 1; the popcount-of-not then counts the cleared
    /// (leading-zero) bits.
    fn build_clz_expansion(&mut self, x: BvTerm) -> BvTerm {
        let w = self.ctx.width_of(x);
        debug_assert!(w >= 2, "single-bit CLZ short-circuited in bv_clz");
        let mut y = x;
        let mut k = 1u32;
        while k < w {
            let shift = self.ctx.bv_const(k as u128, w);
            let shifted = self.ctx.bv_lshr(y, shift);
            y = self.ctx.bv_or(y, shifted);
            k <<= 1;
        }
        let ny = self.ctx.bv_not(y);
        self.build_popcount_expansion(ny)
    }

    /// Build the CTZ expansion as `popcount(~x & (x - 1))`. `x - 1` clears
    /// the lowest set bit and sets all bits below it; ANDing with `~x`
    /// isolates exactly the trailing-zero positions, so their popcount is
    /// the CTZ. For `x == 0` the masks collapse to all-ones whose popcount
    /// is `width`, matching the SMT-LIB convention.
    fn build_ctz_expansion(&mut self, x: BvTerm) -> BvTerm {
        let w = self.ctx.width_of(x);
        debug_assert!(w >= 2, "single-bit CTZ short-circuited in bv_ctz");
        let one = self.ctx.bv_const(1, w);
        let xm1 = self.ctx.bv_sub(x, one);
        let nx = self.ctx.bv_not(x);
        let m = self.ctx.bv_and(nx, xm1);
        self.build_popcount_expansion(m)
    }

    /// Build a symbolic-amount rotation as a log-tree of conditional
    /// constant rotations: for each bit `k` of `amount`, conditionally
    /// rotate by `2^k`. Each step costs only the per-bit ITE since the
    /// constant rotation lowers to extract + concat with zero SAT gates.
    /// For non-power-of-two widths, fall back to the `urem` + shifts form
    /// (rare in real pcode — instruction widths are 8/16/32/64).
    fn build_rotate_dyn_expansion(
        &mut self,
        x: BvTerm,
        amount: BvTerm,
        left: bool,
    ) -> BvTerm {
        let w = self.ctx.width_of(x);
        debug_assert!(w >= 2, "single-bit rotate short-circuited in builder");
        if w.is_power_of_two() {
            let log_w = w.trailing_zeros();
            let one_bit = self.ctx.bv_const(1, 1);
            let mut rot = x;
            for k in 0..log_w {
                let bit_k = self.ctx.bv_extract(amount, k, k);
                let bit_set = self.ctx.bv_eq(bit_k, one_bit);
                let shift = 1u32 << k;
                let rotated = if left {
                    self.ctx.bv_rotate_left(rot, shift)
                } else {
                    self.ctx.bv_rotate_right(rot, shift)
                };
                rot = self.ctx.bv_ite(bit_set, rotated, rot);
            }
            rot
        } else {
            let w_const = self.ctx.bv_const(w as u128, w);
            let amt_mod = self.ctx.bv_urem(amount, w_const);
            let complement = self.ctx.bv_sub(w_const, amt_mod);
            let (left_term, right_term) = if left {
                (
                    self.ctx.bv_shl(x, amt_mod),
                    self.ctx.bv_lshr(x, complement),
                )
            } else {
                (
                    self.ctx.bv_lshr(x, amt_mod),
                    self.ctx.bv_shl(x, complement),
                )
            };
            self.ctx.bv_or(left_term, right_term)
        }
    }

    fn ripple_carry_add(&mut self, a: &[Lit], b: &[Lit], cin: Lit) -> (Vec<Lit>, Lit) {
        assert_eq!(a.len(), b.len());
        let mut sum = Vec::with_capacity(a.len());
        let mut carry = cin;
        for i in 0..a.len() {
            let (s, c) = self.mk_full_adder(a[i], b[i], carry);
            sum.push(s);
            carry = c;
        }
        (sum, carry)
    }

    fn mk_bitwise_eq(&mut self, a: &[Lit], b: &[Lit]) -> Lit {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            return self.get_true_lit();
        }
        let mut eq = !self.mk_xor(a[0], b[0]);
        for i in 1..a.len() {
            let bit_eq = !self.mk_xor(a[i], b[i]);
            eq = self.mk_and(eq, bit_eq);
        }
        eq
    }

    /// Unsigned less-than via the borrow of `a - b` = `a + ~b + 1`. If the
    /// final carry-out is 0, `a < b` (a borrow happened); if 1, `a >= b`.
    fn mk_ult(&mut self, a: &[Lit], b: &[Lit]) -> Lit {
        assert_eq!(a.len(), b.len());
        let b_neg: Vec<Lit> = b.iter().map(|&l| !l).collect();
        let cin = self.get_true_lit();
        let (_sum, cout) = self.ripple_carry_add(a, &b_neg, cin);
        !cout
    }

    /// OR-reduction: returns 1 iff all bits are zero.
    fn mk_all_zero(&mut self, bits: &[Lit]) -> Lit {
        assert!(!bits.is_empty());
        let mut z = !bits[0];
        for i in 1..bits.len() {
            z = self.mk_and(z, !bits[i]);
        }
        z
    }

    /// OR-reduction: returns 1 iff any bit is set.
    fn mk_any_set(&mut self, bits: &[Lit]) -> Lit {
        assert!(!bits.is_empty());
        let mut any = bits[0];
        for i in 1..bits.len() {
            any = self.mk_or(any, bits[i]);
        }
        any
    }

    /// If `t` is a BV constant, return its raw 64-bit value (already masked
    /// to the term's width at construction time in `BvContext::bv_const`).
    fn const_bv_value(&self, t: BvTerm) -> Option<u128> {
        let node = self.ctx.bv_nodes[t.0 as usize];
        if matches!(node.op, BvOp::Const) {
            Some(node.value)
        } else {
            None
        }
    }

    /// Sparse shift-and-add multiplication with one constant operand. Runs
    /// through only the 1-bits of `c`, which is the big win: for a common
    /// case like `x * 3` on 64-bit BVs, we emit 2 ripple-carry adds
    /// (popcount=2) instead of 64 — and each bit-AND gate collapses via the
    /// mk_and short-circuits.
    fn mk_mul_const(&mut self, a: &[Lit], c: u128, n: usize) -> Vec<Lit> {
        let zero = !self.get_true_lit();
        let one = self.get_true_lit();
        // Canonical Signed Digit (NAF) recoding of `c`: represents the
        // constant as a sum of ±(powers of 2) with at most half as many
        // non-zero terms as the raw binary form in the worst case — long
        // runs of 1-bits collapse because `2^(k+1) - 1 = 2^(k+1) - 2^0`.
        // So e.g. `x * 15` emits one subtract (`(x << 4) - x`) instead of
        // four adds, `x * 255` emits one subtract instead of eight.
        //
        // Positions ≥ n represent `2^n · x`, which is zero mod 2^n — drop
        // those digits. For `n > 64` we still only consider the low 64
        // bits of `c` (the caller only ever passes a u64-sized constant).
        let max_bit = n.min(64);
        let digits = naf_recode(c & mask_u128(max_bit as u32), n as u32);
        let mut result: Vec<Lit> = vec![zero; n];
        for (sign, pos) in digits {
            let pos = pos as usize;
            // Build `a << pos`, truncated to n bits.
            let shifted: Vec<Lit> = (0..n)
                .map(|j| if j < pos { zero } else { a[j - pos] })
                .collect();
            if sign > 0 {
                let (new_result, _) = self.ripple_carry_add(&result, &shifted, zero);
                result = new_result;
            } else {
                // `result - shifted = result + (¬shifted) + 1`. The bit
                // inversions are polarity flips on the literal — no gates.
                let neg_shifted: Vec<Lit> = shifted.iter().map(|&l| !l).collect();
                let (new_result, _) =
                    self.ripple_carry_add(&result, &neg_shifted, one);
                result = new_result;
            }
        }
        result
    }

    /// If `t` is a BV constant, return its value clamped to `usize` so we
    /// can reshape it into a shift amount. Used to dispatch constant-amount
    /// shifts into the pure-wiring path.
    ///
    /// Handles both inline (width ≤ 128, value stored in `node.value`) and
    /// wide (width > 128, value stored in the wide-limbs table) constants.
    /// Wide shift amounts far exceeding the shiftee's width are common (e.g.
    /// a 184-bit shift-by-8 over a 184-bit BV) and must still be recognised
    /// as constants or the solver falls back to the symbolic-shift path and
    /// silently bitblasts as shift-by-zero.
    fn const_shift_amt(&self, t: BvTerm) -> Option<usize> {
        let node = self.ctx.bv_nodes[t.0 as usize];
        if !matches!(node.op, BvOp::Const) {
            return None;
        }
        // Inline: value fits in u128 (width ≤ 128).
        if node.wide == crate::bv::WIDE_NONE {
            return Some(node.value.min(usize::MAX as u128) as usize);
        }
        // Wide: read from the limb table. A shift amount above usize::MAX
        // saturates — the wiring path then treats it as ≥ width and zero-fills.
        let limbs = self.ctx.bv_const_value_limbs(t);
        if limbs.iter().skip(2).any(|&l| l != 0) {
            return Some(usize::MAX);
        }
        let lo = *limbs.first().unwrap_or(&0);
        let hi = *limbs.get(1).unwrap_or(&0);
        if hi != 0 {
            // Value doesn't fit in u64 → saturate if usize < 128-bit.
            if (usize::BITS as usize) < 128 {
                return Some(usize::MAX);
            }
            let v128 = (hi as u128) << 64 | (lo as u128);
            return Some(v128.min(usize::MAX as u128) as usize);
        }
        Some((lo as u128).min(usize::MAX as u128) as usize)
    }

    /// Constant-amount left shift: zero new gates, just rewiring.
    fn mk_shl_const(&mut self, a: &[Lit], amt: usize) -> Vec<Lit> {
        let n = a.len();
        let zero = !self.get_true_lit();
        let amt = amt.min(n); // ≥width clears the vector
        (0..n)
            .map(|i| if i < amt { zero } else { a[i - amt] })
            .collect()
    }

    /// Constant-amount right shift with explicit fill (zero for lshr, sign
    /// bit for ashr).
    fn mk_shr_const(&mut self, a: &[Lit], amt: usize, fill: Lit) -> Vec<Lit> {
        let n = a.len();
        let amt = amt.min(n);
        (0..n)
            .map(|i| {
                let src = i + amt;
                if src < n { a[src] } else { fill }
            })
            .collect()
    }

    /// Unsigned left shift with variable amount. Log-layer barrel shifter:
    /// at stage i, conditionally shift by 2^i iff bit i of the amount is set.
    /// If the amount is >= width, the result is all zeros.
    fn mk_shl(&mut self, a: &[Lit], amt: &[Lit]) -> Vec<Lit> {
        let n = a.len();
        assert_eq!(amt.len(), n);
        let zero = !self.get_true_lit();
        let log_n = ceil_log2(n);

        let mut cur = a.to_vec();
        for i in 0..log_n {
            let shift = 1usize << i;
            if shift >= n { break; }
            let shifted: Vec<Lit> = (0..n)
                .map(|j| if j < shift { zero } else { cur[j - shift] })
                .collect();
            cur = self.mux_vec(amt[i], &shifted, &cur);
        }

        // Overflow: if any of amt[log_n..n] is set the shift ≥ n, so clear.
        self.maybe_zero_on_overflow(&cur, amt, log_n, zero)
    }

    /// Right shift (logical or arithmetic) with variable amount. The
    /// `fill` literal determines what streams in from the top.
    fn mk_shr(&mut self, a: &[Lit], amt: &[Lit], fill: Lit) -> Vec<Lit> {
        let n = a.len();
        assert_eq!(amt.len(), n);
        let log_n = ceil_log2(n);

        let mut cur = a.to_vec();
        for i in 0..log_n {
            let shift = 1usize << i;
            if shift >= n { break; }
            let shifted: Vec<Lit> = (0..n)
                .map(|j| if j + shift < n { cur[j + shift] } else { fill })
                .collect();
            cur = self.mux_vec(amt[i], &shifted, &cur);
        }

        // Overflow: amt >= n. Replace all bits with `fill` (0 for lshr,
        // sign for ashr).
        self.maybe_fill_on_overflow(&cur, amt, log_n, fill)
    }

    /// After the main barrel stages, if any high bit of `amt` is set the
    /// requested shift was >= width — zero out the result in that case.
    fn maybe_zero_on_overflow(
        &mut self,
        cur: &[Lit],
        amt: &[Lit],
        log_n: usize,
        zero: Lit,
    ) -> Vec<Lit> {
        if log_n >= amt.len() {
            return cur.to_vec();
        }
        let high = &amt[log_n..];
        let any_high = self.mk_any_set(high);
        cur.iter()
            .map(|&bit| self.mk_mux(any_high, zero, bit))
            .collect()
    }

    fn maybe_fill_on_overflow(
        &mut self,
        cur: &[Lit],
        amt: &[Lit],
        log_n: usize,
        fill: Lit,
    ) -> Vec<Lit> {
        if log_n >= amt.len() {
            return cur.to_vec();
        }
        let high = &amt[log_n..];
        let any_high = self.mk_any_set(high);
        cur.iter()
            .map(|&bit| self.mk_mux(any_high, fill, bit))
            .collect()
    }

    /// Wallace-tree multiplication. Same gate count as shift-and-add, but
    /// the critical path collapses from O(N) to O(log N) via carry-save
    /// reduction — shallower implication chains, which matters a lot for
    /// SAT propagation on symbolic multiplies.
    ///
    /// Algorithm:
    ///   1. Build the partial-product triangle as a list of bits per output
    ///      column.
    ///   2. Repeatedly reduce: for each column with ≥3 bits, apply a full
    ///      adder (3:2 compressor). The `sum` stays in the column; the
    ///      `carry` spills into the next column. Leftover 1 or 2 bits pass
    ///      through unchanged.
    ///   3. After log_{3/2}(N/2) rounds every column has at most 2 bits —
    ///      do a single ripple-carry add for the final result.
    fn mk_mul(&mut self, a: &[Lit], b: &[Lit]) -> Vec<Lit> {
        let n = a.len();
        assert_eq!(b.len(), n);
        let zero = !self.get_true_lit();

        // Step 1: partial products collected by output column. Skip any
        // product whose operand is the false literal — these are the bits
        // that zero-extensions, masked-away positions, and bits-known folds
        // reduce to at bitblast time. Pushing the false lit would correctly
        // short-circuit through `mk_and` but needlessly inflates column
        // lengths, causing extra 3:2 compressions in the Wallace reduction
        // below. Skipping at source keeps columns as tight as they can be.
        let mut columns: Vec<Vec<Lit>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..n {
            if b[i] == zero {
                continue; // entire "row" shifted by i contributes nothing
            }
            for j in i..n {
                let ajm = a[j - i];
                if ajm == zero {
                    continue; // this single partial product is zero
                }
                columns[j].push(self.mk_and(ajm, b[i]));
            }
        }

        // Step 2: reduce to ≤ 2 bits per column.
        loop {
            let max_len = columns.iter().map(|c| c.len()).max().unwrap_or(0);
            if max_len <= 2 {
                break;
            }
            let mut next: Vec<Vec<Lit>> = (0..n).map(|_| Vec::new()).collect();
            for k in 0..n {
                let col = std::mem::take(&mut columns[k]);
                let mut i = 0;
                while i + 2 < col.len() {
                    let (sum, carry) = self.mk_full_adder(col[i], col[i + 1], col[i + 2]);
                    next[k].push(sum);
                    if k + 1 < n {
                        next[k + 1].push(carry);
                    }
                    // else: carry falls off the top (truncated width).
                    i += 3;
                }
                while i < col.len() {
                    next[k].push(col[i]);
                    i += 1;
                }
            }
            columns = next;
        }

        // Step 3: final ripple-carry add of the (≤ 2) remaining rows.
        let row0: Vec<Lit> = columns
            .iter()
            .map(|c| if c.is_empty() { zero } else { c[0] })
            .collect();
        let row1: Vec<Lit> = columns
            .iter()
            .map(|c| if c.len() < 2 { zero } else { c[1] })
            .collect();
        let cin = zero;
        self.ripple_carry_add(&row0, &row1, cin).0
    }

    /// Unsigned division + remainder via non-restoring division. Returns
    /// (quotient, remainder). Saves one ripple-add + one mux-vec per
    /// iteration compared to restoring: we always either add or subtract
    /// the divisor based on the current remainder's sign, and recover the
    /// correct quotient bit from the new sign. At the end, a single
    /// conditional restoration fixes up a negative remainder.
    ///
    /// Arithmetic is done in (N+2)-bit signed form. The extra bit past
    /// the sign keeps the shifted remainder `2*prev_r` from overflowing
    /// when `|prev_r|` approaches `|b|` near `2^N`. Division-by-zero is
    /// handled in the callers.
    fn mk_udivmod(&mut self, a: &[Lit], b: &[Lit]) -> (Vec<Lit>, Vec<Lit>) {
        let n = a.len();
        assert_eq!(b.len(), n);
        let zero = !self.get_true_lit();

        // N+2 bits: one sign bit plus one slack bit for the `2 * r` step.
        let ext = n + 2;
        let mut r: Vec<Lit> = vec![zero; ext];
        let mut b_ext: Vec<Lit> = b.to_vec();
        b_ext.push(zero); // sign bit = 0 (b is always non-negative)
        b_ext.push(zero); // slack bit

        let mut q: Vec<Lit> = vec![zero; n];

        for i in (0..n).rev() {
            // r := (r << 1) | a[i]  — shift up by one, introduce next
            // bit of the dividend at the LSB. Width stays N+2 (top bit
            // falls off, but since we picked ext = n + 2, the worst-case
            // |2*r| ≤ 2*b < 2^(n+1) still fits as signed).
            let mut shifted = vec![zero; ext];
            shifted[0] = a[i];
            for j in 1..ext {
                shifted[j] = r[j - 1];
            }
            r = shifted;

            // Sign of current r (the top bit of the (N+2)-bit value).
            // If r ≥ 0 we subtract b, else add b. The XOR + carry-in pair
            // encodes the choice without any mux.
            let sign = r[ext - 1];
            let not_sign = !sign;
            let effective_b: Vec<Lit> =
                b_ext.iter().map(|&bb| self.mk_xor(bb, not_sign)).collect();
            let (new_r, _cout) = self.ripple_carry_add(&r, &effective_b, not_sign);
            r = new_r;

            // Quotient bit = 1 iff the new remainder is non-negative.
            q[i] = !r[ext - 1];
        }

        // Final restoration: if r went negative, add b back once.
        let final_sign = r[ext - 1];
        let cin = zero;
        let (restored, _cout) = self.ripple_carry_add(&r, &b_ext, cin);
        let r_final = self.mux_vec(final_sign, &restored, &r);

        // Truncate remainder back to N bits (high bits are zero after
        // the restoration step).
        (q, r_final[..n].to_vec())
    }

    /// High N bits of the unsigned 2N-bit product of two N-bit operands.
    /// Used by unsigned-multiplication overflow detection.
    fn mk_umul_hi(&mut self, a: &[Lit], b: &[Lit]) -> Vec<Lit> {
        let n = a.len();
        assert_eq!(b.len(), n);
        let zero = !self.get_true_lit();
        let double_n = 2 * n;

        // Zero-extend both operands to 2N bits and run the same shift-and-add
        // we use for regular multiplication. Keep the top N bits.
        let mut a_ext = a.to_vec();
        a_ext.resize(double_n, zero);
        let mut b_ext = b.to_vec();
        b_ext.resize(double_n, zero);

        let mut result: Vec<Lit> = vec![zero; double_n];
        for i in 0..double_n {
            let partial: Vec<Lit> = (0..double_n)
                .map(|j| {
                    if j < i {
                        zero
                    } else {
                        self.mk_and(a_ext[j - i], b_ext[i])
                    }
                })
                .collect();
            let cin = zero;
            let (new_result, _) = self.ripple_carry_add(&result, &partial, cin);
            result = new_result;
        }
        result[n..].to_vec()
    }

    /// Full 2N-bit signed product: (low N bits, high N bits). Sign-extends
    /// both operands to 2N bits then multiplies. Used by signed-multiplication
    /// overflow detection.
    fn mk_smul_full(&mut self, a: &[Lit], b: &[Lit]) -> (Vec<Lit>, Vec<Lit>) {
        let n = a.len();
        assert_eq!(b.len(), n);
        let double_n = 2 * n;

        // Sign-extend both to 2N bits.
        let a_sign = a[n - 1];
        let b_sign = b[n - 1];
        let mut a_ext = a.to_vec();
        a_ext.resize(double_n, a_sign);
        let mut b_ext = b.to_vec();
        b_ext.resize(double_n, b_sign);

        let prod = self.mk_mul(&a_ext, &b_ext);
        let (lo, hi) = prod.split_at(n);
        (lo.to_vec(), hi.to_vec())
    }

    /// Two's-complement negation: `-x = ~x + 1`.
    fn mk_neg(&mut self, x: &[Lit]) -> Vec<Lit> {
        let neg: Vec<Lit> = x.iter().map(|&l| !l).collect();
        let zero: Vec<Lit> = vec![!self.get_true_lit(); x.len()];
        let cin = self.get_true_lit();
        self.ripple_carry_add(&zero, &neg, cin).0
    }

    /// Absolute value of a signed BV: returns `-x` if x's MSB is set, else x.
    fn mk_abs(&mut self, x: &[Lit]) -> Vec<Lit> {
        let sign = *x.last().unwrap();
        let neg = self.mk_neg(x);
        self.mux_vec(sign, &neg, x)
    }

    /// Signed division with SMT-LIB semantics. Computes absolute values,
    /// does an unsigned divide, then flips the sign of the result when
    /// exactly one operand was negative. Division-by-zero follows from the
    /// underlying udiv-by-zero (all-ones) case-split.
    fn mk_sdiv(&mut self, a: &[Lit], b: &[Lit]) -> Vec<Lit> {
        let n = a.len();
        let a_sign = a[n - 1];
        let b_sign = b[n - 1];

        let a_abs = self.mk_abs(a);
        let b_abs = self.mk_abs(b);
        let (q_abs, _) = self.mk_udivmod(&a_abs, &b_abs);

        // Flip sign of quotient iff exactly one operand was negative.
        let sign_diff = self.mk_xor(a_sign, b_sign);
        let q_neg = self.mk_neg(&q_abs);
        let q = self.mux_vec(sign_diff, &q_neg, &q_abs);

        // Divide-by-zero: sdiv(x, 0) = 1 if x signed-negative, else ~0.
        let b_zero = self.mk_all_zero(b);
        let tl = self.get_true_lit();
        let all_ones = vec![tl; n];
        // Constant 1 of width n.
        let mut one = vec![!tl; n];
        one[0] = tl;
        let dz = self.mux_vec(a_sign, &one, &all_ones);

        self.mux_vec(b_zero, &dz, &q)
    }

    /// Signed remainder — sign of result follows the dividend.
    /// Division-by-zero: srem(x, 0) = x (following SMT-LIB).
    fn mk_srem(&mut self, a: &[Lit], b: &[Lit]) -> Vec<Lit> {
        let n = a.len();
        let a_sign = a[n - 1];

        let a_abs = self.mk_abs(a);
        let b_abs = self.mk_abs(b);
        let (_q, r_abs) = self.mk_udivmod(&a_abs, &b_abs);

        let r_neg = self.mk_neg(&r_abs);
        let r = self.mux_vec(a_sign, &r_neg, &r_abs);

        let b_zero = self.mk_all_zero(b);
        self.mux_vec(b_zero, a, &r)
    }

    /// Signed modulo — sign of result follows the divisor.
    /// Definition: `smod(a, b) = ite(r == 0, 0, ite(sign(a) == sign(b), r, r + b))`
    /// where `r = srem(a, b)` (before the sign adjustment).
    /// Division-by-zero: smod(x, 0) = x.
    fn mk_smod(&mut self, a: &[Lit], b: &[Lit]) -> Vec<Lit> {
        let n = a.len();
        let a_sign = a[n - 1];
        let b_sign = b[n - 1];

        let a_abs = self.mk_abs(a);
        let b_abs = self.mk_abs(b);
        let (_q, r_abs) = self.mk_udivmod(&a_abs, &b_abs);

        // Magnitude-signed remainder (matches srem semantics).
        let r_neg = self.mk_neg(&r_abs);
        let r_srem = self.mux_vec(a_sign, &r_neg, &r_abs);

        // Zero remainder: result is 0 regardless of signs.
        let r_is_zero = self.mk_all_zero(&r_srem);

        // When sign(a) != sign(b), add b to push the result into the
        // divisor's sign half-plane.
        let r_plus_b = {
            let cin = !self.get_true_lit();
            self.ripple_carry_add(&r_srem, b, cin).0
        };
        let sign_diff = self.mk_xor(a_sign, b_sign);
        let adjusted = self.mux_vec(sign_diff, &r_plus_b, &r_srem);

        // If the raw remainder was zero, the answer is zero.
        let zero = vec![!self.get_true_lit(); n];
        let with_zero = self.mux_vec(r_is_zero, &zero, &adjusted);

        let b_zero = self.mk_all_zero(b);
        self.mux_vec(b_zero, a, &with_zero)
    }
}

impl Default for SmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Flip the sign bit (MSB) of a bitblasted BV — used for signed comparisons.
fn flip_msb(bits: &[Lit]) -> Vec<Lit> {
    let mut r = bits.to_vec();
    let last = r.len() - 1;
    r[last] = !r[last];
    r
}

/// Low `w` bits of a u128 mask, clamped. `w >= 128` returns all-ones.
#[inline]
fn mask_u128(w: u32) -> u128 {
    if w >= 128 {
        u128::MAX
    } else if w == 0 {
        0
    } else {
        (1u128 << w) - 1
    }
}

/// Compute the non-adjacent form (NAF, a.k.a. canonical signed digit) of
/// `c`: a sequence of signed digits `d[i] ∈ {-1, 0, 1}` such that
/// `sum(d[i] * 2^i) == c` and no two adjacent non-zero digits appear.
/// Returns only the non-zero digits as `(sign, position)` pairs, with
/// any digit at position ≥ `limit` dropped (they contribute `c * 2^limit`
/// which is zero under mod 2^limit arithmetic). Worst-case weight is
/// `⌈(width+1)/2⌉`, and long runs of 1-bits collapse to two digits —
/// exactly what we want when computing `x * c` via shift-and-add.
fn naf_recode(c: u128, limit: u32) -> Vec<(i8, u32)> {
    if c == 0 {
        return Vec::new();
    }
    // Standard Reitwiesner algorithm: at each bit, if `c` is odd, the
    // current digit is `1` or `-1` depending on whether `c mod 4` is 1
    // or 3. Subtract the digit and shift. This produces NAF incrementally.
    let mut digits = Vec::new();
    let mut c = c;
    let mut pos = 0u32;
    while c != 0 {
        if c & 1 != 0 {
            let digit: i8 = if c & 3 == 1 { 1 } else { -1 };
            if pos < limit {
                digits.push((digit, pos));
            }
            // `c - digit`: either subtract 1 (c&3==1) or add 1 (c&3==3).
            // We use wrapping arithmetic on u128 — adding 1 to a large c
            // can overflow, but only if c had every bit set, in which
            // case we've already emitted every useful digit.
            c = if digit == 1 { c - 1 } else { c.wrapping_add(1) };
        }
        c >>= 1;
        pos += 1;
    }
    digits
}

/// `ceil(log2(n))` for n >= 1. Zero for n <= 1.
fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

/// Sanity helper — unused by the solver itself but useful for tests.
#[inline]
pub fn bv_value_fits(value: u128, width: u32) -> bool {
    value & !mask(width) == 0
}

/// Pack up to two little-endian u64 limbs into a u128. Extra limbs are
/// ignored — the caller is responsible for ensuring the width fits.
#[inline]
fn limbs_to_u128(limbs: &[u64]) -> u128 {
    let lo = limbs.first().copied().unwrap_or(0) as u128;
    let hi = limbs.get(1).copied().unwrap_or(0) as u128;
    lo | (hi << 64)
}

/// Interpret `limbs` as a two's-complement integer of `width` bits and
/// return the sign-extended i128. Width must be ≤ 128.
#[inline]
fn sign_extend_limbs_i128(limbs: &[u64], width: u32) -> i128 {
    let v = limbs_to_u128(limbs);
    if width == 128 {
        v as i128
    } else {
        let shift = 128 - width;
        ((v as i128) << shift) >> shift
    }
}
