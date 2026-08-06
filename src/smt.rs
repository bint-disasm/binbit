//! SMT solver for quantifier-free bitvector logic (QF_BV).
//!
//! Strategy: eager bitblasting through an And-Inverter Graph. Every BV term
//! of width N becomes N AIG references (one per bit, LSB-first); every
//! Boolean term becomes one AIG reference. No CNF is emitted while
//! bitblasting — clauses are generated at `solve*` time by walking the AIG
//! cone reachable from the asserted roots (`lit_of`). Three wins over the
//! older Lit-first Tseitin pipeline:
//!
//!   - **Cross-operator structural dedup.** `bvor(a, b)` and
//!     `bvnot(bvand(bvnot(a), bvnot(b)))` are the same AIG node because OR
//!     is `!and(!, !)` with the inversions on the edges. The old per-gate
//!     `(Lit, Lit) → Lit` caches only caught pairwise coincidences.
//!   - **Cone-of-influence CNF.** Logic that never feeds an asserted root
//!     (dead quotients, unread flag bits, shadowed branches) never reaches
//!     the SAT solver at all.
//!   - **A substrate for AIG-level rewriting** (fraiging via
//!     `Aig::simulate`, cut sweeping, …) before any clause is committed.
//!
//! CNF quality is preserved by shape-aware emission: `lit_of` recognizes
//! the 3-node And patterns that `xor` / `mux` construction produces and
//! emits the direct 4-clause, single-output-var Tseitin encodings for
//! them, so adders and muxes cost exactly what they did under the old
//! direct encoder (5 vars / 17 clauses per full adder) while everything
//! still participates in AIG dedup.
//!
//! Supported BV ops: not, and, or, xor, add, sub, mul, udiv, urem, shl,
//! lshr, ashr, extract, concat, zero-extend, sign-extend, ite.
//! Comparisons: eq/ne, ult/ule/ugt/uge (unsigned), slt/sle/sgt/sge (signed).

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use crate::aig::{Aig, AigNode, AigRef};
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

/// Cut roots with at most this many defining clauses stay eliminable by
/// BVE. A classic AND gate is 3 clauses and an XOR/MUX 4, so this lets
/// the eliminator keep working on everything the mapper happened to
/// leave gate-shaped while protecting the genuinely wide covers.
const CUT_BVE_MAX_CLAUSES: usize = 4;

/// Which gate produced a SAT variable. Kept deliberately small so that
/// downstream code can `match` exhaustively. With the AIG pipeline, `And`
/// covers plain AND nodes; `Xor` / `Ite` cover the pattern-mapped 3-node
/// shapes that get the direct 4-clause encodings.
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
    /// Root of a WIDE cut chosen by CNF technology mapping
    /// (`crate::cnfmap`) — one whose definition is bigger than a Tseitin
    /// gate's. Bounded variable elimination must not resolve these away:
    /// the definition is two multi-cube ISOP covers, so eliminating the
    /// root splices them into far wider resolvents and destroys the
    /// propagation structure the mapper built.
    ///
    /// Cut roots whose definition is gate-sized (see
    /// `CUT_BVE_MAX_CLAUSES`) are tagged as ordinary gate outputs
    /// instead, so BVE still gets to eliminate the cheap ones — the two
    /// passes divide the work rather than competing for it.
    Cut,
}

/// Recorded ITE gate: semantically `o ↔ (sel ∧ t) ∨ (¬sel ∧ e)`. Registered
/// at flush time for every `mk_mux` whose output was actually materialized
/// to CNF (dead-code muxes never show up). Stored in `SmtSolver::ite_gates`
/// so ITE-aware propagation / branching can look gates up cheaply.
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
    /// Decision levels preserved across restarts by reuse-trail.
    pub reused_levels: u64,
    /// Vivification: probed / shortened / deleted-as-implied / root units.
    pub viv_checked: u64,
    pub viv_strengthened: u64,
    pub viv_deleted: u64,
    pub viv_units: u64,
    /// Flush-phase wall clocks (seconds): term-level front end, AIG
    /// bitblast + CNF emission, CNF preprocessing, SAT search.
    pub time_front: f64,
    pub time_emit: f64,
    pub time_preprocess: f64,
    pub time_sat: f64,
    pub learned: u64,
    pub propagations: u64,
    /// Learned-DB reductions and clause-arena garbage collections.
    pub reductions: u64,
    pub gcs: u64,
    pub bv_aliased: usize,
    pub bool_aliased: usize,
    pub bv_var_total: usize,
    pub bv_nodes_total: usize,
    pub bv_vars_bitblasted: usize,
    /// Variables substituted away at the term level (`x = t` roots).
    pub pp_substituted: u64,
    /// Gate variables removed by CNF preprocessing (bounded VE).
    pub pp_eliminated: u64,
    /// Clauses removed by (self-)subsumption during CNF preprocessing.
    pub pp_subsumed: u64,
    /// VE-eliminated AIG nodes later re-materialized (fresh var + clauses,
    /// typically by an assumption probe) — each is wasted elimination work
    /// plus CNF bloat. High values ⇒ run that phase with `set_bve(false)`.
    pub pp_remat: u64,
    /// Literals removed by strengthening during CNF preprocessing.
    pub pp_strengthened: u64,
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

/// One row of the bitblast-cost report — see [`SmtSolver::bitblast_cost_report`].
/// `sat_vars` / `sat_clauses` are *exclusive* of subterms: gate vars are
/// charged to the BV term that was being bitblasted when their AIG node was
/// first created (`Aig::src_terms`, first-writer-wins), so a shared subterm's
/// cost stays on its own row.
#[derive(Copy, Clone, Debug)]
pub struct BitblastCostEntry {
    pub term: BvTerm,
    pub width: u32,
    pub sat_vars: usize,
    pub sat_clauses: usize,
}

/// A `mk_mux` that produced a real AIG mux structure (no fold applied).
/// Held until flush; converted into a public [`IteGate`] iff the output
/// node actually got materialized to CNF.
#[derive(Copy, Clone)]
struct PendingIte {
    sel: AigRef,
    t: AigRef,
    e: AigRef,
    out: AigRef,
    src: Option<BvTerm>,
}

/// Shape classification for an And node during CNF emission. See
/// [`SmtSolver::detect_shape`].
#[derive(Copy, Clone)]
enum NodeShape {
    /// `node ≡ x ⊕ y`.
    Xor(AigRef, AigRef),
    /// `node ≡ ¬mux(s, t, e)` — note the negation: the mux VALUE is the
    /// complement of the matched And node.
    NotMux { s: AigRef, t: AigRef, e: AigRef },
}

/// Which assignment [`SmtSolver::eval_refs_from`] and friends read: the
/// live SAT trail (valid right after a Sat solve) or the banked copy kept
/// across trail rewinds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelSource {
    Trail,
    Banked,
}

/// Flat clause batch for one flush: every literal in one buffer, clause
/// `i` occupying `ends[i-1]..ends[i]`. The emission sink, the CNF
/// pre-filter, the preprocessor arena, and the final SAT commit all share
/// this storage — see `commit_batch`.
#[derive(Default)]
struct CnfBuffer {
    data: Vec<Lit>,
    ends: Vec<u32>,
}

impl CnfBuffer {
    #[inline]
    fn push_slice(&mut self, c: &[Lit]) {
        self.data.extend_from_slice(c);
        self.ends.push(self.data.len() as u32);
    }

    fn clear(&mut self) {
        self.data.clear();
        self.ends.clear();
    }
}

pub struct SmtSolver {
    pub ctx: BvContext,
    sat: Solver,

    /// The bitblaster's intermediate representation. All Boolean structure
    /// lands here; CNF is emitted lazily by `lit_of` for the cone reachable
    /// from asserted roots.
    aig: Aig,

    /// AIG node idx → SAT lit of the node's positive output. `None` until
    /// the node is materialized by `lit_of`. Dense Vec keyed by node index.
    aig_lit: Vec<Option<Lit>>,

    /// Reverse map: SAT var idx → AIG node idx (`u32::MAX` = none). Needed
    /// so CNF preprocessing can un-bind eliminated gate variables from
    /// their AIG nodes (a later consumer then re-materializes the node
    /// under a fresh variable with fresh defining clauses).
    lit_node: Vec<u32>,

    /// FRAIG-merged alias nodes bound to another node's SAT lit: var idx →
    /// the alias node indices sharing it. `lit_node` is one-to-one and
    /// stays pointing at the node that emitted the defining clauses; this
    /// side map lets `commit_batch` invalidate every alias binding when
    /// bounded VE eliminates the shared variable (else an alias would keep
    /// serving a lit whose defining clauses are gone — unsound).
    aig_lit_aliases: HashMap<u32, Vec<u32>>,

    /// FRAIG sweep (off by default — changes search trajectory): prove
    /// sim-equivalence candidates in the batch AIG with bounded SAT
    /// queries and merge them before CNF emission. See `crate::fraig`.
    fraig_enabled: bool,
    /// Nodes below this index were candidates in a previous sweep.
    fraig_swept_upto: u32,
    fraig_stats: crate::fraig::FraigStats,
    fraig_time: std::time::Duration,
    /// Cumulative flush-phase wall clocks — front-end attribution for
    /// instances the SAT engine barely touches (`--stats` prints them):
    /// term-level work (substitution / Gaussian / normalization),
    /// AIG bitblast + CNF emission, CNF preprocessing, and SAT search.
    time_front: std::time::Duration,
    time_emit: std::time::Duration,
    time_preprocess: std::time::Duration,
    time_sat: std::time::Duration,

    /// Sharing-aware post-build substitution pass (`--aig2-post`): the
    /// two-level substitution family applied only where it cannot strand
    /// a co-parent. See `Aig::substitute_pass`.
    aig2_post: bool,

    /// Cut-based CNF technology mapping at materialization (see
    /// `crate::cnfmap`): cover each unmaterialized cone with k-feasible
    /// cuts and give SAT variables to cut roots only, defined by the
    /// ISOP of the cut function.
    ///
    /// OFF by default, and the reason is worth recording: on every
    /// single-shot artifact available here mapping wins (4 of 5 corpus
    /// instances, and nobranch.smt2 at 27.3s vs 32.8s classic), but in
    /// bint's real incremental session the SAME workload is 2.4× SLOWER
    /// (66.7s vs 27.8s) with BVE eliminating 4× fewer variables
    /// (164k → 40k) and propagations up 2.45× against only +39%
    /// conflicts — i.e. much wider clauses. Whatever bint builds
    /// incrementally is not what the .smt2 dumps replay (their variable
    /// counts differ by ~30%), so the dumps cannot currently validate
    /// this feature. Do not flip this default again without measuring in
    /// the real tool.
    cnf_mapping: bool,
    /// Persistent mapper scratch (buffers survive across cones).
    /// One mapper per effort level. The cut stride is a type parameter
    /// so the Fast mapper's per-node cut array is genuinely half the
    /// size (the mapper's hottest structure — measured ~6%); the unused
    /// one holds only empty buffers.
    cnfmap_mapper: crate::cnfmap::Mapper<{ crate::cnfmap::FAST_CUTS }>,
    cnfmap_mapper_full: crate::cnfmap::Mapper<{ crate::cnfmap::MAX_CUTS }>,
    /// Reused plan arena and leaf-literal scratch for mapped emission —
    /// the plan is a flat arena (cf. `ClauseArena`), so a cone costs no
    /// allocations once these are warm.
    cnfmap_plan: crate::cnfmap::Plan,
    cnfmap_effort: crate::cnfmap::Effort,
    cnfmap_leaf_lits: Vec<Lit>,
    /// Cross-cone ISOP cache for the CNF mapper (see `cnfmap::IsopCache`).
    cnfmap_cache: crate::cnfmap::IsopCache,
    aig2_post_stats: crate::aig::PostPassStats,

    /// AIG nodes whose SAT binding was dropped by bounded VE. If a later
    /// consumer (typically an assumption probe under bint's
    /// solve_under_assumptions usage) re-materializes one, the SAT core
    /// ends up holding BOTH the elimination resolvents and the fresh gate
    /// clauses — VE work wasted plus CNF bloat. `pp_remat` counts these
    /// re-materializations; a high count on a workload means VE is
    /// dissolving cones the assumptions keep probing and should be
    /// disabled for that phase (`set_bve(false)`).
    elim_nodes: HashSet<u32>,
    pp_remat: u64,

    /// Gate-mix counters: how CNF emission encoded each materialized gate.
    /// Diagnostic for encoding-shape changes (e.g. AIG rewriting breaking
    /// the XOR/MUX pattern paths and demoting them to generic ANDs).
    stats_and_gates: u64,
    stats_xor_gates: u64,
    stats_mux_gates: u64,

    /// When `Some`, clause emission is buffered here instead of going to
    /// the SAT core — active during `flush_pending` so the whole batch can
    /// be preprocessed (subsumption + bounded variable elimination) before
    /// commitment. `None` = direct mode (model probes, assumptions,
    /// named assertions). Flat CSR layout (all literals in one buffer,
    /// exclusive end offsets per clause): the buffer flows straight into
    /// the preprocessor's arena and back out — no per-clause `Vec`
    /// anywhere on the flush path.
    cnf_buffer: Option<CnfBuffer>,
    /// Retired flush buffers, reused across flushes so a steady-state
    /// session stops allocating for emission entirely.
    cnf_buffer_pool: CnfBuffer,
    /// VE gate-substitution policy: `None` = automatic (enabled unless
    /// two-level AIG rewriting is active — the two circuit minimizers
    /// stacked measured strongly net-negative on Sage2-class instances,
    /// +122% on bench_16728 under --aig2, while EACH ALONE wins);
    /// `Some(x)` forces it. The override exists so bint can A/B the
    /// combination on real incremental sessions — the one validated
    /// aig2 dump (nobranch2) actually LIKED the combination (−9% props),
    /// so the auto policy is conservative, not optimal, under aig2.
    ve_gate_subst: Option<bool>,
    /// Recycled per-flush preprocessor storage (see `PreprocessPool`).
    pp_pool: crate::preprocess::PreprocessPool,
    /// Recycled compact-remap buffers for `commit_batch`.
    pp_to_orig: Vec<u32>,
    pp_to_compact: HashMap<u32, u32>,

    /// Top-level variable substitution: `x → t` for every flush-time
    /// assertion root `(= x t)` where `x` is an un-bitblasted variable not
    /// occurring in `t` (transitively, through the map). Consulted by
    /// `bitblast_bv`'s Var arm — a substituted variable never allocates
    /// SAT bits; model reads evaluate `t` instead, so `get_bv_value(x)`
    /// stays consistent. Keyed by union-find ROOT variable id.
    bv_var_subst: HashMap<u32, BvTerm>,
    /// Memo maps for applying the substitution to assertion DAGs. Cleared
    /// whenever the substitution map grows (older rewrites may be stale
    /// with respect to newly-accepted substitutions).
    subst_bool_memo: HashMap<BoolTerm, BoolTerm>,
    subst_bv_memo: HashMap<BvTerm, BvTerm>,
    /// Cumulative number of installed substitutions (stats).
    pp_substituted: u64,

    /// Arithmetic normalization (flatten/cancel bvadd chains under
    /// comparisons) applied to every pending assertion at flush. Memo maps
    /// persist across flushes — terms are immutable, so a term's normal
    /// form never changes.
    normalize_enabled: bool,
    norm_bool_memo: HashMap<BoolTerm, BoolTerm>,
    norm_bv_memo: HashMap<BvTerm, BvTerm>,

    /// Per-pass preprocessing toggles, all on by default. Independent
    /// switches so a pathological instance can be bisected — variable
    /// elimination in particular can *hurt* search on some formulas
    /// (eliminating gate vars that gave the SAT solver short resolution
    /// proofs, lengthening conflict analysis) even though it shrinks the
    /// formula. `subst_enabled` gates single-variable `(= x t)`
    /// substitution; `gauss_enabled` gates the coupled-system Gaussian
    /// elimination; `bve_enabled` gates the CNF-level bounded variable
    /// elimination + subsumption pass (subsumption alone is cheap and
    /// rarely harmful, but they share the `Preprocessor` so the toggle
    /// covers both).
    subst_enabled: bool,
    gauss_enabled: bool,
    bve_enabled: bool,

    /// Cumulative preprocessing counters (vars eliminated / clauses
    /// subsumed / literals strengthened) — exposed via [`sat_stats`].
    pp_eliminated: u64,
    pp_subsumed: u64,
    pp_strengthened: u64,

    // Bitblast caches: each BV/Bool term is translated exactly once and the
    // result reused on subsequent references. Critical for shared DAGs —
    // without this, a shared subterm could be re-encoded combinatorially
    // many times.
    bv_cache: HashMap<BvTerm, Vec<AigRef>>,
    bool_cache: HashMap<BoolTerm, AigRef>,

    // AIG input encoding for symbolic variables. Populated lazily on first
    // use so we don't allocate SAT vars for unused symbols. Each bit is an
    // `Aig::input` wrapping a freshly-allocated SAT literal.
    bv_var_refs: HashMap<u32, Vec<AigRef>>,
    bool_var_refs: HashMap<u32, AigRef>,

    // Union-find over BV / Bool variable ids. After `alias_bv_vars(x, y)`,
    // both BvVar(x_id) and BvVar(y_id) resolve to the same AIG inputs, so
    // `(= x y)` becomes a free no-op. Populated lazily from the SMT-LIB
    // layer when `(assert (= X Y))` is seen with X and Y both declared vars.
    bv_var_parent: Vec<u32>,
    bool_var_parent: Vec<u32>,

    // Reusable single SAT lit pinned to true — only needed when a constant
    // AIG ref must appear in a clause (e.g. `(assert true)` roots). All
    // other constant handling folds inside the AIG.
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

    // Every AIG root a flushed (or named) assertion's clauses reference —
    // the permanent pin set for `retire_dead_cones`: anything reachable
    // from these must keep its SAT bindings, because their clauses are
    // semantic, not definitional. Grows monotonically; roots of popped
    // scopes stay pinned (conservative — their clauses are vacuous but
    // still reference the cones).
    asserted_roots: Vec<AigRef>,

    // Last solve result if the formula hasn't been modified since. Reading
    // model values before any `solve*` or after a state-changing command
    // (assert, push, pop) is meaningless per SMT-LIB — `has_model()` lets
    // callers check first.
    last_result: Option<SmtResult>,

    // Banked model: a copy of the SAT assignment from a past Sat solve
    // (literal-indexed, same layout as the solver's own table), kept at
    // this layer because the trail model is destroyed by far more than
    // semantic change — known-bits probes, Tseitin emission for a new
    // term, even a later Unsat solve all rewind the trail while every
    // model of the OLD clause set is still a model of the new one. The
    // copy is what `solve_each_under_assumptions` screens candidates
    // against ("warm start"). Only genuinely semantic events invalidate
    // it: flushing pending assertions (which can also act by pure
    // substitution, hence poisoning the generation, see below). Scope
    // pops and `solve_many_u` sessions keep it valid — their clauses are
    // guarded by activation literals that occur only negatively, so the
    // banked model extends to a true model by completing those guards
    // with false.
    banked_model: Vec<LBool>,
    // Which `Solver::model_gen` the copy came from. Doubles as a poison
    // marker: invalidation records the CURRENT generation with
    // `banked_valid = false`, so a standing trail model whose semantics
    // were changed without emitting a clause (top-level substitution)
    // cannot be re-banked.
    banked_gen: u64,
    banked_valid: bool,

    // Assumption-prefix caches. Symbex queries pass an append-only
    // constraint list, so consecutive calls share a long prefix; a u32
    // term compare replaces a hash lookup per term, turning per-query
    // assumption processing from O(list) hashing into O(new suffix) real
    // work. `asmp_*` caches the previous `build_assumption_lits` call
    // (terms → materialized lits); `gate_*` caches the previous
    // `banked_model_holds` walk (terms → AIG refs, no CNF). Both are
    // cleared whenever a flush lands real work (rewrites/BVE can rebind
    // term meanings and lits) and on cone retirement.
    asmp_terms_cache: Vec<BoolTerm>,
    asmp_lits_cache: Vec<Lit>,
    gate_terms_cache: Vec<BoolTerm>,
    gate_refs_cache: Vec<AigRef>,

    // Dead-cone retirement, off by default. When off, the `asserted_roots`
    // pin list below is not recorded at all (nothing else reads it) and
    // `retire_dead_cones` is a no-op — so a solver that never retires
    // pays nothing for the feature. Must be enabled before the first
    // assertion, since the pin list is built as assertions are emitted.
    retirement_enabled: bool,

    // True once `retire_dead_cones` has run. Until then no variable was
    // ever un-branched by retirement, so `set_node_lit` can skip its
    // re-enable write — a per-gate-materialization cache-line store that
    // measured 1.5-2% on front-end-bound sessions that never retire.
    retirement_used: bool,

    // When false, the [activations | named controls] block and the
    // assumption-lit cache both mirror the list the SAT core solved last,
    // so hinted solves may vouch that prefix as unchanged and the solver
    // skips scanning it. Set by anything that breaks the mirror: scope
    // push/pop, named assertions, and build-without-solve paths (probes,
    // failed_assumptions). Cleared by every hinted solve.
    built_prefix_dirty: bool,

    // Structural-evaluation memo (`eval_refs_from`), node-indexed:
    // `eval_stamp[i] == eval_epoch` marks `eval_value[i]` as computed for
    // the current walk. Persistent buffers with an epoch bump per walk
    // replace a per-call hash map — model screening is on the batch fast
    // path where a solve is often near-pure propagation, so the check
    // must not pay allocation or hashing per node.
    eval_stamp: Vec<u32>,
    eval_value: Vec<bool>,
    eval_epoch: u32,
    eval_stack: Vec<u32>,

    // --- Metadata layer -----------------------------------------------------
    // Parallel to the SAT variable table: `var_origin[i]` records what SAT
    // variable `i` is for — a BV input bit, a gate output, an activation
    // literal, etc. Populated at allocation time.
    var_origin: Vec<VarOrigin>,
    // While bitblasting a given BV term, this holds its handle so that AIG
    // nodes created during the translation can be tagged with the enclosing
    // term (`Aig::tag_src`). Push/pop via save-and-restore in `bitblast_bv`.
    current_bv_ctx: Option<BvTerm>,

    // --- Bitblast cost attribution ----------------------------------------
    // Opt-in: when `bitblast_cost_enabled` is true, every SAT var / clause
    // emitted during materialization is charged to the BV term recorded in
    // the AIG node's `src_terms` tag. Because tagging is first-writer-wins
    // at node creation, shared subterms keep their cost on their own row —
    // the report is exclusive per term by construction.
    bitblast_cost_enabled: bool,
    bitblast_cost: HashMap<BvTerm, (usize, usize)>,

    // Deferred ITE-gate records. `mk_mux` can't hand out SAT lits (nothing
    // is materialized during bitblasting), so it queues the AigRefs here;
    // `flush_pending` converts records whose output node got a lit into
    // public `IteGate`s.
    pending_ite_gates: Vec<PendingIte>,
    // Every ITE gate whose CNF was emitted, in flush order.
    ite_gates: Vec<IteGate>,
    // Reverse index: for each SAT literal that's an ITE output, which gate
    // (by index into `ite_gates`) produced it.
    ite_out_to_gate: HashMap<Lit, usize>,

    // When true (the default), each live ITE gate bumps its `sel`
    // variable's VSIDS activity at flush time. This steers the SAT solver
    // toward branching on selectors first — a single decision on `sel`
    // resolves the whole ITE subtree, which is a huge win on symex memory
    // reads that are encoded as deep ITE chains over the address variable.
    ite_branching_hints: bool,
}

impl SmtSolver {
    pub fn new() -> Self {
        SmtSolver {
            ctx: BvContext::new(),
            sat: Solver::new(),
            aig: Aig::new(),
            aig_lit: Vec::new(),
            lit_node: Vec::new(),
            aig_lit_aliases: HashMap::default(),
            fraig_enabled: false,
            fraig_swept_upto: 0,
            fraig_stats: crate::fraig::FraigStats::default(),
            fraig_time: std::time::Duration::ZERO,
            time_front: std::time::Duration::ZERO,
            time_emit: std::time::Duration::ZERO,
            time_preprocess: std::time::Duration::ZERO,
            time_sat: std::time::Duration::ZERO,
            stats_and_gates: 0,
            stats_xor_gates: 0,
            stats_mux_gates: 0,
            aig2_post: false,
            cnf_mapping: false,
            cnfmap_mapper: Default::default(),
            cnfmap_mapper_full: Default::default(),
            cnfmap_plan: Default::default(),
            cnfmap_effort: Default::default(),
            cnfmap_leaf_lits: Vec::new(),
            cnfmap_cache: Default::default(),
            aig2_post_stats: crate::aig::PostPassStats::default(),
            elim_nodes: HashSet::default(),
            pp_remat: 0,
            cnf_buffer: None,
            cnf_buffer_pool: CnfBuffer::default(),
            ve_gate_subst: None,
            pp_pool: Default::default(),
            pp_to_orig: Vec::new(),
            pp_to_compact: HashMap::default(),
            bv_var_subst: HashMap::default(),
            subst_bool_memo: HashMap::default(),
            subst_bv_memo: HashMap::default(),
            pp_substituted: 0,
            normalize_enabled: true,
            norm_bool_memo: HashMap::default(),
            norm_bv_memo: HashMap::default(),
            banked_model: Vec::new(),
            banked_gen: 0,
            banked_valid: false,
            asmp_terms_cache: Vec::new(),
            asmp_lits_cache: Vec::new(),
            gate_terms_cache: Vec::new(),
            gate_refs_cache: Vec::new(),
            retirement_enabled: false,
            retirement_used: false,
            built_prefix_dirty: true,
            eval_stamp: Vec::new(),
            eval_value: Vec::new(),
            eval_epoch: 0,
            eval_stack: Vec::new(),
            subst_enabled: true,
            gauss_enabled: true,
            bve_enabled: true,
            pp_eliminated: 0,
            pp_subsumed: 0,
            pp_strengthened: 0,
            // Modest seeds only. Pre-sizing these to 16k entries shaved
            // ~4% of rehashing off long sessions but cost every solver
            // construction ~600KB of allocate-and-zero — a bad trade for
            // workloads that run many small queries.
            bv_cache: HashMap::with_capacity_and_hasher(256, Default::default()),
            bool_cache: HashMap::with_capacity_and_hasher(256, Default::default()),
            bv_var_refs: HashMap::default(),
            bool_var_refs: HashMap::default(),
            bv_var_parent: Vec::new(),
            bool_var_parent: Vec::new(),
            true_lit: None,
            activation_stack: Vec::new(),
            pending: vec![Vec::new()],
            named_controls: Vec::new(),
            asserted_roots: Vec::new(),
            last_result: None,
            var_origin: Vec::new(),
            current_bv_ctx: None,
            bitblast_cost_enabled: false,
            bitblast_cost: HashMap::default(),
            pending_ite_gates: Vec::new(),
            ite_gates: Vec::new(),
            ite_out_to_gate: HashMap::default(),
            ite_branching_hints: true,
        }
    }

    /// Enable or disable arithmetic normalization of assertions at flush
    /// (bitwuzla-style bvadd flattening/cancellation under comparisons).
    /// On by default; off is useful for ablation benchmarks.
    pub fn set_normalization(&mut self, on: bool) {
        self.normalize_enabled = on;
    }

    /// Enable/disable single-variable `(= x t)` substitution. On by default.
    pub fn set_substitution(&mut self, on: bool) {
        self.subst_enabled = on;
    }

    /// Enable/disable Gaussian elimination of coupled linear systems. On by
    /// default. (Independent of [`set_substitution`], though both feed the
    /// same substitution map.)
    pub fn set_gaussian(&mut self, on: bool) {
        self.gauss_enabled = on;
    }

    /// Enable/disable the CNF-level preprocessing pass (bounded variable
    /// elimination + subsumption). On by default. Off makes `commit_batch`
    /// send clauses straight to the SAT core — useful to check whether
    /// variable elimination is helping or hurting search on a given
    /// instance (it can hurt: eliminating gate variables sometimes
    /// lengthens conflict analysis).
    pub fn set_bve(&mut self, on: bool) {
        self.bve_enabled = on;
    }

    /// Enable/disable SAT-level unsat-core construction on Unsat results
    /// (default on). With tracking off, [`unsat_core_names`] and
    /// [`failed_assumptions`] degrade to reporting only the single
    /// assumption whose installation clashed; callers that only ever ask
    /// "feasible or not?" (symbex branch loops) save an O(trail) core
    /// walk on every Unsat answer.
    pub fn set_core_tracking(&mut self, on: bool) {
        self.sat.set_core_tracking(on);
    }

    /// Enable dead-cone retirement (off by default).
    ///
    /// Must be called BEFORE the first assertion: retirement needs a pin
    /// list of asserted AIG roots, and that list is built as assertions
    /// are emitted. While disabled the list is not recorded at all and
    /// [`retire_dead_cones`] is a no-op, so a solver that never retires
    /// pays nothing.
    ///
    /// Retirement only pays when a long session's history is genuinely
    /// dead — it measured 5.7× SLOWER when called with a live set that
    /// still covers most of the formula (the sweep is O(everything),
    /// finds nothing to delete, and drops the standing trail).
    pub fn set_cone_retirement(&mut self, on: bool) {
        self.retirement_enabled = on;
    }

    /// Enable/disable cut-based CNF mapping at materialization (see the
    /// `cnf_mapping` field). Takes effect for cones materialized after
    /// the call; already-emitted CNF is untouched.
    pub fn set_cnf_mapping(&mut self, on: bool) {
        self.cnf_mapping = on;
    }

    /// Mapping effort for cut-based CNF mapping — see
    /// [`cnfmap::Mapper::set_effort`]. Default (false) is byte-identical
    /// to full effort on symbex-shaped instances at ~40% less mapping
    /// cost; `true` pays off on dense arithmetic (multiplier arrays).
    pub fn set_cnf_mapping_effort(&mut self, full: bool) {
        self.cnfmap_effort = if full {
            crate::cnfmap::Effort::Full
        } else {
            crate::cnfmap::Effort::Fast
        };
    }

    /// Enable or disable the ITE-aware branching hint. On (the default)
    /// means every live ITE gate boosts its selector's VSIDS activity at
    /// flush; off disables that boost entirely. Useful to benchmark the
    /// impact of the heuristic on a given workload.
    pub fn set_ite_branching_hints(&mut self, on: bool) {
        self.ite_branching_hints = on;
    }

    /// Enable/disable Brummayer-Biere two-level AIG rewriting in the
    /// bitblaster's `and()` (see `Aig::set_two_level`). Off by default —
    /// changes circuit structure and search trajectory.
    pub fn set_aig_two_level(&mut self, on: bool) {
        self.aig.set_two_level(on);
    }

    /// Restrict two-level rewriting to its safe subset (pure deletions),
    /// skipping the node-bypassing substitution / idem-4 families — the
    /// two rules that fragment the learned-clause vocabulary on shared
    /// DAGs. Only meaningful with [`set_aig_two_level`] on.
    pub fn set_aig_two_level_subst(&mut self, on: bool) {
        self.aig.set_two_level_subst(on);
    }

    /// Sharing-aware two-level rewriting: construction-time safe subset
    /// (pure deletions) + a post-build substitution pass gated on parent
    /// counts, so a substitution never bypasses a shared interior node.
    /// The compatible successor to plain `set_aig_two_level`.
    pub fn set_aig_two_level_post(&mut self, on: bool) {
        self.aig.set_two_level(on);
        self.aig.set_two_level_subst(false);
        self.aig2_post = on;
    }

    pub fn aig2_post_report(&self) -> crate::aig::PostPassStats {
        self.aig2_post_stats
    }

    /// Enable/disable the flush-time FRAIG sweep (see `crate::fraig`).
    /// Off by default: merging changes CNF shape and therefore search
    /// trajectory — benchmark per-corpus before adopting.
    pub fn set_fraig(&mut self, on: bool) {
        self.fraig_enabled = on;
    }

    /// Override the automatic VE gate-substitution policy (see the
    /// `ve_gate_subst` field doc). `set_ve_gate_substitution(None)`
    /// restores the automatic rule.
    pub fn set_ve_gate_substitution(&mut self, forced: Option<bool>) {
        self.ve_gate_subst = forced;
    }

    /// Enable clause vivification in the SAT core (see
    /// [`crate::solver::Solver::set_vivification`] for the corpus verdict
    /// that keeps it off by default).
    pub fn set_vivification(&mut self, on: bool) {
        self.sat.set_vivification(on);
    }

    /// Cumulative FRAIG sweep statistics and wall-clock time spent.
    pub fn fraig_report(&self) -> (crate::fraig::FraigStats, std::time::Duration) {
        (self.fraig_stats, self.fraig_time)
    }

    /// Two-level rewrite rule firings by family (see `Aig::rw_counts`).
    pub fn aig_rw_counts(&self) -> [u64; 6] {
        self.aig.rw_counts
    }

    /// Gate-mix report: (plain AND, XOR-pattern, MUX-pattern) gates emitted.
    pub fn gate_mix(&self) -> (u64, u64, u64) {
        (
            self.stats_and_gates,
            self.stats_xor_gates,
            self.stats_mux_gates,
        )
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
        if self.bv_var_refs.contains_key(&xid) || self.bv_var_refs.contains_key(&yid) {
            return false;
        }
        let rx = self.find_bv_var_root(xid);
        let ry = self.find_bv_var_root(yid);
        if self.bv_var_subst.contains_key(&rx) || self.bv_var_subst.contains_key(&ry) {
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
        if self.bool_var_refs.contains_key(&xid) || self.bool_var_refs.contains_key(&yid) {
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
        self.built_prefix_dirty = true;
        let phi_ref = self.bitblast_bool(t);
        if self.retirement_enabled {
            self.asserted_roots.push(phi_ref);
        }
        let phi = self.lit_of(phi_ref);
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
        self.built_prefix_dirty = true;
        let act = self.new_sat_lit_tagged(VarOrigin::Activation);
        self.activation_stack.push(act);
        self.pending.push(Vec::new());
    }

    /// Close the most recently-opened scope. All assertions made inside it
    /// become vacuous. Ignored if no scope is open.
    pub fn pop(&mut self) {
        self.last_result = None;
        self.built_prefix_dirty = true;
        if let Some(act) = self.activation_stack.pop() {
            // Retraction only weakens the formula, so a standing model
            // survives it semantically — bank it before the retire clause
            // rewinds the trail. (`act` occurs only negatively, so the
            // banked model completes to a true model with `act = false`.)
            self.bank_model();
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

    /// Copy the standing trail model (if any, not yet banked, and not
    /// poisoned) into the banked model. Called before operations that
    /// rewind the trail; the copy stays usable for warm screening until
    /// assertion semantics change. One memcpy of the assignment table,
    /// and only when a new model actually appeared since the last bank.
    fn bank_model(&mut self) {
        if self.sat.has_model() && self.banked_gen != self.sat.model_gen() {
            self.banked_gen = self.sat.model_gen();
            self.banked_valid = true;
            self.sat.copy_model_into(&mut self.banked_model);
        }
    }

    /// Semantic change: the banked model can no longer vouch for SAT
    /// answers. Recording the current generation also blocks re-banking a
    /// standing trail model that the change made stale without touching
    /// the trail (a flush absorbed by pure substitution).
    fn invalidate_banked_model(&mut self) {
        self.banked_valid = false;
        self.banked_gen = self.sat.model_gen();
    }

    /// Retire every materialized cone unreachable from the asserted
    /// formula and `live` — the assumption terms the caller may still
    /// use. Retired cones lose their SAT bindings, their CNF is deleted
    /// (along with every learned clause touching it — no longer implied
    /// once the definitions go), and their variables leave the decision
    /// set. This is the antidote to unbounded accretion in long
    /// assumption-driven sessions: without it, every query pays
    /// propagation, decision, and rewind costs proportional to all cones
    /// ever materialized; after retirement, cost tracks the live set.
    ///
    /// Sound because an unreachable materialized cone is purely
    /// definitional — its clauses only relate its own gate variables to
    /// its inputs, so deleting them cannot change satisfiability of the
    /// remaining formula. Retired cones re-materialize transparently
    /// (fresh variables and CNF) if a later query reaches them again —
    /// correctness is unaffected, the cost is re-emission.
    ///
    /// Call with the complete set of terms that may reappear in future
    /// queries (a symbex would pass its live states' constraint terms).
    /// Terms never bitblasted are fine to include (no cone, nothing to
    /// do). The standing SAT trail is dropped; the banked model survives
    /// (clause deletion only weakens the formula). `last_result` is
    /// cleared. Returns `(retired_sat_vars, deleted_clauses)`.
    pub fn retire_dead_cones(&mut self, live: &[BoolTerm]) -> (u64, u64) {
        if !self.retirement_enabled {
            // Without the pin list there is no sound live set to retire
            // against — every asserted cone would look dead.
            debug_assert!(
                false,
                "retire_dead_cones requires set_cone_retirement(true) before asserting"
            );
            return (0, 0);
        }
        self.last_result = None;
        self.retirement_used = true;
        self.flush_pending();

        // Live closure over the AIG: the constant, every asserted root,
        // and every live term's cached cone.
        let mut live_nodes = vec![false; self.aig.num_nodes()];
        let mut stack: Vec<u32> = vec![0];
        for &r in &self.asserted_roots {
            stack.push(r.node_idx());
        }
        for t in live {
            if let Some(r) = self.bool_cache.get(t) {
                stack.push(r.node_idx());
            }
        }
        while let Some(n) = stack.pop() {
            if live_nodes[n as usize] {
                continue;
            }
            live_nodes[n as usize] = true;
            if let AigNode::And(a, b) = self.aig.node(n) {
                stack.push(a.node_idx());
                stack.push(b.node_idx());
            }
        }

        // Unbind every variable whose defining node is dead and mark it
        // for the SAT-level purge.
        let mut marked = vec![false; self.sat.num_vars()];
        let mut retired = 0u64;
        for v in 0..self.lit_node.len().min(marked.len()) {
            let node = self.lit_node[v];
            if node != u32::MAX && !live_nodes[node as usize] {
                marked[v] = true;
                retired += 1;
                self.aig_lit[node as usize] = None;
                self.lit_node[v] = u32::MAX;
                self.elim_nodes.insert(node);
                if let Some(aliases) = self.aig_lit_aliases.remove(&(v as u32)) {
                    for n in aliases {
                        self.aig_lit[n as usize] = None;
                    }
                }
            }
        }
        // ITE-gate registry entries whose output variable is retired are
        // stale — drop them so `ite_gate_for_output` can't resurrect one.
        self.ite_out_to_gate
            .retain(|l, _| !marked.get(l.var_idx()).copied().unwrap_or(false));
        // Cached assumption lits may point at purged variables (their
        // clauses are gone — assuming them would be vacuous): drop the
        // prefix caches.
        self.asmp_terms_cache.clear();
        self.asmp_lits_cache.clear();
        self.gate_terms_cache.clear();
        self.gate_refs_cache.clear();

        let deleted = self.sat.purge_vars(&marked);
        (retired, deleted)
    }

    /// True iff the banked model is valid and satisfies the standing
    /// controls (scope activations, named-assertion controls) and every
    /// assumption term — i.e. it genuinely witnesses "assertions ∧
    /// assumptions" as Sat, so Sat answers can be read off it for free.
    /// Costs O(#assumptions) direct reads when the terms were
    /// materialized by an earlier solve; structural walks happen only for
    /// cones the model predates, and the scan exits at the first
    /// falsified assumption. No CNF is emitted (`lit_of` is never
    /// called): a negative answer leaves the solver byte-identical.
    fn banked_model_holds(&mut self, assumptions: &[BoolTerm]) -> bool {
        if !self.banked_valid {
            return false;
        }
        let controls_ok = self
            .activation_stack
            .iter()
            .copied()
            .chain(self.named_controls.iter().map(|(_, l)| *l))
            .all(|l| self.model_value_of(l, ModelSource::Banked) == LBool::True);
        if !controls_ok {
            return false;
        }
        // Prefix-cached term→ref mapping from the previous walk: shared
        // entries skip the bitblast-cache hash lookup entirely (a term's
        // ref never changes once built — the cache-clear on flush/retire
        // is purely defensive). Updated in place so an early exit leaves
        // exactly the processed prefix cached.
        let cached = self.gate_terms_cache.len();
        let shared = if assumptions.len() >= cached
            && assumptions[..cached] == self.gate_terms_cache[..]
        {
            cached
        } else {
            self.gate_terms_cache
                .iter()
                .zip(assumptions)
                .take_while(|(a, b)| *a == *b)
                .count()
        };
        self.gate_terms_cache.truncate(shared);
        self.gate_refs_cache.truncate(shared);
        for (i, &t) in assumptions.iter().enumerate() {
            let r = if i < shared {
                self.gate_refs_cache[i]
            } else {
                let r = self.bitblast_bool(t);
                self.gate_terms_cache.push(t);
                self.gate_refs_cache.push(r);
                r
            };
            let holds = match self.model_ref_fast(r, ModelSource::Banked) {
                Some(v) => v,
                None => self.eval_refs_from(&[r], ModelSource::Banked)[0],
            };
            if !holds {
                return false;
            }
        }
        true
    }

    pub fn solve(&mut self) -> SmtResult {
        self.flush_pending();
        let asmps = self.built_assumptions(&[]);
        let trusted = self.trusted_for(0);
        let result = match self.sat_solve_hinted(&asmps, trusted) {
            SolveResult::Sat => SmtResult::Sat,
            SolveResult::Unsat => SmtResult::Unsat,
        };
        self.last_result = Some(result);
        result
    }

    pub fn solve_under_assumptions(&mut self, assumptions: &[BoolTerm]) -> SmtResult {
        self.flush_pending();
        let (extras, shared) = self.build_assumption_lits_counted(assumptions);
        let asmps = self.built_assumptions(&extras);
        let trusted = self.trusted_for(shared);
        let result = match self.sat_solve_hinted(&asmps, trusted) {
            SolveResult::Sat => SmtResult::Sat,
            SolveResult::Unsat => SmtResult::Unsat,
        };
        self.last_result = Some(result);
        result
    }

    /// Bitblast each assumption to an AigRef, then materialize its SAT lit.
    /// Runs AFTER `flush_pending` so assumption expressions get their CNF
    /// emitted before the SAT call.
    ///
    /// Prefix-cached: the longest prefix shared with the previous call
    /// reuses its lits outright; only the tail pays bitblast-cache and
    /// materialization lookups. The tail keeps the historical two-phase
    /// order (bitblast everything, then materialize) so SAT variable
    /// allocation order is unchanged — a cold cache is byte-identical to
    /// the uncached implementation.
    fn build_assumption_lits(&mut self, assumptions: &[BoolTerm]) -> Vec<Lit> {
        self.build_assumption_lits_counted(assumptions).0
    }

    /// [`build_assumption_lits`] also returning how many leading lits are
    /// shared with the previous call — the basis for hinted solves.
    fn build_assumption_lits_counted(
        &mut self,
        assumptions: &[BoolTerm],
    ) -> (Vec<Lit>, usize) {
        // Append-only fast path: one vectorized slice compare instead of a
        // branchy per-element scan.
        let cached = self.asmp_terms_cache.len();
        let shared = if assumptions.len() >= cached
            && assumptions[..cached] == self.asmp_terms_cache[..]
        {
            cached
        } else {
            self.asmp_terms_cache
                .iter()
                .zip(assumptions)
                .take_while(|(a, b)| *a == *b)
                .count()
        };
        let mut lits = Vec::with_capacity(assumptions.len());
        lits.extend_from_slice(&self.asmp_lits_cache[..shared]);
        let mut refs: Vec<AigRef> = Vec::with_capacity(assumptions.len() - shared);
        for &t in &assumptions[shared..] {
            refs.push(self.bitblast_bool(t));
        }
        for r in refs {
            lits.push(self.lit_of(r));
        }
        // O(suffix) cache maintenance: the shared prefix is already right.
        self.asmp_terms_cache.truncate(shared);
        self.asmp_terms_cache.extend_from_slice(&assumptions[shared..]);
        self.asmp_lits_cache.truncate(shared);
        self.asmp_lits_cache.extend_from_slice(&lits[shared..]);
        (lits, shared)
    }

    /// Caller-vouched prefix for the next hinted solve. The standard
    /// assumption-list shape is [activations | named controls |
    /// assumption lits]; when the controls block still mirrors the last
    /// solved list, its length plus the shared lit prefix is provably
    /// unchanged.
    fn trusted_for(&self, shared_lits: usize) -> usize {
        if self.built_prefix_dirty {
            0
        } else {
            self.activation_stack.len() + self.named_controls.len() + shared_lits
        }
    }

    /// Hinted solve through the standard list shape; re-arms prefix trust.
    fn sat_solve_hinted(&mut self, asmps: &[Lit], trusted: usize) -> SolveResult {
        let t0 = std::time::Instant::now();
        let r = self.sat.solve_under_assumptions_hinted(asmps, trusted);
        self.time_sat += t0.elapsed();
        self.built_prefix_dirty = false;
        r
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
    pub fn solve_under_assumptions_bounded(
        &mut self,
        assumptions: &[BoolTerm],
        max_conflicts: u64,
    ) -> Option<SmtResult> {
        self.flush_pending();
        let extras = self.build_assumption_lits(assumptions);
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
    /// probes).
    pub fn solve_under_assumptions_timed(
        &mut self,
        assumptions: &[BoolTerm],
        timeout: std::time::Duration,
    ) -> Option<SmtResult> {
        self.flush_pending();
        let extras = self.build_assumption_lits(assumptions);
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

    /// Known bits of `x` under assumption terms, derived from bitblasting
    /// plus a single unit-propagation pass — no search, no conflicts, no
    /// learning. The assumption semantics match [`solve_under_assumptions`]
    /// exactly (same literal construction, including scope activation and
    /// named-assertion controls).
    ///
    /// Returns `(known_ones, known_zeros)` in the same shape as
    /// [`BvContext::bv_known_bits`], whose construction-time masks are
    /// folded in. Sound but conservative: a reported bit is 1 (resp. 0) in
    /// EVERY model of assertions + assumptions; an unreported bit may
    /// still be semantically forced — propagation just couldn't see it.
    /// A conservative unsigned range falls out as
    /// `[ones, ones | !(ones | zeros) & mask]`.
    ///
    /// Returns `None` when unit propagation alone refutes the formula
    /// under the assumptions — a real UNSAT proof (same guarantee as an
    /// Unsat result from `solve_under_assumptions`).
    ///
    /// Costs one bitblast of `x`'s cone (cached, and its CNF stays in the
    /// SAT core — later solves reuse it) plus one propagation pass. Like
    /// any state-changing call, it invalidates the model of a previous
    /// solve. Panics if `x` is wider than 128 bits.
    pub fn bv_known_bits_under_assumptions(
        &mut self,
        x: BvTerm,
        assumptions: &[BoolTerm],
    ) -> Option<(u128, u128)> {
        let w = self.ctx.width_of(x);
        assert!(w <= 128, "bv_known_bits_under_assumptions: width > 128");
        self.last_result = None;
        self.flush_pending();
        let refs = self.bitblast_bv(x);
        let bits: Vec<Lit> = refs.iter().map(|&r| self.lit_of(r)).collect();
        let extras = self.build_assumption_lits(assumptions);
        // Build-without-solve: the assumption cache no longer mirrors the
        // solver's last-solved list, so later hinted solves must not vouch
        // for it.
        self.built_prefix_dirty = true;
        let asmps = self.built_assumptions(&extras);
        let (o, z) = self.sat.probe_under_assumptions(&asmps, |s| {
            let (mut o, mut z) = (0u128, 0u128);
            for (i, &l) in bits.iter().enumerate() {
                match s.value_of(l) {
                    LBool::True => o |= 1 << i,
                    LBool::False => z |= 1 << i,
                    LBool::Undef => {}
                }
            }
            (o, z)
        })?;
        let (co, cz) = self.ctx.bv_known_bits(x);
        let (ones, zeros) = (o | co, z | cz);
        if ones & zeros != 0 {
            // Construction-time and propagated facts contradict on a bit:
            // the formula is UNSAT under these assumptions even though
            // propagation alone didn't close the refutation.
            return None;
        }
        Some((ones, zeros))
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

    /// [`solve_min_u`] with assumption terms held through the whole hunt —
    /// the exact minimum of `x` over models satisfying assertions AND
    /// assumptions (same semantics as [`solve_under_assumptions`]).
    /// Returns `None` if unsat under the assumptions.
    pub fn solve_min_u_under_assumptions(
        &mut self,
        x: BvTerm,
        assumptions: &[BoolTerm],
    ) -> Option<u128> {
        assert!(self.ctx.width_of(x) <= 128, "solve_min_u: width > 128");
        let (bits, extras) = self.opt_prologue_with(x, assumptions)?;
        let limbs = self.bit_hunt_with(&bits, &extras, |_| false);
        Some(limbs_to_u128(&limbs))
    }

    /// Enumerate up to `limit` distinct values of `x` under the current
    /// assertions plus `assumptions`. Exact: the returned flag is `true`
    /// iff the returned values are ALL values `x` can take (the
    /// enumeration exhausted the space); `false` means the limit cut it
    /// short. Values arrive in model-discovery order, not sorted.
    ///
    /// Each found value is excluded with a single SAT clause over `x`'s
    /// already-materialized bit literals — no `x != c` comparator circuit,
    /// no term-graph growth, no push/pop churn — and every blocking clause
    /// is guarded by one per-call activation literal that is retired on
    /// return, so the enumeration leaves no semantic residue and learned
    /// clauses remain valid for later solves. Panics if `x` is wider than
    /// 128 bits.
    pub fn solve_many_u_under_assumptions(
        &mut self,
        x: BvTerm,
        limit: usize,
        assumptions: &[BoolTerm],
    ) -> (Vec<u128>, bool) {
        assert!(self.ctx.width_of(x) <= 128, "solve_many_u: width > 128");
        self.last_result = None;
        self.flush_pending();
        let refs = self.bitblast_bv(x);
        let bits: Vec<Lit> = refs.iter().map(|&r| self.lit_of(r)).collect();
        let (mut extras, shared) = self.build_assumption_lits_counted(assumptions);
        let act = self.new_sat_lit_tagged(VarOrigin::Activation);
        extras.push(act);
        let asmps = self.built_assumptions(&extras);
        let mut trusted = self.trusted_for(shared);
        let mut values: Vec<u128> = Vec::new();
        let exhausted = loop {
            if values.len() >= limit {
                break false;
            }
            let t = trusted.min(asmps.len());
            trusted = asmps.len();
            match self.sat_solve_hinted(&asmps, t) {
                SolveResult::Unsat => break true,
                SolveResult::Sat => {
                    // Read the value and build the blocking clause BEFORE
                    // add_clause — it rewinds the trail (and the model).
                    let mut v = 0u128;
                    let mut clause = Vec::with_capacity(bits.len() + 1);
                    clause.push(!act);
                    for (i, &l) in bits.iter().enumerate() {
                        match self.sat.value_of(l) {
                            LBool::True => {
                                v |= 1u128 << i;
                                clause.push(!l);
                            }
                            LBool::False => clause.push(l),
                            // Materialized bit vars are decision vars; a
                            // SAT result assigns all of them.
                            LBool::Undef => unreachable!("unassigned bit in SAT model"),
                        }
                    }
                    values.push(v);
                    self.sat.add_clause(clause);
                }
            }
        };
        // Retire the session: the blocking clauses become permanently
        // vacuous and can never constrain a future solve.
        self.sat.add_clause(vec![!act]);
        (values, exhausted)
    }

    /// [`solve_many_u_under_assumptions`] with no assumptions.
    pub fn solve_many_u(&mut self, x: BvTerm, limit: usize) -> (Vec<u128>, bool) {
        self.solve_many_u_under_assumptions(x, limit, &[])
    }

    /// Batch feasibility: for each `candidates[i]`, decide whether
    /// assertions ∧ `assumptions` ∧ `candidates[i]` is satisfiable — the
    /// result is exactly what `solve_under_assumptions(assumptions ++
    /// [candidates[i]])` would return for each, but usually much cheaper.
    ///
    /// The win is model reuse: after any SAT solve, every still-undecided
    /// candidate is first EVALUATED against the current model (a
    /// structural AIG walk, no SAT work); candidates the model satisfies
    /// are SAT for free. Only model-falsified candidates get a real solve,
    /// and each new model re-screens the remaining ones. For symbex
    /// branch/target fan-outs, typically a large fraction of candidates
    /// resolve without touching the SAT core.
    ///
    /// Warm start: if a model from some previous solve is still
    /// semantically valid (the banked model — it survives trail rewinds
    /// from known-bits probes, CNF emission, even Unsat solves, and dies
    /// only when new assertions land) and it satisfies `assumptions`,
    /// screening starts from it directly: the baseline solve is skipped,
    /// and candidates the model satisfies cost zero SAT work and zero
    /// CNF. A branch fan-out `solve_each_under_assumptions(&[cond,
    /// not_cond], pc)` anywhere downstream of a solve of `pc` does at
    /// most one real solve instead of two. Validating the model against
    /// the assumptions is O(#assumptions) reads when they were
    /// materialized by an earlier solve, and exits at the first
    /// falsified one — a stale model costs no cone walks over old
    /// constraints.
    ///
    /// `last_result` is cleared. The batch's final model (from its last
    /// internal Sat solve, or the still-valid banked model if everything
    /// screened) is banked and can warm the next batch.
    pub fn solve_each_under_assumptions(
        &mut self,
        candidates: &[BoolTerm],
        assumptions: &[BoolTerm],
    ) -> Vec<SmtResult> {
        self.last_result = None;
        // Banks a standing trail model; a flush with real work instead
        // invalidates the banked model (new assertions change semantics).
        self.flush_pending();
        // Bitblast every candidate up front — AIG only, no CNF emission —
        // so model screening can evaluate them structurally.
        let cand_refs: Vec<AigRef> =
            candidates.iter().map(|&t| self.bitblast_bool(t)).collect();

        // Warm gate: the banked model screens candidates iff it also
        // satisfies the base.
        let warm = self.banked_model_holds(assumptions);

        let mut results: Vec<Option<SmtResult>> = vec![None; candidates.len()];
        // Which model screens undecided candidates: the banked one until
        // the first internal solve, the live trail afterwards.
        let mut screen_src = ModelSource::Banked;
        let mut model_valid = warm;
        // Built lazily, at the first candidate needing a real solve — a
        // warm run that screens everything never touches the SAT core.
        let mut base_asmps: Option<Vec<Lit>> = None;
        // Prefix of the base list vouched unchanged for the next hinted
        // solve; becomes the full base once any in-batch solve ran.
        let mut cand_trusted = 0usize;
        loop {
            if model_valid {
                // One structural walk screens every undecided candidate
                // (shared memo across their cones).
                let pending: Vec<usize> =
                    (0..results.len()).filter(|&i| results[i].is_none()).collect();
                let refs: Vec<AigRef> = pending.iter().map(|&i| cand_refs[i]).collect();
                for (&i, sat) in pending.iter().zip(self.eval_refs_from(&refs, screen_src)) {
                    if sat {
                        results[i] = Some(SmtResult::Sat);
                    }
                }
            }
            let Some(next) = results.iter().position(|r| r.is_none()) else {
                break;
            };
            if base_asmps.is_none() {
                let (extras, shared) = self.build_assumption_lits_counted(assumptions);
                cand_trusted = self.trusted_for(shared);
                // A real solve is unavoidable now. Materialize every
                // still-undecided candidate up front: later per-candidate
                // solves then emit no CNF between SAT calls, so each one
                // reuses the previous solve's assumption-prefix trail
                // (solver trail reuse) instead of rebuilding it. Candidates
                // a mid-batch model screens Sat after this point pay their
                // (definitional) CNF without needing it — a bounded price;
                // fully-screened warm batches never reach here and still
                // emit nothing.
                for i in 0..cand_refs.len() {
                    if results[i].is_none() {
                        self.lit_of(cand_refs[i]);
                    }
                }
                let asmps = self.built_assumptions(&extras);
                if !warm {
                    // Cold start: baseline solve with no candidate. Unsat
                    // here decides everything at once; Sat provides the
                    // first screening model. A warm run skips this — the
                    // standing model already witnessed the base as sat.
                    let t = cand_trusted.min(asmps.len());
                    match self.sat_solve_hinted(&asmps, t) {
                        SolveResult::Sat => {
                            cand_trusted = asmps.len();
                            // Bank the baseline model: even if every
                            // later per-candidate solve is Unsat, this
                            // witness of the base warms the next query.
                            self.bank_model();
                            base_asmps = Some(asmps);
                            model_valid = true;
                            screen_src = ModelSource::Trail;
                            continue;
                        }
                        SolveResult::Unsat => {
                            return vec![SmtResult::Unsat; candidates.len()];
                        }
                    }
                }
                base_asmps = Some(asmps);
            }
            // Materializing the candidate lit may emit CNF, which rewinds
            // the trail — safe for the banked model, but done only now so
            // a fully-screened batch emits nothing. The candidate lit is
            // pushed onto the shared base list and popped after the solve
            // (no per-candidate clone).
            let lit = self.lit_of(cand_refs[next]);
            let mut asmps = base_asmps.take().unwrap();
            asmps.push(lit);
            let t = cand_trusted.min(asmps.len() - 1);
            let res = self.sat_solve_hinted(&asmps, t);
            asmps.pop();
            cand_trusted = asmps.len();
            base_asmps = Some(asmps);
            match res {
                SolveResult::Sat => {
                    results[next] = Some(SmtResult::Sat);
                    model_valid = true;
                    screen_src = ModelSource::Trail;
                }
                SolveResult::Unsat => {
                    results[next] = Some(SmtResult::Unsat);
                    model_valid = false;
                }
            }
        }
        // Bank the batch's final model (if its last solve was Sat) so the
        // next batch can start warm even after intervening probes or CNF
        // emission. A fully-screened warm batch never touched the SAT
        // core, so this is a no-op and the banked model just survives.
        self.bank_model();
        results.into_iter().map(|r| r.unwrap()).collect()
    }

    /// Branch feasibility for a complementary pair: exactly
    /// `solve_each_under_assumptions(&[cond, ¬cond], assumptions)`,
    /// returned as `(result_cond, result_not_cond)`, with the pair
    /// structure exploited outright — any model of the base satisfies
    /// exactly one side, so a single structural evaluation hands that
    /// side Sat for free and the one real solve goes to the other side.
    /// There is never a reason to screen the second side against a model
    /// that satisfied the first.
    ///
    /// Solve count: warm (a banked model witnesses the base) — exactly 1;
    /// cold — a baseline solve plus 1, or just the baseline when it comes
    /// back Unsat (both sides Unsat). This is the floor for deciding both
    /// sides of a genuine branch: two answers need two witnesses, one of
    /// which the standing model supplies when warm.
    pub fn solve_pair_under_assumptions(
        &mut self,
        cond: BoolTerm,
        assumptions: &[BoolTerm],
    ) -> (SmtResult, SmtResult) {
        self.solve_pair_impl(cond, assumptions, false)
    }

    /// [`solve_pair_under_assumptions`] for callers that KNOW the base
    /// (assertions ∧ assumptions) is satisfiable — the symbex invariant
    /// that the current path was already reached. When the banked model
    /// can't screen, the baseline solve is skipped and side `cond` is
    /// solved directly; if it comes back Unsat, every base model
    /// satisfies `¬cond`, so the whole pair is decided with ONE solve.
    /// Forced branches (the majority in real programs) drop from two
    /// solves to one; both-feasible pairs still take two.
    ///
    /// CONTRACT: if the guarantee is violated (the base is in fact
    /// Unsat), the result is `(Unsat, Sat)` whose second component is
    /// vacuous. Use [`solve_pair_under_assumptions`] when base
    /// feasibility is not established.
    pub fn solve_pair_assuming_base_sat(
        &mut self,
        cond: BoolTerm,
        assumptions: &[BoolTerm],
    ) -> (SmtResult, SmtResult) {
        self.solve_pair_impl(cond, assumptions, true)
    }

    fn solve_pair_impl(
        &mut self,
        cond: BoolTerm,
        assumptions: &[BoolTerm],
        base_known_sat: bool,
    ) -> (SmtResult, SmtResult) {
        self.last_result = None;
        self.flush_pending();
        let r = self.bitblast_bool(cond);

        // Which model witnesses the base, and does it satisfy `cond`?
        let (cond_sat, mut asmps) = if self.banked_model_holds(assumptions) {
            (self.eval_refs_from(&[r], ModelSource::Banked)[0], None)
        } else {
            let (extras, shared) = self.build_assumption_lits_counted(assumptions);
            // Materialize the candidate BEFORE the first solve: no CNF
            // then lands between the solves, so the second one reuses the
            // first's assumption-prefix trail (solver trail reuse)
            // instead of rebuilding it from level 0.
            let lit = self.lit_of(r);
            let mut asmps = self.built_assumptions(&extras);
            let trusted0 = self.trusted_for(shared);
            if base_known_sat {
                // Base feasibility is vouched for — no baseline solve.
                // Decide side `cond` directly; Unsat here means the pair
                // is done in a single solve.
                asmps.push(lit);
                let t = trusted0.min(asmps.len() - 1);
                match self.sat_solve_hinted(&asmps, t) {
                    SolveResult::Unsat => {
                        return (SmtResult::Unsat, SmtResult::Sat);
                    }
                    SolveResult::Sat => {}
                }
                // Bank this side's model now: if ¬cond turns out Unsat
                // (forced-true branch), this witness still warms the next
                // query.
                self.bank_model();
                asmps.pop();
                asmps.push(!lit);
                let other = match self.sat_solve_hinted(&asmps, asmps.len() - 1) {
                    SolveResult::Sat => SmtResult::Sat,
                    SolveResult::Unsat => SmtResult::Unsat,
                };
                self.bank_model();
                return (SmtResult::Sat, other);
            }
            match self.sat_solve_hinted(&asmps, trusted0) {
                SolveResult::Unsat => return (SmtResult::Unsat, SmtResult::Unsat),
                SolveResult::Sat => {}
            }
            // Bank the baseline model: if the one real solve below comes
            // back Unsat, this witness of the base still warms the next
            // query.
            self.bank_model();
            (self.eval_refs_from(&[r], ModelSource::Trail)[0], Some(asmps))
        };

        // The model-falsified side gets the one real solve. Only now may
        // CNF be emitted (assumption lits on the warm path, the candidate
        // lit always) — screening is done reading models.
        let (mut asmps, trusted) = match asmps.take() {
            // The baseline just solved exactly this list.
            Some(a) => {
                let t = a.len();
                (a, t)
            }
            None => {
                let (extras, shared) = self.build_assumption_lits_counted(assumptions);
                let t = self.trusted_for(shared);
                (self.built_assumptions(&extras), t)
            }
        };
        let lit = self.lit_of(r);
        asmps.push(if cond_sat { !lit } else { lit });
        let t = trusted.min(asmps.len() - 1);
        let other = match self.sat_solve_hinted(&asmps, t) {
            SolveResult::Sat => SmtResult::Sat,
            SolveResult::Unsat => SmtResult::Unsat,
        };
        // Bank the fresh model (if Sat) for the next warm start.
        self.bank_model();
        if cond_sat {
            (SmtResult::Sat, other)
        } else {
            (other, SmtResult::Sat)
        }
    }

    /// Exact unsigned `[min, max]` of `x` under assumptions — one shared
    /// prologue (flush + bitblast + feasibility solve), then two bit-hunts
    /// on the hot solver. `None` iff unsat under the assumptions.
    pub fn solve_range_u_under_assumptions(
        &mut self,
        x: BvTerm,
        assumptions: &[BoolTerm],
    ) -> Option<(u128, u128)> {
        assert!(self.ctx.width_of(x) <= 128, "solve_range_u: width > 128");
        let (bits, extras) = self.opt_prologue_with(x, assumptions)?;
        let min = limbs_to_u128(&self.bit_hunt_with(&bits, &extras, |_| false));
        let max = limbs_to_u128(&self.bit_hunt_with(&bits, &extras, |_| true));
        Some((min, max))
    }

    /// After [`solve_under_assumptions`] (or a per-candidate Unsat from
    /// [`solve_each_under_assumptions`]) returns Unsat, the indices into
    /// `assumptions` whose lits appear in the SAT-level unsat core — the
    /// subset of assumptions that jointly caused the conflict. Pass the
    /// SAME slice that was passed to the failing solve; bitblast caching
    /// makes the lit lookup free. Meaningless after a Sat result.
    pub fn failed_assumptions(&mut self, assumptions: &[BoolTerm]) -> Vec<usize> {
        let extras = self.build_assumption_lits(assumptions);
        // Build-without-solve (see bv_known_bits): drop prefix trust.
        self.built_prefix_dirty = true;
        let core: std::collections::HashSet<Lit> =
            self.sat.unsat_core().iter().copied().collect();
        extras
            .iter()
            .enumerate()
            .filter(|(_, l)| core.contains(l))
            .map(|(i, _)| i)
            .collect()
    }

    /// [`solve_max_u`] with assumption terms — see
    /// [`solve_min_u_under_assumptions`].
    pub fn solve_max_u_under_assumptions(
        &mut self,
        x: BvTerm,
        assumptions: &[BoolTerm],
    ) -> Option<u128> {
        assert!(self.ctx.width_of(x) <= 128, "solve_max_u: width > 128");
        let (bits, extras) = self.opt_prologue_with(x, assumptions)?;
        let limbs = self.bit_hunt_with(&bits, &extras, |_| true);
        Some(limbs_to_u128(&limbs))
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

    /// Shared opt-query setup: flush, bitblast + materialize the target
    /// term (so its SAT lits exist and the formula incorporates all its
    /// clauses), and do the initial feasibility check. Returns `None` when
    /// unsat (and updates `last_result` accordingly); returns the LSB-first
    /// SAT lits of `x` when sat, with `last_result = Sat`.
    fn opt_prologue(&mut self, x: BvTerm) -> Option<Vec<Lit>> {
        self.opt_prologue_with(x, &[]).map(|(bits, _)| bits)
    }

    /// [`opt_prologue`] with assumption terms: also bitblasts the
    /// assumptions and returns their lits so callers can keep them
    /// installed for every solve of the hunt.
    fn opt_prologue_with(
        &mut self,
        x: BvTerm,
        assumptions: &[BoolTerm],
    ) -> Option<(Vec<Lit>, Vec<Lit>)> {
        self.flush_pending();
        // Materialize BEFORE the initial solve so the feasibility check
        // sees any clauses the target term's cone adds — otherwise a sat
        // model from the smaller formula could be falsified by the extra
        // gates.
        let refs = self.bitblast_bv(x);
        let bits: Vec<Lit> = refs.iter().map(|&r| self.lit_of(r)).collect();
        let (extras, shared) = self.build_assumption_lits_counted(assumptions);
        let asmps = self.built_assumptions(&extras);
        let trusted = self.trusted_for(shared);
        match self.sat_solve_hinted(&asmps, trusted) {
            SolveResult::Sat => {
                self.last_result = Some(SmtResult::Sat);
                Some((bits, extras))
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
        self.bit_hunt_with(bits, &[], want_one)
    }

    /// [`bit_hunt`] with a fixed assumption-lit prefix held through every
    /// solve of the hunt (the under-assumptions min/max entry points).
    fn bit_hunt_with(
        &mut self,
        bits: &[Lit],
        extras: &[Lit],
        want_one: impl Fn(usize) -> bool,
    ) -> Vec<u64> {
        let w = bits.len();
        let nlimbs = (w + 63) / 64;
        let mut limbs = vec![0u64; nlimbs];
        // One list for the whole hunt, grown in place: [controls |
        // extras | bit lits...]. `valid` tracks how much of it provably
        // equals the SAT core's last-solved list (the prologue solved
        // exactly [controls | extras]; every hunt solve re-vouches its
        // own list) — assumption plumbing is O(1) per bit.
        let mut asmps = self.built_assumptions(extras);
        let mut valid = asmps.len();
        for i in (0..w).rev() {
            let b = bits[i];
            let prefer_one = want_one(i);
            let first_try = if prefer_one { b } else { !b };
            // Model screening: the standing model (the prologue's, or the
            // last hunt solve's) may already witness prefix ∧ preferred —
            // exactly what a Sat solve would conclude, so the bit locks
            // with zero SAT work. Solves happen only where the model
            // disagrees with the preference.
            if self.sat.has_model() && self.sat.value_of(first_try) == LBool::True {
                asmps.push(first_try);
                if prefer_one {
                    limbs[i / 64] |= 1u64 << (i % 64);
                }
                continue;
            }
            asmps.push(first_try);
            let t = valid.min(asmps.len() - 1);
            let sat = matches!(self.sat_solve_hinted(&asmps, t), SolveResult::Sat);
            valid = asmps.len();
            if sat {
                if prefer_one {
                    limbs[i / 64] |= 1u64 << (i % 64);
                }
            } else {
                // The opposite polarity must be sat under the prefix, by
                // exhaustion of the two possibilities.
                asmps.pop();
                asmps.push(!first_try);
                valid = asmps.len() - 1;
                if !prefer_one {
                    limbs[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        // Leave the SAT solver in a state whose current model realizes
        // the returned optimum, so the caller can read other terms'
        // values via `get_bv_value_*` afterward. A standing model at loop
        // exit already realizes it (every locked bit was screened or
        // solved against it) — only a trailing Unsat needs the re-solve.
        if !self.sat.has_model() {
            let t = valid.min(asmps.len());
            let _ = self.sat_solve_hinted(&asmps, t);
        }
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
            // Skip width-1: the AIG folds 1-bit equality to an XNOR (or a
            // direct ref when one side is constant) — the general path
            // emits at most one gate there anyway. Only the wide case wins.
            let w = self.ctx.width_of(a);
            if w >= 2 {
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                // Materialize side a fully, then side b, then emit the
                // biconditionals — matches the variable-allocation order of
                // the pre-AIG encoder (which bitblasted each side to
                // completion before touching the clauses). Interleaving
                // per-bit shuffles SAT var numbering, which perturbs VSIDS
                // tie-breaking / watch selection enough to flip near-cliff
                // instances.
                if self.retirement_enabled {
                    self.asserted_roots.extend_from_slice(&ab);
                    self.asserted_roots.extend_from_slice(&bb);
                }
                let als: Vec<Lit> = ab.iter().map(|&r| self.lit_of(r)).collect();
                let bls: Vec<Lit> = bb.iter().map(|&r| self.lit_of(r)).collect();
                for i in 0..ab.len() {
                    let al = als[i];
                    let bl = bls[i];
                    match act_lit {
                        None => {
                            self.emit_clause(vec![!al, bl]);
                            self.emit_clause(vec![al, !bl]);
                        }
                        Some(act) => {
                            self.emit_clause(vec![!act, !al, bl]);
                            self.emit_clause(vec![!act, al, !bl]);
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
                    // `¬(x = y)` = some bit differs. Build per-bit XOR refs
                    // and OR them in one clause. Gate vars still needed for
                    // symbolic bit pairs, but we skip the AND chain on top.
                    let ab = self.bitblast_bv(a);
                    let bb = self.bitblast_bv(b);
                    // Same ordering rationale as the Eq path above: fully
                    // materialize each side before the xor outputs.
                    for &r in &ab {
                        let _ = self.lit_of(r);
                    }
                    for &r in &bb {
                        let _ = self.lit_of(r);
                    }
                    let mut clause = Vec::with_capacity(ab.len() + 1);
                    if let Some(act) = act_lit {
                        clause.push(!act);
                    }
                    for i in 0..ab.len() {
                        let x = self.mk_xor(ab[i], bb[i]);
                        if self.retirement_enabled {
                            self.asserted_roots.push(x);
                        }
                        let xl = self.lit_of(x);
                        clause.push(xl);
                    }
                    self.emit_clause(clause);
                    return;
                }
            }
        }
        // General path: bitblast to one AIG root, materialize, emit unit.
        let r = self.bitblast_bool(t);
        if self.retirement_enabled {
            self.asserted_roots.push(r);
        }
        let lit = self.lit_of(r);
        match act_lit {
            None => {
                self.emit_clause(vec![lit]);
            }
            Some(act) => {
                self.emit_clause(vec![!act, lit]);
            }
        }
    }


    /// Scan batch roots for `(= x t)` substitution candidates and install
    /// the sound ones into `bv_var_subst`. Returns how many were added.
    ///
    /// Eligibility for `x → t`: `x` is a bare variable whose union-find
    /// root has no SAT bits yet (nothing outside this batch can reference
    /// it), the root isn't already substituted, and `x` does not occur in
    /// `t` — transitively through already-accepted substitutions, which
    /// also guarantees the map stays acyclic. Var=var equalities go to the
    /// cheaper union-find alias when possible.
    fn collect_substitutions(&mut self, batch: &[(usize, Vec<BoolTerm>)]) -> usize {
        let mut installed = 0usize;
        for (_, ts) in batch {
            for &t in ts {
                let BoolOp::Eq(a, b) = self.ctx.bool_op(t) else { continue };
                // var = var: alias (free) — falls through to substitution
                // if one side is already bitblasted.
                if self.alias_bv_vars(a, b) {
                    continue;
                }
                for (x, rhs) in [(a, b), (b, a)] {
                    let BvOp::Var(id) = self.ctx.bv_op(x) else { continue };
                    let root = self.find_bv_var_root(id);
                    if self.bv_var_refs.contains_key(&root)
                        || self.bv_var_subst.contains_key(&root)
                    {
                        continue;
                    }
                    if self.subst_occurs(root, rhs) {
                        continue;
                    }
                    self.bv_var_subst.insert(root, rhs);
                    self.pp_substituted += 1;
                    installed += 1;
                    break;
                }
            }
        }
        installed
    }

    /// Does variable root `x_root` occur in `t`, chasing already-installed
    /// substitutions? Iterative DAG walk with a visited set.
    fn subst_occurs(&mut self, x_root: u32, t: BvTerm) -> bool {
        let mut visited: HashMap<BvTerm, ()> = HashMap::default();
        let mut stack = vec![t];
        while let Some(cur) = stack.pop() {
            if visited.contains_key(&cur) {
                continue;
            }
            visited.insert(cur, ());
            match self.ctx.bv_op(cur) {
                BvOp::Var(id) => {
                    let root = self.find_bv_var_root(id);
                    if root == x_root {
                        return true;
                    }
                    if let Some(&sub) = self.bv_var_subst.get(&root) {
                        stack.push(sub);
                    }
                }
                BvOp::Const => {}
                BvOp::Not(x) | BvOp::Neg(x) | BvOp::Popcount(x) | BvOp::Clz(x)
                | BvOp::Ctz(x) | BvOp::Extract(x, _, _) | BvOp::ZeroExtend(x, _)
                | BvOp::SignExtend(x, _) => stack.push(x),
                BvOp::And(x, y) | BvOp::Or(x, y) | BvOp::Xor(x, y)
                | BvOp::Add(x, y) | BvOp::Sub(x, y) | BvOp::Mul(x, y)
                | BvOp::Udiv(x, y) | BvOp::Urem(x, y) | BvOp::Sdiv(x, y)
                | BvOp::Srem(x, y) | BvOp::Smod(x, y) | BvOp::Shl(x, y)
                | BvOp::Lshr(x, y) | BvOp::Ashr(x, y) | BvOp::Concat(x, y)
                | BvOp::RotateLeft(x, y) | BvOp::RotateRight(x, y) => {
                    stack.push(x);
                    stack.push(y);
                }
                BvOp::Ite(c, x, y) => {
                    stack.push(x);
                    stack.push(y);
                    self.push_bool_bv_children(c, &mut stack, &mut visited);
                }
                BvOp::Select(idx) => {
                    let table = self.ctx.select_tables[idx as usize].clone();
                    for &v in table.values.iter() {
                        stack.push(v);
                    }
                    stack.push(table.default);
                    for &s in table.selectors.iter() {
                        self.push_bool_bv_children(s, &mut stack, &mut visited);
                    }
                }
            }
        }
        false
    }

    /// Gaussian elimination over Z/2^w on the batch's top-level linear
    /// equalities. Solves coupled systems that single-variable substitution
    /// can't — `x + y = a, x - y = b` has no side in `x = t` form, but the
    /// system determines both `x` and `y`.
    ///
    /// Each eligible `(= lhs rhs)` becomes a row `Σ cᵢ·varᵢ = k (mod 2^w)`;
    /// rows are grouped by width (a bvadd chain is single-width, and a
    /// variable has one width, so widths never mix within a row). Forward
    /// elimination pivots only on *odd* coefficients — odd constants are the
    /// units of Z/2^w and the only ones with a modular inverse; a variable
    /// appearing solely with even coefficients can't be cleanly solved and
    /// its row is left for the SAT core.
    ///
    /// Solved variables are installed into `bv_var_subst` (so model reads
    /// stay consistent) and their defining equality is dropped from the
    /// batch. A row that reduces to `0 = nonzero` proves the whole formula
    /// UNSAT. Only runs at scope depth 0 (like substitution) — a popped
    /// scope must not leave permanent solutions behind.
    ///
    /// Returns `true` if an inconsistent row was found (caller emits the
    /// empty clause).
    fn gaussian_eliminate(&mut self, batch: &mut [(usize, Vec<BoolTerm>)]) -> bool {
        // Cap the system size — real linear cores here are tiny (tens of
        // rows); this only guards against a pathological blow-up.
        const MAX_ROWS: usize = 4096;

        struct Row {
            coeffs: std::collections::BTreeMap<BvTerm, u128>,
            rhs: u128,
            src: BoolTerm,
            w: u32,
        }

        // Collect candidate rows from depth-0 equality roots.
        let mut rows: Vec<Row> = Vec::new();
        for (depth, ts) in batch.iter() {
            if *depth != 0 {
                continue;
            }
            for &t in ts {
                if rows.len() >= MAX_ROWS {
                    break;
                }
                let BoolOp::Eq(a, b) = self.ctx.bool_op(t) else { continue };
                let w = self.ctx.width_of(a);
                if w > 128 {
                    continue;
                }
                let mut coeffs = std::collections::BTreeMap::new();
                let mut konst = 0u128;
                // lhs − rhs: move both sides into one coefficient map.
                if !self.flatten_linear(a, 1, w, &mut coeffs, &mut konst) {
                    continue;
                }
                let neg1 = mask(w); // −1 mod 2^w
                if !self.flatten_linear(b, neg1, w, &mut coeffs, &mut konst) {
                    continue;
                }
                // Row is `Σ coeff·atom = −konst` (everything moved left, so
                // the constant flips to the RHS).
                let m = mask(w);
                coeffs.retain(|_, c| {
                    *c &= m;
                    *c != 0
                });
                let rhs = 0u128.wrapping_sub(konst) & m;
                // A row with a lone solvable variable is already handled by
                // plain substitution; only keep rows worth GE (≥1 solvable
                // var, and either coupled or with a non-unit-ready form).
                if coeffs.is_empty() {
                    // 0 = rhs: redundant (rhs==0) or inconsistent.
                    if rhs != 0 {
                        return true; // UNSAT
                    }
                    continue;
                }
                rows.push(Row { coeffs, rhs, src: t, w });
            }
        }
        if rows.is_empty() {
            return false;
        }

        // Group row indices by width.
        let mut by_width: HashMap<u32, Vec<usize>> = HashMap::default();
        for (i, r) in rows.iter().enumerate() {
            by_width.entry(r.w).or_default().push(i);
        }

        let mut consumed: std::collections::HashSet<BoolTerm> =
            std::collections::HashSet::new();
        let mut solved: Vec<(u32, BvTerm, u128, std::collections::BTreeMap<BvTerm, u128>)> =
            Vec::new(); // (root_id, pivot_var_term, rhs, other coeffs) — normalized so pivot coeff = 1

        for (w, idxs) in by_width.iter() {
            let w = *w;
            let m = mask(w);
            // Working copies (owned) of this group's rows.
            let mut group: Vec<(std::collections::BTreeMap<BvTerm, u128>, u128, BoolTerm)> =
                idxs.iter()
                    .map(|&i| (rows[i].coeffs.clone(), rows[i].rhs, rows[i].src))
                    .collect();

            let mut pivot_rows: Vec<usize> = Vec::new(); // indices into `group`
            let mut used_pivot_var: std::collections::HashSet<BvTerm> =
                std::collections::HashSet::new();

            for i in 0..group.len() {
                // Find a solvable pivot variable with an odd coefficient in row i.
                let pivot = {
                    let (coeffs, _, _) = &group[i];
                    let mut found: Option<(BvTerm, u128)> = None;
                    for (&atom, &c) in coeffs.iter() {
                        if c & 1 == 1
                            && !used_pivot_var.contains(&atom)
                            && self.is_solvable_var(atom)
                        {
                            found = Some((atom, c));
                            break;
                        }
                    }
                    found
                };
                let Some((pvar, pc)) = pivot else { continue };
                let inv = match crate::bv::mod_inverse_pow2(pc, w) {
                    Some(v) => v,
                    None => continue, // even — unreachable (odd checked), defensive
                };
                // Normalize row i so the pivot coefficient becomes 1.
                {
                    let (coeffs, rhs, _) = &mut group[i];
                    for c in coeffs.values_mut() {
                        *c = c.wrapping_mul(inv) & m;
                    }
                    *rhs = rhs.wrapping_mul(inv) & m;
                }
                used_pivot_var.insert(pvar);
                pivot_rows.push(i);

                // Eliminate `pvar` from every other row in the group.
                let (pcoeffs, prhs) = {
                    let (c, r, _) = &group[i];
                    (c.clone(), *r)
                };
                for j in 0..group.len() {
                    if j == i {
                        continue;
                    }
                    let factor = group[j].0.get(&pvar).copied().unwrap_or(0) & m;
                    if factor == 0 {
                        continue;
                    }
                    let (cj, rj, _) = &mut group[j];
                    for (&atom, &pc2) in pcoeffs.iter() {
                        let sub = factor.wrapping_mul(pc2) & m;
                        let e = cj.entry(atom).or_insert(0);
                        *e = e.wrapping_sub(sub) & m;
                    }
                    cj.retain(|_, c| *c != 0);
                    *rj = rj.wrapping_sub(factor.wrapping_mul(prhs)) & m;
                }
            }

            // Read out results.
            for (i, (coeffs, rhs, src)) in group.iter().enumerate() {
                if pivot_rows.contains(&i) {
                    // Pivot row: pivot var (coeff 1) = rhs − Σ others.
                    // Find the pivot var (the one solvable var with coeff 1
                    // that's in used set for this row — it's the atom we
                    // normalized). Recover it as the solvable var with
                    // coeff 1; there is exactly one we pivoted on.
                    let pvar = coeffs
                        .iter()
                        .find(|&(atom, &c)| c == 1 && used_pivot_var.contains(atom))
                        .map(|(&a, _)| a);
                    let Some(pvar) = pvar else { continue };
                    let mut others = coeffs.clone();
                    others.remove(&pvar);
                    let BvOp::Var(id) = self.ctx.bv_op(pvar) else { continue };
                    let root = self.find_bv_var_root(id);
                    solved.push((root, pvar, *rhs, others));
                    consumed.insert(*src);
                } else if coeffs.is_empty() {
                    if *rhs != 0 {
                        return true; // 0 = nonzero ⇒ UNSAT
                    }
                    consumed.insert(*src); // 0 = 0 ⇒ redundant, drop
                }
            }
        }

        // Install solved variables, guarding against cycles. Build the RHS
        // expression `rhs − Σ cᵢ·atomᵢ` with the term builders.
        for (root, _pvar, rhs, others) in solved {
            if self.bv_var_refs.contains_key(&root) || self.bv_var_subst.contains_key(&root) {
                continue;
            }
            let w = self.ctx.bv_var_widths[root as usize];
            let expr = self.build_linear_expr(rhs, &others, w);
            if self.subst_occurs(root, expr) {
                continue; // would form a cycle — leave the equation to SAT
            }
            self.bv_var_subst.insert(root, expr);
            self.pp_substituted += 1;
        }

        // Drop consumed equalities from the batch.
        if !consumed.is_empty() {
            for (depth, ts) in batch.iter_mut() {
                if *depth == 0 {
                    ts.retain(|t| !consumed.contains(t));
                }
            }
        }
        false
    }

    /// Is `t` a bare variable that can still be solved for — i.e. not yet
    /// bitblasted (nothing outside the batch references its bits) and not
    /// already substituted?
    fn is_solvable_var(&mut self, t: BvTerm) -> bool {
        let BvOp::Var(id) = self.ctx.bv_op(t) else { return false };
        let root = self.find_bv_var_root(id);
        !self.bv_var_refs.contains_key(&root) && !self.bv_var_subst.contains_key(&root)
    }

    /// Flatten `t·coeff` into a linear coefficient map + constant over
    /// Z/2^w. Returns `false` if `t` contains any non-linear or opaque
    /// subterm (shift, extract, ite, mul-of-two-vars, …). Substituted
    /// variables are expanded so rows are stated over live atoms. Iterative
    /// (DAG-aware) to stay linear on deep shared add chains.
    fn flatten_linear(
        &mut self,
        t: BvTerm,
        coeff: u128,
        w: u32,
        coeffs: &mut std::collections::BTreeMap<BvTerm, u128>,
        konst: &mut u128,
    ) -> bool {
        let m = mask(w);
        let mut work = vec![(t, coeff & m)];
        while let Some((cur, c)) = work.pop() {
            if c == 0 {
                continue;
            }
            match self.ctx.bv_op(cur) {
                BvOp::Const => {
                    let node = &self.ctx.bv_nodes[cur.0 as usize];
                    if node.wide != crate::bv::WIDE_NONE {
                        return false; // wide const — outside u128 coefficient math
                    }
                    *konst = konst.wrapping_add(c.wrapping_mul(node.value)) & m;
                }
                BvOp::Var(id) => {
                    let root = self.find_bv_var_root(id);
                    if let Some(&sub) = self.bv_var_subst.get(&root) {
                        work.push((sub, c));
                    } else {
                        let e = coeffs.entry(cur).or_insert(0);
                        *e = e.wrapping_add(c) & m;
                    }
                }
                BvOp::Add(x, y) => {
                    work.push((x, c));
                    work.push((y, c));
                }
                BvOp::Sub(x, y) => {
                    work.push((x, c));
                    work.push((y, 0u128.wrapping_sub(c) & m));
                }
                BvOp::Neg(x) => work.push((x, 0u128.wrapping_sub(c) & m)),
                BvOp::Not(x) => {
                    // ~x = −x − 1
                    *konst = konst.wrapping_sub(c) & m;
                    work.push((x, 0u128.wrapping_sub(c) & m));
                }
                BvOp::Mul(x, y) => {
                    if let Some(v) = self.ctx.try_bv_const_value(y) {
                        work.push((x, c.wrapping_mul(v) & m));
                    } else if let Some(v) = self.ctx.try_bv_const_value(x) {
                        work.push((y, c.wrapping_mul(v) & m));
                    } else {
                        return false; // var·var — non-linear
                    }
                }
                BvOp::Shl(x, y) => {
                    // Constant left shift is linear: `x << k = x · 2^k`.
                    // k ≥ w shifts everything out (→ 0).
                    let Some(k) = self.ctx.try_bv_const_value(y) else {
                        return false; // variable shift amount — non-linear
                    };
                    if k >= w as u128 {
                        continue; // shifted entirely out: contributes 0
                    }
                    let scale = 1u128 << k;
                    work.push((x, c.wrapping_mul(scale) & m));
                }
                BvOp::Concat(hi, lo) => {
                    // The constant-left-shift builder lowers `x << k` to
                    // `concat(extract(x, w-1-k, 0), 0_k)` (structural, zero
                    // gates). That concatenation equals `x · 2^k (mod 2^w)`
                    // — the extract drops exactly the bits the shift pushes
                    // out — so recover the power-of-two coefficient instead
                    // of treating it as opaque. Only the shl shape (zero in
                    // the LOW part) is linear; lshr's `concat(0, extract…)`
                    // drops low bits and is not.
                    let wlo = self.ctx.width_of(lo);
                    if self.ctx.try_bv_const_value(lo) == Some(0) {
                        if let BvOp::Extract(src, ehi, elo) = self.ctx.bv_op(hi) {
                            if elo == 0
                                && self.ctx.width_of(src) == w
                                && ehi == w - wlo - 1
                            {
                                let scale = 1u128 << wlo;
                                work.push((src, c.wrapping_mul(scale) & m));
                                continue;
                            }
                        }
                    }
                    return false; // general concat — not linear
                }
                _ => return false, // opaque / non-linear
            }
        }
        true
    }

    /// Build the term `rhs − Σ cᵢ·atomᵢ (mod 2^w)` for a solved variable.
    fn build_linear_expr(
        &mut self,
        rhs: u128,
        others: &std::collections::BTreeMap<BvTerm, u128>,
        w: u32,
    ) -> BvTerm {
        let m = mask(w);
        let mut acc: Option<BvTerm> = None;
        for (&atom, &c) in others.iter() {
            let neg = 0u128.wrapping_sub(c) & m; // −cᵢ
            if neg == 0 {
                continue;
            }
            let scaled = if neg == 1 {
                atom
            } else if neg == m {
                self.ctx.bv_neg(atom)
            } else {
                let cc = self.ctx.bv_const(neg, w);
                self.ctx.bv_mul(atom, cc)
            };
            acc = Some(match acc {
                None => scaled,
                Some(prev) => self.ctx.bv_add(prev, scaled),
            });
        }
        let rhs = rhs & m;
        match acc {
            None => self.ctx.bv_const(rhs, w),
            Some(sum) => {
                if rhs == 0 {
                    sum
                } else {
                    let cc = self.ctx.bv_const(rhs, w);
                    self.ctx.bv_add(sum, cc)
                }
            }
        }
    }

    /// Push the BV children of a Bool term (and recurse through its Bool
    /// structure) onto an occurs-check stack.
    fn push_bool_bv_children(
        &mut self,
        t: BoolTerm,
        stack: &mut Vec<BvTerm>,
        visited: &mut HashMap<BvTerm, ()>,
    ) {
        let _ = visited;
        let mut bools = vec![t];
        let mut seen: HashMap<BoolTerm, ()> = HashMap::default();
        while let Some(bt) = bools.pop() {
            if seen.contains_key(&bt) {
                continue;
            }
            seen.insert(bt, ());
            match self.ctx.bool_op(bt) {
                BoolOp::True | BoolOp::False | BoolOp::Var(_) => {}
                BoolOp::Not(a) => bools.push(a),
                BoolOp::And(a, b) | BoolOp::Or(a, b) | BoolOp::Implies(a, b) => {
                    bools.push(a);
                    bools.push(b);
                }
                BoolOp::Eq(a, b) | BoolOp::Ult(a, b) | BoolOp::Ule(a, b)
                | BoolOp::Slt(a, b) | BoolOp::Sle(a, b)
                | BoolOp::UaddOverflow(a, b) | BoolOp::SaddOverflow(a, b)
                | BoolOp::UsubOverflow(a, b) | BoolOp::SsubOverflow(a, b)
                | BoolOp::UmulOverflow(a, b) | BoolOp::SmulOverflow(a, b)
                | BoolOp::SdivOverflow(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                BoolOp::NegOverflow(a) => stack.push(a),
            }
        }
    }

    /// Apply the substitution map to a Bool assertion DAG (memoized). Runs
    /// through the term builders so rewrites fire on substituted forms —
    /// that's where most of the payoff is (constants flow through folds,
    /// comparisons collapse via bits-known, adder chains cancel).
    fn apply_subst_bool(
        &mut self,
        t: BoolTerm,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) -> BoolTerm {
        if let Some(&r) = bm.get(&t) {
            return r;
        }
        let op = self.ctx.bool_op(t);
        let r = match op {
            BoolOp::True | BoolOp::False | BoolOp::Var(_) => t,
            BoolOp::Not(x) => {
                let nx = self.apply_subst_bool(x, bm, vm);
                if nx == x { t } else { self.ctx.bool_not(nx) }
            }
            BoolOp::And(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bool(x, bm, vm), self.apply_subst_bool(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bool_and(nx, ny) }
            }
            BoolOp::Or(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bool(x, bm, vm), self.apply_subst_bool(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bool_or(nx, ny) }
            }
            BoolOp::Implies(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bool(x, bm, vm), self.apply_subst_bool(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bool_implies(nx, ny) }
            }
            BoolOp::Eq(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_eq(na, nb) }
            }
            BoolOp::Ult(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_ult(na, nb) }
            }
            BoolOp::Ule(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_ule(na, nb) }
            }
            BoolOp::Slt(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_slt(na, nb) }
            }
            BoolOp::Sle(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_sle(na, nb) }
            }
            BoolOp::UaddOverflow(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_uadd_overflow(na, nb) }
            }
            BoolOp::SaddOverflow(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_sadd_overflow(na, nb) }
            }
            BoolOp::UsubOverflow(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_usub_overflow(na, nb) }
            }
            BoolOp::SsubOverflow(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_ssub_overflow(na, nb) }
            }
            BoolOp::UmulOverflow(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_umul_overflow(na, nb) }
            }
            BoolOp::SmulOverflow(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_smul_overflow(na, nb) }
            }
            BoolOp::NegOverflow(a) => {
                let na = self.apply_subst_bv(a, bm, vm);
                if na == a { t } else { self.ctx.bv_neg_overflow(na) }
            }
            BoolOp::SdivOverflow(a, b) => {
                let (na, nb) =
                    (self.apply_subst_bv(a, bm, vm), self.apply_subst_bv(b, bm, vm));
                if na == a && nb == b { t } else { self.ctx.bv_sdiv_overflow(na, nb) }
            }
        };
        bm.insert(t, r);
        r
    }

    fn apply_subst_bv(
        &mut self,
        t: BvTerm,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) -> BvTerm {
        if let Some(&r) = vm.get(&t) {
            return r;
        }
        let op = self.ctx.bv_op(t);
        let r = match op {
            BvOp::Var(id) => {
                let root = self.find_bv_var_root(id);
                match self.bv_var_subst.get(&root).copied() {
                    // Substitution targets may themselves contain
                    // substituted vars — recurse (map is acyclic).
                    Some(sub) => self.apply_subst_bv(sub, bm, vm),
                    None => t,
                }
            }
            BvOp::Const => t,
            BvOp::Not(x) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_not(nx) }
            }
            BvOp::Neg(x) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_neg(nx) }
            }
            BvOp::And(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_and(nx, ny) }
            }
            BvOp::Or(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_or(nx, ny) }
            }
            BvOp::Xor(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_xor(nx, ny) }
            }
            BvOp::Add(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_add(nx, ny) }
            }
            BvOp::Sub(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_sub(nx, ny) }
            }
            BvOp::Mul(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_mul(nx, ny) }
            }
            BvOp::Udiv(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_udiv(nx, ny) }
            }
            BvOp::Urem(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_urem(nx, ny) }
            }
            BvOp::Sdiv(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_sdiv(nx, ny) }
            }
            BvOp::Srem(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_srem(nx, ny) }
            }
            BvOp::Smod(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_smod(nx, ny) }
            }
            BvOp::Shl(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_shl(nx, ny) }
            }
            BvOp::Lshr(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_lshr(nx, ny) }
            }
            BvOp::Ashr(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_ashr(nx, ny) }
            }
            BvOp::RotateLeft(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_rotate_left_dyn(nx, ny) }
            }
            BvOp::RotateRight(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_rotate_right_dyn(nx, ny) }
            }
            BvOp::Popcount(x) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_popcount(nx) }
            }
            BvOp::Clz(x) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_clz(nx) }
            }
            BvOp::Ctz(x) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_ctz(nx) }
            }
            BvOp::Extract(x, hi, lo) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_extract(nx, hi, lo) }
            }
            BvOp::Concat(x, y) => {
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.ctx.bv_concat(nx, ny) }
            }
            BvOp::ZeroExtend(x, n) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_zero_extend(nx, n) }
            }
            BvOp::SignExtend(x, n) => {
                let nx = self.apply_subst_bv(x, bm, vm);
                if nx == x { t } else { self.ctx.bv_sign_extend(nx, n) }
            }
            BvOp::Ite(c, x, y) => {
                let nc = self.apply_subst_bool(c, bm, vm);
                let (nx, ny) =
                    (self.apply_subst_bv(x, bm, vm), self.apply_subst_bv(y, bm, vm));
                if nc == c && nx == x && ny == y {
                    t
                } else {
                    self.ctx.bv_ite(nc, nx, ny)
                }
            }
            BvOp::Select(idx) => {
                let table = self.ctx.select_tables[idx as usize].clone();
                let sels: Vec<BoolTerm> = table
                    .selectors
                    .iter()
                    .map(|&s| self.apply_subst_bool(s, bm, vm))
                    .collect();
                let vals: Vec<BvTerm> = table
                    .values
                    .iter()
                    .map(|&v| self.apply_subst_bv(v, bm, vm))
                    .collect();
                let ndef = self.apply_subst_bv(table.default, bm, vm);
                let unchanged = ndef == table.default
                    && sels.iter().zip(table.selectors.iter()).all(|(a, b)| a == b)
                    && vals.iter().zip(table.values.iter()).all(|(a, b)| a == b);
                if unchanged {
                    t
                } else {
                    self.ctx.bv_select(&sels, &vals, ndef)
                }
            }
        };
        vm.insert(t, r);
        r
    }

    /// Bitblast every pending assertion, emitting SAT clauses for the AIG
    /// cone each assertion actually reaches. After flush, the pending queues
    /// for every scope are empty and all those assertions live in the SAT
    /// core (guarded by activation literals for scopes ≥ 1).
    fn flush_pending(&mut self) {
        let has_work = self.pending.iter().any(|q| !q.is_empty());
        if !has_work {
            // Flush is the funnel every solve/probe/batch entry point runs
            // through, right before the SAT core (possibly) rewinds the
            // trail — the one reliable moment to bank a standing model.
            self.bank_model();
            self.resolve_pending_ites();
            return;
        }
        {
            // New assertions strengthen the formula: neither the banked
            // model nor the standing trail model can vouch for it anymore.
            // (This also poisons the current generation — a flush absorbed
            // entirely by top-level substitution changes semantics without
            // emitting a single clause, leaving the trail intact.)
            self.invalidate_banked_model();
            // Flush work can rebind term meanings (substitutions) and
            // dissolve materialized lits (BVE) — drop the prefix caches.
            self.asmp_terms_cache.clear();
            self.asmp_lits_cache.clear();
            self.gate_terms_cache.clear();
            self.gate_refs_cache.clear();
            let t_front = std::time::Instant::now();
            // Variables at or above this index were allocated by (and are
            // only visible to) this batch — eligible for elimination if
            // they're gate outputs. Everything older may be referenced by
            // clauses already in the SAT core and must survive.
            let batch_start_var = self.sat.num_vars();
            // Collect the batch per scope depth.
            let mut batch: Vec<(usize, Vec<BoolTerm>)> = Vec::new();
            for depth in 0..self.pending.len() {
                let terms = std::mem::take(&mut self.pending[depth]);
                if !terms.is_empty() {
                    batch.push((depth, terms));
                }
            }

            // Top-level variable substitution (bitwuzla's varsubst-lite):
            // only sound outside push scopes — a popped scope must not
            // leave permanent substitutions behind.
            if self.activation_stack.is_empty() {
                let before = self.bv_var_subst.len();
                if self.subst_enabled {
                    self.collect_substitutions(&batch);
                }
                // Gaussian elimination catches coupled linear systems that
                // single-variable substitution leaves untouched. Runs on
                // the equalities that survived collect_substitutions.
                let ge_unsat = if self.gauss_enabled {
                    self.gaussian_eliminate(&mut batch)
                } else {
                    false
                };
                if ge_unsat {
                    // A linear row reduced to `0 = nonzero`: the formula is
                    // unconditionally UNSAT. Commit the empty clause so
                    // every later solve returns UNSAT, and skip the rest.
                    self.sat.add_clause(vec![]);
                    return;
                }
                if self.bv_var_subst.len() != before {
                    // New substitutions invalidate prior rewrites.
                    self.subst_bool_memo.clear();
                    self.subst_bv_memo.clear();
                }
                if !self.bv_var_subst.is_empty() {
                    let mut bm = std::mem::take(&mut self.subst_bool_memo);
                    let mut vm = std::mem::take(&mut self.subst_bv_memo);
                    for (_, ts) in batch.iter_mut() {
                        for t in ts.iter_mut() {
                            *t = self.apply_subst_bool(*t, &mut bm, &mut vm);
                        }
                    }
                    self.subst_bool_memo = bm;
                    self.subst_bv_memo = vm;
                }
            }

            // Arithmetic normalization with size-based acceptance: rewrite
            // the batch, then bitblast BOTH variants into the AIG (pure —
            // no CNF is emitted by bitblasting) and count fresh nodes.
            // Keep the normalized batch only if it builds a smaller
            // circuit. Reassociation can either merge thousands of
            // permuted adder chains into one (huge win) or destroy
            // sharing with non-additive consumers (huge loss) — measuring
            // is the only reliable arbiter, and the loser's AIG nodes are
            // memory-only garbage.
            if self.normalize_enabled {
                let cancelled_before = self.ctx.norm_cancelled;
                let merged_before = self.ctx.norm_merged;
                let mut bm = std::mem::take(&mut self.norm_bool_memo);
                let mut vm = std::mem::take(&mut self.norm_bv_memo);
                let normalized: Vec<(usize, Vec<BoolTerm>)> = batch
                    .iter()
                    .map(|(d, ts)| {
                        (
                            *d,
                            ts.iter()
                                .map(|&t| self.ctx.normalize_assertion(t, &mut bm, &mut vm))
                                .collect(),
                        )
                    })
                    .collect();
                self.norm_bool_memo = bm;
                self.norm_bv_memo = vm;

                let changed = normalized
                    .iter()
                    .zip(batch.iter())
                    .any(|((_, na), (_, oa))| na != oa);
                if changed {
                    let cancelled = self.ctx.norm_cancelled - cancelled_before;
                    let merged = self.ctx.norm_merged - merged_before;
                    // Accept the normalized batch only when equality
                    // flattening actually cancelled addends across sides —
                    // a semantic simplification that reliably predicts a
                    // search win even when coefficient rebuilds grow the
                    // raw circuit (bench_5906: 4× more AIG nodes, 15×
                    // faster solve). Pure reassociation (no cancellation)
                    // was tried under several acceptance schemes —
                    // unconditional, fresh-node scoring, reachable-cone
                    // scoring — and consistently traded wins on one family
                    // for losses on another; circuit size is a poor proxy
                    // for search hardness, so without the cancellation
                    // signal we keep the original formula.
                    // Threshold on merge volume: the decisive wins on
                    // this workload collapse tens of thousands of
                    // duplicated addends (bench_5906/64/13728: ~19-20K
                    // merges alongside ~300 cancellations); the one
                    // observed counterexample where cancellation alone
                    // mispredicts (bench_7373, unsat: 0.9s → 12s) sits at
                    // 7K. Empirical, and deliberately conservative —
                    // a rejected batch behaves exactly like the
                    // pre-normalization solver.
                    let accept = cancelled > 0 && merged >= 10_000;
                    if accept {
                        batch = normalized;
                    }
                }
            }

            // Sharing-aware substitution pass (gated): bitblast the batch
            // purely (memoized — the assert loop below re-hits the caches),
            // then rewrite with full parent-count knowledge so substitution
            // never bypasses a shared interior. Must run before
            // materialization so emission sees the rewritten structure.
            if self.aig2_post {
                let mut roots: Vec<crate::aig::AigRef> = Vec::new();
                for (_, terms) in &batch {
                    for &t in terms {
                        roots.push(self.bitblast_bool(t));
                    }
                }
                let pinned: Vec<bool> = self.aig_lit.iter().map(|l| l.is_some()).collect();
                let stats = self.aig.substitute_pass(&roots, &pinned);
                self.aig2_post_stats.accumulate(stats);
            }

            // FRAIG sweep (gated): bitblast the whole batch into the AIG
            // purely first (bitblasting emits no CNF; these calls are
            // memoized, so the assert loop below re-hits the caches), then
            // prove-and-merge equivalence candidates so materialization
            // reuses one SAT lit per proven class. Unconditional
            // equivalences — sound across scopes and later assertions.
            if self.fraig_enabled {
                let t0 = std::time::Instant::now();
                for (_, terms) in &batch {
                    for &t in terms {
                        let _ = self.bitblast_bool(t);
                    }
                }
                let start = self.fraig_swept_upto;
                let stats = crate::fraig::sweep(
                    &mut self.aig,
                    start,
                    FRAIG_MAX_QUERIES,
                    FRAIG_MAX_CONFLICTS,
                    0x5EED_CAFE_F00D_D00D ^ start as u64,
                );
                self.fraig_swept_upto = self.aig.num_nodes() as u32;
                self.fraig_stats.accumulate(stats);
                self.fraig_time += t0.elapsed();
            }

            self.time_front += t_front.elapsed();
            let t_emit = std::time::Instant::now();
            // Reuse the retired flush buffer — steady-state emission
            // allocates nothing.
            let mut buf = std::mem::take(&mut self.cnf_buffer_pool);
            buf.clear();
            self.cnf_buffer = Some(buf);
            for (depth, terms) in batch {
                let act_lit = if depth == 0 {
                    None
                } else {
                    Some(self.activation_stack[depth - 1])
                };
                for t in terms {
                    self.assert_toplevel_direct(t, act_lit);
                }
            }
            let buffer = self.cnf_buffer.take().unwrap_or_default();
            self.time_emit += t_emit.elapsed();
            // Resolve ITE metadata + selector boosts BEFORE the batch is
            // preprocessed: bounded VE may dissolve a live mux's output
            // var entirely (a good outcome — its function got resolved
            // into the neighbours), but the gate was real and its selector
            // still deserves the branching hint.
            self.resolve_pending_ites();
            let t_pp = std::time::Instant::now();
            self.commit_batch(buffer, batch_start_var);
            self.time_preprocess += t_pp.elapsed();
        }
        self.resolve_pending_ites();
    }

    /// Promote deferred ITE-gate records into the Lit-keyed public
    /// registry, and apply queued selector VSIDS boosts. Only gates whose
    /// output actually got materialized (i.e. reachable from an asserted
    /// root) are registered — dead-code muxes don't pollute the
    /// introspection view. `try_lit_of` on the operands keeps this
    /// metadata-only: if an operand lacks a lit (its cone materialized
    /// through a non-pattern path), the record is skipped rather than
    /// forcing CNF emission for bookkeeping. Note: a registered gate's
    /// output var may later be dissolved by CNF preprocessing.
    fn resolve_pending_ites(&mut self) {
        let pending_ites = std::mem::take(&mut self.pending_ite_gates);
        // Canonical processing order: by materialized output variable.
        // Var indices follow structural materialization order, so gate
        // registration (and the VSIDS bump sequence it drives) is
        // independent of how many bitblast passes queued the records.
        let mut resolved: Vec<(Lit, PendingIte)> = pending_ites
            .into_iter()
            .filter_map(|rec| self.try_lit_of(rec.out).map(|o| (o, rec)))
            .collect();
        resolved.sort_by_key(|(o, _)| o.var_idx());
        for (o, rec) in resolved {
            if self.ite_out_to_gate.contains_key(&o) {
                continue;
            }
            let (Some(sel), Some(t), Some(e)) = (
                self.try_lit_of(rec.sel),
                self.try_lit_of(rec.t),
                self.try_lit_of(rec.e),
            ) else {
                continue;
            };
            let idx = self.ite_gates.len();
            self.ite_gates.push(IteGate {
                sel,
                t,
                e,
                o,
                source_term: rec.src,
            });
            self.ite_out_to_gate.insert(o, idx);
            // Branching hint: deciding `sel` resolves the whole ITE
            // subtree, so bump it once per live gate. Width-N ITEs stack
            // N bumps on the same selector (one per bit-level gate),
            // giving deep / wide ITE fan-outs a proportionally strong
            // priority. Registration is deduped by output lit, so the
            // bump count is independent of how many bitblast passes
            // (e.g. normalization scoring) touched the mux.
            if self.ite_branching_hints {
                self.sat.boost_var_activity(sel.var());
            }
        }
    }

    /// Preprocess one flush batch (subsumption + bounded variable
    /// elimination via [`crate::preprocess`]) and commit the survivors to
    /// the SAT core.
    ///
    /// Frozen (non-eliminable) variables: anything allocated before this
    /// batch (older clauses in the SAT core may mention it) and anything
    /// that isn't a Tseitin gate output — input bits are read by model
    /// evaluation, activation literals by push/pop and unsat cores, the
    /// true-lit by its pinned unit. Freshly-allocated gate variables are
    /// invisible outside the batch, so eliminating them is sound: their
    /// AIG-node binding is dropped, and any later consumer re-materializes
    /// the node under a fresh variable with fresh defining clauses.
    fn commit_batch(&mut self, buffer: CnfBuffer, batch_start_var: usize) {
        let CnfBuffer { mut data, mut ends } = buffer;
        if ends.is_empty() {
            self.cnf_buffer_pool = CnfBuffer { data, ends };
            return;
        }
        // Remap the batch's literals into a compact variable space [0, k)
        // where k = distinct variables appearing in this batch. Every
        // preprocessing array (occurrence lists, counts, frozen flags) and
        // every scan is then O(batch), not O(total variables ever
        // allocated) — the difference between linear and quadratic cost
        // across an incremental session that keeps adding small batches.
        //
        // The remap is folded into the pre-filter pass that already
        // rewrites the flat buffer in place: drop clauses satisfied by a
        // root-level fact (the pinned true-lit especially) and strip
        // level-0-false literals via `value_fixed`, which ignores stale
        // search-trail assignments above level 0. Both cursors trail the
        // read position, so the compaction never moves a literal forward.
        let mut to_orig = std::mem::take(&mut self.pp_to_orig); // compact → original
        let mut to_compact = std::mem::take(&mut self.pp_to_compact);
        to_orig.clear();
        to_compact.clear();
        let mut w = 0usize; // literal write cursor
        let mut kept = 0usize; // clause write cursor
        let mut start = 0usize;
        for r in 0..ends.len() {
            let end = ends[r] as usize;
            let clause_w = w;
            let mut satisfied = false;
            for i in start..end {
                let l = data[i];
                match self.sat.value_fixed(l) {
                    LBool::True => {
                        satisfied = true;
                        break;
                    }
                    LBool::False => {}
                    LBool::Undef => {
                        let ov = l.var().0;
                        let cv = *to_compact.entry(ov).or_insert_with(|| {
                            let id = to_orig.len() as u32;
                            to_orig.push(ov);
                            id
                        });
                        data[w] = Lit::new(Var(cv), l.is_negated());
                        w += 1;
                    }
                }
            }
            if satisfied {
                w = clause_w; // discard the partial write
            } else {
                ends[kept] = w as u32;
                kept += 1;
            }
            start = end;
        }
        data.truncate(w);
        ends.truncate(kept);

        // BVE disabled: commit the (pre-filtered) clauses directly, mapping
        // compact lits back to originals in place. Skips the Preprocessor.
        if !self.bve_enabled {
            let mut start = 0usize;
            for &e in &ends {
                let end = e as usize;
                for l in data[start..end].iter_mut() {
                    *l = Lit::new(Var(to_orig[l.var_idx()]), l.is_negated());
                }
                self.sat.add_clause_from_slice(&data[start..end]);
                start = end;
            }
            self.cnf_buffer_pool = CnfBuffer { data, ends };
            self.pp_to_orig = to_orig;
            self.pp_to_compact = to_compact;
            return;
        }

        let k = to_orig.len();
        // frozen[compact] — a variable survives elimination unless it is a
        // gate output allocated by THIS batch (older vars may be referenced
        // by clauses already committed to the SAT core; inputs / activation
        // lits / the true-lit are read elsewhere).
        // Cut roots are frozen too. A Tseitin gate output is defined by a
        // handful of narrow clauses, so resolving it away is cheap and
        // usually a win; a cut root is defined by two wide ISOP covers,
        // and eliminating one splices them into far wider resolvents,
        // destroying the propagation structure the mapper chose. On the
        // XOR-heavy spear instances — where BVE is otherwise the single
        // biggest lever (36% of variables) — letting BVE at the cut roots
        // cost 2.2× (mapped-full 0.42s → 0.92s) and more than doubled
        // conflicts (9053 → 21428).
        let mut frozen = std::mem::take(&mut self.pp_pool.frozen);
        frozen.clear();
        frozen.reserve(k);
        for &ov in &to_orig {
            let ov = ov as usize;
            let f = ov < batch_start_var
                || !matches!(
                    self.var_origin[ov],
                    VarOrigin::GateOut { gate, .. } if gate != GateKind::Cut
                );
            frozen.push(f);
        }

        let mut pp = crate::preprocess::Preprocessor::from_flat(
            data,
            &ends,
            k,
            frozen,
            std::mem::take(&mut self.pp_pool),
        );
        // Gate substitution and two-level AIG rewriting are both circuit
        // minimizers; stacked they measured strongly net-negative on the
        // Sage2 family (see the field doc in preprocess.rs). One at a
        // time by default: aig2 sessions keep classic full VE only,
        // unless the caller overrides (see `set_ve_gate_substitution`).
        pp.set_gate_substitution(
            self.ve_gate_subst
                .unwrap_or_else(|| !self.aig.two_level_enabled()),
        );
        let result = pp.run();
        self.pp_eliminated += result.eliminated.len() as u64;
        self.pp_subsumed += result.subsumed as u64;
        self.pp_strengthened += result.strengthened as u64;

        // Un-bind eliminated gate vars from their AIG nodes so later
        // consumers re-materialize them freshly instead of referencing
        // variables whose defining clauses no longer exist. Compact ids map
        // back through `to_orig`.
        for &cv in &result.eliminated {
            let ov = to_orig[cv as usize] as usize;
            let node = self.lit_node[ov];
            if node != u32::MAX {
                self.aig_lit[node as usize] = None;
                self.lit_node[ov] = u32::MAX;
                self.elim_nodes.insert(node);
            }
            // FRAIG alias nodes bound to this variable's lit must be
            // invalidated too — their defining clauses are the ones just
            // eliminated. They re-materialize through their merge target.
            if let Some(aliases) = self.aig_lit_aliases.remove(&(ov as u32)) {
                for n in aliases {
                    self.aig_lit[n as usize] = None;
                }
            }
            // The variable has no clauses left — exclude it from branching
            // so model completion doesn't pay a decision for it.
            self.sat.set_decision_var(Var(ov as u32), false);
        }

        // Map surviving clauses back to original variable ids in place
        // (`result.data` is the same arena the emission buffer became —
        // no per-clause storage exists anywhere on this path) and commit.
        let mut arena = result.data;
        for &(off, len) in &result.clauses {
            let range = &mut arena[off as usize..(off + len) as usize];
            for l in range.iter_mut() {
                *l = Lit::new(Var(to_orig[l.var_idx()]), l.is_negated());
            }
            self.sat.add_clause_from_slice(&arena[off as usize..(off + len) as usize]);
        }
        // Retire every reusable buffer for the next flush.
        self.cnf_buffer_pool = CnfBuffer { data: arena, ends };
        self.pp_pool = result.pool;
        self.pp_to_orig = to_orig;
        self.pp_to_compact = to_compact;
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
    /// Run the FRAIG feasibility diagnostic over the accumulated AIG —
    /// see [`Aig::sim_sweep`] for what the numbers mean.
    pub fn fraig_diagnostic(&self) -> crate::aig::SimSweepStats {
        self.aig.sim_sweep(0x5EED_CAFE_F00D_D00D)
    }

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
            reused_levels: self.sat.stats_reused_levels,
            viv_checked: self.sat.stats_viv_checked,
            viv_strengthened: self.sat.stats_viv_strengthened,
            viv_deleted: self.sat.stats_viv_deleted,
            viv_units: self.sat.stats_viv_units,
            time_front: self.time_front.as_secs_f64(),
            time_emit: self.time_emit.as_secs_f64(),
            time_preprocess: self.time_preprocess.as_secs_f64(),
            time_sat: self.time_sat.as_secs_f64(),
            learned: self.sat.stats_learned,
            propagations: self.sat.stats_propagations,
            reductions: self.sat.stats_reductions,
            gcs: self.sat.stats_gcs,
            bv_aliased,
            bool_aliased,
            bv_var_total: self.ctx.bv_var_widths.len(),
            bv_nodes_total: self.ctx.bv_nodes.len(),
            bv_vars_bitblasted: self.bv_var_refs.len(),
            pp_substituted: self.pp_substituted,
            pp_eliminated: self.pp_eliminated,
            pp_subsumed: self.pp_subsumed,
            pp_remat: self.pp_remat,
            pp_strengthened: self.pp_strengthened,
        }
    }

    /// Number of AIG nodes currently in the bitblaster's graph (including
    /// the constant sentinel). Useful for measuring structural dedup.
    pub fn aig_nodes(&self) -> usize {
        self.aig.num_nodes()
    }


    // ---------- Metadata accessors ----------

    /// Turn on bitblast cost attribution. From this point onward every SAT
    /// var / clause emitted during CNF materialization is charged to the BV
    /// term whose bitblast first created the corresponding AIG node.
    /// Cheap to leave on — one HashMap update per emitted gate. Call
    /// [`bitblast_cost_report`] after solving to read out a ranked table.
    ///
    /// Calling this resets any previously-collected data.
    pub fn enable_bitblast_cost_tracking(&mut self) {
        self.bitblast_cost_enabled = true;
        self.bitblast_cost.clear();
    }

    /// Snapshot of the bitblast cost map, sorted by clause count
    /// descending. Empty if [`enable_bitblast_cost_tracking`] was never
    /// called. Each entry tells you how many SAT vars / clauses that BV
    /// term contributed to the formula on its own.
    ///
    /// Bin't (or any caller) is expected to keep its own
    /// `BvTerm → source-instruction` mapping and join on `.term` to map
    /// back to symex-level operations.
    pub fn bitblast_cost_report(&self) -> Vec<BitblastCostEntry> {
        let mut out: Vec<BitblastCostEntry> = self
            .bitblast_cost
            .iter()
            .map(|(&t, &(v, c))| BitblastCostEntry {
                term: t,
                width: self.ctx.width_of(t),
                sat_vars: v,
                sat_clauses: c,
            })
            .collect();
        out.sort_by(|a, b| b.sat_clauses.cmp(&a.sat_clauses));
        out
    }

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

    // ---------- Model reads (AIG evaluation) ----------

    /// Read a BV value out of the current SAT model. Widths up to 64 are
    /// safe; for wider BVs the upper bits are truncated — use
    /// [`get_bv_value_u128`] for the full range.
    pub fn get_bv_value(&mut self, t: BvTerm) -> u64 {
        self.get_bv_value_u128(t) as u64
    }

    /// Full-precision model read: supports widths up to 128.
    pub fn get_bv_value_u128(&mut self, t: BvTerm) -> u128 {
        let bits = self.bitblast_bv(t);
        let vals = self.eval_refs(&bits);
        let mut v = 0u128;
        for (i, &bit) in vals.iter().enumerate() {
            if bit {
                v |= 1u128 << i;
            }
        }
        v
    }

    /// Arbitrary-width model read: returns little-endian u64 limbs. Works
    /// for any BV width including those exceeding 128 bits.
    pub fn get_bv_value_limbs(&mut self, t: BvTerm) -> Vec<u64> {
        let bits = self.bitblast_bv(t);
        let vals = self.eval_refs(&bits);
        let nlimbs = (bits.len() + 63) / 64;
        let mut limbs = vec![0u64; nlimbs];
        for (i, &bit) in vals.iter().enumerate() {
            if bit {
                limbs[i / 64] |= 1u64 << (i % 64);
            }
        }
        limbs
    }

    pub fn get_bool_value(&mut self, t: BoolTerm) -> bool {
        let r = self.bitblast_bool(t);
        self.eval_refs(&[r])[0]
    }

    /// Evaluate AIG refs under the current SAT model *without* forcing any
    /// CNF emission. See [`eval_refs_from`].
    fn eval_refs(&mut self, refs: &[AigRef]) -> Vec<bool> {
        self.eval_refs_from(refs, ModelSource::Trail)
    }

    /// A literal's value in the chosen model: the live SAT trail, or the
    /// banked copy (where literals from variables created after banking
    /// read Undef).
    #[inline]
    fn model_value_of(&self, l: Lit, src: ModelSource) -> LBool {
        match src {
            ModelSource::Trail => self.sat.value_of(l),
            ModelSource::Banked => self
                .banked_model
                .get(l.idx())
                .copied()
                .unwrap_or(LBool::Undef),
        }
    }

    /// Model value of `r` via its materialized SAT lit only — no
    /// structural walk. `None` when the node is unmaterialized or its lit
    /// is unassigned in the chosen model; callers fall back to
    /// [`eval_refs_from`]. This is the O(1) fast path that keeps warm-gate
    /// validation linear in the number of assumptions for the common case
    /// where they were materialized by an earlier solve.
    #[inline]
    fn model_ref_fast(&self, r: AigRef, src: ModelSource) -> Option<bool> {
        let idx = r.node_idx();
        let base = match self.aig.node(idx) {
            AigNode::ConstTrue => true,
            AigNode::Input(l) => match self.model_value_of(l, src) {
                LBool::Undef => return None,
                v => v == LBool::True,
            },
            AigNode::And(..) => match self.node_lit(idx) {
                Some(l) => match self.model_value_of(l, src) {
                    LBool::Undef => return None,
                    v => v == LBool::True,
                },
                None => return None,
            },
        };
        Some(base ^ r.is_negated())
    }

    /// Evaluate AIG refs under the chosen model *without* forcing any CNF
    /// emission. Nodes with an assigned materialized lit read straight
    /// from the model (and don't recurse — Tseitin consistency guarantees
    /// the stored value matches the recomputed one); everything else is
    /// computed structurally from the inputs. Unassigned inputs (vars
    /// created after the model) default to false, matching a "model
    /// completion with 0" convention.
    ///
    /// Hot path for model screening: the memo is the persistent
    /// epoch-stamped `eval_stamp`/`eval_value` pair (no allocation, no
    /// hashing), and an And with one false child never visits the other
    /// child's cone. Within one call refs share the memo, so evaluating
    /// `cond` makes `¬cond` free.
    fn eval_refs_from(&mut self, refs: &[AigRef], src: ModelSource) -> Vec<bool> {
        // Buffers move to locals so the walk can mutate them while
        // reading `self` (AIG, models) — restored before returning.
        let mut stamp = std::mem::take(&mut self.eval_stamp);
        let mut value = std::mem::take(&mut self.eval_value);
        let mut stack = std::mem::take(&mut self.eval_stack);
        self.eval_epoch = self.eval_epoch.wrapping_add(1);
        if self.eval_epoch == 0 {
            // Epoch wrapped: stale stamps could collide. Hard reset.
            stamp.clear();
            self.eval_epoch = 1;
        }
        let epoch = self.eval_epoch;
        if stamp.len() < self.aig.num_nodes() {
            stamp.resize(self.aig.num_nodes(), 0);
            value.resize(self.aig.num_nodes(), false);
        }

        let mut out = Vec::with_capacity(refs.len());
        for &r in refs {
            let root = r.node_idx() as usize;
            if stamp[root] != epoch {
                stack.push(r.node_idx());
                while let Some(&top) = stack.last() {
                    let ti = top as usize;
                    if stamp[ti] == epoch {
                        stack.pop();
                        continue;
                    }
                    // Materialized + assigned: read the model directly.
                    if let Some(Some(l)) = self.aig_lit.get(ti) {
                        let v = self.model_value_of(*l, src);
                        if v != LBool::Undef {
                            stamp[ti] = epoch;
                            value[ti] = v == LBool::True;
                            stack.pop();
                            continue;
                        }
                    }
                    match self.aig.node(top) {
                        AigNode::ConstTrue => {
                            stamp[ti] = epoch;
                            value[ti] = true;
                            stack.pop();
                        }
                        AigNode::Input(l) => {
                            stamp[ti] = epoch;
                            value[ti] = self.model_value_of(l, src) == LBool::True;
                            stack.pop();
                        }
                        AigNode::And(a, b) => {
                            let ai = a.node_idx() as usize;
                            let bi = b.node_idx() as usize;
                            let av = (stamp[ai] == epoch)
                                .then(|| value[ai] ^ a.is_negated());
                            // Short-circuit: one false child decides the
                            // node without visiting the other's cone.
                            if av == Some(false) {
                                stamp[ti] = epoch;
                                value[ti] = false;
                                stack.pop();
                                continue;
                            }
                            let bv = (stamp[bi] == epoch)
                                .then(|| value[bi] ^ b.is_negated());
                            if bv == Some(false) {
                                stamp[ti] = epoch;
                                value[ti] = false;
                                stack.pop();
                                continue;
                            }
                            match (av, bv) {
                                // Neither child is false: both true.
                                (Some(_), Some(_)) => {
                                    stamp[ti] = epoch;
                                    value[ti] = true;
                                    stack.pop();
                                }
                                (None, _) => stack.push(a.node_idx()),
                                (_, None) => stack.push(b.node_idx()),
                            }
                        }
                    }
                }
            }
            out.push(value[root] ^ r.is_negated());
        }
        stack.clear();
        self.eval_stamp = stamp;
        self.eval_value = value;
        self.eval_stack = stack;
        out
    }

    // ---------- Bitblasting (term → AIG) ----------

    /// Produce N AIG refs (LSB-first) representing the bits of `t`. No CNF
    /// is emitted — materialization happens at flush via `lit_of`.
    fn bitblast_bv(&mut self, t: BvTerm) -> Vec<AigRef> {
        if let Some(cached) = self.bv_cache.get(&t) {
            return cached.clone();
        }
        // Record the current BV term so AIG nodes created inside gate
        // helpers (mk_and, mk_mux, ripple_carry_add, …) inherit it as their
        // source tag. Save / restore so recursive bitblast calls see their
        // own enclosing term.
        let prev_ctx = self.current_bv_ctx;
        self.current_bv_ctx = Some(t);
        let node = self.ctx.bv_nodes[t.0 as usize];
        let bits = match node.op {
            BvOp::Var(id) => {
                // Substituted variables never allocate SAT bits — they ARE
                // their target term. Keeps model reads consistent:
                // get_bv_value(x) evaluates t's cone.
                let root = self.find_bv_var_root(id);
                match self.bv_var_subst.get(&root).copied() {
                    Some(sub) => self.bitblast_bv(sub),
                    None => self.get_or_make_bv_var(root, node.width),
                }
            }
            BvOp::Const => {
                if node.wide == crate::bv::WIDE_NONE {
                    // Fast path: value lives inline as a u128.
                    let value = node.value;
                    (0..node.width)
                        .map(|i| {
                            if (value >> i) & 1 == 1 { AigRef::TRUE } else { AigRef::FALSE }
                        })
                        .collect()
                } else {
                    // Wide path: read bit-i from the context's limb pool.
                    let limbs = self.ctx.wide_limbs(node.wide).to_vec();
                    (0..node.width)
                        .map(|i| {
                            let li = (i as usize) / 64;
                            let bi = i % 64;
                            if (limbs[li] >> bi) & 1 == 1 { AigRef::TRUE } else { AigRef::FALSE }
                        })
                        .collect()
                }
            }
            BvOp::Not(x) => {
                let xb = self.bitblast_bv(x);
                xb.iter().map(|&r| !r).collect()
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
                self.ripple_carry_add(&xb, &yb, AigRef::FALSE).0
            }
            BvOp::Sub(x, y) => {
                // a - b = a + ~b + 1
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                let y_neg: Vec<AigRef> = yb.iter().map(|&r| !r).collect();
                self.ripple_carry_add(&xb, &y_neg, AigRef::TRUE).0
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
                let y_is_zero = self.mk_all_zero(&yb);
                let ones = vec![AigRef::TRUE; q.len()];
                self.mux_vec(y_is_zero, &ones, &q)
            }
            BvOp::Urem(x, y) => {
                let xb = self.bitblast_bv(x);
                let yb = self.bitblast_bv(y);
                let (_q, r) = self.mk_udivmod(&xb, &yb);
                // bvurem(x, 0) = x.
                let y_is_zero = self.mk_all_zero(&yb);
                self.mux_vec(y_is_zero, &xb, &r)
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
                if let Some(amt) = self.const_shift_amt(y) {
                    self.mk_shr_const(&xb, amt, AigRef::FALSE)
                } else {
                    let yb = self.bitblast_bv(y);
                    self.mk_shr(&xb, &yb, AigRef::FALSE)
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
                for _ in 0..n {
                    result.push(AigRef::FALSE);
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
                let sel_refs: Vec<AigRef> = table
                    .selectors
                    .iter()
                    .map(|&s| self.bitblast_bool(s))
                    .collect();
                let value_bit_vecs: Vec<Vec<AigRef>> = table
                    .values
                    .iter()
                    .map(|&v| self.bitblast_bv(v))
                    .collect();
                let n_bits = default_bits.len();
                let mut output: Vec<AigRef> = default_bits.clone();
                // Walk right-to-left so the FIRST (outermost) selector ends
                // up taking priority — mux(sel_i, val_i, acc) means "if
                // sel_i, val_i, else whatever the tail produced".
                for i in (0..sel_refs.len()).rev() {
                    let sel = sel_refs[i];
                    let value = &value_bit_vecs[i];
                    for bit in 0..n_bits {
                        output[bit] = self.mk_mux(sel, value[bit], output[bit]);
                    }
                }
                output
            }
            BvOp::Popcount(x) => {
                let xb = self.bitblast_bv(x);
                let w = self.ctx.width_of(x) as usize;
                self.mk_popcount(&xb, w)
            }
            BvOp::Clz(x) => {
                let xb = self.bitblast_bv(x);
                let w = self.ctx.width_of(x) as usize;
                self.mk_clz(&xb, w)
            }
            BvOp::Ctz(x) => {
                let xb = self.bitblast_bv(x);
                let w = self.ctx.width_of(x) as usize;
                self.mk_ctz(&xb, w)
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

    /// Produce a single AIG ref for `t`.
    fn bitblast_bool(&mut self, t: BoolTerm) -> AigRef {
        if let Some(&cached) = self.bool_cache.get(&t) {
            return cached;
        }
        let op = self.ctx.bool_nodes[t.0 as usize];
        let r = match op {
            BoolOp::True => AigRef::TRUE,
            BoolOp::False => AigRef::FALSE,
            BoolOp::Var(id) => {
                let id = self.find_bool_var_root(id);
                if let Some(&cached) = self.bool_var_refs.get(&id) {
                    cached
                } else {
                    let l = self.new_sat_lit_tagged(VarOrigin::Bool { term: t });
                    let input = self.aig.input(l);
                    self.set_node_lit(input.node_idx(), l);
                    self.bool_var_refs.insert(id, input);
                    input
                }
            }
            BoolOp::Not(x) => {
                let xr = self.bitblast_bool(x);
                !xr
            }
            BoolOp::And(x, y) => {
                let xr = self.bitblast_bool(x);
                let yr = self.bitblast_bool(y);
                self.mk_and(xr, yr)
            }
            BoolOp::Or(x, y) => {
                let xr = self.bitblast_bool(x);
                let yr = self.bitblast_bool(y);
                self.mk_or(xr, yr)
            }
            BoolOp::Implies(x, y) => {
                let xr = self.bitblast_bool(x);
                let yr = self.bitblast_bool(y);
                self.mk_or(!xr, yr)
            }
            BoolOp::Eq(a, b) => {
                // Width-1 fast path: 1-bit equality is a single XNOR, and
                // the AIG folds constants on either side for free (e.g.
                // `(= x (_ bv1 1))` is a pure lift of x's bit — no node).
                if self.ctx.width_of(a) == 1 {
                    let ar = self.bitblast_bv(a)[0];
                    let br = self.bitblast_bv(b)[0];
                    !self.mk_xor(ar, br)
                } else {
                    let ab = self.bitblast_bv(a);
                    let bb = self.bitblast_bv(b);
                    self.mk_bitwise_eq(&ab, &bb)
                }
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
                let (_sum, cout) = self.ripple_carry_add(&ab, &bb, AigRef::FALSE);
                cout
            }
            BoolOp::SaddOverflow(a, b) => {
                // Signed add overflows iff: sign(a) == sign(b) && sign(sum) != sign(a).
                let ab = self.bitblast_bv(a);
                let bb = self.bitblast_bv(b);
                let (sum, _) = self.ripple_carry_add(&ab, &bb, AigRef::FALSE);
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
                let b_neg: Vec<AigRef> = bb.iter().map(|&r| !r).collect();
                let (diff, _) = self.ripple_carry_add(&ab, &b_neg, AigRef::TRUE);
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
                let diffs: Vec<AigRef> =
                    hi.iter().map(|&h| self.mk_xor(h, sign_of_lo)).collect();
                self.mk_any_set(&diffs)
            }
            BoolOp::NegOverflow(a) => {
                // -x overflows iff x = INT_MIN = sign-bit-set, all-others-zero.
                let ab = self.bitblast_bv(a);
                let n = ab.len();
                let mut acc = ab[n - 1]; // MSB must be 1
                for i in 0..n - 1 {
                    acc = self.mk_and(acc, !ab[i]); // others must be 0
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
        self.bool_cache.insert(t, r);
        r
    }

    // ---------- CNF materialization (AIG → clauses) ----------

    /// Materialize an AIG ref to a SAT literal, emitting Tseitin clauses for
    /// every not-yet-emitted node in its cone. Iterative post-order walk —
    /// never recurses, so arbitrarily deep AIGs are fine.
    ///
    /// Shape-aware emission: an And node whose structure matches the 3-node
    /// XOR / MUX patterns produced by `Aig::xor` / `Aig::mux` gets the
    /// direct 4-clause single-var encoding over the pattern's *operands*;
    /// the two interior And nodes are skipped entirely (they get their own
    /// vars only if some other consumer independently materializes them).
    /// This keeps CNF size at parity with a hand-written Tseitin encoder
    /// while everything still flows through one AIG.
    /// Materialize `root`'s unmaterialized cone through cut-based CNF
    /// mapping (`crate::cnfmap`): plan a cut cover treating bound nodes,
    /// inputs, and constants as leaves, then give each chosen cut root a
    /// variable defined by the ISOP cubes of its cut function over the
    /// leaf literals. Interior nodes stay unbound (structural model eval
    /// handles them); clauses are ≤ MAX_K+1 literals, so the wide-clause
    /// machinery never engages.
    fn materialize_mapped(&mut self, root: u32) {
        self.materialize_mapped_multi(&[root]);
    }


    /// Multi-root variant: one cut-mapping plan covering every root's
    /// unmaterialized cone at once. Used per flush batch so sharing
    /// across assertions is visible to the mapper and the per-plan fixed
    /// costs amortize.
    fn materialize_mapped_multi(&mut self, roots: &[u32]) {
        let mut cache = std::mem::take(&mut self.cnfmap_cache);
        let mut plan = std::mem::take(&mut self.cnfmap_plan);
        let is_leaf = |this: &Self, n: u32| {
            this.node_lit(n).is_some() || !matches!(this.aig.node(n), AigNode::And(..))
        };
        match self.cnfmap_effort {
            crate::cnfmap::Effort::Fast => {
                let mut mapper = std::mem::take(&mut self.cnfmap_mapper);
                mapper.plan(&self.aig, roots, |n| is_leaf(self, n), &mut cache, &mut plan);
                self.cnfmap_mapper = mapper;
            }
            crate::cnfmap::Effort::Full => {
                let mut mapper = std::mem::take(&mut self.cnfmap_mapper_full);
                mapper.plan(&self.aig, roots, |n| is_leaf(self, n), &mut cache, &mut plan);
                self.cnfmap_mapper_full = mapper;
            }
        }
        self.cnfmap_cache = cache;
        let mut clause: Vec<Lit> = Vec::with_capacity(crate::cnfmap::MAX_K + 1);
        for e in 0..plan.len() {
            self.emit_plan_node(&plan, plan.entry(e), &mut clause);
        }
        self.cnfmap_plan = plan;
    }

    /// Emit one plan node: a fresh variable defined by the ISOP cubes of
    /// its cut function over the leaf literals, bound via `set_node_lit`.
    fn emit_plan_node(
        &mut self,
        plan: &crate::cnfmap::Plan,
        pn: crate::cnfmap::PlanEntry,
        clause: &mut Vec<Lit>,
    ) {
        // Leaf literals: inputs/constants bind through the classic path
        // (no clauses); And leaves are already bound. Reused scratch —
        // one plan node at a time, so a single buffer suffices.
        let mut leaf_lits = std::mem::take(&mut self.cnfmap_leaf_lits);
        leaf_lits.clear();
        for idx in 0..plan.leaves(pn).len() {
            let l = plan.leaves(pn)[idx];
            let lit = self.lit_of(AigRef::from_parts(l, false));
            leaf_lits.push(lit);
        }
        let ncl = plan.on_cubes(pn).len() + plan.off_cubes(pn).len();
        // A cut root is only worth protecting from BVE when its
        // definition is genuinely wider than a Tseitin gate's; a cut that
        // came out gate-sized resolves just as cheaply as one, so leaving
        // it eliminable keeps BVE productive on the mapped CNF.
        let origin = VarOrigin::GateOut {
            gate: if ncl > CUT_BVE_MAX_CLAUSES {
                GateKind::Cut
            } else {
                GateKind::And
            },
            term: self.aig.src_term(pn.node),
        };
        let t = self.new_sat_lit_tagged(origin);
        // cube → t   becomes   (t ∨ ¬cube);  cube → ¬t   (¬t ∨ ¬cube).
        for (cubes, tl) in [(plan.on_cubes(pn), t), (plan.off_cubes(pn), !t)] {
            for cube in cubes {
                clause.clear();
                clause.push(tl);
                for (i, &ll) in leaf_lits.iter().enumerate() {
                    if cube.pos & (1 << i) != 0 {
                        clause.push(!ll);
                    }
                    if cube.neg & (1 << i) != 0 {
                        clause.push(ll);
                    }
                }
                self.emit_clause_slice(clause);
            }
        }
        self.cnfmap_leaf_lits = leaf_lits;
        self.set_node_lit(pn.node, t);
        self.charge_cost(pn.node, 1, ncl);
    }

    fn lit_of(&mut self, r: AigRef) -> Lit {
        let root_idx = r.node_idx();
        if self.cnf_mapping
            && self.node_lit(root_idx).is_none()
            && matches!(self.aig.node(root_idx), AigNode::And(..))
        {
            self.materialize_mapped(root_idx);
        }
        if self.node_lit(root_idx).is_none() {
            let mut worklist: Vec<u32> = vec![root_idx];
            while let Some(&top) = worklist.last() {
                if self.node_lit(top).is_some() {
                    worklist.pop();
                    continue;
                }
                match self.aig.node(top) {
                    AigNode::ConstTrue => {
                        let tl = self.get_true_lit();
                        self.set_node_lit(top, tl);
                        worklist.pop();
                    }
                    AigNode::Input(lit) => {
                        self.set_node_lit(top, lit);
                        worklist.pop();
                    }
                    AigNode::And(a, b) if a == b => {
                        // FRAIG alias node (`Aig::merge_equiv`): the node is
                        // a proven copy of `a` — bind it to a's lit, emit
                        // nothing. Normal construction can't produce
                        // And(x, x) (the builder folds it), so this shape
                        // is unambiguous.
                        match self.node_lit(a.node_idx()) {
                            Some(base) => {
                                let l = if a.is_negated() { !base } else { base };
                                self.alias_node_lit(top, l);
                                worklist.pop();
                            }
                            None => worklist.push(a.node_idx()),
                        }
                    }
                    AigNode::And(a, b) => {
                        // Plain-And shortcut: if both children already have
                        // lits, the 3-clause encoding is the cheapest form —
                        // don't bother pattern-matching.
                        let a_lit = self.node_lit(a.node_idx());
                        let b_lit = self.node_lit(b.node_idx());
                        if let (Some(al_base), Some(bl_base)) = (a_lit, b_lit) {
                            let al = if a.is_negated() { !al_base } else { al_base };
                            let bl = if b.is_negated() { !bl_base } else { bl_base };
                            self.emit_and_gate(top, al, bl);
                            worklist.pop();
                            continue;
                        }
                        // Pattern-mapped shapes bypass the interior nodes.
                        if let Some(shape) = self.detect_shape(top) {
                            let needed: [Option<AigRef>; 3] = match shape {
                                NodeShape::Xor(x, y) => [Some(x), Some(y), None],
                                NodeShape::NotMux { s, t, e } => {
                                    [Some(s), Some(t), Some(e)]
                                }
                            };
                            let mut missing = false;
                            for opnd in needed.iter().flatten() {
                                if self.node_lit(opnd.node_idx()).is_none() {
                                    worklist.push(opnd.node_idx());
                                    missing = true;
                                }
                            }
                            if missing {
                                continue;
                            }
                            match shape {
                                NodeShape::Xor(x, y) => {
                                    let xl = self.ref_lit(x);
                                    let yl = self.ref_lit(y);
                                    self.emit_xor_gate(top, xl, yl);
                                }
                                NodeShape::NotMux { s, t, e } => {
                                    let sl = self.ref_lit(s);
                                    let tl = self.ref_lit(t);
                                    let el = self.ref_lit(e);
                                    self.emit_mux_gate(top, sl, tl, el);
                                }
                            }
                            worklist.pop();
                            continue;
                        }
                        // Plain And: make sure both children are materialized.
                        let mut missing = false;
                        if a_lit.is_none() {
                            worklist.push(a.node_idx());
                            missing = true;
                        }
                        if b_lit.is_none() {
                            worklist.push(b.node_idx());
                            missing = true;
                        }
                        if missing {
                            continue;
                        }
                        unreachable!("both children materialized — handled above");
                    }
                }
            }
        }
        let base = self.node_lit(root_idx).expect("cone materialized");
        if r.is_negated() { !base } else { base }
    }

    /// `lit_of` that returns `None` if the node hasn't been materialized
    /// yet, instead of doing so. Used for model readback and deferred
    /// metadata resolution where forcing CNF emission would be wrong.
    fn try_lit_of(&self, r: AigRef) -> Option<Lit> {
        self.aig_lit
            .get(r.node_idx() as usize)
            .copied()
            .flatten()
            .map(|base| if r.is_negated() { !base } else { base })
    }

    #[inline]
    fn node_lit(&self, idx: u32) -> Option<Lit> {
        self.aig_lit.get(idx as usize).copied().flatten()
    }

    #[inline]
    fn set_node_lit(&mut self, idx: u32, l: Lit) {
        if self.aig_lit.len() <= idx as usize {
            self.aig_lit.resize(idx as usize + 1, None);
        }
        self.aig_lit[idx as usize] = Some(l);
        self.lit_node[l.var_idx()] = idx;
        // The variable is (re-)gaining defining clauses: make sure it is
        // branchable again — load-bearing only when a cone retired by
        // `retire_dead_cones` (which un-branches its vars) re-materializes
        // over the same input literals. Gated so sessions that never
        // retire skip the per-gate store entirely.
        if self.retirement_used {
            self.sat.set_decision_var(l.var(), true);
        }
        // VE dissolved this node once and now something re-materialized it
        // — the elimination was wasted work (see `elim_nodes`).
        if !self.elim_nodes.is_empty() && self.elim_nodes.remove(&idx) {
            self.pp_remat += 1;
        }
    }

    /// Bind a FRAIG alias node to another node's (possibly negated) lit
    /// WITHOUT claiming `lit_node` — that stays pointing at the node whose
    /// materialization emitted the variable's defining clauses. The side
    /// map lets variable elimination invalidate every alias binding too.
    fn alias_node_lit(&mut self, idx: u32, l: Lit) {
        if self.aig_lit.len() <= idx as usize {
            self.aig_lit.resize(idx as usize + 1, None);
        }
        self.aig_lit[idx as usize] = Some(l);
        self.aig_lit_aliases
            .entry(l.var_idx() as u32)
            .or_default()
            .push(idx);
    }

    /// Signed lookup: the node behind `r` must already have a lit.
    #[inline]
    fn ref_lit(&self, r: AigRef) -> Lit {
        let base = self.node_lit(r.node_idx()).expect("operand materialized");
        if r.is_negated() { !base } else { base }
    }

    /// Classify an And node as one of the compound shapes worth a direct
    /// encoding. Both children must be negated And nodes; then:
    ///   - XOR: the grandchild pairs are elementwise complementary —
    ///     `n = ¬(x∧y) ∧ ¬(¬x∧¬y) ≡ x ⊕ y`.
    ///   - MUX: exactly one grandchild pair is complementary (the selector)
    ///     — `n = ¬(s∧t) ∧ ¬(¬s∧e) ≡ ¬mux(s, t, e)`.
    fn detect_shape(&self, idx: u32) -> Option<NodeShape> {
        let AigNode::And(a, b) = self.aig.node(idx) else {
            return None;
        };
        if !a.is_negated() || !b.is_negated() {
            return None;
        }
        let AigNode::And(x0, x1) = self.aig.node(a.node_idx()) else {
            return None;
        };
        let AigNode::And(y0, y1) = self.aig.node(b.node_idx()) else {
            return None;
        };
        // XOR first — it's the fully-complementary special case of MUX and
        // we want it tagged/encoded as XOR.
        if (y0 == !x0 && y1 == !x1) || (y0 == !x1 && y1 == !x0) {
            return Some(NodeShape::Xor(x0, x1));
        }
        if y0 == !x0 {
            return Some(NodeShape::NotMux { s: x0, t: x1, e: y1 });
        }
        if y1 == !x0 {
            return Some(NodeShape::NotMux { s: x0, t: x1, e: y0 });
        }
        if y0 == !x1 {
            return Some(NodeShape::NotMux { s: x1, t: x0, e: y1 });
        }
        if y1 == !x1 {
            return Some(NodeShape::NotMux { s: x1, t: x0, e: y0 });
        }
        None
    }

    /// Emit `o ↔ (al ∧ bl)` for node `idx` (3 clauses, 1 var).
    fn emit_and_gate(&mut self, idx: u32, al: Lit, bl: Lit) {
        self.stats_and_gates += 1;
        let origin = VarOrigin::GateOut {
            gate: GateKind::And,
            term: self.aig.src_term(idx),
        };
        let o = self.new_sat_lit_tagged(origin);
        self.emit_clause_slice(&[!al, !bl, o]);
        self.emit_clause_slice(&[al, !o]);
        self.emit_clause_slice(&[bl, !o]);
        self.set_node_lit(idx, o);
        self.charge_cost(idx, 1, 3);
    }

    /// Emit `o ↔ (xl ⊕ yl)` for node `idx` (4 clauses, 1 var). The node
    /// itself IS the xor of the operands (see `detect_shape`).
    fn emit_xor_gate(&mut self, idx: u32, xl: Lit, yl: Lit) {
        self.stats_xor_gates += 1;
        let origin = VarOrigin::GateOut {
            gate: GateKind::Xor,
            term: self.aig.src_term(idx),
        };
        let o = self.new_sat_lit_tagged(origin);
        self.emit_clause_slice(&[!xl, !yl, !o]);
        self.emit_clause_slice(&[xl, yl, !o]);
        self.emit_clause_slice(&[xl, !yl, o]);
        self.emit_clause_slice(&[!xl, yl, o]);
        self.set_node_lit(idx, o);
        self.charge_cost(idx, 1, 4);
    }

    /// Emit a mux gate for node `idx`, which satisfies `idx ≡ ¬mux(s,t,e)`.
    /// The fresh var `o` encodes the mux VALUE; the node's stored lit is
    /// therefore `¬o`. (4 clauses, 1 var.)
    fn emit_mux_gate(&mut self, idx: u32, sl: Lit, tl: Lit, el: Lit) {
        self.stats_mux_gates += 1;
        let origin = VarOrigin::GateOut {
            gate: GateKind::Ite,
            term: self.aig.src_term(idx),
        };
        let o = self.new_sat_lit_tagged(origin);
        self.emit_clause_slice(&[!sl, !tl, o]);
        self.emit_clause_slice(&[!sl, tl, !o]);
        self.emit_clause_slice(&[sl, !el, o]);
        self.emit_clause_slice(&[sl, el, !o]);
        self.set_node_lit(idx, !o);
        self.charge_cost(idx, 1, 4);
    }

    /// Charge emitted CNF to the BV term tagged on the AIG node, if cost
    /// tracking is on.
    #[inline]
    fn charge_cost(&mut self, idx: u32, vars: usize, clauses: usize) {
        if !self.bitblast_cost_enabled {
            return;
        }
        if let Some(term) = self.aig.src_term(idx) {
            let entry = self.bitblast_cost.entry(term).or_insert((0, 0));
            entry.0 += vars;
            entry.1 += clauses;
        }
    }

    // ---------- Low-level helpers ----------

    /// Allocate a SAT literal tagged with the given metadata. The id may
    /// be recycled from a retired cone (see `Solver::purge_vars`) — then
    /// the per-var metadata is overwritten in place, and the id's slots
    /// in the banked model are poisoned: they hold values from the
    /// variable's previous life, and warm screening must recompute this
    /// node structurally instead of trusting them.
    fn new_sat_lit_tagged(&mut self, origin: VarOrigin) -> Lit {
        let v = self.sat.new_var();
        let vi = v.idx();
        if vi < self.var_origin.len() {
            self.var_origin[vi] = origin;
            self.lit_node[vi] = u32::MAX;
            let li = vi << 1;
            if li + 1 < self.banked_model.len() {
                self.banked_model[li] = LBool::Undef;
                self.banked_model[li + 1] = LBool::Undef;
            }
        } else {
            // Fresh id: keep var_origin aligned 1-to-1 with SAT variables.
            debug_assert_eq!(self.var_origin.len(), vi);
            self.var_origin.push(origin);
            self.lit_node.push(u32::MAX);
        }
        Lit::new(v, false)
    }

    /// Route a clause to the active sink: the flush buffer when
    /// preprocessing is collecting a batch, the SAT core otherwise.
    #[inline]
    fn emit_clause(&mut self, c: Vec<Lit>) {
        match self.cnf_buffer.as_mut() {
            Some(buf) => buf.push_slice(&c),
            None => {
                self.sat.add_clause(c);
            }
        }
    }

    /// [`emit_clause`] from a stack slice: no allocation in either mode —
    /// direct mode routes through the solver's reused buffer, buffered
    /// mode appends to the flat CSR arena.
    #[inline]
    fn emit_clause_slice(&mut self, c: &[Lit]) {
        match self.cnf_buffer.as_mut() {
            Some(buf) => buf.push_slice(c),
            None => {
                self.sat.add_clause_from_slice(c);
            }
        }
    }

    fn get_or_make_bv_var(&mut self, id: u32, width: u32) -> Vec<AigRef> {
        // Route through the union-find root: aliased vars share SAT literals
        // (and therefore AIG input nodes).
        let id = self.find_bv_var_root(id);
        if let Some(cached) = self.bv_var_refs.get(&id) {
            return cached.clone();
        }
        // The BvTerm handle tags each bit's metadata — O(1) reverse map
        // (this was a linear scan of every term ever built, which made
        // fresh-variable creation quadratic over a long session).
        let term = self.ctx.var_term(id);
        let refs: Vec<AigRef> = (0..width)
            .map(|bit| {
                let origin = match term {
                    Some(t) => VarOrigin::BvBit { term: t, bit },
                    None => VarOrigin::Unknown,
                };
                let l = self.new_sat_lit_tagged(origin);
                let input = self.aig.input(l);
                self.set_node_lit(input.node_idx(), l);
                input
            })
            .collect();
        self.bv_var_refs.insert(id, refs.clone());
        refs
    }

    /// A literal pinned to true. Allocated once on first use, backed by a
    /// unit clause. Only needed when a constant AIG ref must appear in an
    /// emitted clause (constant roots / constant bits of a direct-encoded
    /// equality); all other constant handling folds inside the AIG.
    fn get_true_lit(&mut self) -> Lit {
        if let Some(l) = self.true_lit {
            return l;
        }
        let l = self.new_sat_lit_tagged(VarOrigin::TrueLit);
        self.sat.add_clause(vec![l]);
        self.true_lit = Some(l);
        // The ConstTrue node is index 0; pin its lit so `lit_of` on
        // constants resolves without re-entering this path.
        self.set_node_lit(0, l);
        l
    }

    fn zipwith<F>(&mut self, a: &[AigRef], b: &[AigRef], mut f: F) -> Vec<AigRef>
    where
        F: FnMut(&mut Self, AigRef, AigRef) -> AigRef,
    {
        assert_eq!(a.len(), b.len());
        (0..a.len()).map(|i| f(self, a[i], b[i])).collect()
    }

    /// AND gate — delegates to the AIG (construction-time folds + hash-
    /// cons). No CNF is emitted here. Tags the node with the enclosing BV
    /// term for metadata / cost attribution.
    fn mk_and(&mut self, a: AigRef, b: AigRef) -> AigRef {
        let r = self.aig.and(a, b);
        if let Some(term) = self.current_bv_ctx {
            self.aig.tag_src(r, term);
        }
        r
    }

    /// OR gate — `¬and(¬a, ¬b)`.
    fn mk_or(&mut self, a: AigRef, b: AigRef) -> AigRef {
        let r = self.aig.or(a, b);
        if let Some(term) = self.current_bv_ctx {
            self.aig.tag_src(r, term);
        }
        r
    }

    /// XOR gate — the 3-node AIG shape; `lit_of` recognizes it at emission
    /// time and produces the direct 4-clause encoding.
    fn mk_xor(&mut self, a: AigRef, b: AigRef) -> AigRef {
        let r = self.aig.xor(a, b);
        if let Some(term) = self.current_bv_ctx {
            self.aig.tag_src(r, term);
        }
        r
    }

    /// 2:1 MUX. Structural folds happen at the AIG level; a genuine mux
    /// gets queued for ITE metadata + the selector VSIDS boost, resolved at
    /// flush for gates that actually reach an asserted root.
    fn mk_mux(&mut self, sel: AigRef, t: AigRef, e: AigRef) -> AigRef {
        // Replicate the AIG's folds up front so we can tell "real mux
        // structure" apart from a degenerate case — only the former gets
        // ITE-gate metadata and branching hints.
        if t == e {
            return t;
        }
        if sel == AigRef::TRUE {
            return t;
        }
        if sel == AigRef::FALSE {
            return e;
        }
        if t == AigRef::TRUE && e == AigRef::FALSE {
            return sel;
        }
        if t == AigRef::FALSE && e == AigRef::TRUE {
            return !sel;
        }
        if t == sel {
            return self.mk_or(sel, e);
        }
        if t == !sel {
            return self.mk_and(!sel, e);
        }
        if e == sel {
            return self.mk_and(sel, t);
        }
        if e == !sel {
            return self.mk_or(!sel, t);
        }
        let out = self.aig.mux(sel, t, e);
        if let Some(term) = self.current_bv_ctx {
            self.aig.tag_src(out, term);
        }
        self.pending_ite_gates.push(PendingIte {
            sel,
            t,
            e,
            out,
            src: self.current_bv_ctx,
        });
        out
    }

    /// Bit-parallel MUX: pick `t[i]` when `sel` is true, else `e[i]`.
    fn mux_vec(&mut self, sel: AigRef, t: &[AigRef], e: &[AigRef]) -> Vec<AigRef> {
        assert_eq!(t.len(), e.len());
        (0..t.len()).map(|i| self.mk_mux(sel, t[i], e[i])).collect()
    }

    /// Full adder for one bit. Returns (sum, cout).
    ///   sum = a ⊕ b ⊕ cin
    ///   cout = majority(a, b, cin)
    fn mk_full_adder(&mut self, a: AigRef, b: AigRef, cin: AigRef) -> (AigRef, AigRef) {
        let a_xor_b = self.mk_xor(a, b);
        let sum = self.mk_xor(a_xor_b, cin);
        let a_and_b = self.mk_and(a, b);
        let cin_and_xor = self.mk_and(cin, a_xor_b);
        let cout = self.mk_or(a_and_b, cin_and_xor);
        (sum, cout)
    }

    /// Wallace-tree population count. Takes `W` 1-bit inputs and produces
    /// the popcount in `out_width` bits, zero-extended.
    ///
    /// Each input contributes to "column 0" (weight 2⁰). Repeated 3:2
    /// compression — three bits in a column → one sum bit (same column) +
    /// one carry bit (next column) via a full adder — drives every column
    /// down to ≤2 rows. A final ripple-carry add combines the two rows.
    fn mk_popcount(&mut self, inputs: &[AigRef], out_width: usize) -> Vec<AigRef> {
        if inputs.is_empty() {
            return vec![AigRef::FALSE; out_width];
        }
        if inputs.len() == 1 {
            // 1-bit popcount is the bit itself, zero-extended.
            let mut out = vec![AigRef::FALSE; out_width];
            out[0] = inputs[0];
            return out;
        }
        // columns[k] holds the bits at column k (weight 2^k). Each
        // compression round walks all columns, pulls groups of 3, and
        // pushes (sum, carry) into the next round's columns. Leftover bits
        // (0 to 2 per column per round) pass through. Loop ends when every
        // column has ≤ 2 rows.
        let mut columns: Vec<Vec<AigRef>> = vec![inputs.to_vec()];
        loop {
            let mut any_above_two = false;
            // Allocate one extra slot at the top for carries out of the
            // highest current column.
            let mut next: Vec<Vec<AigRef>> = vec![Vec::new(); columns.len() + 1];
            for k in 0..columns.len() {
                let col = std::mem::take(&mut columns[k]);
                let mut i = 0;
                while i + 3 <= col.len() {
                    let (sum, cout) =
                        self.mk_full_adder(col[i], col[i + 1], col[i + 2]);
                    next[k].push(sum);
                    next[k + 1].push(cout);
                    i += 3;
                }
                while i < col.len() {
                    next[k].push(col[i]);
                    i += 1;
                }
                if next[k].len() > 2 {
                    any_above_two = true;
                }
            }
            // Trim trailing empty columns so the loop bound stays tight.
            while next.last().map(|c| c.is_empty()).unwrap_or(false) {
                next.pop();
            }
            columns = next;
            if !any_above_two {
                break;
            }
        }
        // Pad each column out to two rows (with constant zeros where
        // missing) and ripple-carry-add.
        let n_cols = columns.len();
        let mut row1: Vec<AigRef> = Vec::with_capacity(n_cols);
        let mut row2: Vec<AigRef> = Vec::with_capacity(n_cols);
        for k in 0..n_cols {
            match columns[k].len() {
                0 => {
                    row1.push(AigRef::FALSE);
                    row2.push(AigRef::FALSE);
                }
                1 => {
                    row1.push(columns[k][0]);
                    row2.push(AigRef::FALSE);
                }
                2 => {
                    row1.push(columns[k][0]);
                    row2.push(columns[k][1]);
                }
                _ => unreachable!("post-compression column has > 2 bits"),
            }
        }
        let (sum, cout) = self.ripple_carry_add(&row1, &row2, AigRef::FALSE);
        // Combine sum + overflow into one output vector, then fit to out_width.
        let mut result = sum;
        result.push(cout);
        if result.len() < out_width {
            result.resize(out_width, AigRef::FALSE);
        } else if result.len() > out_width {
            result.truncate(out_width);
        }
        result
    }

    /// CLZ via `popcount(~(x | x>>1 | x>>2 | ... | x>>(w/2)))` at the AIG
    /// level. The shifts are structural (just slot-shift the ref vector
    /// with FALSE filling the gaps), so the only real gates come from the
    /// OR-fold and the final popcount Wallace tree.
    fn mk_clz(&mut self, inputs: &[AigRef], out_width: usize) -> Vec<AigRef> {
        let w = inputs.len();
        if w == 0 {
            return vec![AigRef::FALSE; out_width];
        }
        if w == 1 {
            // clz of a 1-bit value is just !x.
            let mut out = vec![AigRef::FALSE; out_width];
            out[0] = !inputs[0];
            return out;
        }
        // OR-fold: every bit at-or-below the highest set bit becomes 1.
        // Shifts are by 1, 2, 4, ... up to < w. Bit i of `y >>L k` is
        // y[i+k] for i+k < w, else false.
        let mut y: Vec<AigRef> = inputs.to_vec();
        let mut k = 1usize;
        while k < w {
            let shifted: Vec<AigRef> = (0..w)
                .map(|i| if i + k < w { y[i + k] } else { AigRef::FALSE })
                .collect();
            for i in 0..w {
                y[i] = self.mk_or(y[i], shifted[i]);
            }
            k <<= 1;
        }
        let ny: Vec<AigRef> = y.iter().map(|&r| !r).collect();
        self.mk_popcount(&ny, out_width)
    }

    /// CTZ via `popcount(~x & (x - 1))` at the AIG level. The mask
    /// `~x & (x - 1)` has exactly one bit set for each trailing zero of `x`
    /// (and is all-ones when `x == 0`, giving the SMT-LIB convention of
    /// `ctz(0) = width`).
    fn mk_ctz(&mut self, inputs: &[AigRef], out_width: usize) -> Vec<AigRef> {
        let w = inputs.len();
        if w == 0 {
            return vec![AigRef::FALSE; out_width];
        }
        if w == 1 {
            let mut out = vec![AigRef::FALSE; out_width];
            out[0] = !inputs[0];
            return out;
        }
        // x - 1 = x + all-ones (mod 2^w) with no carry-in.
        let all_ones: Vec<AigRef> = vec![AigRef::TRUE; w];
        let (xm1, _cout) = self.ripple_carry_add(inputs, &all_ones, AigRef::FALSE);
        // m = ~x & (x - 1)
        let m: Vec<AigRef> = (0..w)
            .map(|i| self.mk_and(!inputs[i], xm1[i]))
            .collect();
        self.mk_popcount(&m, out_width)
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

    fn ripple_carry_add(
        &mut self,
        a: &[AigRef],
        b: &[AigRef],
        cin: AigRef,
    ) -> (Vec<AigRef>, AigRef) {
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

    fn mk_bitwise_eq(&mut self, a: &[AigRef], b: &[AigRef]) -> AigRef {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            return AigRef::TRUE;
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
    fn mk_ult(&mut self, a: &[AigRef], b: &[AigRef]) -> AigRef {
        assert_eq!(a.len(), b.len());
        let b_neg: Vec<AigRef> = b.iter().map(|&r| !r).collect();
        let (_sum, cout) = self.ripple_carry_add(a, &b_neg, AigRef::TRUE);
        !cout
    }

    /// AND-reduction: returns 1 iff all bits are zero.
    fn mk_all_zero(&mut self, bits: &[AigRef]) -> AigRef {
        assert!(!bits.is_empty());
        let mut z = !bits[0];
        for i in 1..bits.len() {
            z = self.mk_and(z, !bits[i]);
        }
        z
    }

    /// OR-reduction: returns 1 iff any bit is set.
    fn mk_any_set(&mut self, bits: &[AigRef]) -> AigRef {
        assert!(!bits.is_empty());
        let mut any = bits[0];
        for i in 1..bits.len() {
            any = self.mk_or(any, bits[i]);
        }
        any
    }

    /// If `t` is a BV constant, return its raw value (already masked
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
    /// through only the non-zero NAF digits of `c`: for a common case like
    /// `x * 3` on 64-bit BVs, we emit 2 ripple-carry adds instead of 64 —
    /// and each bit-AND collapses via the AIG constant folds.
    fn mk_mul_const(&mut self, a: &[AigRef], c: u128, n: usize) -> Vec<AigRef> {
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
        let mut result: Vec<AigRef> = vec![AigRef::FALSE; n];
        for (sign, pos) in digits {
            let pos = pos as usize;
            // Build `a << pos`, truncated to n bits.
            let shifted: Vec<AigRef> = (0..n)
                .map(|j| if j < pos { AigRef::FALSE } else { a[j - pos] })
                .collect();
            if sign > 0 {
                let (new_result, _) =
                    self.ripple_carry_add(&result, &shifted, AigRef::FALSE);
                result = new_result;
            } else {
                // `result - shifted = result + (¬shifted) + 1`. The bit
                // inversions are polarity flips on the refs — no gates.
                let neg_shifted: Vec<AigRef> = shifted.iter().map(|&r| !r).collect();
                let (new_result, _) =
                    self.ripple_carry_add(&result, &neg_shifted, AigRef::TRUE);
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
    fn mk_shl_const(&mut self, a: &[AigRef], amt: usize) -> Vec<AigRef> {
        let n = a.len();
        let amt = amt.min(n); // ≥width clears the vector
        (0..n)
            .map(|i| if i < amt { AigRef::FALSE } else { a[i - amt] })
            .collect()
    }

    /// Constant-amount right shift with explicit fill (zero for lshr, sign
    /// bit for ashr).
    fn mk_shr_const(&mut self, a: &[AigRef], amt: usize, fill: AigRef) -> Vec<AigRef> {
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
    fn mk_shl(&mut self, a: &[AigRef], amt: &[AigRef]) -> Vec<AigRef> {
        let n = a.len();
        assert_eq!(amt.len(), n);
        let log_n = ceil_log2(n);

        let mut cur = a.to_vec();
        for i in 0..log_n {
            let shift = 1usize << i;
            if shift >= n { break; }
            let shifted: Vec<AigRef> = (0..n)
                .map(|j| if j < shift { AigRef::FALSE } else { cur[j - shift] })
                .collect();
            cur = self.mux_vec(amt[i], &shifted, &cur);
        }

        // Overflow: if any of amt[log_n..n] is set the shift ≥ n, so clear.
        self.maybe_fill_on_overflow(&cur, amt, log_n, AigRef::FALSE)
    }

    /// Right shift (logical or arithmetic) with variable amount. The
    /// `fill` ref determines what streams in from the top.
    fn mk_shr(&mut self, a: &[AigRef], amt: &[AigRef], fill: AigRef) -> Vec<AigRef> {
        let n = a.len();
        assert_eq!(amt.len(), n);
        let log_n = ceil_log2(n);

        let mut cur = a.to_vec();
        for i in 0..log_n {
            let shift = 1usize << i;
            if shift >= n { break; }
            let shifted: Vec<AigRef> = (0..n)
                .map(|j| if j + shift < n { cur[j + shift] } else { fill })
                .collect();
            cur = self.mux_vec(amt[i], &shifted, &cur);
        }

        // Overflow: amt >= n. Replace all bits with `fill` (0 for lshr,
        // sign for ashr).
        self.maybe_fill_on_overflow(&cur, amt, log_n, fill)
    }

    /// After the main barrel stages, if any high bit of `amt` is set the
    /// requested shift was >= width — replace the result with `fill`.
    fn maybe_fill_on_overflow(
        &mut self,
        cur: &[AigRef],
        amt: &[AigRef],
        log_n: usize,
        fill: AigRef,
    ) -> Vec<AigRef> {
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
    fn mk_mul(&mut self, a: &[AigRef], b: &[AigRef]) -> Vec<AigRef> {
        let n = a.len();
        assert_eq!(b.len(), n);

        // Step 1: partial products collected by output column. Skip any
        // product whose operand is constant-false — these are the bits
        // that zero-extensions, masked-away positions, and bits-known folds
        // reduce to at bitblast time. Pushing FALSE would correctly
        // short-circuit through `mk_and` but needlessly inflates column
        // lengths, causing extra 3:2 compressions in the Wallace reduction
        // below. Skipping at source keeps columns as tight as they can be.
        let mut columns: Vec<Vec<AigRef>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..n {
            if b[i] == AigRef::FALSE {
                continue; // entire "row" shifted by i contributes nothing
            }
            for j in i..n {
                let ajm = a[j - i];
                if ajm == AigRef::FALSE {
                    continue; // this single partial product is zero
                }
                let pp = self.mk_and(ajm, b[i]);
                if pp != AigRef::FALSE {
                    columns[j].push(pp);
                }
            }
        }

        // Step 2: reduce to ≤ 2 bits per column.
        loop {
            let max_len = columns.iter().map(|c| c.len()).max().unwrap_or(0);
            if max_len <= 2 {
                break;
            }
            let mut next: Vec<Vec<AigRef>> = (0..n).map(|_| Vec::new()).collect();
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
        let row0: Vec<AigRef> = columns
            .iter()
            .map(|c| if c.is_empty() { AigRef::FALSE } else { c[0] })
            .collect();
        let row1: Vec<AigRef> = columns
            .iter()
            .map(|c| if c.len() < 2 { AigRef::FALSE } else { c[1] })
            .collect();
        self.ripple_carry_add(&row0, &row1, AigRef::FALSE).0
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
    fn mk_udivmod(&mut self, a: &[AigRef], b: &[AigRef]) -> (Vec<AigRef>, Vec<AigRef>) {
        let n = a.len();
        assert_eq!(b.len(), n);

        // N+2 bits: one sign bit plus one slack bit for the `2 * r` step.
        let ext = n + 2;
        let mut r: Vec<AigRef> = vec![AigRef::FALSE; ext];
        let mut b_ext: Vec<AigRef> = b.to_vec();
        b_ext.push(AigRef::FALSE); // sign bit = 0 (b is always non-negative)
        b_ext.push(AigRef::FALSE); // slack bit

        let mut q: Vec<AigRef> = vec![AigRef::FALSE; n];

        for i in (0..n).rev() {
            // r := (r << 1) | a[i]  — shift up by one, introduce next
            // bit of the dividend at the LSB. Width stays N+2 (top bit
            // falls off, but since we picked ext = n + 2, the worst-case
            // |2*r| ≤ 2*b < 2^(n+1) still fits as signed).
            let mut shifted = vec![AigRef::FALSE; ext];
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
            let effective_b: Vec<AigRef> =
                b_ext.iter().map(|&bb| self.mk_xor(bb, not_sign)).collect();
            let (new_r, _cout) = self.ripple_carry_add(&r, &effective_b, not_sign);
            r = new_r;

            // Quotient bit = 1 iff the new remainder is non-negative.
            q[i] = !r[ext - 1];
        }

        // Final restoration: if r went negative, add b back once.
        let final_sign = r[ext - 1];
        let (restored, _cout) = self.ripple_carry_add(&r, &b_ext, AigRef::FALSE);
        let r_final = self.mux_vec(final_sign, &restored, &r);

        // Truncate remainder back to N bits (high bits are zero after
        // the restoration step).
        (q, r_final[..n].to_vec())
    }

    /// High N bits of the unsigned 2N-bit product of two N-bit operands.
    /// Used by unsigned-multiplication overflow detection.
    fn mk_umul_hi(&mut self, a: &[AigRef], b: &[AigRef]) -> Vec<AigRef> {
        let n = a.len();
        assert_eq!(b.len(), n);
        let double_n = 2 * n;

        // Zero-extend both operands to 2N bits and multiply via the same
        // Wallace tree used for regular multiplication. Keep the top N bits.
        let mut a_ext = a.to_vec();
        a_ext.resize(double_n, AigRef::FALSE);
        let mut b_ext = b.to_vec();
        b_ext.resize(double_n, AigRef::FALSE);

        let prod = self.mk_mul(&a_ext, &b_ext);
        prod[n..].to_vec()
    }

    /// Full 2N-bit signed product: (low N bits, high N bits). Sign-extends
    /// both operands to 2N bits then multiplies. Used by signed-multiplication
    /// overflow detection.
    fn mk_smul_full(&mut self, a: &[AigRef], b: &[AigRef]) -> (Vec<AigRef>, Vec<AigRef>) {
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
    fn mk_neg(&mut self, x: &[AigRef]) -> Vec<AigRef> {
        let neg: Vec<AigRef> = x.iter().map(|&r| !r).collect();
        let zero: Vec<AigRef> = vec![AigRef::FALSE; x.len()];
        self.ripple_carry_add(&zero, &neg, AigRef::TRUE).0
    }

    /// Absolute value of a signed BV: returns `-x` if x's MSB is set, else x.
    fn mk_abs(&mut self, x: &[AigRef]) -> Vec<AigRef> {
        let sign = *x.last().unwrap();
        let neg = self.mk_neg(x);
        self.mux_vec(sign, &neg, x)
    }

    /// Signed division with SMT-LIB semantics. Computes absolute values,
    /// does an unsigned divide, then flips the sign of the result when
    /// exactly one operand was negative. Division-by-zero follows from the
    /// underlying udiv-by-zero (all-ones) case-split.
    fn mk_sdiv(&mut self, a: &[AigRef], b: &[AigRef]) -> Vec<AigRef> {
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
        let all_ones = vec![AigRef::TRUE; n];
        // Constant 1 of width n.
        let mut one = vec![AigRef::FALSE; n];
        one[0] = AigRef::TRUE;
        let dz = self.mux_vec(a_sign, &one, &all_ones);

        self.mux_vec(b_zero, &dz, &q)
    }

    /// Signed remainder — sign of result follows the dividend.
    /// Division-by-zero: srem(x, 0) = x (following SMT-LIB).
    fn mk_srem(&mut self, a: &[AigRef], b: &[AigRef]) -> Vec<AigRef> {
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
    fn mk_smod(&mut self, a: &[AigRef], b: &[AigRef]) -> Vec<AigRef> {
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
        let r_plus_b = self.ripple_carry_add(&r_srem, b, AigRef::FALSE).0;
        let sign_diff = self.mk_xor(a_sign, b_sign);
        let adjusted = self.mux_vec(sign_diff, &r_plus_b, &r_srem);

        // If the raw remainder was zero, the answer is zero.
        let zero = vec![AigRef::FALSE; n];
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

/// FRAIG sweep budget with an env override — lets budget experiments run
/// (`FRAIG_MAX_QUERIES`, `FRAIG_MAX_CONFLICTS`). Cold
/// path: read once per flush.
/// Per-sweep FRAIG budgets: SAT queries, and conflicts per query.
/// Constants rather than tunables — FRAIG is off by default (measured
/// strongly net-negative; see the module docs) and these values are the
/// ones its evaluation used.
const FRAIG_MAX_QUERIES: u64 = 20_000;
const FRAIG_MAX_CONFLICTS: u64 = 100;

/// Flip the sign bit (MSB) of a bitblasted BV — used for signed comparisons.
fn flip_msb(bits: &[AigRef]) -> Vec<AigRef> {
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
