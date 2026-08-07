//! Post-preprocess propagation augmentation.
//!
//! Gate-by-gate Tseitin encoding is arc-consistent per gate but not
//! generalized-arc-consistent for the *composition*: a multi-gate cone
//! can semantically force a literal that unit propagation never derives
//! because the responsible prime implicate spans several gates (the
//! classic case is the adder carry — `a ∧ cin ⇒ cout` holds for
//! `MAJ(a,b,cin)` but no single gate clause fires). Cut-based CNF
//! mapping fixes this by re-encoding, but re-encoding starves the
//! gate-driven bounded VE that binbit's pipeline leans on (measured
//! 2026-08-06: `--cnfmap-full` collapses `pp_elim` 164k → 27k on
//! nobranch and regresses the symbex corpus badly).
//!
//! This pass takes the third road: leave the encoding AND the
//! preprocessor untouched, and afterwards ADD the missing prime
//! implicates as redundant clauses. For every var-bearing gate of the
//! freshly-committed batch it enumerates 4-feasible cuts over the
//! *derived gate graph* (shape-mapped XOR/MUX nodes are single gates,
//! exactly mirroring emission), classifies each cut function through
//! the [`crate::npn4`] tables, and consults the class's
//! propagation-complete cover — all prime implicates of `y ↔ f(cut)`,
//! which the npn4 generator proves is the unique minimal
//! propagation-complete definition. Each cover clause is kept only if
//! the emitted gate encoding of the cut's interior fails to make its
//! propagation (a UP-redundancy check per clause), so single-gate AND /
//! XOR cuts contribute nothing and arithmetic consensus clauses fall
//! out exactly where the encoding is blind.
//!
//! Soundness: an added clause mentions only surviving variables (cut
//! leaves and the root), and is implied by the original gate
//! definitions. Bounded VE, subsumption and strengthening all preserve
//! logical equivalence over surviving variables, so the clause is
//! implied by the post-preprocess formula too — adding it changes no
//! model and is safe mid-incremental-session.
//!
//! Conservativeness of the filter: the check runs against the
//! *reconstructed* gate encoding of the cut interior. If an interior
//! variable was eliminated, its resolvents connect the same
//! implications in fewer steps, so anything the reconstructed chain
//! derives the live formula derives too — the filter can only
//! under-drop (emit a clause that is already derivable), never
//! over-drop.
//!
//! Two delivery modes, both off by default:
//!
//! **Eager** (`SmtSolver::set_pcaug`) adds every derived clause to the
//! formula at flush time. **Measured negative** (2026-08-06): on
//! nobranch it adds 131k clauses, leaves `pp_elim` untouched at 164,533
//! exactly as designed, yet conflicts go 628k → 644k and propagations
//! +21%; on the corpus bench_13728 −57% but bench_7373 +676% and
//! bench_6554 +347%. Depth-gating (`BINBIT_PCAUG_MIN_GATES` 1→3) barely
//! changes the added set — the holes live in deep compositions — so the
//! losses persist at every depth. The reading: when the interior
//! variables EXIST, CDCL derives these consensus resolvents on demand
//! and keeps only the useful ones; pre-adding all of them taxes every
//! `propagate()` forever. (The same covers DO pay inside `cnfmap` prime
//! emission, where interiors have no variables and the holes are not
//! self-healable.)
//!
//! **On demand** (`SmtSolver::set_pcaug_lazy`) hands the same clauses to
//! the SAT core's reserve instead
//! ([`crate::solver::Solver::bank_implied_clause`]), which injects one
//! only when every variable it names is currently hot in VSIDS terms —
//! i.e. the search is demonstrably working inside that region. Banking
//! costs a few literals of arena and nothing in `propagate`.
//!
//! On demand is **much** better than eager — it removes the catastrophes
//! outright (bench_6554 +347% → −22%, bench_7373 +676% → unchanged). With
//! a **bounded working set** on top (below), it is no longer negative
//! anywhere measured:
//!
//! - Corpus: wall **+0%**, conflicts net-positive — bench_5906 −31%,
//!   bench_6554 −22%, bench_4351 −6%, against one loss, bench_5965 +19%.
//! - nobranch: **neutral**. Unseeded 628,280 → 605,452; over three
//!   seeds, baseline mean 585,996 vs 587,533. (Before the working set it
//!   was +13.6% over six seeds — a real regression, not a lottery.)
//!
//! Still OFF by default: wall-neutral with one standing regression is
//! not enough evidence to change a default in a pipeline this tuned.
//! What it now is, is a lever that costs nothing to carry.
//!
//! ## The bounded working set
//!
//! Injection alone is monotonic, so over a long run every region looks
//! hot at some sweep and the whole bank drains in — nobranch injected
//! 100% of its 131k clauses, degenerating into eager-with-delay. The fix
//! is a capacity on live augmentation clauses
//! (`Solver::set_augmentation_capacity`, default 4,000) with eviction by
//! clause activity, which conflict analysis already maintains.
//!
//! Two eviction details that each cost a measurement to learn:
//!
//! 1. **Evict only under capacity pressure, never on a fixed activity
//!    threshold.** Thresholding discarded clauses while the set was
//!    nearly empty: bench_6554 injected 162 and evicted 155, thrashing to
//!    a WORSE result (27,110 conflicts) than never evicting at all
//!    (23,693).
//! 2. **Eviction is terminal.** Returning the slot to the reserve reads
//!    as obviously right — the region may heat up again — but the
//!    variables that made the clause hot still are, so it is re-injected
//!    next sweep and evicted again. nobranch: 151,171 evictions against a
//!    ceiling of 4,000 (~38× churn) and 781k conflicts. Discarding
//!    instead gives 605k. `BINBIT_AUG_RECYCLE=1` restores the cycling
//!    behaviour for anyone who wants to re-test it.
//!
//! ## Cost
//!
//! Derivation was originally the dominant cost — 2.1s on bench_5906,
//! 3.7s on nobranch — which made the pass a net tax even where its
//! clauses helped. It is now ~30× cheaper (nobranch 131,368 clauses in
//! 0.116s vs 3.688s; bench_64 31×), which is what took the corpus from
//! +8% wall to +0%. The four things that mattered, in order:
//!
//! 1. **The shape cache.** The UP redundancy filter is the innermost
//!    loop, and arithmetic circuits present the same local wiring
//!    thousands of times over (every adder bit is the same MAJ/XOR3), so
//!    caching its verdict turns most cuts into a hash lookup. Worth ~9×
//!    on its own. Keys are compared EXACTLY, and cover both the local
//!    encoding and the NPN transform — see `shape_find`.
//! 2. **No per-gate allocation.** Cut candidates and the three operand
//!    choice lists were heap `Vec`s built per gate — roughly a million
//!    allocations on a large instance. They are now reused buffers and
//!    fixed-size arrays.
//! 3. **Dense `gate_of`.** Node→gate was a hash map probed several times
//!    per gate plus once per interior-walk step; node ids are already
//!    dense, so it is now a flat array reset in O(gates).
//! 4. **Bitmask unit propagation.** `up` was a per-literal loop over a
//!    `[u8; 16]`; it is now a few word ops per clause against two
//!    bitmasks.
//!
//! `BINBIT_PCAUG_NOCACHE=1` forces every cache lookup to miss, so the
//! cache's output-identity is checkable on any real instance — and it
//! is worth checking, because keying the cache on the local encoding
//! alone (the obvious choice; the encoding does determine the cut
//! function) silently changed the emitted set on two corpus instances.

use crate::npn4;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

/// A gate operand: (AIG node id, edge negated).
pub type Op = (u32, bool);

/// One derived gate — the value of the node its SAT variable binds.
#[derive(Clone, Copy)]
pub enum Gate {
    And(Op, Op),
    Xor(Op, Op),
    /// Node VALUE is `¬mux(s, t, e)` (matches `detect_shape` /
    /// `emit_mux_gate`; the sign is baked into the node's stored lit,
    /// which is what the caller resolves).
    NotMux(Op, Op, Op),
}

#[derive(Default, Clone, Copy, Debug)]
pub struct AugStats {
    /// Gates examined as cut roots.
    pub roots: u32,
    /// Multi-gate cuts that went through the redundancy filter.
    pub cuts: u32,
    /// Clauses emitted (post-filter, post-dedup).
    pub added: u32,
}

/// Truth tables of the four leaf positions over a 16-row table.
const VAR4: [u16; 4] = [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00];

/// Cuts kept per gate (slot 0 is always the unit cut over the gate's
/// own operands).
const KEEP: usize = 4;
/// A cut whose interior exceeds this many gates is skipped — the
/// redundancy check would be checking a cone this pass has no business
/// re-deriving.
const MAX_INTERIOR: usize = 8;

#[derive(Clone, Copy)]
struct Cut {
    leaves: [u32; 4],
    n: u8,
    /// Gates strictly inside the cut (root not counted).
    ngates: u8,
    /// Function of the ROOT VALUE over `leaves[0..n]` (leaf i = table
    /// var i), rows beyond `2^n` replicated.
    tt: u16,
    /// Bloom of the leaf set: one bit per `leaf % 32`. Candidate dedup
    /// is quadratic in the candidate count, so rejecting on a single
    /// word compare before touching the leaf array matters.
    sig: u32,
}

const EMPTY_CUT: Cut = Cut { leaves: [0; 4], n: 0, ngates: 0, tt: 0, sig: 0 };

/// Leaf-set bloom for [`Cut::sig`].
#[inline]
fn leaf_sig(leaves: &[u32; 4], n: u8) -> u32 {
    let mut s = 0u32;
    for &l in &leaves[..n as usize] {
        s |= 1 << (l & 31);
    }
    s
}

/// Child cut choices for one operand: its trivial single-leaf cut,
/// then its stored priority cuts when the operand is itself a gate of
/// this batch. The bool records "came from a gate" (drives the
/// interior-gate count). Fixed-size — no allocation per operand.
#[inline]
fn fill_choices(
    out: &mut [(Cut, bool); KEEP + 1],
    gate_of: &[u32],
    cuts: &[Cut],
    ncuts: &[u8],
    op: Op,
) -> usize {
    out[0] = (
        Cut {
            leaves: [op.0, 0, 0, 0],
            n: 1,
            ngates: 0,
            tt: VAR4[0],
            sig: 1 << (op.0 & 31),
        },
        false,
    );
    let mut k = 1;
    let cgi = gate_of[op.0 as usize];
    if cgi != u32::MAX {
        let base = cgi as usize * KEEP;
        for c in 0..ncuts[cgi as usize] as usize {
            out[k] = (cuts[base + c], true);
            k += 1;
        }
    }
    k
}

/// Insert a don't-care variable at position `j`: blocks of `2^j` rows
/// duplicate (mask-pyramid, mirrors `cnfmap::insert64` at 16 rows).
#[inline]
fn insert_var16(t: u16, j: usize) -> u16 {
    let mut x = (t & 0x00FF) as u32;
    if j <= 2 {
        x = (x | (x << 4)) & 0x0F0F;
    }
    if j <= 1 {
        x = (x | (x << 2)) & 0x3333;
    }
    if j == 0 {
        x = (x | (x << 1)) & 0x5555;
    }
    let x = x as u16;
    x | (x << (1usize << j))
}

/// Re-express `tt` over the merged leaf basis: `present` bit j set iff
/// merged position j is one of the source cut's own leaves (in order).
#[inline]
fn expand16(tt: u16, present: u8, n: usize) -> u16 {
    let full = (1u8 << n).wrapping_sub(1);
    if present == full && n > 0 {
        return tt;
    }
    let mut t = tt;
    for j in 0..n {
        if present & (1 << j) == 0 {
            t = insert_var16(t, j);
        }
    }
    t
}

/// Merge two sorted leaf sets; `None` if the union exceeds 4. Returns
/// the union plus the presence masks of each source.
fn merge_leaves(a: &Cut, b: &Cut) -> Option<([u32; 4], u8, u8, u8)> {
    let mut out = [0u32; 4];
    let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
    let (mut ma, mut mb) = (0u8, 0u8);
    while i < a.n as usize || j < b.n as usize {
        if k == 4 {
            return None;
        }
        let ai = (i < a.n as usize).then(|| a.leaves[i]);
        let bj = (j < b.n as usize).then(|| b.leaves[j]);
        match (ai, bj) {
            (Some(x), Some(y)) if x == y => {
                out[k] = x;
                ma |= 1 << k;
                mb |= 1 << k;
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x < y => {
                out[k] = x;
                ma |= 1 << k;
                i += 1;
            }
            (Some(_), Some(y)) => {
                out[k] = y;
                mb |= 1 << k;
                j += 1;
            }
            (Some(x), None) => {
                out[k] = x;
                ma |= 1 << k;
                i += 1;
            }
            (None, Some(y)) => {
                out[k] = y;
                mb |= 1 << k;
                j += 1;
            }
            (None, None) => unreachable!(),
        }
        k += 1;
    }
    Some((out, k as u8, ma, mb))
}

/// A clause over local var ids as pos/neg masks (≤ 16 locals).
#[derive(Clone, Copy, PartialEq, Eq)]
struct LClause {
    pos: u16,
    neg: u16,
}

impl LClause {
    /// Pack into one word — the shape-cache key is a flat `u32` slice.
    #[inline]
    fn pack(self) -> u32 {
        (self.pos as u32) << 16 | self.neg as u32
    }
}

/// A partial assignment over ≤ 16 locals as two disjoint bitmasks.
#[derive(Clone, Copy, Default)]
struct Assign {
    t: u16,
    f: u16,
}

/// Unit propagation to fixpoint over the local encoding. Returns false
/// on conflict.
///
/// Bitmask form: a clause is satisfied iff it shares a bit with the
/// matching mask, and its free literals are one AND-NOT away, so the
/// per-clause test is a handful of word ops with no per-literal loop.
/// This is the innermost loop of the whole pass — it runs once per
/// candidate literal per cover clause per cut.
///
/// Precondition: no clause carries both polarities of a variable. Such
/// a clause is a tautology and is dropped at construction — a bitmask
/// propagator cannot represent one faithfully (it would read as a unit
/// when the variable is free), and neither can the per-literal form
/// (which reads it as unsatisfied, and can then report a false
/// conflict).
fn up(clauses: &[LClause], a: &mut Assign) -> bool {
    debug_assert!(
        clauses.iter().all(|c| c.pos & c.neg == 0),
        "tautological clause reached unit propagation"
    );
    loop {
        let mut changed = false;
        for cl in clauses {
            if cl.pos & a.t != 0 || cl.neg & a.f != 0 {
                continue; // satisfied
            }
            let free = (cl.pos | cl.neg) & !(a.t | a.f);
            if free == 0 {
                return false; // all literals falsified
            }
            if free & (free - 1) == 0 {
                // Exactly one free literal — assign it.
                if cl.pos & free != 0 {
                    a.t |= free;
                } else {
                    a.f |= free;
                }
                changed = true;
            }
        }
        if !changed {
            return true;
        }
    }
}

/// Is `c` propagation-redundant given the local encoding? For every
/// literal of `c`, falsify the others and check unit propagation derives
/// it (or conflicts). If all pass, the encoding already makes every
/// propagation `c` would.
fn up_redundant(clauses: &[LClause], c: LClause) -> bool {
    let all = c.pos | c.neg;
    let mut lits = all;
    while lits != 0 {
        let bit = lits & lits.wrapping_neg();
        lits &= lits - 1;
        let rest = all & !bit;
        // Falsifying a literal assigns its variable the opposite value:
        // positively-occurring vars go false, negatives go true.
        let mut a = Assign { t: rest & c.neg, f: rest & c.pos };
        let ok = up(clauses, &mut a);
        let derived = !ok
            || if c.pos & bit != 0 { a.t & bit != 0 } else { a.f & bit != 0 };
        if !derived {
            return false;
        }
    }
    true
}

/// Sequence hash for the shape-cache key / emitted-clause dedup.
/// splitmix64's finalizer over an additive accumulator — `add` rather
/// than `xor` because `xor` collapses to zero whenever the running hash
/// equals the incoming word, which would erase the prefix.
#[inline]
fn mix(h: u64, v: u64) -> u64 {
    let mut x = h.wrapping_add(v).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^ (x >> 33)
}

/// One cached verdict: which cover clauses survived the redundancy
/// filter for a given local encoding. Chained by `next` off the hash
/// bucket so a lookup never allocates.
struct ShapeEntry {
    /// Slice of `shape_arena` holding the exact key (packed clauses).
    off: u32,
    len: u16,
    /// Leaf count the key was built for.
    n: u8,
    /// Bit i set = cover clause i is a genuine propagation hole.
    mask: u32,
    next: u32,
}

pub struct Augmenter {
    canon: npn4::Canon,
    /// node id → index into the batch's gate list, `u32::MAX` for
    /// non-gates. Dense rather than hashed: this is read two or three
    /// times per gate during cut enumeration and again for every
    /// interior walk, and node ids are already a dense space. Only the
    /// entries a batch touches are written and cleared, so the cost is
    /// O(gates) per batch, not O(nodes).
    gate_of: Vec<u32>,
    /// Flat per-gate cut storage, stride KEEP.
    cuts: Vec<Cut>,
    ncuts: Vec<u8>,
    /// Candidate cuts for the gate being processed (reused; a fresh Vec
    /// per gate was ~1 allocation per gate).
    cand: Vec<Cut>,
    /// Dedup of emitted clauses, by order-sensitive hash of the literal
    /// sequence — matching the previous `HashSet<Vec<_>>` semantics
    /// without the per-clause heap allocation. A collision drops a
    /// redundant clause, which is harmless.
    seen: HashSet<u64>,
    /// Shape cache: local encoding → surviving cover clauses. The UP
    /// filter is by far the most expensive part of the pass, and
    /// arithmetic circuits present the same local shapes over and over
    /// (every adder bit is the same MAJ/XOR3 wiring), so this converts
    /// most cuts into a hash lookup. Keys are compared exactly — a
    /// collision here would emit a clause that is NOT implied.
    shape_arena: Vec<u32>,
    shape_buckets: HashMap<u64, u32>,
    shape_entries: Vec<ShapeEntry>,
    pub shape_hits: u64,
    pub shape_misses: u64,
    scratch_interior: Vec<u32>,
    scratch_clauses: Vec<LClause>,
    scratch_lits: Vec<(u32, bool)>,
    /// Force every shape-cache lookup to miss; see
    /// [`Augmenter::set_shape_cache_enabled`].
    nocache: bool,
    /// Selection-depth knob; see [`Augmenter::set_min_gates`].
    min_gates: u32,
}


impl Default for Augmenter {
    fn default() -> Self {
        Self {
            canon: npn4::Canon::default(),
            gate_of: Vec::new(),
            cuts: Vec::new(),
            ncuts: Vec::new(),
            cand: Vec::new(),
            seen: HashSet::default(),
            shape_arena: Vec::new(),
            shape_buckets: HashMap::default(),
            shape_entries: Vec::new(),
            shape_hits: 0,
            shape_misses: 0,
            scratch_interior: Vec::new(),
            scratch_clauses: Vec::new(),
            scratch_lits: Vec::new(),
            nocache: false,
            min_gates: Self::MIN_GATES,
        }
    }
}

impl Augmenter {
    /// Default for [`Augmenter::set_min_gates`]. Note this is 1, not 0 —
    /// a derived `Default` would get it wrong, which is why `Default` is
    /// written out above.
    pub const MIN_GATES: u32 = 1;

    /// Enable or disable the shape cache. It must be output-identical to
    /// recomputing, and turning it off is how that is checked on a real
    /// instance. On by default.
    pub fn set_shape_cache_enabled(&mut self, on: bool) {
        self.nocache = !on;
    }

    /// Fewest interior gates a cut must cover to be worth emitting (see
    /// `worth` in [`Augmenter::run`]).
    pub fn set_min_gates(&mut self, n: u32) {
        self.min_gates = n;
    }

    /// Run over one batch. `gates` is topologically ordered (operands
    /// precede users — AIG indices give this for free). `leaf_ok` says a
    /// node can serve as a cut leaf (it still has a live SAT binding).
    /// `emit` receives each surviving clause as (node, negated) pairs —
    /// the LAST pair is the root's output literal.
    pub fn run(
        &mut self,
        gates: &[(u32, Gate)],
        max_added: u32,
        per_root: u32,
        mut leaf_ok: impl FnMut(u32) -> bool,
        mut emit: impl FnMut(&[(u32, bool)]),
    ) -> AugStats {
        let min_gates = self.min_gates;
        let mut stats = AugStats::default();
        // Dense node → gate index. Sized to the batch's largest node id
        // and reused across batches; only touched entries are reset (at
        // the end of the pass), so nothing here is O(nodes) per call.
        let max_node = gates.last().map_or(0, |&(n, _)| n) as usize;
        if self.gate_of.len() <= max_node {
            self.gate_of.resize(max_node + 1, u32::MAX);
        }
        for (gi, &(n, _)) in gates.iter().enumerate() {
            self.gate_of[n as usize] = gi as u32;
        }
        self.cuts.clear();
        self.cuts.resize(gates.len() * KEEP, EMPTY_CUT);
        self.ncuts.clear();
        self.ncuts.resize(gates.len(), 0);
        self.seen.clear();
        // Cut storage and the candidate buffer move out for the duration:
        // enumeration reads them while `filter_and_emit` needs `&mut
        // self`. (Same idiom as `cnfmap::Mapper::plan`.)
        let mut cuts = std::mem::take(&mut self.cuts);
        let mut ncuts = std::mem::take(&mut self.ncuts);
        let mut cand = std::mem::take(&mut self.cand);
        let mut ca_buf = [(EMPTY_CUT, false); KEEP + 1];
        let mut cb_buf = [(EMPTY_CUT, false); KEEP + 1];
        let mut cc_buf = [(EMPTY_CUT, false); KEEP + 1];

        for (gi, &(node, gate)) in gates.iter().enumerate() {
            // Budget exhausted: stop the pass outright rather than keep
            // enumerating cuts nothing will consume. Cut enumeration, not
            // emission, is the bulk of this pass's cost, so the early
            // exit is what makes a tight budget actually cheap.
            if stats.added >= max_added {
                break;
            }
            stats.roots += 1;
            // ---- enumerate cuts ----
            cand.clear();
            let ops: [Option<Op>; 3] = match gate {
                Gate::And(a, b) | Gate::Xor(a, b) => [Some(a), Some(b), None],
                Gate::NotMux(s, t, e) => [Some(s), Some(t), Some(e)],
            };
            let nops = if ops[2].is_some() { 3 } else { 2 };
            let na =
                fill_choices(&mut ca_buf, &self.gate_of, &cuts, &ncuts, ops[0].unwrap());
            let nb =
                fill_choices(&mut cb_buf, &self.gate_of, &cuts, &ncuts, ops[1].unwrap());
            let nc = match ops[2] {
                Some(o) => fill_choices(&mut cc_buf, &self.gate_of, &cuts, &ncuts, o),
                None => {
                    cc_buf[0] = (EMPTY_CUT, false);
                    1
                }
            };
            for &(cut_a, ga) in &ca_buf[..na] {
                for &(cut_b, gb) in &cb_buf[..nb] {
                    let Some((l2, n2, ma, mb)) = merge_leaves(&cut_a, &cut_b) else {
                        continue;
                    };
                    for &(cut_c, gc) in &cc_buf[..nc] {
                        let ab = Cut { leaves: l2, n: n2, ngates: 0, tt: 0, sig: 0 };
                        let (leaves, n, mab, mc) = if nops == 3 {
                            match merge_leaves(&ab, &cut_c) {
                                Some(v) => v,
                                None => continue,
                            }
                        } else {
                            (l2, n2, (1u8 << n2).wrapping_sub(1), 0)
                        };
                        // Tables of each operand over the merged basis.
                        let sub = |cut: &Cut, m_outer: u8, m_inner: u8, neg: bool| -> u16 {
                            // Source leaves sit at inner positions within
                            // the a/b merge, which sit at outer positions
                            // within the final merge.
                            let mut pres = 0u8;
                            let mut ii = 0u8;
                            for j in 0..4u8 {
                                if m_outer & (1 << j) != 0 {
                                    if m_inner & (1 << ii) != 0 {
                                        pres |= 1 << j;
                                    }
                                    ii += 1;
                                }
                            }
                            let t = expand16(cut.tt, pres, n as usize);
                            if neg { !t } else { t }
                        };
                        let ta = sub(&cut_a, mab, ma, ops[0].unwrap().1);
                        let tb = sub(&cut_b, mab, mb, ops[1].unwrap().1);
                        let tt = match gate {
                            Gate::And(..) => ta & tb,
                            Gate::Xor(..) => ta ^ tb,
                            Gate::NotMux(..) => {
                                // `mc` is already cut_c's presence mask in
                                // the FINAL basis (it came from the outer
                                // merge, unlike ma/mb which need the
                                // two-level composition above).
                                let t = expand16(cut_c.tt, mc, n as usize);
                                let tc = if ops[2].unwrap().1 { !t } else { t };
                                !((ta & tb) | (!ta & tc))
                            }
                        };
                        let ngates = (if ga { 1 + cut_a.ngates } else { 0 })
                            + (if gb { 1 + cut_b.ngates } else { 0 })
                            + (if nops == 3 && gc { 1 + cut_c.ngates } else { 0 });
                        let cut =
                            Cut { leaves, n, ngates, tt, sig: leaf_sig(&leaves, n) };
                        if !cand.iter().any(|e| {
                            e.sig == cut.sig && e.n == cut.n && e.leaves == cut.leaves
                        }) {
                            cand.push(cut);
                        }
                    }
                }
            }
            // Keep: unit cut first (it came from all-trivial choices and
            // is always candidate 0 — parents build their compositions
            // from it), then the deepest compositions: a cut spanning
            // more gates covers propagation paths no smaller cut sees,
            // so those must survive the truncation.
            cand[1..].sort_by_key(|c| std::cmp::Reverse(c.ngates));
            let keep = cand.len().min(KEEP);
            let base = gi * KEEP;
            cuts[base..base + keep].copy_from_slice(&cand[..keep]);
            ncuts[gi] = keep as u8;

            // ---- filter + emit ----
            let mut added_here = 0u32;
            for c in 0..keep {
                let cut = cuts[base + c];
                // Single-gate AND/XOR encodings are already GAC — only
                // compositions (or a bare mux) can have holes. The depth
                // threshold is tunable while the selection policy is
                // being calibrated (v1 measured: taking every ≥1-gate
                // hole adds ~13% clauses and loses — CDCL learns the
                // shallow consensus resolvents on demand for free).
                let worth = cut.ngates as u32 >= min_gates
                    || (min_gates <= 1 && matches!(gate, Gate::NotMux(..)));
                if !worth || added_here >= per_root {
                    continue;
                }
                if (0..cut.n as usize).any(|i| !leaf_ok(cut.leaves[i])) {
                    continue;
                }
                if self.filter_and_emit(
                    gates, node, &cut, max_added, per_root, &mut added_here, &mut stats,
                    &mut emit,
                ) {
                    stats.cuts += 1;
                }
            }
        }
        // Reset only the entries this batch claimed, and hand the
        // buffers back for the next one.
        for &(n, _) in gates {
            self.gate_of[n as usize] = u32::MAX;
        }
        self.cuts = cuts;
        self.ncuts = ncuts;
        self.cand = cand;
        stats
    }

    /// Reconstruct the interior encoding, test each PC-cover clause for
    /// propagation redundancy, and emit the holes. Returns true if the
    /// cut was actually processed.
    #[allow(clippy::too_many_arguments)]
    fn filter_and_emit(
        &mut self,
        gates: &[(u32, Gate)],
        root: u32,
        cut: &Cut,
        max_added: u32,
        per_root: u32,
        added_here: &mut u32,
        stats: &mut AugStats,
        emit: &mut impl FnMut(&[(u32, bool)]),
    ) -> bool {
        let n = cut.n as usize;
        let mut interior = std::mem::take(&mut self.scratch_interior);
        let mut clauses = std::mem::take(&mut self.scratch_clauses);
        let mut out = std::mem::take(&mut self.scratch_lits);
        let processed = 'done: {
        // Collect interior gates (root included) by walking gate edges
        // until cut leaves.
        interior.clear();
        interior.push(root);
        let mut i = 0usize;
        while i < interior.len() {
            if interior.len() > MAX_INTERIOR {
                break 'done false;
            }
            let g = gates[self.gate_of[interior[i] as usize] as usize].1;
            let ops: [Option<Op>; 3] = match g {
                Gate::And(a, b) | Gate::Xor(a, b) => [Some(a), Some(b), None],
                Gate::NotMux(s, t, e) => [Some(s), Some(t), Some(e)],
            };
            for op in ops.iter().flatten() {
                let c = op.0;
                if cut.leaves[..n].contains(&c) || interior.contains(&c) {
                    continue;
                }
                // Non-leaf operands inside a merged cut are gates by
                // construction; a lookup miss means the cut sliced
                // through a non-batch node — bail out.
                if self.gate_of[c as usize] == u32::MAX {
                    break 'done false;
                }
                interior.push(c);
            }
            i += 1;
        }
        // Local ids: 0..n leaves (table order), n = y (root value),
        // n+1.. interior values in walk order (root's own slot is y).
        let local_of = |node: u32, interior: &[u32]| -> usize {
            if let Some(p) = cut.leaves[..n].iter().position(|&l| l == node) {
                return p;
            }
            let p = interior.iter().position(|&g| g == node).unwrap();
            if p == 0 { n } else { n + p }
        };
        clauses.clear();
        for (idx, &g) in interior.iter().enumerate() {
            let v = if idx == 0 { n } else { n + idx };
            let gate = gates[self.gate_of[g as usize] as usize].1;
            let lit = |op: Op, interior: &[u32]| -> (usize, bool) {
                (local_of(op.0, interior), op.1)
            };
            let mut cl = |lits: &[(usize, bool)]| {
                let (mut pos, mut neg) = (0u16, 0u16);
                for &(id, ng) in lits {
                    if ng {
                        neg |= 1 << id;
                    } else {
                        pos |= 1 << id;
                    }
                }
                // Tautology — vacuous, and unrepresentable for the
                // propagator (see `up`). Arises when two operands of a
                // gate resolve to the same node with opposite signs.
                if pos & neg != 0 {
                    return;
                }
                clauses.push(LClause { pos, neg });
            };
            match gate {
                Gate::And(a, b) => {
                    let (ai, an) = lit(a, &interior);
                    let (bi, bn) = lit(b, &interior);
                    // (¬a ∨ ¬b ∨ v), (a ∨ ¬v), (b ∨ ¬v)
                    cl(&[(ai, !an), (bi, !bn), (v, false)]);
                    cl(&[(ai, an), (v, true)]);
                    cl(&[(bi, bn), (v, true)]);
                }
                Gate::Xor(a, b) => {
                    let (ai, an) = lit(a, &interior);
                    let (bi, bn) = lit(b, &interior);
                    cl(&[(ai, !an), (bi, !bn), (v, true)]);
                    cl(&[(ai, an), (bi, bn), (v, true)]);
                    cl(&[(ai, an), (bi, !bn), (v, false)]);
                    cl(&[(ai, !an), (bi, bn), (v, false)]);
                }
                Gate::NotMux(s, t, e) => {
                    let (si, sn) = lit(s, &interior);
                    let (ti, tn) = lit(t, &interior);
                    let (ei, en) = lit(e, &interior);
                    // value v = ¬mux(s,t,e):
                    // (¬s∨¬t∨¬v), (¬s∨t∨v), (s∨¬e∨¬v), (s∨e∨v)
                    cl(&[(si, !sn), (ti, !tn), (v, true)]);
                    cl(&[(si, !sn), (ti, tn), (v, false)]);
                    cl(&[(si, sn), (ei, !en), (v, true)]);
                    cl(&[(si, sn), (ei, en), (v, false)]);
                }
            }
        }

        // The class cover, transported onto this cut.
        let (canon_tt, npn) = self.canon.get(pad16(cut.tt, n));
        let Some(pc) = npn4::pc_cover_for(canon_tt) else {
            break 'done false;
        };
        let (on_src, off_src) =
            if npn.out_neg { (pc.off, pc.on) } else { (pc.on, pc.off) };
        if on_src.len() + off_src.len() > 32 {
            break 'done false; // survivor mask is a u32 (npn4 max is 26)
        }

        // Which cover clauses are genuine holes? This is the expensive
        // question, and the answer depends only on the LOCAL encoding —
        // node identities are already erased — so it caches across every
        // structurally identical cut in the circuit.
        // Key on exactly what the filter reads: the local encoding AND
        // the transform, since the candidate clauses come from the class
        // cover mapped through `npn`. Keying on the encoding alone is
        // NOT sufficient — measured (bench_64, bench_13728) to change
        // the emitted set — so the transform is folded in rather than
        // assumed to be implied by the structure.
        let npn_word = (canon_tt as u32)
            | (npn.perm[0] as u32) << 16
            | (npn.perm[1] as u32) << 18
            | (npn.perm[2] as u32) << 20
            | (npn.perm[3] as u32) << 22
            | (npn.in_neg as u32) << 24
            | (npn.out_neg as u32) << 28;
        let mut h = mix(0x51ED_5EED, n as u64);
        h = mix(h, npn_word as u64);
        for c in clauses.iter() {
            h = mix(h, c.pack() as u64);
        }
        let mask = match self.shape_find(h, npn_word, &clauses, cut.n) {
            Some(m) => {
                self.shape_hits += 1;
                m
            }
            None => {
                self.shape_misses += 1;
                let mut m = 0u32;
                for (i, (cube, y_pos)) in on_src
                    .iter()
                    .map(|c| (c, true))
                    .chain(off_src.iter().map(|c| (c, false)))
                    .enumerate()
                {
                    let (cp, cn) = npn4::map_cube(*cube, &npn);
                    let cl = LClause {
                        pos: cn as u16 | if y_pos { 1 << n } else { 0 },
                        neg: cp as u16 | if y_pos { 0 } else { 1 << n },
                    };
                    if !up_redundant(&clauses, cl) {
                        m |= 1 << i;
                    }
                }
                self.shape_insert(h, npn_word, &clauses, cut.n, m);
                m
            }
        };

        // Emit the holes over real nodes, within budget.
        for (i, (cube, y_pos)) in on_src
            .iter()
            .map(|c| (c, true))
            .chain(off_src.iter().map(|c| (c, false)))
            .enumerate()
        {
            if mask >> i & 1 == 0 {
                continue;
            }
            if *added_here >= per_root || stats.added >= max_added {
                break;
            }
            let (cp, cn) = npn4::map_cube(*cube, &npn);
            out.clear();
            for i in 0..n {
                if cn >> i & 1 == 1 {
                    out.push((cut.leaves[i], false));
                } else if cp >> i & 1 == 1 {
                    out.push((cut.leaves[i], true));
                }
            }
            out.push((root, !y_pos));
            let mut ch = 0xC1A5_E5EEu64;
            for &(nd, ng) in out.iter() {
                ch = mix(ch, (nd as u64) << 1 | ng as u64);
            }
            if self.seen.insert(ch) {
                emit(&out);
                stats.added += 1;
                *added_here += 1;
            }
        }
        true
        };
        self.scratch_interior = interior;
        self.scratch_clauses = clauses;
        self.scratch_lits = out;
        processed
    }

    /// Exact shape-cache probe: hash bucket, then a full key compare
    /// down the chain. Inexact matching here would emit clauses that are
    /// not implied, so the comparison is never elided.
    fn shape_find(
        &self,
        hash: u64,
        npn_word: u32,
        clauses: &[LClause],
        n: u8,
    ) -> Option<u32> {
        if self.nocache {
            return None;
        }
        let mut e = *self.shape_buckets.get(&hash)?;
        while e != u32::MAX {
            let en = &self.shape_entries[e as usize];
            if en.n == n && en.len as usize == clauses.len() + 1 {
                let k = en.off as usize;
                let key = &self.shape_arena[k..k + en.len as usize];
                if key[0] == npn_word
                    && key[1..].iter().zip(clauses).all(|(&a, b)| a == b.pack())
                {
                    return Some(en.mask);
                }
            }
            e = en.next;
        }
        None
    }

    fn shape_insert(
        &mut self,
        hash: u64,
        npn_word: u32,
        clauses: &[LClause],
        n: u8,
        mask: u32,
    ) {
        let off = self.shape_arena.len() as u32;
        self.shape_arena.push(npn_word);
        self.shape_arena.extend(clauses.iter().map(|c| c.pack()));
        let next = self.shape_buckets.get(&hash).copied().unwrap_or(u32::MAX);
        self.shape_entries.push(ShapeEntry {
            off,
            len: clauses.len() as u16 + 1,
            n,
            mask,
            next,
        });
        self.shape_buckets
            .insert(hash, self.shape_entries.len() as u32 - 1);
    }
}

/// Replicate an n-var table (meaningful low 2^n rows) to the full
/// 16-row representation.
#[inline]
fn pad16(tt: u16, n: usize) -> u16 {
    let mut t = tt;
    for i in n..4 {
        let sh = 1usize << i;
        let mask = (1u32 << sh) as u16 - 1;
        let low = t & mask;
        t = low | (low << sh);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluate the derived gate graph over one assignment to the leaf
    /// nodes. Returns the value of every node (leaves pass through).
    fn eval(
        gates: &[(u32, Gate)],
        leaves: &[u32],
        assign: usize,
    ) -> HashMap<u32, bool> {
        let mut val: HashMap<u32, bool> = HashMap::default();
        for (i, &l) in leaves.iter().enumerate() {
            val.insert(l, assign >> i & 1 == 1);
        }
        let get = |v: &HashMap<u32, bool>, op: Op| v[&op.0] ^ op.1;
        for &(n, g) in gates {
            let r = match g {
                Gate::And(a, b) => get(&val, a) & get(&val, b),
                Gate::Xor(a, b) => get(&val, a) ^ get(&val, b),
                Gate::NotMux(s, t, e) => {
                    !(if get(&val, s) { get(&val, t) } else { get(&val, e) })
                }
            };
            val.insert(n, r);
        }
        val
    }

    /// Run the augmenter and return the emitted clauses, checking each
    /// one is a genuine implicate of the gate graph (holds under every
    /// leaf assignment).
    fn run_checked(gates: &[(u32, Gate)], leaves: &[u32]) -> Vec<Vec<(u32, bool)>> {
        let mut aug = Augmenter::default();
        let mut out: Vec<Vec<(u32, bool)>> = Vec::new();
        aug.run(gates, 10_000, 64, |_| true, |cl| out.push(cl.to_vec()));
        for cl in &out {
            for assign in 0..1usize << leaves.len() {
                let val = eval(gates, leaves, assign);
                let sat = cl.iter().any(|&(n, neg)| val[&n] ^ neg);
                assert!(
                    sat,
                    "emitted clause {cl:?} violated at leaf assignment {assign:#b}"
                );
            }
        }
        out
    }

    /// XOR and AND trees are propagation complete gate-by-gate — the
    /// filter must find nothing to add.
    #[test]
    fn xor_and_trees_add_nothing() {
        let (a, b, c) = (1u32, 2, 3);
        let xor_tree = vec![
            (10u32, Gate::Xor((a, false), (b, false))),
            (11u32, Gate::Xor((10, false), (c, false))),
        ];
        assert!(run_checked(&xor_tree, &[a, b, c]).is_empty());
        let and_tree = vec![
            (10u32, Gate::And((a, false), (b, false))),
            (11u32, Gate::And((10, false), (c, false))),
        ];
        assert!(run_checked(&and_tree, &[a, b, c]).is_empty());
    }

    /// The adder-carry composition (MAJ through AND/XOR/NOR gates) has
    /// the classic propagation holes — `a ∧ cin` forces the carry but no
    /// gate clause fires. The augmenter must add consensus clauses, and
    /// among them the `(¬a ∨ ¬cin ∨ carry)` family.
    #[test]
    fn adder_carry_gets_consensus_clauses() {
        let (a, b, cin) = (1u32, 2, 3);
        let gates = vec![
            (10u32, Gate::And((a, false), (b, false))), // ab
            (11u32, Gate::Xor((a, false), (b, false))), // a^b
            (12u32, Gate::And((cin, false), (11, false))), // cin & (a^b)
            // OR via the AIG's NOR: value = ¬(ab ∨ cin(a^b)) = ¬carry.
            (13u32, Gate::And((10, true), (12, true))),
        ];
        let out = run_checked(&gates, &[a, b, cin]);
        assert!(!out.is_empty(), "carry composition must yield holes");
        // The consensus family: two of {a, b, cin} true forces node 13
        // (= ¬carry) FALSE: clause (¬a ∨ ¬cin ∨ ¬n13).
        let want = [(a, true), (cin, true), (13, true)];
        assert!(
            out.iter().any(|cl| {
                cl.len() == want.len() && want.iter().all(|l| cl.contains(l))
            }),
            "missing carry consensus clause; got {out:?}"
        );
    }

    /// A bare mux gate's 4-clause encoding misses the `t = e ⇒ out`
    /// propagation; the augmenter must supply the two redundant clauses.
    #[test]
    fn mux_gets_redundant_clauses() {
        let (s, t, e) = (1u32, 2, 3);
        let gates = vec![(10u32, Gate::NotMux((s, false), (t, false), (e, false)))];
        let out = run_checked(&gates, &[s, t, e]);
        // value = ¬mux: t ∧ e forces value false, ¬t ∧ ¬e forces true.
        assert_eq!(out.len(), 2, "mux should get exactly its two consensus clauses; got {out:?}");
    }

    /// Reference unit propagation: the straightforward per-literal form
    /// the bitmask version replaced. `val[i]`: 0 unassigned, 1 true,
    /// 2 false.
    fn up_reference(clauses: &[LClause], val: &mut [u8; 16]) -> bool {
        loop {
            let mut changed = false;
            for cl in clauses {
                let mut lits = cl.pos | cl.neg;
                let mut sat = false;
                let mut free = 0u32;
                let mut last = (0usize, false);
                while lits != 0 {
                    let i = lits.trailing_zeros() as usize;
                    lits &= lits - 1;
                    let p = cl.pos >> i & 1 == 1;
                    match val[i] {
                        0 => {
                            free += 1;
                            last = (i, p);
                        }
                        1 if p => sat = true,
                        2 if !p => sat = true,
                        _ => {}
                    }
                }
                if sat {
                    continue;
                }
                match free {
                    0 => return false,
                    1 => {
                        val[last.0] = if last.1 { 1 } else { 2 };
                        changed = true;
                    }
                    _ => {}
                }
            }
            if !changed {
                return true;
            }
        }
    }

    /// The bitmask propagator must agree with the reference on every
    /// input, INCLUDING clauses that carry both polarities of a
    /// variable — the two disagree there unless satisfaction is tested
    /// per polarity, and such clauses do arise when a gate's operands
    /// resolve to the same node.
    #[test]
    fn bitmask_up_matches_reference() {
        let mut s = 0x243F_6A88_85A3_08D3u64;
        let mut rng = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let nlocals = 8u32;
        for _ in 0..20_000 {
            let ncl = 1 + (rng() % 6) as usize;
            let mut clauses = Vec::with_capacity(ncl);
            for _ in 0..ncl {
                let mut pos = 0u16;
                let mut neg = 0u16;
                for _ in 0..(1 + rng() % 3) {
                    let v = (rng() % nlocals as u64) as u16;
                    if rng() & 1 == 0 {
                        pos |= 1 << v;
                    } else {
                        neg |= 1 << v;
                    }
                }
                // Tautologies are dropped at construction in the real
                // pass (see `up`'s precondition), so don't generate them.
                if pos | neg != 0 && pos & neg == 0 {
                    clauses.push(LClause { pos, neg });
                }
            }
            if clauses.is_empty() {
                continue;
            }
            // Random partial assignment.
            let (mut t, mut f) = (0u16, 0u16);
            for v in 0..nlocals as u16 {
                match rng() % 3 {
                    0 => t |= 1 << v,
                    1 => f |= 1 << v,
                    _ => {}
                }
            }
            let mut val = [0u8; 16];
            for v in 0..nlocals as usize {
                val[v] = if t >> v & 1 == 1 {
                    1
                } else if f >> v & 1 == 1 {
                    2
                } else {
                    0
                };
            }
            let mut a = Assign { t, f };
            let got = up(&clauses, &mut a);
            let want = up_reference(&clauses, &mut val);
            assert_eq!(got, want, "conflict verdict differs");
            if got {
                for v in 0..nlocals as usize {
                    let bit = 1u16 << v;
                    let got_v = if a.t & bit != 0 {
                        1
                    } else if a.f & bit != 0 {
                        2
                    } else {
                        0
                    };
                    assert_eq!(got_v, val[v], "local {v} differs");
                }
            }
        }
    }

    /// Table plumbing: expand16/insert_var16 against a brute-force
    /// re-expression.
    #[test]
    fn expand16_matches_bruteforce() {
        for n in 1..=4usize {
            for present in 1u8..(1 << n) {
                let k = present.count_ones() as usize;
                for tt_low in 0..1u32 << (1 << k) {
                    let tt = pad16(tt_low as u16, k);
                    let got = expand16(tt, present, n);
                    // Brute force: row m of the expanded table reads the
                    // source row assembled from the present positions.
                    for m in 0..1usize << n {
                        let mut src = 0usize;
                        let mut si = 0usize;
                        for j in 0..n {
                            if present >> j & 1 == 1 {
                                if m >> j & 1 == 1 {
                                    src |= 1 << si;
                                }
                                si += 1;
                            }
                        }
                        assert_eq!(
                            got >> m & 1,
                            tt >> src & 1,
                            "expand16 mismatch n={n} present={present:#b} tt={tt:#06x} row={m}"
                        );
                    }
                    if k >= 3 {
                        break; // sample large spaces, exhaust small ones
                    }
                }
            }
        }
    }
}
