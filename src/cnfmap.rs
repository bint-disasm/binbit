//! Cut-based CNF technology mapping (Eén/Mishchenko/Sörensson, SAT'07).
//!
//! Instead of one Tseitin variable per AIG node, the materialization cone
//! is covered by k-feasible cuts: only cut ROOTS get SAT variables, and
//! each root is defined by the optimal CNF of its cut function (ISOP of
//! both polarities), computed over the cut's leaves. Interior nodes are
//! absorbed — no variable, no clauses. The mapping is chosen by the
//! paper's two refinement passes over a priority-cut set, with a cut's
//! area measured in CLAUSES (the SAT-specific cost that beats unit-area
//! LUT mapping): an area-flow pass for global shape, then an exact-local-
//! area pass that can only improve the total.
//!
//! Everything here is pure planning over an immutable view of the AIG:
//! the caller (smt.rs) walks the plan and emits variables/clauses,
//! keeping all solver interaction in one place.
//!
//! Performance shape: one hash map translates AIG indices to dense cone
//! ids once per plan; everything after is flat-array indexed, cuts are
//! inline `Copy` values (no per-cut allocation), truth-table re-expression
//! is O(k) word operations per merge, and ISOP results are cached across
//! cones keyed by the (table, arity) pair (cut functions repeat
//! enormously — every adder bit yields the same XOR3/MAJ tables).

use crate::aig::{Aig, AigNode};
use crate::solver::push_unchecked;
use rustc_hash::FxHashMap as HashMap;

/// Maximum cut width. The representation supports up to 8 (256-row
/// tables), but 6 is the measured sweet spot: k=8 was evaluated and
/// found quality-neutral on binbit's shapes at 1.7× the mapping cost —
/// wide cuts lose on ISOP size for XOR-heavy arithmetic (exponential
/// covers), and the compact wide functions (AND trees) barely occur.
pub const MAX_K: usize = 6;
/// Priority cuts kept per node at [`Effort::Full`] — also the storage
/// stride of the per-node cut array.
pub const MAX_CUTS: usize = 4;
/// Priority cuts kept per node at [`Effort::Fast`].
pub const FAST_CUTS: usize = 2;
/// A cut whose ISOP needs more than this many cubes per polarity is
/// rejected (infinite cost) — bounds worst-case clause fan-out.
const MAX_CUBES: usize = 12;

/// Truth table at the current MAX_K = 6: one u64, row r at bit r. (The
/// k=8 experiment used `[u64; 4]` 256-row tables — quality-neutral on
/// binbit's shapes at 1.7× the cost, so the scalar form returned; see
/// the project notes before widening again.)
pub type Tt = u64;

const TT_ZERO: Tt = 0;
const TT_ONES: Tt = !0u64;

#[inline]
fn tt_and(a: Tt, b: Tt) -> Tt {
    a & b
}
#[inline]
fn tt_or(a: Tt, b: Tt) -> Tt {
    a | b
}
#[inline]
fn tt_not(a: Tt) -> Tt {
    !a
}
#[inline]
fn tt_andnot(a: Tt, b: Tt) -> Tt {
    a & !b
}
#[inline]
fn tt_is_zero(a: Tt) -> bool {
    a == 0
}
#[cfg(test)]
#[inline]
fn tt_bit(a: Tt, r: usize) -> bool {
    a & (1u64 << r) != 0
}

// ---------- Truth tables ----------

/// Truth table of variable i (toggling with period 2^i) over 64 rows.
const VAR_MASKS: [Tt; 6] = [
    0xAAAA_AAAA_AAAA_AAAA,
    0xCCCC_CCCC_CCCC_CCCC,
    0xF0F0_F0F0_F0F0_F0F0,
    0xFF00_FF00_FF00_FF00,
    0xFFFF_0000_FFFF_0000,
    0xFFFF_FFFF_0000_0000,
];

/// A product of literals over cut-leaf indices: `pos`/`neg` bit i set
/// means leaf i appears positively/negatively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cube {
    pub pos: u8,
    pub neg: u8,
}

impl Cube {
    #[allow(dead_code)] // exercised by tests; kept as the cube semantics reference
    fn table(self) -> Tt {
        let mut t = TT_ONES;
        for i in 0..MAX_K {
            if self.pos & (1 << i) != 0 {
                t = tt_and(t, VAR_MASKS[i]);
            }
            if self.neg & (1 << i) != 0 {
                t = tt_andnot(t, VAR_MASKS[i]);
            }
        }
        t
    }
}

/// Minato-Morreale irredundant SOP of the interval [l, u] (lower/upper
/// bound on the cover's function), over `nvars` variables. Returns None
/// if the cover exceeds `MAX_CUBES`. Entry point: `isop(f, f, nvars)`.
fn isop(l: Tt, u: Tt, nvars: usize, out: &mut Vec<Cube>) -> Option<Tt> {
    debug_assert!(tt_is_zero(tt_andnot(l, u)), "isop: L must imply U");
    if tt_is_zero(l) {
        return Some(TT_ZERO);
    }
    if u == TT_ONES {
        if out.len() >= MAX_CUBES {
            return None;
        }
        out.push(Cube { pos: 0, neg: 0 });
        return Some(TT_ONES);
    }
    // Split on the highest variable both bounds depend on.
    let x = (0..nvars)
        .rev()
        .find(|&i| tt_dep(l, i) || tt_dep(u, i))
        .expect("non-constant interval must depend on a variable");
    let m = VAR_MASKS[x];
    let (l0, l1) = (tt_cofactor(l, x, false), tt_cofactor(l, x, true));
    let (u0, u1) = (tt_cofactor(u, x, false), tt_cofactor(u, x, true));

    // Cubes that must contain ¬x / x.
    let start0 = out.len();
    let c0 = isop(tt_andnot(l0, u1), u0, x, out)?;
    for c in &mut out[start0..] {
        c.neg |= 1 << x;
    }
    let start1 = out.len();
    let c1 = isop(tt_andnot(l1, u0), u1, x, out)?;
    for c in &mut out[start1..] {
        c.pos |= 1 << x;
    }
    // Remainder cover, free of x.
    let lnew = tt_or(tt_andnot(l0, c0), tt_andnot(l1, c1));
    let cs = isop(lnew, tt_and(u0, u1), x, out)?;
    Some(tt_or(tt_or(tt_andnot(c0, m), tt_and(c1, m)), cs))
}

/// Does the table depend on variable i?
#[inline]
fn tt_dep(f: Tt, i: usize) -> bool {
    let m = VAR_MASKS[i];
    (f & m) >> (1usize << i) != (f & !m)
}

/// Cofactor at variable i, duplicated across both halves so recursion
/// stays in the fixed 64-row representation.
#[inline]
fn tt_cofactor(f: Tt, i: usize, hi: bool) -> Tt {
    let m = VAR_MASKS[i];
    let sh = 1usize << i;
    let half = if hi { (f & m) >> sh } else { f & !m };
    half | (half << sh)
}

/// ISOP cover of `f` over `nvars` vars, or None if it exceeds MAX_CUBES.
pub fn isop_cover(f: Tt, nvars: usize, out: &mut Vec<Cube>) -> Option<()> {
    out.clear();
    let full = pad(f, nvars);
    let got = isop(full, full, nvars, out)?;
    debug_assert_eq!(got, full, "ISOP must reproduce the function");
    Some(())
}

/// Pad an `nvars`-variable table (meaningful in its low 2^nvars rows)
/// to the full 64-row representation by duplication.
pub fn pad(f: Tt, nvars: usize) -> Tt {
    let mut t = f;
    for i in nvars..6 {
        let sh = 1usize << i;
        let mask = (1u128 << sh) as u64 - 1;
        let low = t & mask;
        t = low | (low << sh);
    }
    t
}

/// Like `pad`, but first masks the table down to its meaningful rows —
/// needed when complementing (the pad rows of `!f` would be garbage).
/// The guard is the REPRESENTATION width (64 rows = 6 vars), not MAX_K.
fn pad_masked(f: Tt, nvars: usize) -> Tt {
    if nvars >= 6 {
        return f;
    }
    let low = f & ((1u128 << (1usize << nvars)) as u64 - 1);
    pad(low, nvars)
}

/// Insert a don't-care variable at position `j` of a packed table:
/// every existing variable at position ≥ j shifts up by one. Blocks of
/// 2^j rows are duplicated — a handful of word operations, not a
/// per-row loop.
/// 64-bit block spread: duplicate 2^j-row blocks of the low 32 bits.
#[inline]
fn insert64(t: u64, j: usize) -> u64 {
    // Mask-pyramid spread at 2^j-block granularity, then OR-duplicate:
    // branch-free word ops per level, no row loop.
    let mut x = t & 0xFFFF_FFFF;
    if j <= 4 {
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        if j <= 3 {
            x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
            if j <= 2 {
                x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
                if j <= 1 {
                    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
                    if j == 0 {
                        x = (x | (x << 1)) & 0x5555_5555_5555_5555;
                    }
                }
            }
        }
    }
    x | (x << (1 << j))
}

/// Insert a don't-care variable at position `j` of a packed table:
/// every existing variable at position ≥ j shifts up by one (blocks of
/// 2^j rows duplicate).
#[inline]
fn insert_var_at(t: Tt, j: usize) -> Tt {
    insert64(t, j)
}

/// Re-express `table` over a merged basis of `n` leaves, where `present`
/// has bit j set iff merged position j is one of the source's own
/// leaves. Positions absent from `present` become don't-cares. The mask
/// comes from the union walk in [`merge_union`], so no leaf search is
/// needed here.
#[inline]
fn expand_table(table: Tt, present: u8, n: usize) -> Tt {
    let full = if n >= 8 { u8::MAX } else { (1u8 << n) - 1 };
    // Source already spans the merged basis — nothing to insert.
    if present == full {
        return table;
    }
    // Single-leaf source (every leaf child and every trivial fanin cut):
    // the identity function's expansion is just that leaf's mask at its
    // merged position. Constants take the complementary shortcut.
    if present.count_ones() == 1 {
        let pos = present.trailing_zeros() as usize;
        if table == VAR_MASKS[0] {
            return VAR_MASKS[pos];
        }
        if table == tt_not(VAR_MASKS[0]) {
            return tt_not(VAR_MASKS[pos]);
        }
    }
    let mut t = table;
    for j in 0..n {
        if present & (1 << j) == 0 {
            t = insert_var_at(t, j);
        }
    }
    t
}

/// Number of clauses defining a variable for `table` over `nvars` leaves
/// (both ISOP polarities); `u32::MAX` when the bound is exceeded.
fn isop_cost(table: Tt, nvars: usize, scratch: &mut Vec<Cube>) -> u32 {
    // Cut metric: clauses weighted by width (Σ 1 + cube_width) rather
    // than the paper's pure clause count. BVE declines to eliminate
    // variables whose defining clauses are wide (resolvents blow up),
    // so literal-aware pricing steers the mapping toward cuts the
    // eliminator can still chew: corpus conflicts −5..−26%, post-BVE
    // clauses to −12%, never worse. (Constant, not a runtime knob —
    // this is on the cache-miss path of every candidate cut.)
    let mut n = 0u32;
    for f in [table, tt_not(table)] {
        match isop_cover(pad_masked(f, nvars), nvars, scratch) {
            Some(()) => {
                n += scratch.len() as u32
                    + scratch
                        .iter()
                        .map(|c| c.pos.count_ones() + c.neg.count_ones())
                        .sum::<u32>();
            }
            None => return u32::MAX,
        }
    }
    n
}

/// Cross-call cache of ISOP results keyed by (padded table, nvars).
/// Owned by the caller (one per SmtSolver) and threaded through
/// [`Mapper::plan`].
const COST_CACHE_BITS: usize = 15;

pub struct IsopCache {
    /// Direct-mapped (padded table, nvars) → clause-count cache. Cost is
    /// recomputable, so collisions just evict — no chaining, no probing.
    /// Entry: (key table, nvars+1, cost); nvars+1 == 0 marks empty.
    cost_keys: Vec<(Tt, u8, u32)>,
    /// Cube store, arena-style (cf. `ClauseArena`): all covers live in
    /// one buffer, `cube_idx` maps (table, nvars) → (offset, on-count,
    /// off-count) with the on-set cubes immediately followed by the
    /// off-set cubes. Emission copies out of the arena by slice — no
    /// per-hit Vec clone.
    cube_store: Vec<Cube>,
    cube_idx: HashMap<(Tt, u8), (u32, u16, u16)>,
    /// Scratch for the cache-miss ISOP runs.
    miss_scratch: Vec<Cube>,
}

impl Default for IsopCache {
    fn default() -> Self {
        IsopCache {
            // Allocated lazily on first use: a solver that never maps
            // (the whole classic path) must not pay ~768KB of
            // allocate-and-zero in its constructor.
            cost_keys: Vec::new(),
            cube_store: Vec::new(),
            cube_idx: HashMap::default(),
            miss_scratch: Vec::new(),
        }
    }
}

impl IsopCache {
    fn cost(&mut self, table: Tt, nvars: usize, scratch: &mut Vec<Cube>) -> u32 {
        if self.cost_keys.is_empty() {
            self.cost_keys.resize(1 << COST_CACHE_BITS, (TT_ZERO, 0, 0));
        }
        let key = pad_masked(table, nvars);
        let tag = nvars as u8 + 1;
        let slot = ((key
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(nvars as u64))
            >> (64 - COST_CACHE_BITS)) as usize;
        let e = &self.cost_keys[slot];
        if e.0 == key && e.1 == tag {
            return e.2;
        }
        let c = isop_cost(key, nvars, scratch);
        self.cost_keys[slot] = (key, tag, c);
        c
    }

    /// Append the (on-set, off-set) cover of `table` to `out`, returning
    /// the two lengths. Covers are memoized in the cache's arena; the
    /// append is a slice copy.
    fn cubes_into(&mut self, table: Tt, nvars: usize, out: &mut Vec<Cube>) -> (u16, u16) {
        let key = (pad_masked(table, nvars), nvars as u8);
        if let Some(&(off, non, noff)) = self.cube_idx.get(&key) {
            let s = off as usize;
            out.extend_from_slice(&self.cube_store[s..s + non as usize + noff as usize]);
            return (non, noff);
        }
        let start = self.cube_store.len();
        let mut scratch = std::mem::take(&mut self.miss_scratch);
        isop_cover(key.0, nvars, &mut scratch).expect("chosen cut has bounded ISOP");
        let non = scratch.len() as u16;
        self.cube_store.extend_from_slice(&scratch);
        isop_cover(pad_masked(tt_not(key.0), nvars), nvars, &mut scratch)
            .expect("chosen cut has bounded ISOP");
        let noff = scratch.len() as u16;
        self.cube_store.extend_from_slice(&scratch);
        self.cube_idx.insert(key, (start as u32, non, noff));
        out.extend_from_slice(&self.cube_store[start..]);
        self.miss_scratch = scratch;
        (non, noff)
    }
}

// ---------- Cuts and the mapper ----------

/// A k-feasible cut as flat `Copy` data: sorted dense leaf ids inline,
/// the node's function over them, and its CNF cost.
#[derive(Clone, Copy, Debug)]
// Field order is deliberate: this packs to 48 bytes with `fan` and `n`
// filling padding that existed anyway. Keep it lean — CutC is copied in
// the mapper's hottest loops and growing it measures directly.
struct CutC {
    leaves: [u32; MAX_K],
    table: Tt,
    /// Hashed leaf-set signature: popcount(sig_a | sig_b) lower-bounds
    /// the union size (collisions only shrink it), so oversize merges
    /// reject before the two-pointer walk.
    sig: u64,
    nclauses: u32,
    /// Average leaf fanout ×64, saturating at u16 (verified not to
    /// change any mapping decision on the corpus) — the sharing
    /// tie-break.
    /// Structural fanout is fixed for the whole plan, so this is
    /// computed once at enumeration instead of per refinement visit
    /// (2 passes × up to MAX_CUTS cuts × every node).
    fan: u16,
    n: u8,
}

impl CutC {
    #[inline]
    fn leaves(&self) -> &[u32] {
        &self.leaves[..self.n as usize]
    }
}

/// One entry of the emission plan: give `node` a variable defined by its
/// cut function over the entry's leaves.
#[derive(Copy, Clone)]
pub struct PlanEntry {
    pub node: u32,
    leaf_off: u32,
    leaf_len: u32,
    cube_off: u32,
    on_len: u16,
    off_len: u16,
}

/// The emission plan as a flat arena (same shape as `ClauseArena` /
/// `WatchArena`): entries index into shared leaf and cube buffers, so a
/// plan of N nodes costs zero allocations once the buffers are warm
/// instead of 3N. Owned by the caller and reused across cones.
#[derive(Default)]
pub struct Plan {
    entries: Vec<PlanEntry>,
    leaves: Vec<u32>,
    cubes: Vec<Cube>,
}

impl Plan {
    fn clear(&mut self) {
        self.entries.clear();
        self.leaves.clear();
        self.cubes.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    pub fn entry(&self, i: usize) -> PlanEntry {
        self.entries[i]
    }

    /// AIG node indices of entry `i`'s cut leaves.
    #[inline]
    pub fn leaves(&self, e: PlanEntry) -> &[u32] {
        let s = e.leaf_off as usize;
        &self.leaves[s..s + e.leaf_len as usize]
    }

    /// Cubes implying the node (on-set cover of the cut function).
    #[inline]
    pub fn on_cubes(&self, e: PlanEntry) -> &[Cube] {
        let s = e.cube_off as usize;
        &self.cubes[s..s + e.on_len as usize]
    }

    /// Cubes implying the node's negation (off-set cover).
    #[inline]
    pub fn off_cubes(&self, e: PlanEntry) -> &[Cube] {
        let s = e.cube_off as usize + e.on_len as usize;
        &self.cubes[s..s + e.off_len as usize]
    }

    /// Total clauses the plan will emit — the arbitration metric.
    pub fn total_clauses(&self) -> u32 {
        self.cubes.len() as u32
    }
}

/// Per-cone mapping state, dense-indexed: dense id = rank of the node's
/// AIG index within the cone (AIG indices are topological — children
/// precede parents — so ascending dense order IS child-first order, and
/// leaf orderings match the AIG-index orderings the tie-breaks assume).
#[derive(Default)]
pub struct Mapper<const L: usize> {
    /// AIG idx → dense id for the current cone. Persistent so its
    /// capacity (and hashbrown's tables) survive across plans.
    dense: HashMap<u32, u32>,
    /// dense id → AIG node index, ascending.
    ids: Vec<u32>,
    is_leaf: Vec<bool>,
    /// Structural fanout within the cone (roots count one external
    /// consumer). Cuts whose leaves have high fanout duplicate less
    /// logic — the paper's decisive tie-break.
    fanout: Vec<u32>,
    /// Mapped-fanout counts of the current mapping (nFanouts(M, n)).
    refs: Vec<u32>,
    chosen: Vec<u8>,
    /// Priority cuts: `cuts[d * MAX_CUTS ..]`, `ncuts[d]` live. Cut 0 of
    /// every interior node is its trivial fanin cut, so the pre-mapping
    /// (today's Tseitin shape) is always representable.
    ncuts: Vec<u8>,
    cuts: Vec<CutC>,
    /// Area-flow estimate per node (refinement pass 1).
    flow: Vec<f64>,
    /// Epoch-stamped visited marks for the trial-area walks.
    mark: Vec<u32>,
    epoch: u32,
    /// Reused walk stack for ref/deref/trial-area (they run per node ×
    /// per candidate — a fresh Vec each call was 12% of mapping time).
    /// Holds bare node ids: every node reached by the walk uses its own
    /// `chosen` cut, so no index needs carrying.
    walk: Vec<u32>,
    /// Reused DFS postorder / roots / candidate buffers.
    post: Vec<u32>,
    droots: Vec<u32>,
    /// Reused emission-walk buffers (see `emit_plan`).
    order: Vec<u32>,
    emit_stack: Vec<u32>,
    emit_state: Vec<u8>,
    /// Reused cut-enumeration buffers (candidate set + ISOP scratch).
    cand: Vec<CutC>,
    scratch: Vec<Cube>,
}

/// How hard the mapper searches for a cover.
///
/// `Fast` (the default) keeps 2 priority cuts per node and stops after
/// the area-flow pass; `Full` keeps [`MAX_CUTS`] and adds the
/// exact-local-area pass. Fast is measurably FREE on real instances —
/// emission is byte-identical on every corpus instance tested, because
/// tree-shaped symbex cones have almost no reconvergence and their
/// candidate sets collapse under dedup — for ~40% less mapping time.
/// Full pays only on dense arithmetic: 32×32 multiplier arrays keep
/// ~10% more variables under Fast, ITE chains ~18% more.
///
/// This is a whole-solver choice on purpose. Choosing per cone from its
/// reconvergence was implemented and measured WORSE than either uniform
/// setting (bench_13728 conflicts 7132 → 10850): mixing effort levels
/// fragments the learned-clause vocabulary across shared subfunctions,
/// the same failure mode as per-cone encoder arbitration and the aig2
/// substitution arc.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Effort {
    #[default]
    Fast,
    Full,
}

impl<const L: usize> Mapper<L> {
    /// Plan the emission of every node in `roots`' cones, where
    /// `is_leaf(n)` says a node already has a SAT binding (or is an
    /// input/constant) and needs no emission. Returns the plan in
    /// child-first order; every root is guaranteed active.
    pub fn plan(
        &mut self,
        aig: &Aig,
        roots: &[u32],
        is_leaf: impl Fn(u32) -> bool,
        cache: &mut IsopCache,
        out: &mut Plan,
    ) {

        out.clear();
        self.collect_cone(aig, roots, &is_leaf);
        if self.ids.is_empty() {
            return;
        }
        let n = self.ids.len();
        // Reset per-plan state; buffer capacity persists across plans
        // (`mark`/`epoch` are stamp-based and survive untouched).
        self.fanout.clear();
        self.fanout.resize(n, 0);
        self.refs.clear();
        self.refs.resize(n, 0);
        self.chosen.clear();
        self.chosen.resize(n, 0);
        self.ncuts.clear();
        self.ncuts.resize(n, 0);
        self.cuts.clear();
        self.cuts.resize(
            n * L,
            CutC { leaves: [0; MAX_K], table: TT_ZERO, sig: 0, nclauses: u32::MAX, fan: 0, n: 0 },
        );
        self.flow.clear();
        self.flow.resize(n, 0.0);
        if self.mark.len() < n {
            self.mark.resize(n, 0);
        }
        let mut droots = std::mem::take(&mut self.droots);
        droots.clear();
        droots.extend(roots.iter().map(|r| self.dense[r]));
        // Interior processing order: the DFS postorder of the collection
        // walk (child-first). Any topological order is correct, but the
        // exact-local-area pass is path-dependent through its live ref
        // counts, and postorder measurably beats ascending-index order.
        let mut dpost = std::mem::take(&mut self.post);
        for a in dpost.iter_mut() {
            *a = self.dense[a];
        }
        self.count_fanout(aig, &droots);
        self.enumerate_cuts(aig, &dpost, cache);
        self.init_refs(&droots);
        // Refinement: one area-flow pass, then one exact-local-area pass,
        // both maintaining the mapping's reference counts incrementally
        // (activate/inactivate) so sharing decisions see live state.
        // Effort is encoded in the stride: the exact-local-area pass
        // only runs for the wide (Full) mapper, and `L` being a type
        // parameter keeps the per-node cut array exactly as large as
        // the effort needs — the array is the mapper's hottest
        // structure, and halving its stride measured ~6%.
        self.refine(&dpost, false);
        if L > FAST_CUTS {
            self.refine(&dpost, true);
        }
        self.emit_plan(&droots, cache, out);
        self.droots = droots;
        self.post = dpost;
    }


    fn count_fanout(&mut self, aig: &Aig, droots: &[u32]) {
        for &r in droots {
            self.fanout[r as usize] += 1;
        }
        for d in 0..self.ids.len() {
            if self.is_leaf[d] {
                continue;
            }
            if let AigNode::And(a, b) = aig.node(self.ids[d]) {
                self.fanout[self.dense[&a.node_idx()] as usize] += 1;
                self.fanout[self.dense[&b.node_idx()] as usize] += 1;
            }
        }
    }

    /// Bottom-up priority-cut enumeration with truth tables and clause
    /// costs. Ascending dense order is child-first by construction.
    fn enumerate_cuts(&mut self, aig: &Aig, dpost: &[u32], cache: &mut IsopCache) {
        // Candidate set + ISOP scratch, both reused across plans.
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut cand = std::mem::take(&mut self.cand);
        cand.reserve(L * L + 1);
        for &dd in dpost {
            let d = dd as usize;
            let AigNode::And(a, b) = aig.node(self.ids[d]) else { unreachable!() };
            let (da, db) = (self.dense[&a.node_idx()], self.dense[&b.node_idx()]);
            let (na, nb) = (a.is_negated(), b.is_negated());
            cand.clear();
            // Trivial fanin cut first — always kept at slot 0.
            let ta = trivial_of(da);
            let tb = trivial_of(db);
            if let Some((mut c, ma, mb)) = merge_union(&ta, &tb) {
                fill_cut(&mut c, &ta, na, ma, &tb, nb, mb, cache, &mut scratch);
                c.fan = avg_fanout(&c, &self.fanout);
                cand.push(c);
            }
            // Child cut sets: a leaf child contributes only its identity
            // cut; an interior child contributes its stored priority cuts
            // (whose slot 0 is its own trivial fanin cut).
            let ca = if self.is_leaf[da as usize] { 1 } else { self.ncuts[da as usize] };
            let cb = if self.is_leaf[db as usize] { 1 } else { self.ncuts[db as usize] };
            let leaf_a = self.is_leaf[da as usize];
            let leaf_b = self.is_leaf[db as usize];
            let triv_a = trivial_of(da);
            let triv_b = trivial_of(db);
            let base_a = da as usize * L;
            let base_b = db as usize * L;
            for ia in 0..ca {
                let cuta = if leaf_a { triv_a } else { self.cuts[base_a + ia as usize] };
                for ib in 0..cb {
                    // Read by index into the flat cut array; the copy is
                    // needed anyway for `fill_cut`'s borrow, but hoisting
                    // `cuta` keeps it to one per inner loop.
                    let cutb = if leaf_b { triv_b } else { self.cuts[base_b + ib as usize] };
                    // Union first (cheap); pay for tables/ISOP only for
                    // cuts that survive the width bound and dedup.
                    if let Some((mut c, ma, mb)) = merge_union(&cuta, &cutb) {
                        // Signature-first dedup: one u64 compare rejects
                        // almost every duplicate before touching leaves.
                        // (Full subsumption filtering — dropping cuts whose
                        // leaf set is a superset of another's — was tried
                        // here: identical mapping on every probe shape and
                        // ~8% SLOWER, since the O(k²) subset scan costs
                        // more than the cached ISOP lookups it avoids.)
                        if !cand
                            .iter()
                            .any(|e| e.sig == c.sig && e.leaves() == c.leaves())
                        {
                            fill_cut(
                                &mut c, &cuta, na, ma, &cutb, nb, mb, cache, &mut scratch,
                            );
                            c.fan = avg_fanout(&c, &self.fanout);
                            cand.push(c);
                        }
                    }
                }
            }
            // Order: fewer clauses first, ties broken by HIGHER average
            // leaf fanout (sharing-friendly), then smaller cut. The
            // trivial cut stays at slot 0 so the pre-mapping shape is
            // always representable.
            let base = d * L;
            self.cuts[base] = cand[0];
            // Packed sort keys in a parallel stack array: clauses ↑, avg
            // fanout ↓, size ↑ — keeps CutC lean and evaluates each key
            // exactly once.
            let rest = &cand[1..];
            let mut keys: [(u64, u8); MAX_CUTS * MAX_CUTS] =
                [(u64::MAX, 0); MAX_CUTS * MAX_CUTS];
            for (i, c) in rest.iter().enumerate() {
                let inv_fan = u16::MAX - c.fan;
                keys[i] = (
                    ((c.nclauses as u64) << 40) | ((inv_fan as u64) << 8) | c.n as u64,
                    i as u8,
                );
            }
            keys[..rest.len()].sort_unstable();
            let keep = rest.len().min(L - 1);
            for (slot, &(_, idx)) in keys[..keep].iter().enumerate() {
                self.cuts[base + 1 + slot] = rest[idx as usize];
            }
            self.ncuts[d] = (keep + 1) as u8;
        }
        self.scratch = scratch;
        self.cand = cand;
    }

    /// Build the mapped-fanout counts of the current `chosen` mapping by
    /// referencing every root's cone once.
    fn init_refs(&mut self, droots: &[u32]) {
        for &r in droots {
            if self.is_leaf[r as usize] {
                continue;
            }
            self.refs[r as usize] += 1;
            if self.refs[r as usize] == 1 {
                self.ref_cut(r, self.chosen[r as usize] as usize);
            }
        }
    }

    /// Reference a cut's leaves, recursively activating interiors that
    /// become newly used (their chosen cut gets referenced too).
    fn ref_cut(&mut self, d: u32, ci: usize) {
        // Field destructuring (not mem::take) gives the walk stack and
        // the counters independent borrows, and the cut is read by
        // reference — no 48-byte CutC copy per step, no Vec move per
        // call. Only the entry cut can differ from `chosen`; everything
        // the walk reaches uses its own, so the stack holds bare ids.
        // SAFETY (this walk and `deref_cut`): every id reached is a dense
        // cone id < ids.len() == refs.len() == is_leaf.len() ==
        // chosen.len(), and `id * L + chosen[id] < cuts.len()` because
        // chosen is always < ncuts <= L. `walk` is sized to the cone
        // below, and a node enters it at most once (its ref count
        // transitions 0→1 exactly once per walk).
        let Mapper { walk, refs, cuts, chosen, is_leaf, .. } = self;
        walk.clear();
        walk.reserve(refs.len());
        let mut cur = unsafe { cuts.get_unchecked(d as usize * L + ci) };
        loop {
            for k in 0..cur.n as usize {
                let li = unsafe { *cur.leaves.get_unchecked(k) } as usize;
                unsafe {
                    let r = refs.get_unchecked_mut(li);
                    *r += 1;
                    if *r == 1 && !*is_leaf.get_unchecked(li) {
                        push_unchecked(walk, li as u32);
                    }
                }
            }
            match walk.pop() {
                Some(m) => {
                    cur = unsafe {
                        cuts.get_unchecked(
                            m as usize * L + *chosen.get_unchecked(m as usize) as usize,
                        )
                    }
                }
                None => break,
            }
        }
    }

    /// Inverse of [`ref_cut`]: dereference leaves, recursively
    /// deactivating interiors whose count reaches zero.
    fn deref_cut(&mut self, d: u32, ci: usize) {
        let Mapper { walk, refs, cuts, chosen, is_leaf, .. } = self;
        walk.clear();
        walk.reserve(refs.len());
        let mut cur = unsafe { cuts.get_unchecked(d as usize * L + ci) };
        loop {
            for k in 0..cur.n as usize {
                let li = unsafe { *cur.leaves.get_unchecked(k) } as usize;
                unsafe {
                    let r = refs.get_unchecked_mut(li);
                    debug_assert!(*r > 0, "deref of unreferenced leaf");
                    *r -= 1;
                    if *r == 0 && !*is_leaf.get_unchecked(li) {
                        push_unchecked(walk, li as u32);
                    }
                }
            }
            match walk.pop() {
                Some(m) => {
                    cur = unsafe {
                        cuts.get_unchecked(
                            m as usize * L + *chosen.get_unchecked(m as usize) as usize,
                        )
                    }
                }
                None => break,
            }
        }
    }

    /// One bottom-up refinement pass over every cone node. `exact` false
    /// = area flow (global estimate), true = exact local area (cannot
    /// worsen the total). Active nodes swap their mapped cut in place via
    /// deref/ref so later decisions see live sharing.
    fn refine(&mut self, dpost: &[u32], exact: bool) {
        for &d in dpost {
            let di = d as usize;
            // Single-option nodes have nothing to decide — skipping them
            // avoids an ELA walk per node (the pass's dominant cost).
            if self.ncuts[di] <= 1 {
                if !exact {
                    let cut = &self.cuts[di * L];
                    let mut c = cut.nclauses as f64;
                    for &l in &cut.leaves[..cut.n as usize] {
                        let li = l as usize;
                        if !self.is_leaf[li] {
                            c += self.flow[li] / self.refs[li].max(1) as f64;
                        }
                    }
                    // Same bookkeeping the general path performs: the
                    // stored flow is the tie-adjusted cost. Dropping the
                    // tie term here shifts every ancestor's area-flow and
                    // silently flips near-ties (measured on the corpus).
                    self.flow[di] = (c - cut.fan as f64 * 1e-6).max(0.0);
                }
                continue;
            }
            let old = self.chosen[di] as usize;
            let active = self.refs[di] > 0;
            // ELA evaluates candidates with n's own contribution removed
            // (the paper's "first deactivated").
            if exact && active {
                self.deref_cut(d, old);
            }
            let mut best = old;
            let mut best_cost = f64::INFINITY;
            let base = di * L;
            for ci in 0..self.ncuts[di] as usize {
                let cut = &self.cuts[base + ci];
                if cut.nclauses == u32::MAX {
                    continue;
                }
                let (nclauses, tie) = (cut.nclauses, cut.fan as f64 * 1e-6);
                let cost = if exact {
                    self.trial_area(d, ci, best_cost + tie + 1.0)
                } else {
                    let cut = &self.cuts[base + ci];
                    let mut c = nclauses as f64;
                    for &l in &cut.leaves[..cut.n as usize] {
                        let li = l as usize;
                        if !self.is_leaf[li] {
                            c += self.flow[li] / self.refs[li].max(1) as f64;
                        }
                    }
                    c
                } - tie;
                if cost < best_cost {
                    best_cost = cost;
                    best = ci;
                }
            }
            self.chosen[di] = best as u8;
            self.flow[di] = best_cost.max(0.0);
            if exact && active {
                self.ref_cut(d, best);
            } else if !exact && active && best != old {
                self.deref_cut(d, old);
                self.ref_cut(d, best);
            }
        }
    }

    /// cost_ELA(n, C): clauses added by activating `n` with cut `C`
    /// given the current (n-deactivated) reference counts — C's own
    /// clauses plus, recursively, the chosen-cut clauses of currently
    /// unreferenced interior leaves. Non-mutating; epoch-stamped marks.
    fn trial_area(&mut self, d: u32, ci: usize, bound: f64) -> f64 {
        self.epoch += 1;
        let epoch = self.epoch;
        let Mapper { walk, refs, cuts, chosen, is_leaf, mark, .. } = self;
        mark[d as usize] = epoch;
        let mut total = 0u64;
        walk.clear();
        let mut cur = &cuts[d as usize * L + ci];
        loop {
            total += cur.nclauses as u64;
            // The walk only adds cost — once past the caller's best, the
            // exact figure no longer matters.
            if total as f64 >= bound {
                return f64::INFINITY;
            }
            for &l in &cur.leaves[..cur.n as usize] {
                let li = l as usize;
                if is_leaf[li] || mark[li] == epoch {
                    continue;
                }
                if refs[li] == 0 {
                    mark[li] = epoch;
                    walk.push(l);
                }
            }
            match walk.pop() {
                Some(m) => {
                    cur = &cuts[m as usize * L + chosen[m as usize] as usize]
                }
                None => break,
            }
        }
        total as f64
    }

    /// Walk the final mapping from the roots and produce the emission
    /// plan in child-first order.
    fn emit_plan(&mut self, droots: &[u32], cache: &mut IsopCache, out: &mut Plan) {
        // Postorder over the chosen mapping, reusing the walk buffers.
        // `emit_state`: 0 = unseen, 1 = open (exit entry pending), 2 = done.
        let mut order = std::mem::take(&mut self.order);
        let mut stack = std::mem::take(&mut self.emit_stack);
        order.clear();
        stack.clear();
        self.emit_state.clear();
        self.emit_state.resize(self.ids.len(), 0);
        for &r in droots {
            if !self.is_leaf[r as usize] {
                stack.push(r << 1);
            }
        }
        // Low bit of the stack word is the exit flag — keeps the stack a
        // flat Vec<u32> instead of Vec<(u32, bool)> (8 bytes with padding).
        while let Some(w) = stack.pop() {
            let d = w >> 1;
            let di = d as usize;
            if w & 1 != 0 {
                self.emit_state[di] = 2;
                order.push(d);
                continue;
            }
            if self.emit_state[di] != 0 {
                continue;
            }
            self.emit_state[di] = 1;
            stack.push((d << 1) | 1);
            let cut = &self.cuts[di * L + self.chosen[di] as usize];
            for &l in &cut.leaves[..cut.n as usize] {
                if !self.is_leaf[l as usize] && self.emit_state[l as usize] == 0 {
                    stack.push(l << 1);
                }
            }
        }
        for &d in order.iter() {
            let di = d as usize;
            let cut = self.cuts[di * L + self.chosen[di] as usize];
            let leaf_off = out.leaves.len() as u32;
            for &l in &cut.leaves[..cut.n as usize] {
                out.leaves.push(self.ids[l as usize]);
            }
            let cube_off = out.cubes.len() as u32;
            let (on_len, off_len) =
                cache.cubes_into(cut.table, cut.n as usize, &mut out.cubes);
            out.entries.push(PlanEntry {
                node: self.ids[di],
                leaf_off,
                leaf_len: cut.n as u32,
                cube_off,
                on_len,
                off_len,
            });
        }
        self.order = order;
        self.emit_stack = stack;
    }
}

/// The packing claim above is load-bearing for mapper speed — assert it.
const _: () = assert!(std::mem::size_of::<CutC>() == 48);

/// Average leaf fanout of a cut, scaled ×64 and saturated to u16 —
/// higher is better (high-fanout leaves are natural sharing
/// boundaries, so cuts ending there duplicate less logic).
#[inline]
fn avg_fanout(c: &CutC, fanout: &[u32]) -> u16 {
    let sum: u32 = c.leaves().iter().map(|&l| fanout[l as usize]).sum();
    (sum * 64 / (c.n as u32).max(1)).min(u16::MAX as u32) as u16
}

/// The trivial single-leaf cut of a child node: identity function.
#[inline]
fn trivial_of(d: u32) -> CutC {
    let mut leaves = [0u32; MAX_K];
    leaves[0] = d;
    CutC {
        leaves,
        table: VAR_MASKS[0],
        sig: 1u64 << (d & 63),
        nclauses: 0,
        fan: 0,
        n: 1,
    }
}

/// Sorted leaf union of two child cuts (≤ MAX_K), with table/cost left
/// unfilled — [`fill_cut`] completes survivors. Also returns, for each
/// child, the bitmask of merged positions its own leaves occupy: the
/// union walk knows this for free, and it turns table expansion from a
/// search into a bit test per position.
fn merge_union(a: &CutC, b: &CutC) -> Option<(CutC, u8, u8)> {
    // Signature prefilter: a cheap lower bound on the union size.
    let sig = a.sig | b.sig;
    if sig.count_ones() > MAX_K as u32 {
        return None;
    }
    // Two-pointer union of sorted leaf lists.
    let (la, lb) = (a.leaves(), b.leaves());
    let mut leaves = [0u32; MAX_K];
    let (mut i, mut j, mut n) = (0usize, 0usize, 0usize);
    let (mut ma, mut mb) = (0u8, 0u8);
    while i < la.len() || j < lb.len() {
        if n == MAX_K {
            return None;
        }
        let bit = 1u8 << n;
        let next = match (la.get(i), lb.get(j)) {
            (Some(&x), Some(&y)) if x == y => {
                i += 1;
                j += 1;
                ma |= bit;
                mb |= bit;
                x
            }
            (Some(&x), Some(&y)) if x < y => {
                i += 1;
                ma |= bit;
                x
            }
            (Some(_), Some(&y)) => {
                j += 1;
                mb |= bit;
                y
            }
            (Some(&x), None) => {
                i += 1;
                ma |= bit;
                x
            }
            (None, Some(&y)) => {
                j += 1;
                mb |= bit;
                y
            }
            (None, None) => unreachable!(),
        };
        leaves[n] = next;
        n += 1;
    }
    Some((
        CutC { leaves, table: TT_ZERO, sig, nclauses: u32::MAX, fan: 0, n: n as u8 },
        ma,
        mb,
    ))
}

/// Complete a deduped union cut: conjoin the re-expressed child tables
/// and price the CNF through the cache.
fn fill_cut(
    c: &mut CutC,
    a: &CutC,
    neg_a: bool,
    ma: u8,
    b: &CutC,
    neg_b: bool,
    mb: u8,
    cache: &mut IsopCache,
    scratch: &mut Vec<Cube>,
) {
    let n = c.n as usize;
    let ra = expand_table(a.table, ma, n);
    let rb = expand_table(b.table, mb, n);
    let fa = if neg_a { tt_not(ra) } else { ra };
    let fb = if neg_b { tt_not(rb) } else { rb };
    c.table = tt_and(fa, fb);
    c.nclauses = cache.cost(c.table, c.n as usize, scratch);
}

impl<const L: usize> Mapper<L> {
    /// Depth-first cone collection below `roots`, stopping at leaves.
    /// Fills `ids` (AIG indices ascending = topological, children
    /// first), `is_leaf`, `post` (interior DFS postorder, AIG indices),
    /// and the `dense` translation map.
    fn collect_cone(&mut self, aig: &Aig, roots: &[u32], is_leaf: &impl Fn(u32) -> bool) {
        enum V {
            Enter(u32),
            Exit(u32),
        }
        // `dense` doubles as the visited set during collection (value 0,
        // patched to real dense ids after sorting); `is_leaf` is rebuilt
        // from a temporary bitset keyed by insertion into `ids`.
        self.dense.clear();
        self.post.clear();
        self.ids.clear();
        let mut leaves: Vec<u32> = Vec::new();
        let mut stack: Vec<V> = roots.iter().map(|&r| V::Enter(r)).collect();
        while let Some(v) = stack.pop() {
            match v {
                V::Enter(n) => {
                    if self.dense.contains_key(&n) {
                        continue;
                    }
                    self.dense.insert(n, 0);
                    let leaf = is_leaf(n) || !matches!(aig.node(n), AigNode::And(..));
                    if leaf {
                        leaves.push(n);
                        continue;
                    }
                    if let AigNode::And(a, b) = aig.node(n) {
                        stack.push(V::Exit(n));
                        stack.push(V::Enter(a.node_idx()));
                        stack.push(V::Enter(b.node_idx()));
                    }
                }
                V::Exit(n) => self.post.push(n),
            }
        }
        self.ids.extend(self.dense.keys().copied());
        self.ids.sort_unstable();
        for (d, &a) in self.ids.iter().enumerate() {
            *self.dense.get_mut(&a).unwrap() = d as u32;
        }
        self.is_leaf.clear();
        self.is_leaf.resize(self.ids.len(), false);
        for l in leaves {
            self.is_leaf[self.dense[&l] as usize] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_eval(cubes: &[Cube], row: usize) -> bool {
        cubes.iter().any(|c| {
            (0..MAX_K).all(|i| {
                let bit = row & (1usize << i) != 0;
                !(c.pos & (1 << i) != 0 && !bit) && !(c.neg & (1 << i) != 0 && bit)
            })
        })
    }

    fn rnd(x: &mut u64) -> u64 {
        *x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x
    }

    fn rnd_tt(x: &mut u64) -> Tt {
        rnd(x)
    }

    #[test]
    fn isop_matches_function_on_random_tables() {
        let mut x = 0x1234_5678_9abc_def0u64;
        for _ in 0..300 {
            let raw = rnd_tt(&mut x);
            for nvars in 1..=MAX_K {
                let f = pad_masked(raw, nvars);
                let mut cubes = Vec::new();
                if isop_cover(f, nvars, &mut cubes).is_none() {
                    continue; // bound hit — allowed
                }
                for row in 0..64usize {
                    assert_eq!(
                        brute_eval(&cubes, row),
                        tt_bit(f, row),
                        "isop mismatch f={f:#x} nvars={nvars} row={row}"
                    );
                }
            }
        }
    }

    #[test]
    fn cube_tables_match_definition() {
        let c = Cube { pos: 0b101, neg: 0b010 };
        let t = c.table();
        for row in 0..64usize {
            let want = (row & 1 != 0) && (row & 2 == 0) && (row & 4 != 0);
            assert_eq!(tt_bit(t, row), want);
        }
    }

    /// Row-loop reference for the word-op variable insertion.
    fn insert_var_reference(t: Tt, j: usize) -> Tt {
        let mut out = TT_ZERO;
        for row in 0..64usize {
            let src = ((row >> (j + 1)) << j) | (row & ((1usize << j) - 1));
            if tt_bit(t, src) {
                out |= 1 << row;
            }
        }
        out
    }

    #[test]
    fn insert_var_matches_reference() {
        let mut x = 0xfeed_beef_dead_c0deu64;
        for _ in 0..200 {
            let t = rnd_tt(&mut x);
            for j in 0..MAX_K {
                assert_eq!(
                    insert_var_at(t, j),
                    insert_var_reference(t, j),
                    "insert mismatch t={t:?} j={j}"
                );
            }
        }
    }

    #[test]
    fn pad_and_cofactor_agree_with_rows() {
        let mut x = 0xabad_1dea_0u64;
        for _ in 0..100 {
            let raw = rnd_tt(&mut x);
            for nvars in 1..=MAX_K {
                let f = pad_masked(raw, nvars);
                for i in 0..nvars {
                    for hi in [false, true] {
                        let c = tt_cofactor(f, i, hi);
                        for row in 0..64usize {
                            let fixed = if hi { row | (1 << i) } else { row & !(1 << i) };
                            assert_eq!(
                                tt_bit(c, row),
                                tt_bit(f, fixed),
                                "cofactor mismatch nvars={nvars} i={i} hi={hi} row={row}"
                            );
                        }
                    }
                }
            }
        }
    }
}
