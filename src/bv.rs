//! Bitvector and Boolean term representation for the SMT layer.
//!
//! Terms form two parallel DAGs — one over bitvector-sorted expressions and
//! one over boolean-sorted expressions. Each is stored in a flat `Vec` of
//! nodes; a `BvTerm` / `BoolTerm` is just an opaque index.
//!
//! Hash consing is enabled: building the same structural term twice returns
//! the same `BvTerm`/`BoolTerm` so shared subterms are shared in the arena.
//! Without this the bitblaster would still avoid re-encoding (it has its own
//! cache), but hash consing additionally keeps the term DAG small, which
//! matters for large bitblasted formulas typical of symex.
//!
//! Supported widths: 1..=[`MAX_BV_WIDTH`] (currently 65536). Constants up to
//! 128 bits live inline in the `BvNode` as a `u128` (the fast path); wider
//! constants spill into a separate limb pool on the context, so the common
//! case stays zero-overhead while still handling the occasional 256-bit or
//! larger BV that shows up in real SMT-LIB queries.

use rustc_hash::FxHashMap as HashMap;

/// Maximum supported bitvector width. Widths up to this are fully handled;
/// beyond it we reject at build time to avoid runaway allocations on
/// malformed input. If you need more, raise this — the data structures are
/// not intrinsically bounded.
pub const MAX_BV_WIDTH: u32 = 1 << 16; // 64K bits

/// Sentinel value for `BvNode::wide`: this constant lives inline in the
/// `value` field (width ≤ 128), nothing to look up in the wide pool.
pub const WIDE_NONE: u32 = u32::MAX;

/// Opaque handle to a bitvector-sorted term. `Ord` follows creation order
/// (arena index) — used by the normalization pass for canonical operand
/// ordering.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BvTerm(pub u32);

/// Opaque handle to a boolean-sorted term.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BoolTerm(pub u32);

/// The four relational operators whose constant-bounded forms we track for
/// the and-chain collapse rewrite (see `bool_and` / `extract_const_bound`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum CmpOp {
    Ult, // unsigned strict
    Ule, // unsigned non-strict
    Slt, // signed strict
    Sle, // signed non-strict
}

/// A classified constant-vs-variable comparison. The side the constant
/// appears on matters for direction: `KonstLhs(Slt)` means `k < v` (lower
/// bound on v), `KonstRhs(Slt)` means `v < k` (upper bound).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum BoundKind {
    KonstLhs(CmpOp),
    KonstRhs(CmpOp),
}

impl BoundKind {
    /// Return `true` if the bound expressed by `(kind, new_k)` subsumes the
    /// bound expressed by `(kind, old_k)` — i.e. the formula `(op, k_new, v)`
    /// implies `(op, k_old, v)`. When this holds, the old bound is redundant
    /// in the conjunction and can be dropped. `width` is the BV width so we
    /// can sign-extend for signed comparisons.
    fn is_tighter(&self, new_k: u128, old_k: u128, width: u32) -> bool {
        use BoundKind::*;
        use CmpOp::*;
        let sx = |v: u128| sign_extend_to_i128(v, width);
        match self {
            // k < v: larger k is tighter (lower bound rising).
            KonstLhs(Ult) | KonstLhs(Ule) => new_k > old_k,
            KonstLhs(Slt) | KonstLhs(Sle) => sx(new_k) > sx(old_k),
            // v < k: smaller k is tighter (upper bound falling).
            KonstRhs(Ult) | KonstRhs(Ule) => new_k < old_k,
            KonstRhs(Slt) | KonstRhs(Sle) => sx(new_k) < sx(old_k),
        }
    }
}

/// Shape of a BV term. Leaves (variables/constants) carry their data
/// externally; interior nodes carry references to their children (BvTerm
/// for bitvector children, BoolTerm for the condition of an ITE).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BvOp {
    /// Symbolic BV variable identified by a stable id (assigned at creation).
    Var(u32),
    /// Concrete BV literal; the value lives in `BvNode::value`.
    Const,

    // --- Bitwise ---
    Not(BvTerm),
    And(BvTerm, BvTerm),
    Or(BvTerm, BvTerm),
    Xor(BvTerm, BvTerm),

    // --- Arithmetic ---
    Add(BvTerm, BvTerm),
    Sub(BvTerm, BvTerm),
    /// Two's complement negation: `-x = ~x + 1`.
    Neg(BvTerm),
    Mul(BvTerm, BvTerm),
    /// Unsigned divide. `bvudiv(x, 0)` returns all-ones (per SMT-LIB).
    Udiv(BvTerm, BvTerm),
    /// Unsigned remainder. `bvurem(x, 0)` returns `x`.
    Urem(BvTerm, BvTerm),
    /// Signed divide (rounds toward zero). Follows SMT-LIB definition via
    /// the case split over sign bits.
    Sdiv(BvTerm, BvTerm),
    /// Signed remainder — sign follows the dividend.
    Srem(BvTerm, BvTerm),
    /// Signed modulo — sign follows the divisor.
    Smod(BvTerm, BvTerm),

    // --- Bit-counting ---
    /// Population count: number of 1 bits. Output is the same width as the
    /// input (capped at width, so the count always fits).
    Popcount(BvTerm),
    /// Count of leading zeros — number of zero bits before the highest set
    /// bit. `clz(0) = width`. Output is the same width as the input.
    Clz(BvTerm),
    /// Count of trailing zeros — number of zero bits after the lowest set
    /// bit. `ctz(0) = width`. Output is the same width as the input.
    Ctz(BvTerm),

    /// Rotate left by a symbolic amount, modulo the bit-width. `amount`
    /// has the same width as the value. For constant rotation amounts,
    /// use [`BvContext::bv_rotate_left`] which lowers to extract + concat.
    RotateLeft(BvTerm, BvTerm),
    /// Mirror of `RotateLeft`.
    RotateRight(BvTerm, BvTerm),

    // --- Shifts ---
    Shl(BvTerm, BvTerm),
    /// Logical shift right — fills high bits with 0.
    Lshr(BvTerm, BvTerm),
    /// Arithmetic shift right — fills high bits with the sign bit.
    Ashr(BvTerm, BvTerm),

    // --- Structural ---
    /// Bit-select: `Extract(x, high, low)` produces bits [low..=high] as a
    /// BV of width `high - low + 1`.
    Extract(BvTerm, u32, u32),
    /// `Concat(x, y)` places `x` in the high bits and `y` in the low bits.
    /// Result width = width(x) + width(y).
    Concat(BvTerm, BvTerm),
    /// Zero-extend by `n` bits. New width = width(x) + n.
    ZeroExtend(BvTerm, u32),
    /// Sign-extend by `n` bits, replicating the MSB.
    SignExtend(BvTerm, u32),

    // --- Conditional ---
    /// `Ite(c, t, e)` is `t` when `c` is true, `e` otherwise.
    Ite(BoolTerm, BvTerm, BvTerm),
    /// N-way select (a.k.a. φ-node for state-merging). Variable arity, so
    /// the operands live in `BvContext::select_tables[idx]` rather than
    /// inline. Semantically `Select(idx)` is the first matching branch:
    /// the value of the earliest `selectors[i]` that evaluates to true,
    /// or `default` if none match. See [`SelectTable`].
    Select(u32),
}

/// Variable-arity operand storage for `BvOp::Select`. Lives in
/// `BvContext::select_tables`; nodes reference it by index.
///
/// First-match semantics: the value is `values[i]` for the smallest `i`
/// with `selectors[i] == true`, or `default` if every selector is false.
/// When a caller knows the selectors are pairwise mutually exclusive,
/// first-match becomes equivalent to "pick the one true selector", and
/// [`SmtSolver::assert_mutually_exclusive`] can be called once to install
/// the O(N²) exclusion clauses so SAT propagation collapses the chain
/// into a single decision.
#[derive(Clone, Debug)]
pub struct SelectTable {
    pub selectors: Box<[BoolTerm]>,
    pub values: Box<[BvTerm]>,
    pub default: BvTerm,
}

/// One node in the BV term DAG. Width is always set. For constants:
///   - `wide == WIDE_NONE`:  `value` is the inline constant (width ≤ 128)
///   - `wide != WIDE_NONE`:  `value` is meaningless (always 0); the actual
///      limb content lives at `ctx.wide_values[wide as usize]`
///
/// Wide constants are interned — identical limb content produces the same
/// `wide` index — so two `BvNode`s that represent the same constant always
/// compare equal and hash the same, preserving hash-consing.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BvNode {
    pub op: BvOp,
    pub width: u32,
    pub value: u128,
    pub wide: u32,
}

/// Shape of a boolean term. Some variants bridge into the BV DAG via their
/// children — e.g. `Eq(a, b)` says "BV term `a` equals BV term `b`".
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BoolOp {
    True,
    False,
    Var(u32),
    Not(BoolTerm),
    And(BoolTerm, BoolTerm),
    Or(BoolTerm, BoolTerm),
    Implies(BoolTerm, BoolTerm),

    // --- BV bridges ---
    Eq(BvTerm, BvTerm),
    Ult(BvTerm, BvTerm),
    Ule(BvTerm, BvTerm),
    /// Signed less-than. Equivalent to `Ult(x ^ sign_bit, y ^ sign_bit)`.
    Slt(BvTerm, BvTerm),
    Sle(BvTerm, BvTerm),

    // --- Overflow predicates ---
    /// Unsigned-add carry-out: true iff `x + y` overflows as unsigned.
    UaddOverflow(BvTerm, BvTerm),
    /// Signed-add overflow: operands share a sign but result has the opposite.
    SaddOverflow(BvTerm, BvTerm),
    /// Unsigned-sub borrow: true iff `x - y` wraps (i.e., `x < y`).
    UsubOverflow(BvTerm, BvTerm),
    /// Signed-sub overflow: operands have different signs and result has
    /// the opposite sign of the minuend.
    SsubOverflow(BvTerm, BvTerm),
    /// Unsigned-mul overflow: true iff the full 2N-bit product has any
    /// high-bit set.
    UmulOverflow(BvTerm, BvTerm),
    /// Signed-mul overflow: the full 2N-bit sign-extended product differs
    /// from the sign-extended N-bit truncated product.
    SmulOverflow(BvTerm, BvTerm),
    /// Signed-negation overflow: `-x` overflows iff `x = INT_MIN`.
    NegOverflow(BvTerm),
    /// Signed-division overflow: `x / y` overflows iff `x = INT_MIN` and `y = -1`.
    SdivOverflow(BvTerm, BvTerm),
}

/// The central term arena + naming context for BV and Boolean expressions.
pub struct BvContext {
    pub bv_nodes: Vec<BvNode>,
    pub bool_nodes: Vec<BoolOp>,
    /// Width of each symbolic BV variable, keyed by variable id.
    pub bv_var_widths: Vec<u32>,
    /// Number of boolean variables allocated.
    pub num_bool_vars: u32,

    // Hash-consing tables: reuse an existing term when the structural key
    // already exists in the arena.
    bv_hashcons: HashMap<BvNode, BvTerm>,
    bool_hashcons: HashMap<BoolOp, BoolTerm>,

    /// Memoises the *rewrite* performed by [`bv_extract`], keyed by
    /// `(source term, high, low)` → result term. Hash-consing dedups
    /// the resulting nodes, but the extract-narrowing rule recurses
    /// into BOTH operands of `Add`/`Sub`/`Mul`/`And`/… — so on a term
    /// DAG with shared subterms (e.g. `x` reused across a chain of
    /// `lea [rax + rax*4]` multiply-by-5 steps), an un-memoised
    /// descent re-walks every root-to-leaf path, which is exponential
    /// in depth. Caching the per-`(term, slice)` result collapses that
    /// back to one visit per distinct subproblem. The arena is
    /// append-only and terms are immutable, so a cached extract stays
    /// valid for the context's lifetime.
    extract_cache: HashMap<(BvTerm, u32, u32), BvTerm>,

    /// Storage for wide constants (width > 128). Each entry is the limb
    /// representation of a unique constant value — little-endian, length
    /// exactly `ceil(width / 64)`. Indexed by `BvNode::wide`.
    wide_values: Vec<Box<[u64]>>,
    /// Intern table for `wide_values` — identical limb content maps to the
    /// same index so hash-consing stays correct.
    wide_interner: HashMap<Box<[u64]>, u32>,

    /// Variable-arity operand storage for [`BvOp::Select`] nodes. Each node's
    /// `Select(idx)` references the table at `select_tables[idx as usize]`.
    /// Not hash-consed — merge-style Selects have distinct operand lists by
    /// construction, and scanning every prior table on each build would be
    /// O(N·M) for no measurable dedup gain.
    pub select_tables: Vec<SelectTable>,

    /// Cumulative count of addends cancelled by `norm_eq_add` (terms whose
    /// coefficients zeroed out when the two sides of an equality merged).
    /// Read by the SMT layer's normalization-acceptance heuristic.
    pub norm_cancelled: u64,

    /// Cumulative count of coefficient merges during add-chain flattening
    /// (an addend occurring again while flattening — `x + ... + x` folding
    /// to `2·x`). Second signal for normalization acceptance.
    pub norm_merged: u64,

    /// Parallel to [`bv_nodes`]: for each term, `(known_ones, known_zeros)`
    /// — a 3-valued abstraction of the term's bits. Bit i of `known_ones`
    /// is set iff bit i of the term is provably 1 under every satisfying
    /// assignment; likewise `known_zeros` for provable zeros. Unknown bits
    /// have 0 in both masks. We track these at width ≤ 128 only; for wider
    /// BVs the entry is `(0, 0)` (no info).
    ///
    /// The invariant `known_ones & known_zeros == 0` always holds — a bit
    /// can't be both proved-1 and proved-0. Sometimes we also have the
    /// strong form `known_ones | known_zeros == mask(width)`, meaning the
    /// whole value is forced at construction time; constructors watch for
    /// this and fold straight to a constant.
    pub known_bits: Vec<(u128, u128)>,
}

impl BvContext {
    pub fn new() -> Self {
        BvContext {
            bv_nodes: Vec::new(),
            bool_nodes: Vec::new(),
            bv_var_widths: Vec::new(),
            num_bool_vars: 0,
            bv_hashcons: HashMap::default(),
            bool_hashcons: HashMap::default(),
            extract_cache: HashMap::default(),
            wide_values: Vec::new(),
            wide_interner: HashMap::default(),
            select_tables: Vec::new(),
            norm_cancelled: 0,
            norm_merged: 0,
            known_bits: Vec::new(),
        }
    }

    /// `(known_ones, known_zeros)` for a term's bits. See [`known_bits`].
    #[inline]
    pub fn bv_known_bits(&self, t: BvTerm) -> (u128, u128) {
        self.known_bits[t.0 as usize]
    }

    /// Intern the given limbs (must already be canonicalised: length =
    /// ceil(width/64), high bits masked to fit width). Returns the dedup'd
    /// `wide` index.
    fn intern_wide(&mut self, limbs: Box<[u64]>) -> u32 {
        if let Some(&idx) = self.wide_interner.get(&limbs) {
            return idx;
        }
        let idx = self.wide_values.len() as u32;
        assert!(idx != WIDE_NONE, "too many distinct wide constants");
        self.wide_values.push(limbs.clone());
        self.wide_interner.insert(limbs, idx);
        idx
    }

    /// Access the limb representation of a wide constant. Panics if `idx`
    /// is out of range; caller is expected to have a valid `wide` from a
    /// real `BvNode`.
    pub fn wide_limbs(&self, idx: u32) -> &[u64] {
        &self.wide_values[idx as usize]
    }

    // ---------- BV term builders ----------

    /// Allocate a fresh symbolic BV variable of the given width.
    pub fn bv_var(&mut self, width: u32) -> BvTerm {
        assert!(
            (1..=MAX_BV_WIDTH).contains(&width),
            "BV width must be 1..={}",
            MAX_BV_WIDTH
        );
        let var_id = self.bv_var_widths.len() as u32;
        self.bv_var_widths.push(width);
        self.push_bv(BvOp::Var(var_id), width, 0)
    }

    /// Build a BV literal at widths ≤ 128 (the fast path). Value is masked
    /// to `width` bits. For widths beyond 128, use [`bv_const_wide`].
    pub fn bv_const(&mut self, value: u128, width: u32) -> BvTerm {
        assert!(
            (1..=MAX_BV_WIDTH).contains(&width),
            "BV width must be 1..={}",
            MAX_BV_WIDTH
        );
        if width <= 128 {
            let v = value & mask(width);
            self.push_bv(BvOp::Const, width, v)
        } else {
            // Widen the u128 into a full-width limb array.
            let nlimbs = ((width as usize) + 63) / 64;
            let mut limbs = vec![0u64; nlimbs];
            limbs[0] = value as u64;
            limbs[1] = (value >> 64) as u64;
            // Mask the top limb to zero out bits above `width`.
            mask_top_limb(&mut limbs, width);
            let idx = self.intern_wide(limbs.into_boxed_slice());
            self.push_bv_wide(BvOp::Const, width, idx)
        }
    }

    /// Build a BV literal at any width from little-endian limb input.
    /// `limbs` must have length exactly `ceil(width / 64)` (pad with zeros).
    /// Any bits set above `width` are masked away.
    pub fn bv_const_wide(&mut self, limbs: &[u64], width: u32) -> BvTerm {
        assert!(
            (1..=MAX_BV_WIDTH).contains(&width),
            "BV width must be 1..={}",
            MAX_BV_WIDTH
        );
        let nlimbs = ((width as usize) + 63) / 64;
        assert_eq!(
            limbs.len(),
            nlimbs,
            "bv_const_wide: expected {} limbs for width {}, got {}",
            nlimbs,
            width,
            limbs.len()
        );
        if width <= 128 {
            // Funnel through the fast path for narrow inputs.
            let mut v: u128 = limbs[0] as u128;
            if limbs.len() >= 2 {
                v |= (limbs[1] as u128) << 64;
            }
            return self.bv_const(v, width);
        }
        let mut limbs_v: Vec<u64> = limbs.to_vec();
        mask_top_limb(&mut limbs_v, width);
        let idx = self.intern_wide(limbs_v.into_boxed_slice());
        self.push_bv_wide(BvOp::Const, width, idx)
    }

    pub fn bv_not(&mut self, x: BvTerm) -> BvTerm {
        let w = self.width_of(x);
        // Constant fold (only for widths ≤ 128 — wide constants fall through
        // to the symbolic path, losing the `not` optimization but preserving
        // correctness).
        if let Some(v) = self.const_val(x) {
            return self.bv_const(!v, w);
        }
        // Double-negation.
        if let BvOp::Not(inner) = self.bv_op(x) {
            return inner;
        }
        // Push `bvnot` through an `ite` whose branches are both constants:
        // `~ite(c, k1, k2)` = `ite(c, ~k1, ~k2)`. This keeps the result a
        // "boolean selector" shape that `bv_eq(const, ...)` can collapse.
        if let BvOp::Ite(c, t, e) = self.bv_op(x) {
            if let (Some(vt), Some(ve)) = (self.const_val(t), self.const_val(e)) {
                let nt = self.bv_const(!vt, w);
                let ne = self.bv_const(!ve, w);
                return self.bv_ite(c, nt, ne);
            }
        }
        self.push_bv(BvOp::Not(x), w, 0)
    }

    pub fn bv_and(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return x; // x & x = x
        }
        let (x, y) = self.canonicalize_commutative(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return self.bv_const(vx & vy, w);
        }
        if let Some(vy) = self.const_val(y) {
            if vy & mask(w) == 0 {
                return self.bv_const(0, w); // x & 0 = 0
            }
            if vy & mask(w) == mask(w) {
                return x; // x & ~0 = x
            }
            // Associative rollup: (a & c) & c' = a & (c & c').
            if let BvOp::And(a, b) = self.bv_op(x) {
                if let Some(vb) = self.const_val(b) {
                    let merged = self.bv_const(vb & vy, w);
                    return self.bv_and(a, merged);
                }
            }
            // NOTE: a constant-mask structural rewrite (concat tree of
            // extracts + zero-consts) was tried here and reverted — the
            // bitblaster's `mk_and` already short-circuits T/F operands at
            // the literal level, so the term-level rewrite is redundant
            // and costs preprocessing time on every formula.
        }
        self.push_bv(BvOp::And(x, y), w, 0)
    }

    pub fn bv_or(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return x;
        }
        let (x, y) = self.canonicalize_commutative(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return self.bv_const(vx | vy, w);
        }
        if let Some(vy) = self.const_val(y) {
            if vy & mask(w) == 0 {
                return x; // x | 0 = x
            }
            if vy & mask(w) == mask(w) {
                return self.bv_const(mask(w), w); // x | ~0 = ~0
            }
            // Associative rollup: (a | c) | c' = a | (c | c').
            if let BvOp::Or(a, b) = self.bv_op(x) {
                if let Some(vb) = self.const_val(b) {
                    let merged = self.bv_const(vb | vy, w);
                    return self.bv_or(a, merged);
                }
            }
            // NOTE: see bv_and — `mk_or` already short-circuits T/F at the
            // literal level, so a term-level constant-mask rewrite was
            // tried here and reverted as redundant.
        }
        self.push_bv(BvOp::Or(x, y), w, 0)
    }

    pub fn bv_xor(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return self.bv_const(0, w); // x ^ x = 0
        }
        let (x, y) = self.canonicalize_commutative(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return self.bv_const(vx ^ vy, w);
        }
        if let Some(vy) = self.const_val(y) {
            let vy_m = vy & mask(w);
            if vy_m == 0 {
                return x; // x ^ 0 = x
            }
            if vy_m == mask(w) {
                return self.bv_not(x); // x ^ ~0 = ~x
            }
            // Associative rollup: (a ^ c) ^ c' = a ^ (c ^ c').
            if let BvOp::Xor(a, b) = self.bv_op(x) {
                if let Some(vb) = self.const_val(b) {
                    let merged = self.bv_const(vb ^ vy_m, w);
                    return self.bv_xor(a, merged);
                }
            }
            // NOTE: a constant-mask structural rewrite (per-bit pass-through
            // or invert via `bvnot`) was tried here but destroys the
            // `BvOp::Xor` shape that downstream `(a ^ c) ^ c'` chains rely
            // on for canonicalization. The bitblaster already specialises
            // `BvOp::Xor(x, const)` to single-lit ops per bit, so the BV-
            // layer rewrite buys nothing.
        }
        self.push_bv(BvOp::Xor(x, y), w, 0)
    }

    pub fn bv_add(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        let (x, y) = self.canonicalize_commutative(x, y);
        // Both constants → fold.
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return self.bv_const(vx.wrapping_add(vy), w);
        }
        // x + 0 = x (constant is on the right after canonicalization).
        if let Some(vy) = self.const_val(y) {
            if vy == 0 {
                return x;
            }
            // Associative rollup: (a + c) + c' = a + (c + c').
            if let BvOp::Add(a, b) = self.bv_op(x) {
                if let Some(vb) = self.const_val(b) {
                    let merged = self.bv_const(vb.wrapping_add(vy), w);
                    return self.bv_add(a, merged);
                }
            }
            // (a - c) + c' = a + (c' - c).
            if let BvOp::Sub(a, b) = self.bv_op(x) {
                if let Some(vb) = self.const_val(b) {
                    let merged = self.bv_const(vy.wrapping_sub(vb), w);
                    return self.bv_add(a, merged);
                }
            }
        }
        self.push_bv(BvOp::Add(x, y), w, 0)
    }

    pub fn bv_sub(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return self.bv_const(0, w);
        }
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return self.bv_const(vx.wrapping_sub(vy), w);
        }
        if let Some(vy) = self.const_val(y) {
            if vy == 0 {
                return x;
            }
            // Normalise `x - c` to `x + (-c)` so associative rollup kicks in
            // and both sub- and add-chains share a single canonical shape.
            let neg_c = self.bv_const(0u128.wrapping_sub(vy), w);
            return self.bv_add(x, neg_c);
        }
        self.push_bv(BvOp::Sub(x, y), w, 0)
    }

    /// Two's-complement negation: `-x`.
    pub fn bv_neg(&mut self, x: BvTerm) -> BvTerm {
        let w = self.width_of(x);
        if let Some(vx) = self.const_val(x) {
            return self.bv_const(0u128.wrapping_sub(vx), w);
        }
        // -(-x) = x
        if let BvOp::Neg(inner) = self.bv_op(x) {
            return inner;
        }
        self.push_bv(BvOp::Neg(x), w, 0)
    }

    /// Population count of `x`'s bits. Output width equals input width
    /// (so the result is always representable). Bitblasts to a balanced
    /// divide-and-conquer adder tree — `O(W log W)` gates vs the naive
    /// chained sum's `O(W²)`.
    pub fn bv_popcount(&mut self, x: BvTerm) -> BvTerm {
        let w = self.width_of(x);
        if let Some(vx) = self.const_val(x) {
            return self.bv_const((vx & mask(w)).count_ones() as u128, w);
        }
        // For width 1 the popcount is just the bit itself.
        if w == 1 {
            return x;
        }
        self.push_bv(BvOp::Popcount(x), w, 0)
    }

    /// Count leading zeros — number of zero bits before the highest set
    /// bit. `bv_clz(0)` is `width`. Output width equals input width.
    pub fn bv_clz(&mut self, x: BvTerm) -> BvTerm {
        let w = self.width_of(x);
        if let Some(vx) = self.const_val(x) {
            let masked = vx & mask(w);
            let r = if masked == 0 {
                w as u128
            } else {
                (masked.leading_zeros() - (128 - w)) as u128
            };
            return self.bv_const(r, w);
        }
        if w == 1 {
            // 1-bit clz: 0 if bit set, 1 otherwise. Equivalent to bvnot(x).
            return self.bv_not(x);
        }
        self.push_bv(BvOp::Clz(x), w, 0)
    }

    /// Count trailing zeros — number of zero bits after the lowest set
    /// bit. `bv_ctz(0)` is `width`. Output width equals input width.
    pub fn bv_ctz(&mut self, x: BvTerm) -> BvTerm {
        let w = self.width_of(x);
        if let Some(vx) = self.const_val(x) {
            let masked = vx & mask(w);
            let r = if masked == 0 {
                w as u128
            } else {
                masked.trailing_zeros() as u128
            };
            return self.bv_const(r, w);
        }
        if w == 1 {
            return self.bv_not(x);
        }
        self.push_bv(BvOp::Ctz(x), w, 0)
    }

    /// Rotate `x` left by a symbolic `amount`. Both terms must have the
    /// same width. The amount is interpreted modulo the width. Folds to
    /// a constant rotate when `amount` is constant; bitblasts via a
    /// log-tree of conditional constant rotations for power-of-2 widths,
    /// and falls back to `urem` + shifts otherwise.
    pub fn bv_rotate_left_dyn(&mut self, x: BvTerm, amount: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, amount);
        if let (Some(vx), Some(vamt)) = (self.const_val(x), self.const_val(amount)) {
            let m = mask(w);
            let a = (vamt % w as u128) as u32;
            let v = vx & m;
            let r = if a == 0 { v } else { ((v << a) | (v >> (w - a))) & m };
            return self.bv_const(r, w);
        }
        if let Some(vamt) = self.const_val(amount) {
            let a = (vamt % w as u128) as u32;
            return self.bv_rotate_left(x, a);
        }
        if w == 1 {
            // 1-bit value is unchanged by any rotation.
            return x;
        }
        self.push_bv(BvOp::RotateLeft(x, amount), w, 0)
    }

    /// Mirror of [`bv_rotate_left_dyn`].
    pub fn bv_rotate_right_dyn(&mut self, x: BvTerm, amount: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, amount);
        if let (Some(vx), Some(vamt)) = (self.const_val(x), self.const_val(amount)) {
            let m = mask(w);
            let a = (vamt % w as u128) as u32;
            let v = vx & m;
            let r = if a == 0 { v } else { ((v >> a) | (v << (w - a))) & m };
            return self.bv_const(r, w);
        }
        if let Some(vamt) = self.const_val(amount) {
            let a = (vamt % w as u128) as u32;
            return self.bv_rotate_right(x, a);
        }
        if w == 1 {
            return x;
        }
        self.push_bv(BvOp::RotateRight(x, amount), w, 0)
    }

    pub fn bv_mul(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        let (x, y) = self.canonicalize_commutative(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return self.bv_const(vx.wrapping_mul(vy), w);
        }
        if let Some(vy) = self.const_val(y) {
            let vy_m = vy & mask(w);
            if vy_m == 0 {
                return self.bv_const(0, w);
            }
            if vy_m == 1 {
                return x;
            }
            // Power-of-2 multiplicand → left shift, then free via the
            // constant-shift wiring path in the bitblaster.
            if let Some(k) = power_of_two_exp(vy_m) {
                let amt = self.bv_const(k as u128, w);
                return self.bv_shl(x, amt);
            }
            // Associative rollup: (a * c) * c' = a * (c * c').
            if let BvOp::Mul(a, b) = self.bv_op(x) {
                if let Some(vb) = self.const_val(b) {
                    let merged = self.bv_const(vb.wrapping_mul(vy), w);
                    return self.bv_mul(a, merged);
                }
            }
        }
        self.push_bv(BvOp::Mul(x, y), w, 0)
    }

    /// Unsigned divide. By SMT-LIB convention, `x / 0 = ~0` (all ones).
    pub fn bv_udiv(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let vy_masked = vy & mask(w);
            let result = if vy_masked == 0 {
                mask(w) // ~0
            } else {
                (vx & mask(w)) / vy_masked
            };
            return self.bv_const(result, w);
        }
        if let Some(vy) = self.const_val(y) {
            let vy_m = vy & mask(w);
            if vy_m == 1 {
                return x; // x / 1 = x
            }
            // Divide by power of two → logical right shift.
            if let Some(k) = power_of_two_exp(vy_m) {
                let amt = self.bv_const(k as u128, w);
                return self.bv_lshr(x, amt);
            }
            // Granlund-Montgomery: reduce `x / d` to a constant-operand
            // multiply + shift when the magic constants fit. For most
            // real-world constant divisors (small `d`, `w` ≤ 64) this
            // replaces an O(N²) restoring division with a much cheaper
            // sparse multiply — the multiplier's popcount is typically
            // around `w/2`, and each 1-bit gets a single ripple-carry add.
            if let Some((m_lo, l)) = unsigned_magic(vy_m, w) {
                return self.bv_udiv_by_magic(x, m_lo, l, w);
            }
        }
        self.push_bv(BvOp::Udiv(x, y), w, 0)
    }

    /// Lower `x / d` to the magic-number recipe
    /// `q = ((x - mulhi) >> 1 + mulhi) >> (l - 1)` where
    /// `mulhi = high-w-bits(x * m_lo)`. Expressed entirely in existing BV
    /// ops so rewrites + bitblaster specializations (especially
    /// constant-multiplicand) carry it the rest of the way.
    fn bv_udiv_by_magic(&mut self, x: BvTerm, m_lo: u128, l: u32, w: u32) -> BvTerm {
        // High w bits of (zero-extend x) * (zero-extend m_lo), both at 2w.
        let ext_w = 2 * w;
        let x_ext = self.bv_zero_extend(x, w);
        let m_const = self.bv_const(m_lo, ext_w);
        let prod = self.bv_mul(x_ext, m_const); // 2w bits
        let mulhi = self.bv_extract(prod, ext_w - 1, w); // high w bits → w bits

        // `(x - mulhi) >> 1`, then + mulhi, then >> (l - 1).
        let diff = self.bv_sub(x, mulhi);
        let one = self.bv_const(1, w);
        let halved = self.bv_lshr(diff, one);
        let t4 = self.bv_add(halved, mulhi);

        if l == 1 {
            // Shouldn't hit — the magic path is skipped for d=2 (power of two).
            return t4;
        }
        let shift_amt = self.bv_const((l - 1) as u128, w);
        self.bv_lshr(t4, shift_amt)
    }

    /// Unsigned remainder. By SMT-LIB convention, `x mod 0 = x`.
    pub fn bv_urem(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let vy_masked = vy & mask(w);
            let result = if vy_masked == 0 {
                vx & mask(w) // x mod 0 = x
            } else {
                (vx & mask(w)) % vy_masked
            };
            return self.bv_const(result, w);
        }
        if let Some(vy) = self.const_val(y) {
            let vy_masked = vy & mask(w);
            if vy_masked == 1 {
                return self.bv_const(0, w); // x mod 1 = 0
            }
            // Mod by power of two → low-bit mask.
            if let Some(_k) = power_of_two_exp(vy_masked) {
                let m = vy_masked - 1;
                let mask_c = self.bv_const(m, w);
                return self.bv_and(x, mask_c);
            }
            // Magic-number path: once udiv is cheap, urem follows as
            // `x - d * (x / d)`. The `d * ...` is a constant-multiplicand
            // multiplication which the bitblaster specializes to a sparse
            // shift-and-add.
            if let Some((m_lo, l)) = unsigned_magic(vy_masked, w) {
                let q = self.bv_udiv_by_magic(x, m_lo, l, w);
                let d_const = self.bv_const(vy_masked, w);
                let dq = self.bv_mul(d_const, q);
                return self.bv_sub(x, dq);
            }
        }
        self.push_bv(BvOp::Urem(x, y), w, 0)
    }

    /// Signed division (rounds toward zero). Follows SMT-LIB semantics —
    /// division by zero returns `1` if the dividend is signed-negative and
    /// `~0` otherwise (this falls out of the case split + bvudiv-by-zero
    /// convention and is done in the bitblaster).
    pub fn bv_sdiv(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            let result = if sy == 0 {
                if sx < 0 { 1 } else { -1i128 }
            } else {
                sx.wrapping_div(sy)
            };
            return self.bv_const(result as u128, w);
        }
        self.push_bv(BvOp::Sdiv(x, y), w, 0)
    }

    /// Signed remainder — the sign of the result follows the dividend.
    pub fn bv_srem(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            let result = if sy == 0 {
                sx
            } else {
                sx.wrapping_rem(sy)
            };
            return self.bv_const(result as u128, w);
        }
        self.push_bv(BvOp::Srem(x, y), w, 0)
    }

    /// Signed modulo — the sign of the result follows the divisor.
    pub fn bv_smod(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            let result = if sy == 0 {
                sx
            } else {
                // SMT-LIB bvsmod: the mathematical modulo with the divisor's
                // sign. Equivalent to `((x % y) + y) % y` for y != 0.
                let r = sx.wrapping_rem(sy);
                if r == 0 {
                    0
                } else if (r < 0) != (sy < 0) {
                    r.wrapping_add(sy)
                } else {
                    r
                }
            };
            return self.bv_const(result as u128, w);
        }
        self.push_bv(BvOp::Smod(x, y), w, 0)
    }

    pub fn bv_shl(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let result = if vy >= w as u128 { 0 } else { vx << vy };
            return self.bv_const(result, w);
        }
        if let Some(vy) = self.const_val(y) {
            if vy == 0 {
                return x;
            }
            // (x << c) << c' = x << (c + c'), saturating at width.
            if let BvOp::Shl(inner, amt) = self.bv_op(x) {
                if let Some(va) = self.const_val(amt) {
                    let total = va.saturating_add(vy).min(w as u128);
                    let combined = self.bv_const(total, w);
                    return self.bv_shl(inner, combined);
                }
            }
            // Constant-amount shift: replace the variable-shift MUX tree
            // with a structural concat. `x << c` is just `x`'s low (w-c)
            // bits placed in the high (w-c) positions, with c zero bits at
            // the bottom. Costs zero SAT gates (extract + const + concat
            // are all structural at bitblast time).
            if vy >= w as u128 {
                return self.bv_const(0, w);
            }
            let c = vy as u32;
            let zero_low = self.bv_const(0, c);
            let kept = self.bv_extract(x, w - 1 - c, 0);
            return self.bv_concat(kept, zero_low);
        }
        self.push_bv(BvOp::Shl(x, y), w, 0)
    }

    pub fn bv_lshr(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let result = if vy >= w as u128 { 0 } else { (vx & mask(w)) >> vy };
            return self.bv_const(result, w);
        }
        if let Some(vy) = self.const_val(y) {
            if vy == 0 {
                return x;
            }
            // (x >>L c) >>L c' = x >>L (c + c').
            if let BvOp::Lshr(inner, amt) = self.bv_op(x) {
                if let Some(va) = self.const_val(amt) {
                    let total = va.saturating_add(vy).min(w as u128);
                    let combined = self.bv_const(total, w);
                    return self.bv_lshr(inner, combined);
                }
            }
            // Constant-amount logical shift right: `x >>L c` is x's high
            // (w-c) bits placed in the low (w-c) positions, with c zero
            // bits at the top. Structural concat — zero SAT gates.
            if vy >= w as u128 {
                return self.bv_const(0, w);
            }
            let c = vy as u32;
            let zero_high = self.bv_const(0, c);
            let kept = self.bv_extract(x, w - 1, c);
            return self.bv_concat(zero_high, kept);
        }
        self.push_bv(BvOp::Lshr(x, y), w, 0)
    }

    /// Circular left shift by `shift` bits. Synthesized as
    /// `(x << k) | (x >>L (w - k))` where `k = shift mod w`. Equivalent to
    /// SMT-LIB `((_ rotate_left shift) x)`.
    pub fn bv_rotate_left(&mut self, x: BvTerm, shift: u32) -> BvTerm {
        let w = self.width_of(x);
        let k = shift % w;
        if k == 0 {
            return x;
        }
        let k_term = self.bv_const(k as u128, w);
        let left = self.bv_shl(x, k_term);
        let r_term = self.bv_const((w - k) as u128, w);
        let right = self.bv_lshr(x, r_term);
        self.bv_or(left, right)
    }

    /// Circular right shift by `shift` bits. Synthesized as
    /// `(x >>L k) | (x << (w - k))` where `k = shift mod w`. Equivalent to
    /// SMT-LIB `((_ rotate_right shift) x)`.
    pub fn bv_rotate_right(&mut self, x: BvTerm, shift: u32) -> BvTerm {
        let w = self.width_of(x);
        let k = shift % w;
        if k == 0 {
            return x;
        }
        let k_term = self.bv_const(k as u128, w);
        let right = self.bv_lshr(x, k_term);
        let l_term = self.bv_const((w - k) as u128, w);
        let left = self.bv_shl(x, l_term);
        self.bv_or(right, left)
    }

    pub fn bv_ashr(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let vx_masked = vx & mask(w);
            let sign = (vx_masked >> (w - 1)) & 1 == 1;
            let effective = vy.min(w as u128);
            let shifted = vx_masked >> effective;
            let result = if sign {
                // Fill the freshly-exposed high bits with 1s.
                let fill = if effective == 0 {
                    0
                } else {
                    mask(w) & !((1u128 << (w as u128 - effective)) - 1)
                };
                (shifted | fill) & mask(w)
            } else {
                shifted
            };
            return self.bv_const(result, w);
        }
        if let Some(vy) = self.const_val(y) {
            if vy == 0 {
                return x;
            }
            // Constant-amount arithmetic shift right: shift logically and
            // fill the top `c` bits with copies of the sign bit. For c >= w
            // the result is a sign-bit replication of full width. Both
            // forms are structural (extract + sign_extend) — zero gates.
            let c = (vy.min(w as u128)) as u32;
            if c >= w {
                let sign = self.bv_extract(x, w - 1, w - 1);
                return self.bv_sign_extend(sign, w - 1);
            }
            let kept = self.bv_extract(x, w - 1, c);
            return self.bv_sign_extend(kept, c);
        }
        self.push_bv(BvOp::Ashr(x, y), w, 0)
    }

    /// Extract bits `[low..=high]` of `x`. `high` must be in range and
    /// `low <= high`.
    pub fn bv_extract(&mut self, x: BvTerm, high: u32, low: u32) -> BvTerm {
        let wx = self.width_of(x);
        assert!(high < wx, "extract high={} out of range for width {}", high, wx);
        assert!(low <= high, "extract low={} > high={}", low, high);
        // Full-width extract is a no-op — cheap, skip the cache.
        if low == 0 && high + 1 == wx {
            return x;
        }
        // Memoise the (possibly recursive) rewrite. Without this the
        // extract-narrowing rule below re-descends shared subterms
        // exponentially; see `extract_cache`'s doc comment.
        let key = (x, high, low);
        if let Some(&cached) = self.extract_cache.get(&key) {
            return cached;
        }
        let result = self.bv_extract_uncached(x, high, low, wx);
        self.extract_cache.insert(key, result);
        result
    }

    /// The body of [`bv_extract`] — its rewrite rules, minus the
    /// memoisation wrapper. Recurses via the cached `bv_extract` so
    /// shared subterms are only rewritten once.
    fn bv_extract_uncached(&mut self, x: BvTerm, high: u32, low: u32, wx: u32) -> BvTerm {
        let new_w = high - low + 1;
        if let Some(vx) = self.const_val(x) {
            let shifted = (vx >> low) & mask(new_w);
            return self.bv_const(shifted, new_w);
        }
        // extract(extract(z, h', l'), high, low) =
        //   extract(z, l' + high, l' + low) — bits rebase onto the underlying
        // BV so multi-step extractions collapse into one.
        if let BvOp::Extract(inner, _inner_hi, inner_lo) = self.bv_op(x) {
            return self.bv_extract(inner, inner_lo + high, inner_lo + low);
        }
        // extract(concat(hi, lo), h, l):
        //   - if entirely within the low part: just extract from `lo`.
        //   - if entirely within the high part: extract from `hi` (bits
        //     rebased by lo's width).
        //   - otherwise: split the slice on the concat boundary, extract
        //     each half from its source, then concat. Each sub-extract
        //     gets folded by the rules above, so a chain of concat+extract
        //     collapses cleanly even across boundaries.
        if let BvOp::Concat(hi_t, lo_t) = self.bv_op(x) {
            let lo_w = self.width_of(lo_t);
            if high < lo_w {
                return self.bv_extract(lo_t, high, low);
            }
            if low >= lo_w {
                return self.bv_extract(hi_t, high - lo_w, low - lo_w);
            }
            let lo_part = self.bv_extract(lo_t, lo_w - 1, low);
            let hi_part = self.bv_extract(hi_t, high - lo_w, 0);
            return self.bv_concat(hi_part, lo_part);
        }
        // extract over zero/sign extend, staying inside the original bits:
        if let BvOp::ZeroExtend(inner, _n) = self.bv_op(x) {
            let iw = self.width_of(inner);
            if high < iw {
                return self.bv_extract(inner, high, low);
            }
        }
        if let BvOp::SignExtend(inner, _n) = self.bv_op(x) {
            let iw = self.width_of(inner);
            if high < iw {
                return self.bv_extract(inner, high, low);
            }
        }
        // Width narrowing through ops whose bit i depends only on bits
        // 0..=i of the operands. We push the extract through, narrowing
        // each operand to width `high+1`, build the op at the narrower
        // width, and then slice out [low..=high] from the result. The
        // recursive `bv_extract(narrowed, high, low)` falls through to the
        // literal Extract case since `narrowed`'s width is exactly
        // `high+1`, so this terminates after one round of narrowing.
        //
        // The big payoff is on traces from symbolic execution where most
        // arithmetic feeds into a final extract that truncates the result
        // — without this, a 32-bit add whose only consumer is `extract 7 0`
        // would emit a full 32-bit ripple-carry adder, of which 24 bits
        // are dead.
        if high + 1 < wx {
            let narrowed = match self.bv_op(x) {
                BvOp::Add(a, b) => {
                    let na = self.bv_extract(a, high, 0);
                    let nb = self.bv_extract(b, high, 0);
                    Some(self.bv_add(na, nb))
                }
                BvOp::Sub(a, b) => {
                    let na = self.bv_extract(a, high, 0);
                    let nb = self.bv_extract(b, high, 0);
                    Some(self.bv_sub(na, nb))
                }
                BvOp::Mul(a, b) => {
                    let na = self.bv_extract(a, high, 0);
                    let nb = self.bv_extract(b, high, 0);
                    Some(self.bv_mul(na, nb))
                }
                BvOp::Neg(a) => {
                    let na = self.bv_extract(a, high, 0);
                    Some(self.bv_neg(na))
                }
                BvOp::And(a, b) => {
                    let na = self.bv_extract(a, high, 0);
                    let nb = self.bv_extract(b, high, 0);
                    Some(self.bv_and(na, nb))
                }
                BvOp::Or(a, b) => {
                    let na = self.bv_extract(a, high, 0);
                    let nb = self.bv_extract(b, high, 0);
                    Some(self.bv_or(na, nb))
                }
                BvOp::Xor(a, b) => {
                    let na = self.bv_extract(a, high, 0);
                    let nb = self.bv_extract(b, high, 0);
                    Some(self.bv_xor(na, nb))
                }
                BvOp::Not(a) => {
                    let na = self.bv_extract(a, high, 0);
                    Some(self.bv_not(na))
                }
                _ => None,
            };
            if let Some(t) = narrowed {
                return self.bv_extract(t, high, low);
            }
        }
        self.push_bv(BvOp::Extract(x, high, low), new_w, 0)
    }

    /// Concatenate `x` (high bits) with `y` (low bits). Resulting width is
    /// `width(x) + width(y)` and must not exceed 64.
    pub fn bv_concat(&mut self, x: BvTerm, y: BvTerm) -> BvTerm {
        let wx = self.width_of(x);
        let wy = self.width_of(y);
        let w = wx + wy;
        assert!(w <= MAX_BV_WIDTH, "concat produces width {} > {}", w, MAX_BV_WIDTH);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let combined = ((vx & mask(wx)) << wy) | (vy & mask(wy));
            return self.bv_const(combined, w);
        }
        // Concat-of-concat with adjacent constants — fold them so chains of
        // constant-shift rewrites collapse to a canonical structural form.
        // Without this, `(x << 2) << 3` and `x << 5` produce equivalent but
        // structurally distinct terms, losing hash-cons-based CSE.
        //
        // Left case: concat(concat(a, b_const), y_const) → concat(a, b++y)
        if let Some(vy) = self.const_val(y) {
            if let BvOp::Concat(a, b) = self.bv_op(x) {
                if let Some(vb) = self.const_val(b) {
                    let wb = self.width_of(b);
                    let merged = ((vb & mask(wb)) << wy) | (vy & mask(wy));
                    let merged_const = self.bv_const(merged, wb + wy);
                    return self.bv_concat(a, merged_const);
                }
            }
        }
        // Right case: concat(x_const, concat(a_const, b)) → concat(x++a, b)
        if let Some(vx) = self.const_val(x) {
            if let BvOp::Concat(a, b) = self.bv_op(y) {
                if let Some(va) = self.const_val(a) {
                    let wa = self.width_of(a);
                    let merged = ((vx & mask(wx)) << wa) | (va & mask(wa));
                    let merged_const = self.bv_const(merged, wx + wa);
                    return self.bv_concat(merged_const, b);
                }
            }
        }
        // concat(extract(z, h1, l1), extract(z, h2, l2)) collapses to a
        // single extract when the slices are adjacent on the same BV.
        if let (BvOp::Extract(za, ha, la), BvOp::Extract(zb, hb, lb)) =
            (self.bv_op(x), self.bv_op(y))
        {
            if za == zb && la == hb + 1 {
                return self.bv_extract(za, ha, lb);
            }
        }
        self.push_bv(BvOp::Concat(x, y), w, 0)
    }

    /// Zero-extend `x` by `extra` bits.
    pub fn bv_zero_extend(&mut self, x: BvTerm, extra: u32) -> BvTerm {
        let wx = self.width_of(x);
        let w = wx + extra;
        assert!(w <= MAX_BV_WIDTH, "zero_extend produces width {} > {}", w, MAX_BV_WIDTH);
        if extra == 0 {
            return x;
        }
        if let Some(vx) = self.const_val(x) {
            return self.bv_const(vx & mask(wx), w);
        }
        // zero_extend(zero_extend(y, a), b) = zero_extend(y, a + b).
        if let BvOp::ZeroExtend(inner, a) = self.bv_op(x) {
            return self.bv_zero_extend(inner, a + extra);
        }
        self.push_bv(BvOp::ZeroExtend(x, extra), w, 0)
    }

    /// Sign-extend `x` by `extra` bits, replicating the MSB.
    pub fn bv_sign_extend(&mut self, x: BvTerm, extra: u32) -> BvTerm {
        let wx = self.width_of(x);
        let w = wx + extra;
        assert!(w <= MAX_BV_WIDTH, "sign_extend produces width {} > {}", w, MAX_BV_WIDTH);
        if extra == 0 {
            return x;
        }
        if let Some(vx) = self.const_val(x) {
            let sign_bit = (vx >> (wx - 1)) & 1 == 1;
            let extended = if sign_bit {
                (vx & mask(wx)) | (mask(w) & !mask(wx))
            } else {
                vx & mask(wx)
            };
            return self.bv_const(extended, w);
        }
        // sign_extend(sign_extend(y, a), b) = sign_extend(y, a + b).
        if let BvOp::SignExtend(inner, a) = self.bv_op(x) {
            return self.bv_sign_extend(inner, a + extra);
        }
        self.push_bv(BvOp::SignExtend(x, extra), w, 0)
    }

    /// If-then-else over bitvectors: `c ? t : e`. `t` and `e` must have the
    /// same width.
    pub fn bv_ite(&mut self, c: BoolTerm, t: BvTerm, e: BvTerm) -> BvTerm {
        let w = self.check_same_width(t, e);
        // Constant condition — collapse to the chosen branch.
        match self.const_bool(c) {
            Some(true) => return t,
            Some(false) => return e,
            None => {}
        }
        if t == e {
            return t; // both branches identical
        }
        // Nested-ite on the same selector: `ite(c, ite(c, x, _), e)` = `ite(c, x, e)`,
        // and `ite(c, t, ite(c, _, y))` = `ite(c, t, y)`. Same with `¬c` on the inner:
        // `ite(c, ite(¬c, _, y), e)` = `ite(c, y, e)`, etc.
        let nc = self.bool_not(c);
        if let BvOp::Ite(ic, it, ie) = self.bv_op(t) {
            if ic == c {
                return self.bv_ite(c, it, e);
            }
            if ic == nc {
                return self.bv_ite(c, ie, e);
            }
        }
        if let BvOp::Ite(ic, it, ie) = self.bv_op(e) {
            if ic == c {
                return self.bv_ite(c, t, ie);
            }
            if ic == nc {
                return self.bv_ite(c, t, it);
            }
        }
        // ITE factoring: when both branches are ITEs on the same inner
        // selector with a common sub-branch, hoist the common branch.
        // `ite(c, ite(d, x, y), ite(d, x, z))` ≡ `ite(d, x, ite(c, y, z))`.
        // `ite(c, ite(d, y, x), ite(d, z, x))` ≡ `ite(d, ite(c, y, z), x)`.
        // Fires naturally in state-merge formulas where two paths agree on
        // one side of a nested condition.
        if let (BvOp::Ite(tc, tt, te), BvOp::Ite(ec, et, ee)) =
            (self.bv_op(t), self.bv_op(e))
        {
            if tc == ec {
                if tt == et {
                    // Common "then" branch (tt == et): hoist it outside.
                    let inner = self.bv_ite(c, te, ee);
                    return self.bv_ite(tc, tt, inner);
                }
                if te == ee {
                    // Common "else" branch (te == ee): hoist it outside.
                    let inner = self.bv_ite(c, tt, et);
                    return self.bv_ite(tc, inner, te);
                }
            }
        }
        self.push_bv(BvOp::Ite(c, t, e), w, 0)
    }

    /// N-way first-match select (φ-node for state merging): the value is
    /// `values[i]` for the smallest `i` with `selectors[i]` true, else
    /// `default`. Pairs are supplied as parallel slices; their lengths must
    /// match, and every value (including `default`) must have the same width.
    ///
    /// Simplifications applied before building the node:
    ///   - Drop pairs where the selector is the constant `false` (unreachable).
    ///   - Return `values[i]` immediately if `selectors[i]` is constant `true`
    ///     (every later pair is shadowed by first-match).
    ///   - Drop pairs whose value equals `default` — they're behaviourally
    ///     indistinguishable from falling through to the default.
    ///   - If no pairs remain, return `default`.
    ///   - If all surviving values (plus default) are structurally identical,
    ///     return that value.
    ///
    /// For mutually-exclusive selectors (the state-merge case), callers
    /// should pair this with [`SmtSolver::assert_mutually_exclusive`] over
    /// the same `selectors` slice so SAT propagation can collapse the chain
    /// into a single decision.
    pub fn bv_select(
        &mut self,
        selectors: &[BoolTerm],
        values: &[BvTerm],
        default: BvTerm,
    ) -> BvTerm {
        assert_eq!(
            selectors.len(),
            values.len(),
            "bv_select: selectors and values must be parallel slices"
        );
        let w = self.width_of(default);
        let mut filtered: Vec<(BoolTerm, BvTerm)> =
            Vec::with_capacity(selectors.len());
        for (&s, &v) in selectors.iter().zip(values.iter()) {
            assert_eq!(
                self.width_of(v),
                w,
                "bv_select: all values must share default's width"
            );
            match self.const_bool(s) {
                Some(true) => {
                    // First const-true selector wins — everything later is
                    // shadowed by first-match semantics.
                    return v;
                }
                Some(false) => continue, // drop unreachable branch
                None => {}
            }
            if v == default {
                // Branch value equals the default: whether `s` is true or
                // false, the observed output is the same. Drop it.
                continue;
            }
            filtered.push((s, v));
        }
        if filtered.is_empty() {
            return default;
        }
        if filtered.iter().all(|&(_, v)| v == filtered[0].1) && filtered[0].1 == default {
            return default;
        }
        let idx = self.select_tables.len() as u32;
        self.select_tables.push(SelectTable {
            selectors: filtered.iter().map(|p| p.0).collect(),
            values: filtered.iter().map(|p| p.1).collect(),
            default,
        });
        self.push_bv(BvOp::Select(idx), w, 0)
    }

    // ---------- Boolean term builders ----------

    pub fn bool_true(&mut self) -> BoolTerm {
        self.push_bool(BoolOp::True)
    }

    pub fn bool_false(&mut self) -> BoolTerm {
        self.push_bool(BoolOp::False)
    }

    pub fn bool_var(&mut self) -> BoolTerm {
        let id = self.num_bool_vars;
        self.num_bool_vars += 1;
        self.push_bool(BoolOp::Var(id))
    }

    pub fn bool_not(&mut self, x: BoolTerm) -> BoolTerm {
        match self.bool_op(x) {
            BoolOp::True => return self.bool_false(),
            BoolOp::False => return self.bool_true(),
            BoolOp::Not(inner) => return inner, // double negation
            _ => {}
        }
        self.push_bool(BoolOp::Not(x))
    }

    pub fn bool_and(&mut self, x: BoolTerm, y: BoolTerm) -> BoolTerm {
        if x == y {
            return x;
        }
        match (self.const_bool(x), self.const_bool(y)) {
            (Some(true), _) => return y,
            (_, Some(true)) => return x,
            (Some(false), _) | (_, Some(false)) => return self.bool_false(),
            _ => {}
        }
        // Constant-bound chain collapse. When `x = and(l, r)` and both `r`
        // and `y` are comparisons of the same BV variable against constants
        // in the same direction (e.g., both `(bvslt k v)` for the same v),
        // keep only the tighter bound. BMC-style loop unrollings spam 1000s
        // of `(bvslt k_i N)` terms chained left-associative; every one after
        // the max is semantically redundant and without this fold each would
        // bitblast into a full 32-bit subtractor (~128 clauses) for nothing.
        if let BoolOp::And(l, r) = self.bool_op(x) {
            if let Some((ya_v, ya_k, ya_kind)) = self.extract_const_bound(y) {
                if let Some((rb_v, rb_k, rb_kind)) = self.extract_const_bound(r) {
                    if ya_v == rb_v && ya_kind == rb_kind {
                        // Same var, same comparison direction. Pick the
                        // bound that subsumes the other.
                        let w = self.width_of(ya_v);
                        let keep_new = ya_kind.is_tighter(ya_k, rb_k, w);
                        return if keep_new {
                            // New one subsumes r — replace r with y.
                            self.bool_and(l, y)
                        } else {
                            // r subsumes new one — drop y.
                            x
                        };
                    }
                }
            }
        }
        self.push_bool(BoolOp::And(x, y))
    }

    /// Classify `t` as a comparison of a BV variable against a constant in a
    /// recognized direction. Returns `(var, const, kind)` where `kind` tells
    /// us which side the constant lives on and whether the comparison is
    /// signed/unsigned, strict/non-strict.
    fn extract_const_bound(&self, t: BoolTerm) -> Option<(BvTerm, u128, BoundKind)> {
        let (op_kind, a, b) = match self.bool_op(t) {
            BoolOp::Ult(a, b) => (CmpOp::Ult, a, b),
            BoolOp::Ule(a, b) => (CmpOp::Ule, a, b),
            BoolOp::Slt(a, b) => (CmpOp::Slt, a, b),
            BoolOp::Sle(a, b) => (CmpOp::Sle, a, b),
            _ => return None,
        };
        let ak = self.const_val(a);
        let bk = self.const_val(b);
        match (ak, bk) {
            // const < var  (i.e., var > const): bound on var from below.
            (Some(k), None) => Some((b, k, BoundKind::KonstLhs(op_kind))),
            // var < const: bound on var from above.
            (None, Some(k)) => Some((a, k, BoundKind::KonstRhs(op_kind))),
            _ => None,
        }
    }

    pub fn bool_or(&mut self, x: BoolTerm, y: BoolTerm) -> BoolTerm {
        if x == y {
            return x;
        }
        match (self.const_bool(x), self.const_bool(y)) {
            (Some(true), _) | (_, Some(true)) => return self.bool_true(),
            (Some(false), _) => return y,
            (_, Some(false)) => return x,
            _ => {}
        }
        self.push_bool(BoolOp::Or(x, y))
    }

    pub fn bool_implies(&mut self, x: BoolTerm, y: BoolTerm) -> BoolTerm {
        // x → y ≡ ¬x ∨ y — route through bool_or so it picks up folds.
        let nx = self.bool_not(x);
        self.bool_or(nx, y)
    }

    // ---------- BV comparisons (bridge to Bool) ----------

    pub fn bv_eq(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return self.bool_true();
        }
        let (x, y) = self.canonicalize_commutative(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return if (vx & mask(w)) == (vy & mask(w)) {
                self.bool_true()
            } else {
                self.bool_false()
            };
        }
        // Bits-known vs constant: if any bit is forced to disagree with the
        // constant, the equality is statically false. And if the constant
        // matches every known bit while no unknown bits remain, it's true
        // (already caught by the both-constants arm after push_bv's full-fold,
        // but handled here for clarity).
        if w <= 128 {
            if let Some(vy) = self.const_val(y) {
                let (ox, zx) = self.bv_known_bits(x);
                let m = mask(w);
                let target = vy & m;
                // Any forced-one bit that should be zero, or forced-zero bit
                // that should be one, kills the equality.
                if (ox & !target) & m != 0 || (zx & target) & m != 0 {
                    return self.bool_false();
                }
            }
        }
        // Constant-vs-ite with constant branches: collapse the equality to a
        // Boolean selector. Common in symbolic-execution encodings that emit
        // `(= 1 (ite cond 1 0))` to lift a Bool into a BV and then test it.
        if let Some(vy) = self.const_val(y) {
            if let BvOp::Ite(c, t, e) = self.bv_op(x) {
                if let (Some(vt), Some(ve)) = (self.const_val(t), self.const_val(e)) {
                    let m = mask(w);
                    let eq_t = (vt & m) == (vy & m);
                    let eq_e = (ve & m) == (vy & m);
                    return match (eq_t, eq_e) {
                        (true, true) => self.bool_true(),
                        (false, false) => self.bool_false(),
                        (true, false) => c,
                        (false, true) => self.bool_not(c),
                    };
                }
            }
        }
        // Arithmetic solving: when the equality has a constant on the right
        // and an add/sub/neg with at least one constant operand on the left,
        // move the constant across so the equality becomes a direct
        // constraint on the remaining variable term. The bvadd's gates die
        // if this is its only consumer (hash-cons may keep it alive
        // otherwise — that's fine, just no win in that case).
        //
        // (a + b_const) = vy   →  a = vy - b_const
        // (a_const - b) = vy   →  b = a_const - vy
        // (a - b_const) = vy   →  a = vy + b_const    (bv_sub normalises
        //                                              const-rhs into add,
        //                                              so this is here for
        //                                              imported terms that
        //                                              skipped the builder)
        // (-a) = vy            →  a = -vy
        if let Some(vy) = self.const_val(y) {
            match self.bv_op(x) {
                BvOp::Add(a, b) => {
                    if let Some(vb) = self.const_val(b) {
                        let target = self.bv_const(vy.wrapping_sub(vb), w);
                        return self.bv_eq(a, target);
                    }
                    if let Some(va) = self.const_val(a) {
                        let target = self.bv_const(vy.wrapping_sub(va), w);
                        return self.bv_eq(b, target);
                    }
                }
                BvOp::Sub(a, b) => {
                    if let Some(va) = self.const_val(a) {
                        let target = self.bv_const(va.wrapping_sub(vy), w);
                        return self.bv_eq(b, target);
                    }
                    if let Some(vb) = self.const_val(b) {
                        let target = self.bv_const(vy.wrapping_add(vb), w);
                        return self.bv_eq(a, target);
                    }
                }
                BvOp::Neg(a) => {
                    let target = self.bv_const(0u128.wrapping_sub(vy), w);
                    return self.bv_eq(a, target);
                }
                _ => {}
            }
        }
        self.push_bool(BoolOp::Eq(x, y))
    }

    /// Bitvector not-equal: `!(x == y)`.
    pub fn bv_ne(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let eq = self.bv_eq(x, y);
        self.bool_not(eq)
    }

    pub fn bv_ult(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return self.bool_false();
        }
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return if (vx & mask(w)) < (vy & mask(w)) {
                self.bool_true()
            } else {
                self.bool_false()
            };
        }
        // Unsigned interval check via bits-known: a term's known-ones form
        // its lower bound and its `w_mask & !known_zeros` form its upper
        // bound (any unknown bit could be one). If the ranges are disjoint
        // the comparison statically decides.
        if w <= 128 {
            if let Some((xl, xh, yl, yh)) = self.unsigned_interval_pair(x, y, w) {
                if xh < yl {
                    return self.bool_true();
                }
                if xl >= yh {
                    return self.bool_false();
                }
            }
        }
        self.push_bool(BoolOp::Ult(x, y))
    }

    pub fn bv_ule(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return self.bool_true();
        }
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            return if (vx & mask(w)) <= (vy & mask(w)) {
                self.bool_true()
            } else {
                self.bool_false()
            };
        }
        if w <= 128 {
            if let Some((xl, xh, yl, yh)) = self.unsigned_interval_pair(x, y, w) {
                if xh <= yl {
                    return self.bool_true();
                }
                if xl > yh {
                    return self.bool_false();
                }
            }
        }
        self.push_bool(BoolOp::Ule(x, y))
    }

    /// Helper: `(x_lo, x_hi, y_lo, y_hi)` unsigned intervals from bits-known.
    /// Returns `None` when the width forbids u128 representation.
    fn unsigned_interval_pair(
        &self,
        x: BvTerm,
        y: BvTerm,
        w: u32,
    ) -> Option<(u128, u128, u128, u128)> {
        let m = mask(w);
        let (ox, zx) = self.bv_known_bits(x);
        let (oy, zy) = self.bv_known_bits(y);
        // Low bound: exactly the forced-one bits. Unknown bits become zero.
        let xl = ox & m;
        let yl = oy & m;
        // High bound: everything that isn't forced zero.
        let xh = (!zx) & m;
        let yh = (!zy) & m;
        Some((xl, xh, yl, yh))
    }

    pub fn bv_ugt(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        self.bv_ult(y, x)
    }

    pub fn bv_uge(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        self.bv_ule(y, x)
    }

    /// Signed less-than.
    pub fn bv_slt(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return self.bool_false();
        }
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            return if sx < sy { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::Slt(x, y))
    }

    pub fn bv_sle(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if x == y {
            return self.bool_true();
        }
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            return if sx <= sy { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::Sle(x, y))
    }

    pub fn bv_sgt(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        self.bv_slt(y, x)
    }

    pub fn bv_sge(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        self.bv_sle(y, x)
    }

    // ---------- Overflow predicates ----------

    pub fn bv_uadd_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let m = mask(w);
            let overflows = (vx & m).checked_add(vy & m).map(|v| v > m).unwrap_or(true);
            return if overflows { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::UaddOverflow(x, y))
    }

    pub fn bv_sadd_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            let sum = sx.wrapping_add(sy);
            let min = -(1i128 << (w - 1));
            let max = (1i128 << (w - 1)) - 1;
            let overflows = sum < min || sum > max;
            return if overflows { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::SaddOverflow(x, y))
    }

    pub fn bv_usub_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        self.check_same_width(x, y);
        // Unsigned subtract borrow ≡ x <u y.
        self.bv_ult(x, y)
    }

    pub fn bv_ssub_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            let diff = sx.wrapping_sub(sy);
            let min = -(1i128 << (w - 1));
            let max = (1i128 << (w - 1)) - 1;
            let overflows = diff < min || diff > max;
            return if overflows { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::SsubOverflow(x, y))
    }

    pub fn bv_umul_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let m = mask(w);
            let overflows = match (vx & m).checked_mul(vy & m) {
                Some(v) => v > m,
                None => true,
            };
            return if overflows { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::UmulOverflow(x, y))
    }

    pub fn bv_smul_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            let prod = sx.checked_mul(sy);
            let overflows = match prod {
                None => true,
                Some(p) => {
                    let min = -(1i128 << (w - 1));
                    let max = (1i128 << (w - 1)) - 1;
                    p < min || p > max
                }
            };
            return if overflows { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::SmulOverflow(x, y))
    }

    pub fn bv_neg_overflow(&mut self, x: BvTerm) -> BoolTerm {
        let w = self.width_of(x);
        if let Some(vx) = self.const_val(x) {
            let sx = sign_extend_to_i128(vx, w);
            let overflows = sx == -(1i128 << (w - 1)); // INT_MIN
            return if overflows { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::NegOverflow(x))
    }

    pub fn bv_sdiv_overflow(&mut self, x: BvTerm, y: BvTerm) -> BoolTerm {
        let w = self.check_same_width(x, y);
        if let (Some(vx), Some(vy)) = (self.const_val(x), self.const_val(y)) {
            let sx = sign_extend_to_i128(vx, w);
            let sy = sign_extend_to_i128(vy, w);
            let overflows = sx == -(1i128 << (w - 1)) && sy == -1;
            return if overflows { self.bool_true() } else { self.bool_false() };
        }
        self.push_bool(BoolOp::SdivOverflow(x, y))
    }

    // ---------- Introspection ----------

    pub fn width_of(&self, t: BvTerm) -> u32 {
        self.bv_nodes[t.0 as usize].width
    }

    /// Is this term a BV constant?
    #[inline]
    pub fn is_const(&self, t: BvTerm) -> bool {
        matches!(self.bv_op(t), BvOp::Const)
    }

    /// Returns `Some(value)` if `t` is a constant that fits in `u128`
    /// (width ≤ 128), `None` otherwise. Wide constants fall through to
    /// `None` so the existing u128-based folding paths skip them — that's
    /// intentional; we haven't implemented limb arithmetic yet, and
    /// "skip folding for wide" is a correct (if less optimized) default.
    #[inline]
    fn const_val(&self, t: BvTerm) -> Option<u128> {
        let node = &self.bv_nodes[t.0 as usize];
        if matches!(node.op, BvOp::Const) && node.wide == WIDE_NONE {
            Some(node.value)
        } else {
            None
        }
    }

    /// Returns `Some(b)` if this boolean term is the literal True/False.
    #[inline]
    fn const_bool(&self, t: BoolTerm) -> Option<bool> {
        match self.bool_op(t) {
            BoolOp::True => Some(true),
            BoolOp::False => Some(false),
            _ => None,
        }
    }

    /// Canonical ordering for commutative ops: constant moves to the right;
    /// among two non-constants, the smaller `BvTerm` id comes first. Giving
    /// every equivalent pair the same order lets hash consing dedupe them.
    #[inline]
    fn canonicalize_commutative(&self, x: BvTerm, y: BvTerm) -> (BvTerm, BvTerm) {
        match (self.is_const(x), self.is_const(y)) {
            (true, false) => (y, x),
            (false, true) => (x, y),
            _ => {
                if x.0 <= y.0 { (x, y) } else { (y, x) }
            }
        }
    }

    pub fn bv_op(&self, t: BvTerm) -> BvOp {
        self.bv_nodes[t.0 as usize].op
    }

    /// Non-panicking constant test: returns `Some(value)` if `t` is a
    /// [`BvOp::Const`] of width ≤ 128, else `None`. Width > 128 constants
    /// — which need limb storage — return `None`; use
    /// [`bv_const_value_limbs`] if you need those. The common caller check
    /// "is this term a folded constant?" is `try_bv_const_value(t).is_some()`.
    pub fn try_bv_const_value(&self, t: BvTerm) -> Option<u128> {
        let node = &self.bv_nodes[t.0 as usize];
        if matches!(node.op, BvOp::Const) && node.wide == WIDE_NONE {
            Some(node.value)
        } else {
            None
        }
    }

    /// Return a BV constant's value as a `u64`. Truncates for widths > 64.
    /// Panics if the constant is wider than 128 (use
    /// [`bv_const_value_limbs`] instead).
    pub fn bv_const_value(&self, t: BvTerm) -> u64 {
        let node = &self.bv_nodes[t.0 as usize];
        assert!(
            node.wide == WIDE_NONE,
            "bv_const_value: width {} > 128 — use bv_const_value_limbs",
            node.width
        );
        node.value as u64
    }

    /// u128 accessor for constant values — valid for widths ≤ 128. Panics
    /// for wider constants; use [`bv_const_value_limbs`] in that case.
    pub fn bv_const_value_u128(&self, t: BvTerm) -> u128 {
        let node = &self.bv_nodes[t.0 as usize];
        assert!(
            node.wide == WIDE_NONE,
            "bv_const_value_u128: width {} > 128 — use bv_const_value_limbs",
            node.width
        );
        node.value
    }

    /// Return a constant's limb representation (little-endian, length
    /// `ceil(width / 64)`). Works at any width — the u128 inline path
    /// gets converted on the fly.
    pub fn bv_const_value_limbs(&self, t: BvTerm) -> Vec<u64> {
        let node = &self.bv_nodes[t.0 as usize];
        if node.wide == WIDE_NONE {
            let nlimbs = ((node.width as usize) + 63) / 64;
            let mut limbs = vec![0u64; nlimbs];
            if nlimbs >= 1 {
                limbs[0] = node.value as u64;
            }
            if nlimbs >= 2 {
                limbs[1] = (node.value >> 64) as u64;
            }
            limbs
        } else {
            self.wide_values[node.wide as usize].to_vec()
        }
    }

    pub fn bool_op(&self, t: BoolTerm) -> BoolOp {
        self.bool_nodes[t.0 as usize]
    }

    // ---------- Internal helpers ----------

    /// Compute `(known_ones, known_zeros)` for a newly-constructed term of
    /// shape `(op, width)` with inline literal `value` (used only when op is
    /// `Const`). Returns `(0, 0)` for widths > 128 where the u128-mask
    /// representation doesn't fit. Propagation rules are conservative: when
    /// an operator can't determine a bit, we leave it unknown — never claim
    /// a bit is forced when it isn't.
    ///
    /// The helper reads `self.known_bits[t.0]` for each operand, which is
    /// already populated because hash-consed children were pushed before us.
    fn compute_known_bits(&self, op: BvOp, width: u32, value: u128) -> (u128, u128) {
        if width > 128 {
            return (0, 0);
        }
        let w_mask = mask(width);
        match op {
            BvOp::Var(_) => (0, 0),
            BvOp::Const => {
                let v = value & w_mask;
                (v, !v & w_mask)
            }
            BvOp::Not(x) => {
                let (ox, zx) = self.known_bits[x.0 as usize];
                (zx & w_mask, ox & w_mask)
            }
            BvOp::And(x, y) => {
                let (ox, zx) = self.known_bits[x.0 as usize];
                let (oy, zy) = self.known_bits[y.0 as usize];
                ((ox & oy) & w_mask, (zx | zy) & w_mask)
            }
            BvOp::Or(x, y) => {
                let (ox, zx) = self.known_bits[x.0 as usize];
                let (oy, zy) = self.known_bits[y.0 as usize];
                ((ox | oy) & w_mask, (zx & zy) & w_mask)
            }
            BvOp::Xor(x, y) => {
                let (ox, zx) = self.known_bits[x.0 as usize];
                let (oy, zy) = self.known_bits[y.0 as usize];
                // A bit is determined iff both operands are determined at
                // that bit. Then the bit value is the XOR of the known
                // values.
                let both_known = (ox | zx) & (oy | zy) & w_mask;
                let ones = (((ox & zy) | (zx & oy)) & both_known) & w_mask;
                let zeros = (((ox & oy) | (zx & zy)) & both_known) & w_mask;
                (ones, zeros)
            }
            BvOp::Add(x, y) => add_known_bits(
                self.known_bits[x.0 as usize],
                self.known_bits[y.0 as usize],
                width,
                0,
            ),
            BvOp::Sub(x, y) => {
                // x - y = x + (~y) + 1. Propagate through as an add with the
                // negated y and carry-in of 1.
                let (ox, zx) = self.known_bits[x.0 as usize];
                let (oy, zy) = self.known_bits[y.0 as usize];
                add_known_bits((ox, zx), (zy, oy), width, 1)
            }
            BvOp::Neg(x) => {
                // -x = ~x + 1.
                let (ox, zx) = self.known_bits[x.0 as usize];
                add_known_bits((zx, ox), (0, w_mask), width, 1)
            }
            BvOp::Mul(x, y) => {
                // Low-bit trailing zeros propagate: if x has k low known-zero
                // bits and y has m, the product has at least k+m low zeros.
                let (_ox, zx) = self.known_bits[x.0 as usize];
                let (_oy, zy) = self.known_bits[y.0 as usize];
                let tz_x = trailing_known_zeros(zx, width);
                let tz_y = trailing_known_zeros(zy, width);
                let low_zero_bits = (tz_x + tz_y).min(width);
                let zeros = if low_zero_bits == 0 {
                    0
                } else {
                    mask(low_zero_bits) & w_mask
                };
                (0, zeros)
            }
            BvOp::Udiv(_, _)
            | BvOp::Urem(_, _)
            | BvOp::Sdiv(_, _)
            | BvOp::Srem(_, _)
            | BvOp::Smod(_, _) => (0, 0),
            BvOp::Shl(x, y) => {
                // Constant shift amount: shift masks left by k and mark the
                // low k bits as known-zero. Symbolic shift amount: no info.
                let ynode = &self.bv_nodes[y.0 as usize];
                if matches!(ynode.op, BvOp::Const) && ynode.wide == WIDE_NONE {
                    let k = (ynode.value & w_mask).min(width as u128) as u32;
                    if k >= width {
                        return (0, w_mask);
                    }
                    let (ox, zx) = self.known_bits[x.0 as usize];
                    let low_zeros = if k == 0 { 0 } else { mask(k) };
                    let ones = (ox << k) & w_mask;
                    let zeros = ((zx << k) | low_zeros) & w_mask;
                    (ones, zeros)
                } else {
                    (0, 0)
                }
            }
            BvOp::Lshr(x, y) => {
                let ynode = &self.bv_nodes[y.0 as usize];
                if matches!(ynode.op, BvOp::Const) && ynode.wide == WIDE_NONE {
                    let k = (ynode.value & w_mask).min(width as u128) as u32;
                    if k >= width {
                        return (0, w_mask);
                    }
                    let (ox, zx) = self.known_bits[x.0 as usize];
                    // Bits shifted in from the top are known-zero.
                    let top_zeros = if k == 0 {
                        0
                    } else {
                        (mask(k) << (width - k)) & w_mask
                    };
                    let ones = ((ox & w_mask) >> k) & w_mask;
                    let zeros = (((zx & w_mask) >> k) | top_zeros) & w_mask;
                    (ones, zeros)
                } else {
                    (0, 0)
                }
            }
            BvOp::Ashr(x, y) => {
                let ynode = &self.bv_nodes[y.0 as usize];
                if matches!(ynode.op, BvOp::Const) && ynode.wide == WIDE_NONE {
                    let k = (ynode.value & w_mask).min(width as u128) as u32;
                    let (ox, zx) = self.known_bits[x.0 as usize];
                    if k == 0 {
                        return (ox & w_mask, zx & w_mask);
                    }
                    let k_eff = k.min(width);
                    let msb = 1u128 << (width - 1);
                    let msb_one = ox & msb != 0;
                    let msb_zero = zx & msb != 0;
                    if k_eff >= width {
                        return if msb_one {
                            (w_mask, 0)
                        } else if msb_zero {
                            (0, w_mask)
                        } else {
                            (0, 0)
                        };
                    }
                    let top_mask = (mask(k_eff) << (width - k_eff)) & w_mask;
                    let (top_ones, top_zeros) = if msb_one {
                        (top_mask, 0)
                    } else if msb_zero {
                        (0, top_mask)
                    } else {
                        (0, 0)
                    };
                    let ones = (((ox & w_mask) >> k_eff) | top_ones) & w_mask;
                    let zeros = (((zx & w_mask) >> k_eff) | top_zeros) & w_mask;
                    (ones, zeros)
                } else {
                    (0, 0)
                }
            }
            BvOp::Extract(x, hi, lo) => {
                let (ox, zx) = self.known_bits[x.0 as usize];
                let new_w = hi - lo + 1;
                let new_mask = mask(new_w);
                ((ox >> lo) & new_mask, (zx >> lo) & new_mask)
            }
            BvOp::Concat(hi_t, lo_t) => {
                let w_hi = self.width_of(hi_t);
                let w_lo = self.width_of(lo_t);
                let (ohi, zhi) = self.known_bits[hi_t.0 as usize];
                let (olo, zlo) = self.known_bits[lo_t.0 as usize];
                let hi_mask = mask(w_hi);
                let lo_mask = mask(w_lo);
                let ones = (((ohi & hi_mask) << w_lo) | (olo & lo_mask)) & w_mask;
                let zeros = (((zhi & hi_mask) << w_lo) | (zlo & lo_mask)) & w_mask;
                (ones, zeros)
            }
            BvOp::ZeroExtend(x, n) => {
                let w_x = self.width_of(x);
                let (ox, zx) = self.known_bits[x.0 as usize];
                let inner_mask = mask(w_x);
                let high_zeros = if n == 0 {
                    0
                } else {
                    (mask(n) << w_x) & w_mask
                };
                let ones = (ox & inner_mask) & w_mask;
                let zeros = ((zx & inner_mask) | high_zeros) & w_mask;
                (ones, zeros)
            }
            BvOp::SignExtend(x, n) => {
                let w_x = self.width_of(x);
                let (ox, zx) = self.known_bits[x.0 as usize];
                let inner_mask = mask(w_x);
                let ones = ox & inner_mask;
                let zeros = zx & inner_mask;
                if n == 0 {
                    return (ones & w_mask, zeros & w_mask);
                }
                let msb = 1u128 << (w_x - 1);
                let msb_one = ox & msb != 0;
                let msb_zero = zx & msb != 0;
                let high_mask = (mask(n) << w_x) & w_mask;
                let (high_ones, high_zeros) = if msb_one {
                    (high_mask, 0)
                } else if msb_zero {
                    (0, high_mask)
                } else {
                    (0, 0)
                };
                ((ones | high_ones) & w_mask, (zeros | high_zeros) & w_mask)
            }
            BvOp::Ite(_c, t, e) => {
                // Both branches agree on a bit → that bit is known.
                let (ot, zt) = self.known_bits[t.0 as usize];
                let (oe, ze) = self.known_bits[e.0 as usize];
                ((ot & oe) & w_mask, (zt & ze) & w_mask)
            }
            BvOp::Select(idx) => {
                // Intersection of all value branches + default.
                let table = &self.select_tables[idx as usize];
                let (mut ones, mut zeros) = self.known_bits[table.default.0 as usize];
                ones &= w_mask;
                zeros &= w_mask;
                for &v in table.values.iter() {
                    let (ov, zv) = self.known_bits[v.0 as usize];
                    ones &= ov;
                    zeros &= zv;
                }
                (ones, zeros)
            }
            BvOp::RotateLeft(_, _) | BvOp::RotateRight(_, _) => {
                // Could in principle rotate the known-bits masks if the
                // amount has narrow known-bits — skipped here for now.
                (0, 0)
            }
            BvOp::Popcount(_) | BvOp::Clz(_) | BvOp::Ctz(_) => {
                // The result is bounded by [0, width]. The bits above the
                // minimum bit-width needed to represent `width` are known
                // zero, which gives downstream `extract`s a free narrow
                // (the high zero-bits fold to constants).
                if width == 0 {
                    (0, 0)
                } else {
                    let k_bits = 32 - width.leading_zeros();
                    if k_bits >= width {
                        (0, 0)
                    } else {
                        (0, !mask(k_bits) & w_mask)
                    }
                }
            }
        }
    }

    fn push_bv(&mut self, op: BvOp, width: u32, value: u128) -> BvTerm {
        let node = BvNode {
            op,
            width,
            value,
            wide: WIDE_NONE,
        };
        if let Some(&t) = self.bv_hashcons.get(&node) {
            return t;
        }
        let known = self.compute_known_bits(op, width, value);
        // Bits-known constant fold: if every bit is determined, the node is
        // a literal regardless of its operator. Route through `bv_const` so
        // we hash-cons against any existing constant of the same value.
        // Skip this short-circuit for `BvOp::Const` itself (otherwise we'd
        // recurse infinitely — `bv_const` calls `push_bv` with `Const`).
        if !matches!(op, BvOp::Const) && width <= 128 {
            let (ones, zeros) = known;
            if ones | zeros == mask(width) {
                return self.bv_const(ones, width);
            }
        }
        let id = self.bv_nodes.len() as u32;
        self.bv_nodes.push(node);
        self.known_bits.push(known);
        let term = BvTerm(id);
        self.bv_hashcons.insert(node, term);
        term
    }

    /// Like `push_bv` but for constants too wide to fit inline — the limb
    /// content lives at `wide_values[wide_idx]`.
    fn push_bv_wide(&mut self, op: BvOp, width: u32, wide_idx: u32) -> BvTerm {
        let node = BvNode {
            op,
            width,
            value: 0,
            wide: wide_idx,
        };
        if let Some(&t) = self.bv_hashcons.get(&node) {
            return t;
        }
        let id = self.bv_nodes.len() as u32;
        self.bv_nodes.push(node);
        // Wide BVs: no bits-known tracking. The helper's (u128, u128) mask
        // pair can't represent width > 128; callers that need bits-known on
        // wide values would have to switch to limb masks, which isn't a
        // pattern symbex formulas tend to need.
        self.known_bits.push((0, 0));
        let term = BvTerm(id);
        self.bv_hashcons.insert(node, term);
        term
    }

    fn push_bool(&mut self, op: BoolOp) -> BoolTerm {
        if let Some(&t) = self.bool_hashcons.get(&op) {
            return t;
        }
        let id = self.bool_nodes.len() as u32;
        self.bool_nodes.push(op);
        let term = BoolTerm(id);
        self.bool_hashcons.insert(op, term);
        term
    }

    fn check_same_width(&self, x: BvTerm, y: BvTerm) -> u32 {
        let wx = self.width_of(x);
        let wy = self.width_of(y);
        assert_eq!(wx, wy, "BV width mismatch: {} vs {}", wx, wy);
        wx
    }

    // ---------- Arithmetic normalization (bitwuzla-style) ----------
    //
    // Flatten bvadd chains under comparisons into coefficient multisets
    // (`term → coefficient mod 2^w`), then rebuild in a canonical order.
    // Two payoffs:
    //
    //   - **Sharing.** `(x + y) + z` and `(z + x) + y` become the same
    //     term, so permuted re-associations across thousands of assertions
    //     bitblast to ONE adder chain instead of many that the SAT solver
    //     must prove equal by search. This is the pass that decides the
    //     Sage2-style `bvadd`/`bvmul` benchmarks (ablating the equivalent
    //     pass in bitwuzla turns 0.7s solves into >30s timeouts).
    //   - **Cancellation.** For equalities, common addends on both sides
    //     cancel outright (`a + b = b + c` → `a = c`), and negative
    //     coefficients move across the relation — both sound in Z/2^w.
    //     For inequalities only the per-side canonicalization applies
    //     (cancellation across `<` is unsound under wraparound).
    //
    // Flattening rules (all ring identities in Z/2^w):
    //   Const v         → constant accumulator += coeff · v
    //   Add(x, y)       → flatten(x, coeff), flatten(y, coeff)
    //   Sub(x, y)       → flatten(x, coeff), flatten(y, −coeff)
    //   Neg(x)          → flatten(x, −coeff)
    //   Not(x)          → accumulator −= coeff; flatten(x, −coeff)   [~x = −x−1]
    //   Mul(x, c)/(c, x)→ flatten(x, coeff · c)
    //   anything else   → leaf: occs[t] += coeff

    /// Normalize one assertion. Rewrites every comparison node reachable in
    /// the Bool/BV DAG whose operands contain `bvadd` chains. Memoized via
    /// the caller-provided maps so shared subterms rewrite once (pass the
    /// same maps across calls for cross-assertion sharing).
    pub fn normalize_assertion(
        &mut self,
        t: BoolTerm,
        bool_memo: &mut HashMap<BoolTerm, BoolTerm>,
        bv_memo: &mut HashMap<BvTerm, BvTerm>,
    ) -> BoolTerm {
        self.norm_bool(t, bool_memo, bv_memo)
    }

    fn norm_bool(
        &mut self,
        t: BoolTerm,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) -> BoolTerm {
        if let Some(&r) = bm.get(&t) {
            return r;
        }
        let op = self.bool_op(t);
        let r = match op {
            BoolOp::True | BoolOp::False | BoolOp::Var(_) => t,
            BoolOp::Not(x) => {
                let nx = self.norm_bool(x, bm, vm);
                if nx == x { t } else { self.bool_not(nx) }
            }
            BoolOp::And(x, y) => {
                let nx = self.norm_bool(x, bm, vm);
                let ny = self.norm_bool(y, bm, vm);
                if nx == x && ny == y { t } else { self.bool_and(nx, ny) }
            }
            BoolOp::Or(x, y) => {
                let nx = self.norm_bool(x, bm, vm);
                let ny = self.norm_bool(y, bm, vm);
                if nx == x && ny == y { t } else { self.bool_or(nx, ny) }
            }
            BoolOp::Implies(x, y) => {
                let nx = self.norm_bool(x, bm, vm);
                let ny = self.norm_bool(y, bm, vm);
                if nx == x && ny == y { t } else { self.bool_implies(nx, ny) }
            }
            BoolOp::Eq(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                self.norm_eq_add(na, nb, bm, vm)
            }
            BoolOp::Ult(a, b) => {
                let (na, nb) = self.norm_cmp_sides(a, b, bm, vm);
                self.bv_ult(na, nb)
            }
            BoolOp::Ule(a, b) => {
                let (na, nb) = self.norm_cmp_sides(a, b, bm, vm);
                self.bv_ule(na, nb)
            }
            BoolOp::Slt(a, b) => {
                let (na, nb) = self.norm_cmp_sides(a, b, bm, vm);
                self.bv_slt(na, nb)
            }
            BoolOp::Sle(a, b) => {
                let (na, nb) = self.norm_cmp_sides(a, b, bm, vm);
                self.bv_sle(na, nb)
            }
            // Overflow predicates: normalize operands, keep the predicate.
            BoolOp::UaddOverflow(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                if na == a && nb == b { t } else { self.bv_uadd_overflow(na, nb) }
            }
            BoolOp::SaddOverflow(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                if na == a && nb == b { t } else { self.bv_sadd_overflow(na, nb) }
            }
            BoolOp::UsubOverflow(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                if na == a && nb == b { t } else { self.bv_usub_overflow(na, nb) }
            }
            BoolOp::SsubOverflow(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                if na == a && nb == b { t } else { self.bv_ssub_overflow(na, nb) }
            }
            BoolOp::UmulOverflow(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                if na == a && nb == b { t } else { self.bv_umul_overflow(na, nb) }
            }
            BoolOp::SmulOverflow(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                if na == a && nb == b { t } else { self.bv_smul_overflow(na, nb) }
            }
            BoolOp::NegOverflow(a) => {
                let na = self.norm_bv(a, bm, vm);
                if na == a { t } else { self.bv_neg_overflow(na) }
            }
            BoolOp::SdivOverflow(a, b) => {
                let na = self.norm_bv(a, bm, vm);
                let nb = self.norm_bv(b, bm, vm);
                if na == a && nb == b { t } else { self.bv_sdiv_overflow(na, nb) }
            }
        };
        bm.insert(t, r);
        r
    }

    /// Recursively rewrite a BV term: nested comparisons (inside ITE
    /// conditions) get normalized, and every `bvadd`-topped subterm is
    /// rebuilt in canonical flattened form so permuted re-associations
    /// share one term.
    fn norm_bv(
        &mut self,
        t: BvTerm,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) -> BvTerm {
        if let Some(&r) = vm.get(&t) {
            return r;
        }
        let op = self.bv_op(t);
        let r = match op {
            BvOp::Var(_) | BvOp::Const => t,
            BvOp::Not(x) => {
                let nx = self.norm_bv(x, bm, vm);
                if nx == x { t } else { self.bv_not(nx) }
            }
            BvOp::And(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_and(nx, ny) }
            }
            BvOp::Or(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_or(nx, ny) }
            }
            BvOp::Xor(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_xor(nx, ny) }
            }
            BvOp::Add(_, _) | BvOp::Sub(_, _) | BvOp::Neg(_) => {
                // Canonical flattened rebuild; acceptance decided at the
                // batch level by the SMT layer (see flush_pending).
                self.canon_add(t, bm, vm)
            }
            BvOp::Mul(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_mul(nx, ny) }
            }
            BvOp::Udiv(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_udiv(nx, ny) }
            }
            BvOp::Urem(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_urem(nx, ny) }
            }
            BvOp::Sdiv(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_sdiv(nx, ny) }
            }
            BvOp::Srem(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_srem(nx, ny) }
            }
            BvOp::Smod(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_smod(nx, ny) }
            }
            BvOp::Shl(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_shl(nx, ny) }
            }
            BvOp::Lshr(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_lshr(nx, ny) }
            }
            BvOp::Ashr(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_ashr(nx, ny) }
            }
            BvOp::RotateLeft(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_rotate_left_dyn(nx, ny) }
            }
            BvOp::RotateRight(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_rotate_right_dyn(nx, ny) }
            }
            BvOp::Popcount(x) => {
                let nx = self.norm_bv(x, bm, vm);
                if nx == x { t } else { self.bv_popcount(nx) }
            }
            BvOp::Clz(x) => {
                let nx = self.norm_bv(x, bm, vm);
                if nx == x { t } else { self.bv_clz(nx) }
            }
            BvOp::Ctz(x) => {
                let nx = self.norm_bv(x, bm, vm);
                if nx == x { t } else { self.bv_ctz(nx) }
            }
            BvOp::Extract(x, hi, lo) => {
                let nx = self.norm_bv(x, bm, vm);
                if nx == x { t } else { self.bv_extract(nx, hi, lo) }
            }
            BvOp::Concat(x, y) => {
                let (nx, ny) = (self.norm_bv(x, bm, vm), self.norm_bv(y, bm, vm));
                if nx == x && ny == y { t } else { self.bv_concat(nx, ny) }
            }
            BvOp::ZeroExtend(x, n) => {
                let nx = self.norm_bv(x, bm, vm);
                if nx == x { t } else { self.bv_zero_extend(nx, n) }
            }
            BvOp::SignExtend(x, n) => {
                let nx = self.norm_bv(x, bm, vm);
                if nx == x { t } else { self.bv_sign_extend(nx, n) }
            }
            BvOp::Ite(c, a, b) => {
                let nc = self.norm_bool(c, bm, vm);
                let (na, nb) = (self.norm_bv(a, bm, vm), self.norm_bv(b, bm, vm));
                if nc == c && na == a && nb == b {
                    t
                } else {
                    self.bv_ite(nc, na, nb)
                }
            }
            BvOp::Select(idx) => {
                let table = self.select_tables[idx as usize].clone();
                let sels: Vec<BoolTerm> = table
                    .selectors
                    .iter()
                    .map(|&s| self.norm_bool(s, bm, vm))
                    .collect();
                let vals: Vec<BvTerm> = table
                    .values
                    .iter()
                    .map(|&v| self.norm_bv(v, bm, vm))
                    .collect();
                let ndef = self.norm_bv(table.default, bm, vm);
                let unchanged = ndef == table.default
                    && sels.iter().zip(table.selectors.iter()).all(|(a, b)| a == b)
                    && vals.iter().zip(table.values.iter()).all(|(a, b)| a == b);
                if unchanged {
                    t
                } else {
                    self.bv_select(&sels, &vals, ndef)
                }
            }
        };
        vm.insert(t, r);
        r
    }

    /// Flatten `t` (an add-chain) into `occs` with the given coefficient.
    /// All arithmetic mod 2^w.
    ///
    /// Iterative and DAG-aware: decomposable nodes are queued with
    /// accumulated coefficients and processed largest-index-first. Since
    /// children always have smaller arena indices than their parents, every
    /// node is decomposed exactly once with its full coefficient — a naive
    /// recursive flatten re-visits shared subterms once per PATH, which is
    /// exponential on deep shared add-DAGs (bench_4351 hangs in it).
    fn flatten_add(
        &mut self,
        t: BvTerm,
        coeff: u128,
        w: u32,
        occs: &mut std::collections::BTreeMap<BvTerm, u128>,
        acc: &mut u128,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) {
        let m = mask(w);
        // Pending decomposition queue: node index → accumulated coefficient.
        let mut pending: std::collections::BTreeMap<u32, u128> =
            std::collections::BTreeMap::new();
        let add_pending = |map: &mut std::collections::BTreeMap<u32, u128>,
                               node: BvTerm,
                               c: u128,
                               merged: &mut u64| {
            match map.entry(node.0) {
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    *merged += 1;
                    let v = e.get_mut();
                    *v = v.wrapping_add(c) & m;
                }
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(c & m);
                }
            }
        };
        let mut merged_local = 0u64;
        add_pending(&mut pending, t, coeff, &mut merged_local);

        while let Some((&idx, _)) = pending.last_key_value() {
            let c = pending.remove(&idx).unwrap();
            if c & m == 0 {
                continue;
            }
            let cur = BvTerm(idx);
            match self.bv_op(cur) {
                BvOp::Const => {
                    if let Some(v) = self.const_val(cur) {
                        *acc = acc.wrapping_add(c.wrapping_mul(v)) & m;
                        continue;
                    }
                }
                BvOp::Add(x, y) => {
                    add_pending(&mut pending, x, c, &mut merged_local);
                    add_pending(&mut pending, y, c, &mut merged_local);
                    continue;
                }
                BvOp::Sub(x, y) => {
                    let neg = 0u128.wrapping_sub(c) & m;
                    add_pending(&mut pending, x, c, &mut merged_local);
                    add_pending(&mut pending, y, neg, &mut merged_local);
                    continue;
                }
                BvOp::Neg(x) => {
                    let neg = 0u128.wrapping_sub(c) & m;
                    add_pending(&mut pending, x, neg, &mut merged_local);
                    continue;
                }
                BvOp::Not(x) => {
                    // ~x = -x - 1
                    *acc = acc.wrapping_sub(c) & m;
                    let neg = 0u128.wrapping_sub(c) & m;
                    add_pending(&mut pending, x, neg, &mut merged_local);
                    continue;
                }
                BvOp::Mul(x, y) => {
                    if let Some(cv) = self.const_val(y) {
                        add_pending(&mut pending, x, c.wrapping_mul(cv) & m, &mut merged_local);
                        continue;
                    }
                    if let Some(cv) = self.const_val(x) {
                        add_pending(&mut pending, y, c.wrapping_mul(cv) & m, &mut merged_local);
                        continue;
                    }
                }
                _ => {}
            }
            // Leaf: normalize it independently, then record the occurrence.
            let nt = self.norm_bv(cur, bm, vm);
            match occs.entry(nt) {
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    self.norm_merged += 1;
                    let v = e.get_mut();
                    *v = v.wrapping_add(c) & m;
                }
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(c & m);
                }
            }
        }
        self.norm_merged += merged_local;
    }

    /// Rebuild a flattened side in canonical order: terms ascending by id,
    /// each scaled by its coefficient (the `bv_mul` builder turns powers
    /// of two into structural shifts and everything else into sparse NAF
    /// adds), constant last.
    fn rebuild_add(
        &mut self,
        occs: &std::collections::BTreeMap<BvTerm, u128>,
        acc: u128,
        w: u32,
    ) -> BvTerm {
        let m = mask(w);
        let mut out: Option<BvTerm> = None;
        for (&t, &c) in occs.iter() {
            let c = c & m;
            if c == 0 {
                continue;
            }
            let scaled = if c == 1 {
                t
            } else if c == m {
                self.bv_neg(t)
            } else {
                let cc = self.bv_const(c, w);
                self.bv_mul(t, cc)
            };
            out = Some(match out {
                None => scaled,
                Some(prev) => self.bv_add(prev, scaled),
            });
        }
        match out {
            None => self.bv_const(acc, w),
            Some(sum) => {
                if acc & m == 0 {
                    sum
                } else {
                    let cc = self.bv_const(acc, w);
                    self.bv_add(sum, cc)
                }
            }
        }
    }

    /// Canonical rebuild of an add-topped term (used per-side under
    /// inequalities and for standalone adds).
    fn canon_add(
        &mut self,
        t: BvTerm,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) -> BvTerm {
        let w = self.width_of(t);
        if w > 128 {
            return t; // coefficient arithmetic is u128-based
        }
        let mut occs = std::collections::BTreeMap::new();
        let mut acc = 0u128;
        self.flatten_add(t, 1, w, &mut occs, &mut acc, bm, vm);
        self.rebuild_add(&occs, acc, w)
    }

    /// Normalize both sides of a comparison (per-side only — cancellation
    /// across an inequality is unsound in modular arithmetic).
    fn norm_cmp_sides(
        &mut self,
        a: BvTerm,
        b: BvTerm,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) -> (BvTerm, BvTerm) {
        let na = self.norm_bv(a, bm, vm);
        let nb = self.norm_bv(b, bm, vm);
        (na, nb)
    }

    /// Equality over adds: flatten both sides, move everything to one map
    /// (lhs − rhs), then split — coefficients with a clear sign bit stay
    /// left, sign-bit-set coefficients move right negated (min-signed stays
    /// left: it is its own negation and would ping-pong). Common terms
    /// cancel automatically in the subtraction. Sound in Z/2^w.
    fn norm_eq_add(
        &mut self,
        a: BvTerm,
        b: BvTerm,
        bm: &mut HashMap<BoolTerm, BoolTerm>,
        vm: &mut HashMap<BvTerm, BvTerm>,
    ) -> BoolTerm {
        let w = self.width_of(a);
        if w > 128 {
            return self.bv_eq(a, b);
        }
        // Only bother when at least one side is an additive shape — plain
        // `x = y` equalities gain nothing from the multiset round-trip.
        let additive = |op: BvOp| {
            matches!(
                op,
                BvOp::Add(_, _) | BvOp::Sub(_, _) | BvOp::Neg(_)
            )
        };
        if !additive(self.bv_op(a)) && !additive(self.bv_op(b)) {
            return self.bv_eq(a, b);
        }
        let m = mask(w);
        let mut occs = std::collections::BTreeMap::new();
        let mut acc = 0u128;
        self.flatten_add(a, 1, w, &mut occs, &mut acc, bm, vm);
        // rhs enters with coefficient −1: lhs − rhs = 0.
        let neg1 = m;
        self.flatten_add(b, neg1, w, &mut occs, &mut acc, bm, vm);

        let min_signed = if w == 0 { 0 } else { 1u128 << (w - 1) };
        let mut lhs = std::collections::BTreeMap::new();
        let mut rhs = std::collections::BTreeMap::new();
        for (&t, &c) in occs.iter() {
            let c = c & m;
            if c == 0 {
                self.norm_cancelled += 1;
                continue;
            }
            let negative = (c & min_signed) != 0 && c != min_signed;
            if negative {
                rhs.insert(t, 0u128.wrapping_sub(c) & m);
            } else {
                lhs.insert(t, c);
            }
        }
        // Constant: keep on the right as −acc so the common SSA shape
        // `sum = k` comes out with the constant alone when possible.
        let rhs_const = 0u128.wrapping_sub(acc) & m;
        let l = self.rebuild_add(&lhs, 0, w);
        let r = self.rebuild_add(&rhs, rhs_const, w);
        self.bv_eq(l, r)
    }
}

impl Default for BvContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Low `width` bits set, as a `u128`. Only defined for widths ≤ 128 — wider
/// values can't be represented as a single u128 and use the limb-based
/// code path instead.
#[inline]
pub fn mask(width: u32) -> u128 {
    if width == 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Bits-known ripple-carry adder. Given the `(ones, zeros)` masks for two
/// operands, a width, and an initial carry-in (0 or 1), walk bit-by-bit
/// from LSB propagating the sum and carry. Stops tracking as soon as either
/// operand bit or the carry becomes unknown — everything beyond is left as
/// "unknown" (both masks 0 for those positions). This mirrors what a full
/// bit-level simulation would derive from the same knowns, without actually
/// enumerating any free bits.
fn add_known_bits(
    (ox, zx): (u128, u128),
    (oy, zy): (u128, u128),
    width: u32,
    cin: u128,
) -> (u128, u128) {
    debug_assert!(cin <= 1);
    let w_mask = mask(width);
    let mut ones = 0u128;
    let mut zeros = 0u128;
    let mut carry = cin;
    for i in 0..width {
        let bit = 1u128 << i;
        let x_known = (ox | zx) & bit != 0;
        let y_known = (oy | zy) & bit != 0;
        if !x_known || !y_known {
            break;
        }
        let x_bit = (ox & bit) >> i;
        let y_bit = (oy & bit) >> i;
        let sum = x_bit + y_bit + carry;
        if sum & 1 == 1 {
            ones |= bit;
        } else {
            zeros |= bit;
        }
        carry = sum >> 1;
    }
    (ones & w_mask, zeros & w_mask)
}

/// Number of low-order bits that are known-zero in `zeros` (i.e. trailing
/// ones of the `zeros` mask), capped at `width`. Used by `bv_mul` bits-known
/// to bound the trailing-zero count of the product.
#[inline]
fn trailing_known_zeros(zeros: u128, width: u32) -> u32 {
    if zeros == 0 {
        return 0;
    }
    // Walk from LSB while the bit is set in `zeros`.
    let mut i = 0u32;
    while i < width && (zeros >> i) & 1 == 1 {
        i += 1;
    }
    i
}

/// Zero any bits in the top limb that lie above the BV's `width`. Call
/// this after any operation that might have set garbage in the spill
/// space of a limb array. Fine on length-0 slices (no-op).
#[inline]
pub fn mask_top_limb(limbs: &mut [u64], width: u32) {
    let bits_in_top = width as usize % 64;
    if bits_in_top == 0 || limbs.is_empty() {
        return;
    }
    let top_idx = limbs.len() - 1;
    let top_mask = (1u64 << bits_in_top) - 1;
    limbs[top_idx] &= top_mask;
}

/// If `v > 0` is a power of two, return its exponent (i.e., `log2(v)`);
/// otherwise `None`. Used by builder-level rewrites that turn
/// mul/div/rem by a power of two into a shift or mask.
#[inline]
pub fn power_of_two_exp(v: u128) -> Option<u32> {
    if v == 0 || v & v.wrapping_sub(1) != 0 {
        None
    } else {
        Some(v.trailing_zeros())
    }
}

/// Granlund-Montgomery magic constants for unsigned division by a constant.
///
/// Given divisor `d` (must be > 1 and not a power of two) and bitwidth `w`
/// (≤ 64 so the internal 2w-bit multiply fits in a u128), returns `(m_lo, l)`
/// such that for every `x` in `[0, 2^w)`:
///
/// ```text
///   floor(x / d)  ==  (((mulhi_w(x, m_lo) + ((x - mulhi_w(x, m_lo)) >> 1))) >> (l - 1))
/// ```
///
/// where `mulhi_w(x, m_lo)` is the high `w` bits of the 2w-bit unsigned
/// product. `l` is `ceil(log2(d))`.
///
/// Returns `None` when `d` is 0, 1, a power of two, or when `w + l > 127`
/// (computing `ceil(2^(w+l) / d)` would overflow u128).
#[inline]
pub fn unsigned_magic(d: u128, w: u32) -> Option<(u128, u32)> {
    if d <= 1 {
        return None;
    }
    if power_of_two_exp(d).is_some() {
        return None;
    }
    // l = ceil(log2(d)).  For d > 1 and not a power of two, 2^(l-1) < d < 2^l.
    let l = 128 - (d - 1).leading_zeros();
    // Need w + l < 128 so 2^(w+l) fits in u128.
    if w + l >= 128 || w == 0 {
        return None;
    }
    let numer: u128 = 1u128 << (w + l);
    // ceil(numer / d)
    let m = numer.div_ceil(d);
    // m lives in (2^w, 2^(w+1)) for non-power-of-two d, so this subtraction
    // is a valid N-bit value.
    if m <= (1u128 << w) {
        return None; // sanity: shouldn't happen for non-power-of-two d
    }
    let m_lo = m - (1u128 << w);
    Some((m_lo, l))
}

/// Reinterpret `value` (low `width` bits) as a signed integer in i128.
/// Used by const-folding paths for signed comparisons at any width up to
/// 128.
#[inline]
pub fn sign_extend_to_i128(value: u128, width: u32) -> i128 {
    if width == 0 || width >= 128 {
        return value as i128;
    }
    let m = mask(width);
    let v = value & m;
    let sign_bit = 1u128 << (width - 1);
    if v & sign_bit != 0 {
        (v | !m) as i128
    } else {
        v as i128
    }
}
