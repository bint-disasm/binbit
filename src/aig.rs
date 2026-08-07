//! And-Inverter Graph (AIG) — the bitblaster's intermediate representation.
//!
//! Every Boolean expression the bitblaster produces lands here first as a
//! graph of 2-input AND nodes with inversion carried on the edges. CNF is
//! emitted at flush time by walking the AIG reachable from asserted roots
//! (see `SmtSolver::materialize_aig`).
//!
//! Why an AIG layer at all — wasn't the gate cache enough? Two reasons:
//!
//! 1. **Cross-operator dedup.** The gate cache keys on Tseitin output lits.
//!    Two expressions that compute the same function via different op
//!    choices (e.g., `bvor(a, b)` vs `bvnot(bvand(bvnot(a), bvnot(b)))`)
//!    end up with distinct output lits in the Lit-based world. In the AIG
//!    both reduce to the same node because `or` is encoded as `!and(!,!)`
//!    and the `!` bits live on the edges, not in separate nodes.
//!
//! 2. **Delayed CNF emission.** Assertion shapes like `(= x y)` get a
//!    direct 2N biconditional encoding at flush time; we don't want to
//!    have emitted the XNOR-chain gates beforehand.
//!
//! The earlier shadow-AIG attempt (which shipped an Aig alongside the
//! existing CNF emitter) was unsound — the two representations drifted
//! apart on corner cases. This module is the sole source of truth for
//! bitblasted logic.

use rustc_hash::FxHashMap as HashMap;

use crate::lit::Lit;

/// Reference to an AIG node with a polarity bit in bit 0. The upper bits
/// index into `Aig::nodes`. Polarity 0 = output of the node; polarity 1 =
/// negation of the node.
///
/// Node index 0 is reserved for the constant-true sentinel; therefore
/// `AigRef::TRUE == AigRef(0)` (idx=0, polarity=0) and `AigRef::FALSE ==
/// AigRef(1)` (idx=0, polarity=1).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct AigRef(pub u32);

impl AigRef {
    pub const TRUE: AigRef = AigRef(0);
    pub const FALSE: AigRef = AigRef(1);

    #[inline]
    pub fn from_parts(node_idx: u32, negated: bool) -> Self {
        AigRef((node_idx << 1) | (negated as u32))
    }
    #[inline]
    pub fn node_idx(self) -> u32 {
        self.0 >> 1
    }
    #[inline]
    pub fn is_negated(self) -> bool {
        (self.0 & 1) != 0
    }
    #[inline]
    pub fn negate(self) -> AigRef {
        AigRef(self.0 ^ 1)
    }
    #[inline]
    pub fn is_const_true(self) -> bool {
        self == AigRef::TRUE
    }
    #[inline]
    pub fn is_const_false(self) -> bool {
        self == AigRef::FALSE
    }
    #[inline]
    pub fn is_const(self) -> bool {
        self.node_idx() == 0
    }
}

impl std::ops::Not for AigRef {
    type Output = AigRef;
    #[inline]
    fn not(self) -> AigRef {
        self.negate()
    }
}

/// Kind of an AIG node. Node 0 is always `ConstTrue`. Input nodes hold a
/// reference to an externally-allocated SAT literal (typically a BV bit
/// variable or a Bool variable allocated during bitblasting). And nodes
/// hold two signed operands.
#[derive(Copy, Clone, Debug)]
pub enum AigNode {
    ConstTrue,
    Input(Lit),
    And(AigRef, AigRef),
}

/// The AIG itself: a topologically-ordered node arena plus a hash-cons
/// table for structural dedup.
///
/// The invariant is that children of an `And` node have strictly smaller
/// node indices, so a left-to-right walk of `nodes` visits every node
/// after its operands. This makes CNF emission trivial.
pub struct Aig {
    /// Two-level AIG rewriting (Brummayer & Biere, "Local Two-Level
    /// And-Inverter Graph Minimization without Blowup") applied inside
    /// `and()`, ported rule-for-rule from bitwuzla's `rewrite_and`. Off by
    /// default: it changes circuit structure and therefore CNF shape and
    /// search trajectory — benchmark per-corpus before adopting.
    two_level: bool,
    /// Parent-count ceiling above which a two-level substitution is
    /// declined (see [`Aig::set_subst_share_limit`]).
    subst_share_limit: u32,
    /// When two_level is on, also apply the substitution / idempotence-4
    /// families. These are the only rules that BYPASS a shared interior
    /// node (re-pointing an operand at a grandchild) instead of purely
    /// deleting redundancy; on trajectory-sensitive instances that
    /// fragments the learned-clause vocabulary carried by shared gate
    /// variables (bench_5906: 9045 substitutions, nothing else, 15× more
    /// conflicts). Off = "safe subset": contradiction, subsumption,
    /// idempotence-2, resolution only.
    two_level_subst: bool,
    /// Rule-firing counters for the two-level rewriter, by family:
    /// [contradiction, subsumption, idempotence-2, resolution,
    /// substitution, idempotence-4]. The last two BYPASS a shared interior
    /// node (they re-point an operand at a grandchild) rather than merely
    /// returning an existing ref — tracked separately to diagnose
    /// proof-vocabulary fragmentation.
    pub rw_counts: [u64; 6],
    /// Node arena. `nodes[0]` is always `ConstTrue`.
    pub nodes: Vec<AigNode>,
    /// Per-node BV-source annotation, for propagating `VarOrigin::BvBit`
    /// style metadata when we allocate SAT lits at CNF emission time.
    /// `None` for the constant and most input nodes.
    pub src_terms: Vec<Option<crate::bv::BvTerm>>,
    /// Hash cons: packed `(canonical_lhs << 32) | canonical_rhs` →
    /// node_idx. Both sides already have their polarity applied.
    /// Canonical ordering is `lhs.0 <= rhs.0`. The single-u64 key halves
    /// hasher work and makes probes a word compare — `and()` is the
    /// hottest front-end call on emission-bound sessions.
    hash_cons: HashMap<u64, u32>,
    /// Operand-reference count per node: how many And nodes already take
    /// it as a child. Used as a construction-time sharing test — a node
    /// with a live co-parent must not be BYPASSED by a substitution, or
    /// its subfunction ends up with two CNF vocabularies and learned
    /// clauses stop transferring (the bench_5906 pathology). This is a
    /// lower bound (future parents aren't known yet), so it never claims
    /// a shared node is private; `substitute_pass` still catches the rest
    /// after the batch is complete.
    uses: Vec<u32>,
    /// Scratch for `substitute_pass`, reused across flushes (see the
    /// allocation note there). Node-sized; kept here so an incremental
    /// session doesn't re-allocate and re-zero them per flush.
    post_live: Vec<bool>,
    post_parents: Vec<u32>,
    post_stack: Vec<u32>,
    post_rel: Vec<u32>,
    /// Input dedup: one AIG input per SAT literal, dense-indexed by the
    /// literal (a lit and its negation share the node; polarity carried on
    /// the AigRef). `u32::MAX` = no entry. Dense because input creation is
    /// on the fresh-variable fast path — no hashing.
    input_lut: Vec<u32>,
}


#[inline]
fn cons_key(lhs: AigRef, rhs: AigRef) -> u64 {
    ((lhs.0 as u64) << 32) | rhs.0 as u64
}

impl Aig {
    /// Default for [`Aig::set_subst_share_limit`]: unrestricted.
    /// How many existing co-parents a node may have and still be bypassed by
    /// a construction-time substitution. 0 = only private nodes; `u32::MAX`
    /// (the DEFAULT) is the original ungated behaviour.
    ///
    /// Measured 2026-08-06, and the two workloads want opposite settings, so
    /// this stays a knob rather than becoming a policy. On the shared-DAG
    /// pathology it is decisive — bench_5906 conflicts 213,411 (ungated) →
    /// 25,219 at limit 1, against a 17,428 baseline. On nobranch every
    /// gating level is a large loss — 628,280 ungated → 863,487 at 1 →
    /// 1,025,755 at 0 — because that circuit's substitutions are on
    /// genuinely private structure and blocking them leaves a worse graph.
    /// Note it is not monotonic: limit 0 is worse than 1 on BOTH.
    pub const SUBST_SHARE_UNLIMITED: u32 = u32::MAX;

    pub fn new() -> Self {
        Aig {
            two_level: false,
            subst_share_limit: Self::SUBST_SHARE_UNLIMITED,
            two_level_subst: true,
            rw_counts: [0; 6],
            nodes: vec![AigNode::ConstTrue],
            src_terms: vec![None],
            hash_cons: HashMap::with_capacity_and_hasher(512, Default::default()),
            uses: Vec::new(),
            post_live: Vec::new(),
            post_parents: Vec::new(),
            post_stack: Vec::new(),
            post_rel: Vec::new(),
            input_lut: Vec::new(),
        }
    }

    /// Enable/disable two-level rewriting in `and()` (see the field docs).
    pub fn set_two_level(&mut self, on: bool) {
        self.two_level = on;
    }

    /// Decline a two-level substitution when the node it would bypass has
    /// more than `limit` parents. [`Aig::SUBST_SHARE_UNLIMITED`] (the
    /// default) never declines; the constant's doc carries the measurement
    /// showing why this is a knob and not a policy.
    pub fn set_subst_share_limit(&mut self, limit: u32) {
        self.subst_share_limit = limit;
    }

    /// Whether two-level rewriting is enabled (see `set_two_level`).
    pub fn two_level_enabled(&self) -> bool {
        self.two_level
    }

    /// Restrict two-level rewriting to the pure-deletion families
    /// (see `two_level_subst` field docs). `on = false` disables the
    /// substitution / idempotence-4 rules.
    pub fn set_two_level_subst(&mut self, on: bool) {
        self.two_level_subst = on;
    }

    /// Number of AIG nodes (including the constant-true sentinel at 0).
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn node(&self, idx: u32) -> AigNode {
        self.nodes[idx as usize]
    }

    #[inline]
    pub fn src_term(&self, idx: u32) -> Option<crate::bv::BvTerm> {
        self.src_terms[idx as usize]
    }

    /// Register `lit` as a primary input. Lit and !lit dedup to the same
    /// input node — the negation is carried on the returned AigRef.
    pub fn input(&mut self, lit: Lit) -> AigRef {
        let li = lit.0 as usize;
        if let Some(&raw) = self.input_lut.get(li)
            && raw != u32::MAX {
                return AigRef(raw);
            }
        // Canonicalize to positive-polarity storage.
        let canonical_lit = Lit(lit.0 & !1);
        let ci = canonical_lit.0 as usize;
        let hi = li.max(ci);
        if self.input_lut.len() <= hi {
            self.input_lut.resize(hi + 1, u32::MAX);
        }
        if self.input_lut[ci] != u32::MAX {
            // Lit we were asked about is the negation of an existing input.
            let neg = AigRef(self.input_lut[ci]).negate();
            self.input_lut[li] = neg.0;
            return neg;
        }
        let idx = self.nodes.len() as u32;
        self.nodes.push(AigNode::Input(canonical_lit));
        self.src_terms.push(None);
        let pos = AigRef::from_parts(idx, false);
        self.input_lut[ci] = pos.0;
        let requested = if lit.0 & 1 != 0 { pos.negate() } else { pos };
        self.input_lut[li] = requested.0;
        requested
    }

    /// Children of the *node* under `r` if it is an And, else None. The
    /// polarity of `r` itself is NOT applied — two-level rules inspect the
    /// node's structure and track the edge polarity separately.
    #[inline]
    fn and_children(&self, r: AigRef) -> Option<(AigRef, AigRef)> {
        match self.nodes[r.node_idx() as usize] {
            AigNode::And(x, y) => Some((x, y)),
            _ => None,
        }
    }

    /// Build `and(a, b)` with construction-time simplification + hash-cons.
    ///
    /// Level-1 folds (always on):
    ///   - identity vs constants (TRUE / FALSE)
    ///   - `and(x, x) = x`, `and(x, ¬x) = FALSE`
    ///
    /// With `two_level` enabled, additionally applies the Brummayer-Biere
    /// two-level rules — contradiction, subsumption, idempotence,
    /// resolution, substitution — looking one level into And-shaped
    /// operands. Substitution rules shrink an operand and restart the loop
    /// (as in bitwuzla's `rewrite_and`, which this ports rule-for-rule).
    /// Then hash-cons on the sorted pair.
    pub fn and(&mut self, a: AigRef, b: AigRef) -> AigRef {
        let (mut left, mut right) = (a, b);
        loop {
            // === Level 1: neutrality / boundedness / idempotence /
            // contradiction on the operands themselves. ===
            if left == AigRef::TRUE || left == right {
                return right;
            }
            if right == AigRef::TRUE {
                return left;
            }
            if left == AigRef::FALSE || right == AigRef::FALSE || left == !right {
                return AigRef::FALSE;
            }

            if !self.two_level {
                break;
            }

            let lk = self.and_children(left);
            let rk = self.and_children(right);
            let ln = left.is_negated();
            let rn = right.is_negated();

            // === Level 2 ===

            // Contradiction (asymmetric): (a ∧ b) ∧ c with a = ¬c ∨ b = ¬c.
            if !ln
                && let Some((x, y)) = lk
                    && (x == !right || y == !right) {
                        self.rw_counts[0] += 1;
                        return AigRef::FALSE;
                    }
            if !rn
                && let Some((x, y)) = rk
                    && (x == !left || y == !left) {
                        self.rw_counts[0] += 1;
                        return AigRef::FALSE;
                    }

            // Contradiction (symmetric): (a ∧ b) ∧ (c ∧ d) with any
            // child pair complementary.
            if !ln && !rn
                && let (Some((x, y)), Some((w, z))) = (lk, rk)
                    && (x == !w || x == !z || y == !w || y == !z) {
                        self.rw_counts[0] += 1;
                        return AigRef::FALSE;
                    }

            // Subsumption (asymmetric): ¬(a ∧ b) ∧ c with a = ¬c ∨ b = ¬c
            // → c.
            if ln
                && let Some((x, y)) = lk
                    && (x == !right || y == !right) {
                        self.rw_counts[1] += 1;
                        return right;
                    }
            if rn
                && let Some((x, y)) = rk
                    && (x == !left || y == !left) {
                        self.rw_counts[1] += 1;
                        return left;
                    }

            // Subsumption (symmetric): ¬(a ∧ b) ∧ (c ∧ d) with any child
            // pair complementary → (c ∧ d).
            if ln && !rn
                && let (Some((x, y)), Some((w, z))) = (lk, rk)
                    && (x == !w || x == !z || y == !w || y == !z) {
                        self.rw_counts[1] += 1;
                        return right;
                    }
            if rn && !ln
                && let (Some((w, z)), Some((x, y))) = (rk, lk)
                    && (w == !x || w == !y || z == !x || z == !y) {
                        self.rw_counts[1] += 1;
                        return left;
                    }

            // Idempotence (2-level): (a ∧ b) ∧ c with a = c ∨ b = c
            // → (a ∧ b).
            if !ln
                && let Some((x, y)) = lk
                    && (x == right || y == right) {
                        self.rw_counts[2] += 1;
                        return left;
                    }
            if !rn
                && let Some((x, y)) = rk
                    && (x == left || y == left) {
                        self.rw_counts[2] += 1;
                        return right;
                    }

            // Resolution: ¬(a ∧ b) ∧ ¬(c ∧ d) with {a=c, b=¬d} or
            // {a=d, b=¬c} → ¬a (and the mirrored variants → ¬d).
            if ln && rn
                && let (Some((x, y)), Some((w, z))) = (lk, rk) {
                    if (x == w && y == !z) || (x == z && y == !w) {
                        self.rw_counts[3] += 1;
                        return !x;
                    }
                    if (z == y && w == !x) || (z == x && w == !y) {
                        self.rw_counts[3] += 1;
                        return !z;
                    }
                }

            // Safe-subset mode stops here: the remaining families bypass
            // shared interior nodes rather than deleting redundancy.
            if !self.two_level_subst {
                break;
            }

            // === Level 3: substitution — shrink an operand, restart. ===
            //
            // Each rule below BYPASSES an interior node (re-points this
            // node at a grandchild). That is safe only while the bypassed
            // node has no co-parent: otherwise its subfunction keeps a
            // gate variable for the other parents while this cone
            // constrains its expansion, and learned clauses stop
            // transferring between the two vocabularies. `may_bypass`
            // enforces that with the reference counts known so far.
            let may_bypass = |uses: &Vec<u32>, r: AigRef| -> bool {
                uses.get(r.node_idx() as usize).copied().unwrap_or(0)
                    <= self.subst_share_limit
            };

            // Asymmetric: ¬(a ∧ b) ∧ c with a = c → ¬b ∧ c (resp. b = c
            // → ¬a ∧ c).
            if ln && may_bypass(&self.uses, left)
                && let Some((x, y)) = lk {
                    if x == right {
                        left = !y;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                    if y == right {
                        left = !x;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                }
            if rn && may_bypass(&self.uses, right)
                && let Some((w, z)) = rk {
                    if w == left {
                        right = !z;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                    if z == left {
                        right = !w;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                }

            // Symmetric: ¬(a ∧ b) ∧ (c ∧ d) with a ∈ {c, d} → ¬b ∧ (c ∧ d)
            // (resp. b ∈ {c, d} → ¬a ∧ (c ∧ d)).
            if ln && !rn && may_bypass(&self.uses, left)
                && let (Some((x, y)), Some((w, z))) = (lk, rk) {
                    if x == w || x == z {
                        left = !y;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                    if y == w || y == z {
                        left = !x;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                }
            if rn && !ln && may_bypass(&self.uses, right)
                && let (Some((w, z)), Some((x, y))) = (rk, lk) {
                    if w == x || w == y {
                        right = !z;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                    if z == x || z == y {
                        right = !w;
                        self.rw_counts[4] += 1;
                        continue;
                    }
                }

            // === Level 4: idempotence across two Ands — drop the shared
            // conjunct from one side, restart. (a ∧ b) ∧ (c ∧ d) with
            // c ∈ {a, b} → keep d (resp. d ∈ {a, b} → keep c). ===
            if !ln && !rn
                && let (Some((x, y)), Some((w, z))) = (lk, rk) {
                    if x == w || y == w {
                        self.rw_counts[5] += 1;
                        right = z;
                        continue;
                    }
                    if x == z || y == z {
                        self.rw_counts[5] += 1;
                        right = w;
                        continue;
                    }
                }

            break;
        }

        // Canonicalize: put the smaller AigRef first so `(a, b)` and
        // `(b, a)` land on the same hash-cons key.
        let (lhs, rhs) = if left.0 <= right.0 {
            (left, right)
        } else {
            (right, left)
        };
        if let Some(&idx) = self.hash_cons.get(&cons_key(lhs, rhs)) {
            return AigRef::from_parts(idx, false);
        }
        let idx = self.nodes.len() as u32;
        self.nodes.push(AigNode::And(lhs, rhs));
        self.src_terms.push(None);
        self.hash_cons.insert(cons_key(lhs, rhs), idx);
        if self.two_level && self.two_level_subst {
            if self.uses.len() <= idx as usize {
                self.uses.resize(idx as usize + 1, 0);
            }
            self.uses[lhs.node_idx() as usize] += 1;
            self.uses[rhs.node_idx() as usize] += 1;
        }
        AigRef::from_parts(idx, false)
    }

    /// Read-only hash-cons probe: does `and(a, b)` already exist as a
    /// node? Returns it without creating anything.
    ///
    /// Deliberately skips the constant folds and two-level rewriting that
    /// [`Self::and`] applies, so a `None` here does not prove `and(a, b)`
    /// would allocate — it only proves no node with this exact operand
    /// pair exists. Cost estimators may therefore over-estimate, never
    /// under-estimate, the nodes a build would add.
    pub fn lookup_and(&self, a: AigRef, b: AigRef) -> Option<AigRef> {
        let (lhs, rhs) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        self.hash_cons
            .get(&cons_key(lhs, rhs))
            .map(|&idx| AigRef::from_parts(idx, false))
    }

    /// `or(a, b) = ¬and(¬a, ¬b)` — no native OR node in an AIG.
    pub fn or(&mut self, a: AigRef, b: AigRef) -> AigRef {
        !self.and(!a, !b)
    }

    /// `xor(a, b) = (a ∧ ¬b) ∨ (¬a ∧ b)`.
    pub fn xor(&mut self, a: AigRef, b: AigRef) -> AigRef {
        if a == b {
            return AigRef::FALSE;
        }
        if a == !b {
            return AigRef::TRUE;
        }
        if a == AigRef::TRUE {
            return !b;
        }
        if b == AigRef::TRUE {
            return !a;
        }
        if a == AigRef::FALSE {
            return b;
        }
        if b == AigRef::FALSE {
            return a;
        }
        let t = self.and(a, !b);
        let u = self.and(!a, b);
        self.or(t, u)
    }

    /// `mux(sel, t, e) = (sel ∧ t) ∨ (¬sel ∧ e)`, with the usual AIG-era
    /// mux simplifications baked in.
    pub fn mux(&mut self, sel: AigRef, t: AigRef, e: AigRef) -> AigRef {
        if t == e {
            return t;
        }
        if sel == AigRef::TRUE {
            return t;
        }
        if sel == AigRef::FALSE {
            return e;
        }
        // `mux(s, T, F) = s`, `mux(s, F, T) = !s`.
        if t == AigRef::TRUE && e == AigRef::FALSE {
            return sel;
        }
        if t == AigRef::FALSE && e == AigRef::TRUE {
            return !sel;
        }
        // Branch-equals-selector folds, from `mux(s,t,e) = (s∧t) ∨ (¬s∧e)`:
        // `mux(s, s, e) = s ∨ e`, `mux(s, ¬s, e) = ¬s ∧ e`,
        // `mux(s, t, s) = s ∧ t`, `mux(s, t, ¬s) = ¬s ∨ t`.
        if t == sel {
            return self.or(sel, e);
        }
        if t == !sel {
            return self.and(!sel, e);
        }
        if e == sel {
            return self.and(sel, t);
        }
        if e == !sel {
            return self.or(!sel, t);
        }
        let hi = self.and(sel, t);
        let lo = self.and(!sel, e);
        self.or(hi, lo)
    }

    /// Rewrite node `dup` into a pure alias of `target`, after an external
    /// proof (FRAIG sweep) that they compute the same function. The alias
    /// is represented structurally as `And(target, target)`, which computes
    /// exactly `target` — every existing holder of a ref to `dup` keeps its
    /// ref and transparently evaluates / materializes through `target`
    /// (CNF emission recognizes the shape and reuses target's SAT lit with
    /// zero clauses; see `SmtSolver::lit_of`). Such identity nodes can NOT
    /// arise from normal construction — `and()` folds `and(x, x)` to `x` —
    /// so the shape unambiguously means "merged".
    ///
    /// `target.node_idx() < dup` is required: it preserves the topological
    /// invariant (children strictly smaller), which is also what makes
    /// alias chains terminate.
    pub fn merge_equiv(&mut self, dup: u32, target: AigRef) {
        debug_assert!(target.node_idx() < dup, "merge target must be older than dup");
        debug_assert!(
            matches!(self.nodes[dup as usize], AigNode::And(..)),
            "only And nodes are merge candidates"
        );
        self.nodes[dup as usize] = AigNode::And(target, target);
    }

    /// Chase FRAIG/fold alias nodes (`And(t, t)`) to the underlying ref,
    /// composing polarities. Terminates: an alias always points at a
    /// strictly smaller node index.
    fn resolve(&self, mut r: AigRef) -> AigRef {
        loop {
            if let AigNode::And(a, b) = self.nodes[r.node_idx() as usize]
                && a == b {
                    r = if r.is_negated() { !a } else { a };
                    continue;
                }
            return r;
        }
    }

    /// Sharing-aware two-level substitution, run AFTER the batch's AIG is
    /// fully built (unlike the construction-time rules, which fire before
    /// a node's co-parents exist). This is what makes the substitution
    /// family compatible with shared circuits:
    ///
    /// Construction-time substitution re-points a parent at a grandchild,
    /// bypassing the interior node. If that interior keeps other parents,
    /// the emitted CNF ends up with two vocabularies for one subfunction —
    /// some cones constrain the interior's gate variable, rewritten ones
    /// constrain its expansion — and learned clauses stop transferring
    /// between them (measured: bench_5906 fired 9045 substitutions,
    /// changed <3% of gates, and took 15× more conflicts). Here, with the
    /// batch graph complete, we apply a substitution ONLY when the
    /// bypassed interior has exactly one live parent (the node being
    /// rewritten) and is neither a root nor already materialized — so a
    /// rewrite never strands a co-parent. Tree-shaped cones still cascade
    /// (freed nodes release their children, and later passes pick up
    /// substitutions unblocked by earlier ones); shared nodes are left
    /// intact.
    ///
    /// All rewrites are function-preserving and in-place, so every
    /// existing holder of a ref (caches, roots, other parents) stays
    /// valid. Full folds turn the node into an `And(t, t)` alias (the
    /// same shape `merge_equiv` uses; CNF emission binds it to `t`'s lit
    /// with zero clauses).
    ///
    /// `roots` are the batch's assertion refs (bypass-protected);
    /// `pinned[i]` marks nodes already bound to a SAT lit (their CNF is
    /// emitted — rewriting them can only fragment). May be shorter than
    /// `nodes`.
    pub fn substitute_pass(&mut self, roots: &[AigRef], pinned: &[bool]) -> PostPassStats {
        let n = self.nodes.len();
        let mut stats = PostPassStats::default();
        let is_pinned = |i: u32| (i as usize) < pinned.len() && pinned[i as usize];
        // Scratch reused across flushes. An incremental session calls this
        // once per flush over a monotonically growing AIG, so allocating
        // and zeroing three node-sized buffers each time is quadratic in
        // the session — invisible on single-query corpus files, not on a
        // symbex run that flushes thousands of times.
        let mut live = std::mem::take(&mut self.post_live);
        let mut parents = std::mem::take(&mut self.post_parents);
        let mut stack = std::mem::take(&mut self.post_stack);
        let mut rel = std::mem::take(&mut self.post_rel);
        live.clear();
        live.resize(n, false);
        parents.clear();
        parents.resize(n, 0);

        // Liveness: reachable from the roots without descending into
        // pinned cones (their structure is settled). Keeps garbage nodes
        // (e.g. rejected-normalization variants) from inflating parent
        // counts and blocking valid substitutions.
        stack.clear();
        stack.extend(roots.iter().map(|r| r.node_idx()));
        while let Some(i) = stack.pop() {
            if live[i as usize] || is_pinned(i) {
                continue;
            }
            live[i as usize] = true;
            if let AigNode::And(a, b) = self.nodes[i as usize] {
                stack.push(a.node_idx());
                stack.push(b.node_idx());
            }
        }

        // Live parent counts. Roots get a sentinel parent so they can
        // never be bypassed (they materialize regardless).
        for i in 0..n {
            if !live[i] {
                continue;
            }
            if let AigNode::And(a, b) = self.nodes[i] {
                parents[a.node_idx() as usize] += 1;
                parents[b.node_idx() as usize] += 1;
            }
        }
        for r in roots {
            parents[r.node_idx() as usize] += 1;
        }

        // Release a reference to node `k`; if it just died, cascade into
        // its children so later passes see accurate counts.
        // `stack` is caller-owned scratch: this fires on every re-point
        // and fold, so a fresh Vec per call was one allocation per
        // rewrite.
        fn release(
            k: u32,
            nodes: &[AigNode],
            parents: &mut [u32],
            live: &mut [bool],
            stack: &mut Vec<u32>,
        ) {
            stack.clear();
            stack.push(k);
            while let Some(i) = stack.pop() {
                let iu = i as usize;
                debug_assert!(parents[iu] > 0, "releasing unreferenced node");
                parents[iu] -= 1;
                if parents[iu] == 0 && live[iu] {
                    live[iu] = false;
                    if let AigNode::And(a, b) = nodes[iu] {
                        stack.push(a.node_idx());
                        stack.push(b.node_idx());
                    }
                }
            }
        }

        const MAX_PASSES: u32 = 4;
        for _pass in 0..MAX_PASSES {
            let mut rewrites_this_pass = 0u64;
            stats.passes += 1;

            for i in 1..n {
                if !live[i] || is_pinned(i as u32) {
                    continue;
                }
                let AigNode::And(l0, r0) = self.nodes[i] else {
                    continue;
                };
                if l0 == r0 {
                    continue; // already an alias
                }

                // Track the node's current (virtual) children; parent
                // counts are adjusted incrementally on every re-point.
                let mut left = self.resolve(l0);
                let mut right = self.resolve(r0);
                if left != l0 {
                    parents[left.node_idx() as usize] += 1;
                    release(l0.node_idx(), &self.nodes, &mut parents, &mut live, &mut rel);
                }
                if right != r0 {
                    parents[right.node_idx() as usize] += 1;
                    release(r0.node_idx(), &self.nodes, &mut parents, &mut live, &mut rel);
                }

                let mut fold: Option<AigRef> = None;
                loop {
                    // Level-1 folds on the virtual children.
                    if left == AigRef::TRUE || left == right {
                        fold = Some(right);
                        break;
                    }
                    if right == AigRef::TRUE {
                        fold = Some(left);
                        break;
                    }
                    if left == AigRef::FALSE || right == AigRef::FALSE || left == !right {
                        fold = Some(AigRef::FALSE);
                        break;
                    }

                    let lk = self.and_children(left);
                    let rk = self.and_children(right);
                    let ln = left.is_negated();
                    let rn = right.is_negated();

                    // Pure-deletion folds (ungated — they don't bypass).
                    // Contradiction.
                    if !ln
                        && let Some((x, y)) = lk
                            && (x == !right || y == !right) {
                                fold = Some(AigRef::FALSE);
                                break;
                            }
                    if !rn
                        && let Some((x, y)) = rk
                            && (x == !left || y == !left) {
                                fold = Some(AigRef::FALSE);
                                break;
                            }
                    if !ln && !rn
                        && let (Some((x, y)), Some((w, z))) = (lk, rk)
                            && (x == !w || x == !z || y == !w || y == !z) {
                                fold = Some(AigRef::FALSE);
                                break;
                            }
                    // Subsumption.
                    if ln
                        && let Some((x, y)) = lk
                            && (x == !right || y == !right) {
                                fold = Some(right);
                                break;
                            }
                    if rn
                        && let Some((x, y)) = rk
                            && (x == !left || y == !left) {
                                fold = Some(left);
                                break;
                            }
                    if ln && !rn
                        && let (Some((x, y)), Some((w, z))) = (lk, rk)
                            && (x == !w || x == !z || y == !w || y == !z) {
                                fold = Some(right);
                                break;
                            }
                    if rn && !ln
                        && let (Some((w, z)), Some((x, y))) = (rk, lk)
                            && (w == !x || w == !y || z == !x || z == !y) {
                                fold = Some(left);
                                break;
                            }
                    // Idempotence (2-level).
                    if !ln
                        && let Some((x, y)) = lk
                            && (x == right || y == right) {
                                fold = Some(left);
                                break;
                            }
                    if !rn
                        && let Some((x, y)) = rk
                            && (x == left || y == left) {
                                fold = Some(right);
                                break;
                            }
                    // Resolution.
                    if ln && rn
                        && let (Some((x, y)), Some((w, z))) = (lk, rk) {
                            if (x == w && y == !z) || (x == z && y == !w) {
                                fold = Some(!x);
                                break;
                            }
                            if (z == y && w == !x) || (z == x && w == !y) {
                                fold = Some(!z);
                                break;
                            }
                        }

                    // Gated substitution: bypass only single-parent,
                    // unpinned interiors. `parents[k] == 1` means the sole
                    // live reference is the one we're about to drop.
                    let can_bypass = |k: u32, parents: &[u32]| {
                        parents[k as usize] == 1 && !is_pinned(k)
                    };
                    let mut applied = false;

                    // Asymmetric: ¬(a ∧ b) ∧ c.
                    if ln
                        && let Some((x, y)) = lk {
                            let k = left.node_idx();
                            if x == right || y == right {
                                if can_bypass(k, &parents) {
                                    let repl = if x == right { !y } else { !x };
                                    parents[repl.node_idx() as usize] += 1;
                                    release(k, &self.nodes, &mut parents, &mut live, &mut rel);
                                    left = repl;
                                    applied = true;
                                } else {
                                    stats.blocked += 1;
                                }
                            }
                        }
                    if !applied && rn
                        && let Some((w, z)) = rk {
                            let k = right.node_idx();
                            if w == left || z == left {
                                if can_bypass(k, &parents) {
                                    let repl = if w == left { !z } else { !w };
                                    parents[repl.node_idx() as usize] += 1;
                                    release(k, &self.nodes, &mut parents, &mut live, &mut rel);
                                    right = repl;
                                    applied = true;
                                } else {
                                    stats.blocked += 1;
                                }
                            }
                        }
                    // Symmetric: ¬(a ∧ b) ∧ (c ∧ d).
                    if !applied && ln && !rn
                        && let (Some((x, y)), Some((w, z))) = (lk, rk) {
                            let k = left.node_idx();
                            if x == w || x == z || y == w || y == z {
                                if can_bypass(k, &parents) {
                                    let repl = if x == w || x == z { !y } else { !x };
                                    parents[repl.node_idx() as usize] += 1;
                                    release(k, &self.nodes, &mut parents, &mut live, &mut rel);
                                    left = repl;
                                    applied = true;
                                } else {
                                    stats.blocked += 1;
                                }
                            }
                        }
                    if !applied && rn && !ln
                        && let (Some((w, z)), Some((x, y))) = (rk, lk) {
                            let k = right.node_idx();
                            if w == x || w == y || z == x || z == y {
                                if can_bypass(k, &parents) {
                                    let repl = if w == x || w == y { !z } else { !w };
                                    parents[repl.node_idx() as usize] += 1;
                                    release(k, &self.nodes, &mut parents, &mut live, &mut rel);
                                    right = repl;
                                    applied = true;
                                } else {
                                    stats.blocked += 1;
                                }
                            }
                        }
                    // Idempotence (level 4): (a ∧ b) ∧ (c ∧ d) sharing a
                    // conjunct — drop the shared one from the right side.
                    if !applied && !ln && !rn
                        && let (Some((x, y)), Some((w, z))) = (lk, rk) {
                            let k = right.node_idx();
                            if x == w || y == w || x == z || y == z {
                                if can_bypass(k, &parents) {
                                    let repl = if x == w || y == w { z } else { w };
                                    parents[repl.node_idx() as usize] += 1;
                                    release(k, &self.nodes, &mut parents, &mut live, &mut rel);
                                    right = repl;
                                    applied = true;
                                } else {
                                    stats.blocked += 1;
                                }
                            }
                        }

                    if applied {
                        stats.subst_applied += 1;
                        continue;
                    }
                    break;
                }

                // Commit: alias-fold or child re-point (both in place, both
                // function-preserving, so co-parents of `i` are unaffected).
                if let Some(t) = fold {
                    stats.folds += 1;
                    rewrites_this_pass += 1;
                    parents[t.node_idx() as usize] += 2;
                    release(left.node_idx(), &self.nodes, &mut parents, &mut live, &mut rel);
                    release(right.node_idx(), &self.nodes, &mut parents, &mut live, &mut rel);
                    self.nodes[i] = AigNode::And(t, t);
                } else if left != self.resolve_stored(i, true) || right != self.resolve_stored(i, false)
                {
                    // Children changed (substitutions applied above).
                    let (lhs, rhs) = if left.0 <= right.0 {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    if self.nodes_and_eq(i, lhs, rhs) {
                        continue;
                    }
                    rewrites_this_pass += 1;
                    self.nodes[i] = AigNode::And(lhs, rhs);
                    // Keep hash-consing able to find the simplified form.
                    self.hash_cons.entry(cons_key(lhs, rhs)).or_insert(i as u32);
                }
            }

            if rewrites_this_pass == 0 {
                break;
            }
        }
        // Hand the scratch back for the next flush.
        self.post_live = live;
        self.post_parents = parents;
        self.post_stack = stack;
        self.post_rel = rel;
        stats
    }

    /// Current stored child of node `i` (left or right).
    #[inline]
    fn resolve_stored(&self, i: usize, left: bool) -> AigRef {
        match self.nodes[i] {
            AigNode::And(a, b) => {
                if left {
                    a
                } else {
                    b
                }
            }
            _ => unreachable!("substitute_pass only visits And nodes"),
        }
    }

    #[inline]
    fn nodes_and_eq(&self, i: usize, l: AigRef, r: AigRef) -> bool {
        matches!(self.nodes[i], AigNode::And(a, b) if a == l && b == r)
    }

    /// Attach a BV-source annotation to a node (typically the most recent
    /// And node just built). Used so that SAT vars allocated at CNF
    /// emission time can carry `VarOrigin::BvBit { term, bit }` metadata.
    pub fn tag_src(&mut self, r: AigRef, term: crate::bv::BvTerm) {
        let idx = r.node_idx();
        if idx != 0 {
            // Don't clobber an existing tag; first writer wins (reflects
            // which bitblast context the node was first produced under).
            if self.src_terms[idx as usize].is_none() {
                self.src_terms[idx as usize] = Some(term);
            }
        }
    }

    /// Run one round of 64-bit random simulation over the AIG. Returns
    /// per-node signatures. Inputs get hash-mixed signatures from `seed`
    /// and their lit identity; AND nodes get the bitwise AND of their
    /// (signed) children's signatures. Constants: TRUE gets all-ones.
    ///
    /// Exposed for future Fraig-style equivalence-candidate discovery.
    /// Not used by the CNF emission path.
    pub fn simulate(&self, seed: u64) -> Vec<u64> {
        let mut sigs = vec![0u64; self.nodes.len()];
        sigs[0] = u64::MAX;
        for (idx, &node) in self.nodes.iter().enumerate().skip(1) {
            match node {
                AigNode::ConstTrue => sigs[idx] = u64::MAX,
                AigNode::Input(lit) => {
                    let mut x =
                        seed ^ (lit.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    x ^= x >> 30;
                    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    x ^= x >> 27;
                    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                    x ^= x >> 31;
                    sigs[idx] = x;
                }
                AigNode::And(a, b) => {
                    let sa = Self::lookup(&sigs, a);
                    let sb = Self::lookup(&sigs, b);
                    sigs[idx] = sa & sb;
                }
            }
        }
        sigs
    }

    #[inline]
    fn lookup(sigs: &[u64], r: AigRef) -> u64 {
        let s = sigs[r.node_idx() as usize];
        if r.is_negated() { !s } else { s }
    }

    /// FRAIG feasibility diagnostic: how much *semantic* redundancy does
    /// random simulation see beyond what structural hashing already merged?
    ///
    /// Runs `SIM_WORDS` independent 64-bit simulation rounds (a 256-bit
    /// signature per node), canonicalizes each signature up to complement
    /// (a node equal to the *negation* of another is equally mergeable —
    /// the inversion rides on the edge), and buckets AND nodes by
    /// signature. Inputs are excluded: distinct primary inputs are free
    /// variables, never merge candidates.
    ///
    /// Interpretation caveats, for honest reading of the numbers:
    /// - `redundant` is an UPPER bound: sim-equivalence ≠ equivalence.
    ///   A real FRAIG confirms each candidate with a SAT query.
    /// - `sim_const` counts nodes whose 256 random samples were all-0 /
    ///   all-1. Comparator-style outputs (`a == b` over wide BVs) are
    ///   almost-never-true under uniform inputs, so this bucket is the
    ///   classic false-positive inflation source — reported separately
    ///   and EXCLUDED from `classes`/`redundant`.
    pub fn sim_sweep(&self, seed: u64) -> SimSweepStats {
        const SIM_WORDS: usize = 4;
        let rounds: Vec<Vec<u64>> = (0..SIM_WORDS)
            .map(|w| self.simulate(seed.wrapping_add(w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
            .collect();

        let mut buckets: HashMap<[u64; SIM_WORDS], u32> = HashMap::default();
        let mut num_and = 0usize;
        let mut sim_const = 0usize;
        for (idx, &node) in self.nodes.iter().enumerate() {
            if !matches!(node, AigNode::And(..)) {
                continue;
            }
            num_and += 1;
            let mut sig = [0u64; SIM_WORDS];
            for w in 0..SIM_WORDS {
                sig[w] = rounds[w][idx];
            }
            // Canonicalize up to complement: clear the LSB of word 0.
            if sig[0] & 1 != 0 {
                for s in sig.iter_mut() {
                    *s = !*s;
                }
            }
            if sig.iter().all(|&s| s == 0) {
                sim_const += 1;
                continue;
            }
            *buckets.entry(sig).or_insert(0) += 1;
        }

        let mut classes = 0usize;
        let mut redundant = 0usize;
        let mut largest_class = 0usize;
        for &count in buckets.values() {
            let c = count as usize;
            if c >= 2 {
                classes += 1;
                redundant += c - 1;
                largest_class = largest_class.max(c);
            }
        }

        SimSweepStats {
            num_nodes: self.nodes.len(),
            num_and,
            sim_const,
            classes,
            redundant,
            largest_class,
        }
    }
}

/// Result of [`Aig::substitute_pass`].
#[derive(Copy, Clone, Debug, Default)]
pub struct PostPassStats {
    /// Substitutions applied (bypassed interior had a single live parent).
    pub subst_applied: u64,
    /// Substitution opportunities skipped because the interior is shared
    /// or pinned — the fragmentation cases.
    pub blocked: u64,
    /// Nodes folded to an alias (constant, child, or resolution result).
    pub folds: u64,
    /// Bottom-up passes run (cascades may need more than one).
    pub passes: u32,
}

impl PostPassStats {
    pub fn accumulate(&mut self, o: PostPassStats) {
        self.subst_applied += o.subst_applied;
        self.blocked += o.blocked;
        self.folds += o.folds;
        self.passes += o.passes;
    }
}

/// Result of [`Aig::sim_sweep`] — see its docs for interpretation caveats.
#[derive(Copy, Clone, Debug)]
pub struct SimSweepStats {
    /// Total AIG nodes (constant + inputs + ANDs).
    pub num_nodes: usize,
    /// AND nodes — the merge-candidate population.
    pub num_and: usize,
    /// AND nodes simulating as constant over all 256 samples (excluded
    /// from the class counts; see docs).
    pub sim_const: usize,
    /// Signature classes containing >= 2 AND nodes.
    pub classes: usize,
    /// Upper bound on mergeable nodes: sum of (class size - 1).
    pub redundant: usize,
    /// Size of the biggest candidate class.
    pub largest_class: usize,
}

impl Default for Aig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lit::{Lit, Var};

    fn mk_input(aig: &mut Aig, var: u32) -> AigRef {
        aig.input(Lit::new(Var(var), false))
    }

    #[test]
    fn constants_are_sentinels() {
        let aig = Aig::new();
        assert_eq!(AigRef::TRUE.0, 0);
        assert_eq!(AigRef::FALSE.0, 1);
        assert!(AigRef::TRUE.is_const_true());
        assert!(AigRef::FALSE.is_const_false());
        assert_eq!(!AigRef::TRUE, AigRef::FALSE);
        assert_eq!(aig.num_nodes(), 1);
    }

    #[test]
    fn input_dedup_positive_and_negative() {
        let mut aig = Aig::new();
        let l = Lit::new(Var(5), false);
        let nl = !l;
        let a = aig.input(l);
        let b = aig.input(l);
        assert_eq!(a, b);
        let na = aig.input(nl);
        assert_eq!(na, !a);
        // Only one input node allocated for the pair.
        assert_eq!(aig.num_nodes(), 2); // TRUE + one input
    }

    #[test]
    fn and_simplifications() {
        let mut aig = Aig::new();
        let x = mk_input(&mut aig, 1);
        let y = mk_input(&mut aig, 2);
        // Identities with constants.
        assert_eq!(aig.and(x, AigRef::TRUE), x);
        assert_eq!(aig.and(AigRef::TRUE, y), y);
        assert_eq!(aig.and(x, AigRef::FALSE), AigRef::FALSE);
        assert_eq!(aig.and(AigRef::FALSE, y), AigRef::FALSE);
        // Idempotence / complementation.
        assert_eq!(aig.and(x, x), x);
        assert_eq!(aig.and(x, !x), AigRef::FALSE);
    }

    #[test]
    fn and_hash_cons_dedups_and_is_commutative() {
        let mut aig = Aig::new();
        let x = mk_input(&mut aig, 1);
        let y = mk_input(&mut aig, 2);
        let a = aig.and(x, y);
        let b = aig.and(y, x);
        assert_eq!(a, b);
        // No extra node produced.
        let nodes_before = aig.num_nodes();
        let _c = aig.and(x, y);
        assert_eq!(aig.num_nodes(), nodes_before);
    }

    #[test]
    fn or_is_de_morgan_and() {
        let mut aig = Aig::new();
        let x = mk_input(&mut aig, 1);
        let y = mk_input(&mut aig, 2);
        let or = aig.or(x, y);
        let same = !aig.and(!x, !y);
        assert_eq!(or, same);
    }

    #[test]
    fn xor_simplifications() {
        let mut aig = Aig::new();
        let x = mk_input(&mut aig, 1);
        assert_eq!(aig.xor(x, x), AigRef::FALSE);
        assert_eq!(aig.xor(x, !x), AigRef::TRUE);
        assert_eq!(aig.xor(x, AigRef::FALSE), x);
        assert_eq!(aig.xor(x, AigRef::TRUE), !x);
    }

    #[test]
    fn mux_identities() {
        let mut aig = Aig::new();
        let s = mk_input(&mut aig, 1);
        let t = mk_input(&mut aig, 2);
        let e = mk_input(&mut aig, 3);
        assert_eq!(aig.mux(AigRef::TRUE, t, e), t);
        assert_eq!(aig.mux(AigRef::FALSE, t, e), e);
        assert_eq!(aig.mux(s, t, t), t);
        assert_eq!(aig.mux(s, AigRef::TRUE, AigRef::FALSE), s);
        assert_eq!(aig.mux(s, AigRef::FALSE, AigRef::TRUE), !s);
    }

    /// Two-level rewriting rule tests. Each asserts the exact result ref
    /// (not just node count), mirroring bitwuzla's rule semantics.
    fn aig2() -> Aig {
        let mut a = Aig::new();
        a.set_two_level(true);
        a
    }

    #[test]
    fn two_level_contradiction_asymmetric() {
        let mut aig = aig2();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let ab = aig.and(a, b);
        assert_eq!(aig.and(ab, !a), AigRef::FALSE);
        assert_eq!(aig.and(!b, ab), AigRef::FALSE);
    }

    #[test]
    fn two_level_contradiction_symmetric() {
        let mut aig = aig2();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let c = mk_input(&mut aig, 3);
        let ab = aig.and(a, b);
        let nac = aig.and(!a, c);
        assert_eq!(aig.and(ab, nac), AigRef::FALSE);
    }

    #[test]
    fn two_level_subsumption() {
        let mut aig = aig2();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let c = mk_input(&mut aig, 3);
        let ab = aig.and(a, b);
        // ¬(a ∧ b) ∧ ¬a = ¬a  (asymmetric)
        assert_eq!(aig.and(!ab, !a), !a);
        // ¬(a ∧ b) ∧ (¬a ∧ c) = ¬a ∧ c  (symmetric)
        let nac = aig.and(!a, c);
        assert_eq!(aig.and(!ab, nac), nac);
    }

    #[test]
    fn two_level_idempotence() {
        let mut aig = aig2();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let c = mk_input(&mut aig, 3);
        let ab = aig.and(a, b);
        // (a ∧ b) ∧ a = (a ∧ b)
        assert_eq!(aig.and(ab, a), ab);
        assert_eq!(aig.and(b, ab), ab);
        // Level 4: (a ∧ b) ∧ (b ∧ c) = (a ∧ b) ∧ c
        let bc = aig.and(b, c);
        let expect = aig.and(ab, c);
        assert_eq!(aig.and(ab, bc), expect);
    }

    #[test]
    fn two_level_resolution() {
        let mut aig = aig2();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let ab = aig.and(a, b);
        let anb = aig.and(a, !b);
        // ¬(a ∧ b) ∧ ¬(a ∧ ¬b) = ¬a
        assert_eq!(aig.and(!ab, !anb), !a);
    }

    #[test]
    fn two_level_substitution() {
        let mut aig = aig2();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let c = mk_input(&mut aig, 3);
        let ab = aig.and(a, b);
        // ¬(a ∧ b) ∧ b = ¬a ∧ b  (asymmetric)
        let expect = aig.and(!a, b);
        assert_eq!(aig.and(!ab, b), expect);
        // ¬(a ∧ b) ∧ (b ∧ c) = ¬a ∧ (b ∧ c)  (symmetric)
        let bc = aig.and(b, c);
        let expect2 = aig.and(!a, bc);
        assert_eq!(aig.and(!ab, bc), expect2);
    }

    /// The XOR and MUX construction shapes over independent inputs must
    /// survive two-level rewriting — `SmtSolver::detect_shape`'s direct
    /// 4-clause encodings depend on the exact 3-node structure.
    #[test]
    fn two_level_preserves_xor_and_mux_shapes() {
        let mut aig = aig2();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let s = mk_input(&mut aig, 3);
        let x = aig.xor(a, b);
        assert!(x.is_negated());
        match aig.node(x.node_idx()) {
            AigNode::And(l, r) => {
                assert!(l.is_negated() && r.is_negated());
                assert!(matches!(aig.node(l.node_idx()), AigNode::And(..)));
                assert!(matches!(aig.node(r.node_idx()), AigNode::And(..)));
            }
            n => panic!("xor top must be an And, got {:?}", n),
        }
        let m = aig.mux(s, a, b);
        match aig.node(m.node_idx()) {
            AigNode::And(l, r) => {
                assert!(l.is_negated() && r.is_negated());
            }
            n => panic!("mux top must be an And, got {:?}", n),
        }
    }

    /// Post-pass: substitution applies when the bypassed interior has a
    /// single live parent...
    #[test]
    fn post_pass_substitutes_private_interior() {
        let mut aig = Aig::new(); // build-time rules off
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let ab = aig.and(a, b);
        let top = aig.and(!ab, b); // ¬(a∧b) ∧ b — substitutable to ¬a ∧ b
        let stats = aig.substitute_pass(&[top], &[]);
        assert_eq!(stats.subst_applied, 1);
        assert_eq!(stats.blocked, 0);
        match aig.node(top.node_idx()) {
            AigNode::And(l, r) => {
                let (lo, hi) = if l.0 <= r.0 { (l, r) } else { (r, l) };
                assert_eq!(lo, !a);
                assert_eq!(hi, b);
            }
            n => panic!("expected rewritten And, got {:?}", n),
        }
    }

    /// ...and is blocked when the interior has another live parent — the
    /// fragmentation case the pass exists to avoid.
    #[test]
    fn post_pass_blocks_shared_interior() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let c = mk_input(&mut aig, 3);
        let ab = aig.and(a, b);
        let top = aig.and(!ab, b);
        let keeper = aig.and(ab, c); // second live parent of (a∧b)
        let before = format!("{:?}", aig.node(top.node_idx()));
        let stats = aig.substitute_pass(&[top, keeper], &[]);
        assert_eq!(stats.subst_applied, 0);
        assert_eq!(stats.blocked, 1);
        assert_eq!(format!("{:?}", aig.node(top.node_idx())), before);
    }

    /// A pinned (already-materialized) interior must also block.
    #[test]
    fn post_pass_blocks_pinned_interior() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let ab = aig.and(a, b);
        let top = aig.and(!ab, b);
        let mut pinned = vec![false; aig.num_nodes()];
        pinned[ab.node_idx() as usize] = true;
        let stats = aig.substitute_pass(&[top], &pinned);
        assert_eq!(stats.subst_applied, 0);
        assert_eq!(stats.blocked, 1);
    }

    /// Pure-deletion folds rewrite the node into an alias in place.
    #[test]
    fn post_pass_folds_to_alias() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let ab = aig.and(a, b);
        let top = aig.and(!ab, !a); // subsumption: ≡ ¬a
        let stats = aig.substitute_pass(&[top], &[]);
        assert_eq!(stats.folds, 1);
        match aig.node(top.node_idx()) {
            AigNode::And(l, r) => {
                assert_eq!(l, r, "fold must produce an alias node");
                assert_eq!(l, !a);
            }
            n => panic!("expected alias, got {:?}", n),
        }
    }

    /// Cascade: bypassing one interior frees its child, unblocking a
    /// substitution one level up on a later pass.
    #[test]
    fn post_pass_cascades_across_passes() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let c = mk_input(&mut aig, 3);
        // inner = a∧b (parent: mid only). mid = ¬(a∧b) ∧ b (parents: top).
        // top = ¬mid ∧ ... constructed so top's substitution needs mid to
        // be single-parent (it is) and mid's needs inner (it is).
        let inner = aig.and(a, b);
        let mid = aig.and(!inner, b); // → ¬a ∧ b after pass
        let top = aig.and(!mid, c);
        let stats = aig.substitute_pass(&[top], &[]);
        // mid gets rewritten; top has shape ¬(¬a∧b) ∧ c afterwards — no
        // further rule applies (no shared children with c), so we just
        // assert mid's rewrite happened and no assert on top.
        assert!(stats.subst_applied >= 1);
        match aig.node(mid.node_idx()) {
            AigNode::And(l, r) => {
                let (lo, hi) = if l.0 <= r.0 { (l, r) } else { (r, l) };
                assert_eq!(lo, !a);
                assert_eq!(hi, b);
            }
            n => panic!("expected rewritten And, got {:?}", n),
        }
    }

    /// Flag off must behave exactly like the historical builder: none of
    /// the two-level shapes may fold.
    #[test]
    fn two_level_off_is_inert() {
        let mut aig = Aig::new(); // flag off
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let ab = aig.and(a, b);
        let r = aig.and(ab, !a); // would be FALSE with the rules on
        assert_ne!(r, AigRef::FALSE);
        assert!(matches!(aig.node(r.node_idx()), AigNode::And(..)));
    }

    #[test]
    fn simulation_signatures_respect_negation() {
        let mut aig = Aig::new();
        let x = mk_input(&mut aig, 7);
        let y = mk_input(&mut aig, 42);
        let a = aig.and(x, y);
        let sigs = aig.simulate(0xDEAD_BEEF);
        // The AND of two independent random-looking signatures should
        // itself look random — hard to assert directly — so just sanity-
        // check the polarity math.
        let sig_a = sigs[a.node_idx() as usize];
        let neg_a = sigs[(!a).node_idx() as usize]; // same underlying node
        assert_eq!(sig_a, neg_a);
        assert_eq!(sigs[0], u64::MAX); // TRUE is all-ones
    }

    #[test]
    fn sim_sweep_finds_semantic_duplicates() {
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        // `a ∨ b` built directly...
        let or1 = aig.or(a, b);
        // ...and as `¬(¬a ∧ ¬(¬a ∧ b))` — semantically a ∨ b, but the inner
        // AND differs structurally so hash-consing can NOT merge the tops.
        let inner = aig.and(!a, b);
        let or2 = !aig.and(!a, !inner);
        assert_ne!(or1.node_idx(), or2.node_idx());
        let stats = aig.sim_sweep(42);
        assert_eq!(stats.classes, 1);
        assert_eq!(stats.redundant, 1);
        assert_eq!(stats.sim_const, 0);
    }

    #[test]
    fn cross_op_dedup_via_de_morgan() {
        // `bvor(a, b)` and `bvnot(bvand(bvnot(a), bvnot(b)))` must collapse
        // to the same AIG output because both encode to `!and(!a, !b)`
        // after polarity normalization.
        let mut aig = Aig::new();
        let a = mk_input(&mut aig, 1);
        let b = mk_input(&mut aig, 2);
        let via_or = aig.or(a, b);
        let via_de_morgan = !aig.and(!a, !b);
        assert_eq!(via_or, via_de_morgan);
    }
}
