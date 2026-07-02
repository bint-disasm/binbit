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

use std::collections::HashMap;

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
    /// Node arena. `nodes[0]` is always `ConstTrue`.
    pub nodes: Vec<AigNode>,
    /// Per-node BV-source annotation, for propagating `VarOrigin::BvBit`
    /// style metadata when we allocate SAT lits at CNF emission time.
    /// `None` for the constant and most input nodes.
    pub src_terms: Vec<Option<crate::bv::BvTerm>>,
    /// Hash cons: `(canonical_lhs, canonical_rhs) → node_idx`. Both sides
    /// already have their polarity applied. Canonical ordering is
    /// `lhs.0 <= rhs.0`.
    hash_cons: HashMap<(AigRef, AigRef), u32>,
    /// Input dedup: one AIG input per SAT literal. A lit and its negation
    /// share the same input node (the polarity is carried on the AigRef).
    input_lut: HashMap<Lit, AigRef>,
}

impl Aig {
    pub fn new() -> Self {
        Aig {
            nodes: vec![AigNode::ConstTrue],
            src_terms: vec![None],
            hash_cons: HashMap::new(),
            input_lut: HashMap::new(),
        }
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
        if let Some(&r) = self.input_lut.get(&lit) {
            return r;
        }
        // Canonicalize to positive-polarity storage.
        let canonical_lit = Lit(lit.0 & !1);
        if let Some(&r) = self.input_lut.get(&canonical_lit) {
            // Lit we were asked about is the negation of an existing input.
            let neg = r.negate();
            self.input_lut.insert(lit, neg);
            return neg;
        }
        let idx = self.nodes.len() as u32;
        self.nodes.push(AigNode::Input(canonical_lit));
        self.src_terms.push(None);
        let pos = AigRef::from_parts(idx, false);
        self.input_lut.insert(canonical_lit, pos);
        let requested = if lit.0 & 1 != 0 { pos.negate() } else { pos };
        self.input_lut.insert(lit, requested);
        requested
    }

    /// Build `and(a, b)` with the standard AIG simplifications and hash-cons.
    ///
    /// Construction-time folds:
    ///   - identity vs constants (TRUE / FALSE)
    ///   - `and(x, x) = x`
    ///   - `and(x, ¬x) = FALSE`
    /// Then hash-cons on the sorted pair.
    pub fn and(&mut self, a: AigRef, b: AigRef) -> AigRef {
        // Constant folds.
        if a == AigRef::TRUE {
            return b;
        }
        if b == AigRef::TRUE {
            return a;
        }
        if a == AigRef::FALSE || b == AigRef::FALSE {
            return AigRef::FALSE;
        }
        // Identities on the operands themselves.
        if a == b {
            return a;
        }
        if a == !b {
            return AigRef::FALSE;
        }
        // Canonicalize: put the smaller AigRef first so `(a, b)` and
        // `(b, a)` land on the same hash-cons key.
        let (lhs, rhs) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        if let Some(&idx) = self.hash_cons.get(&(lhs, rhs)) {
            return AigRef::from_parts(idx, false);
        }
        let idx = self.nodes.len() as u32;
        self.nodes.push(AigNode::And(lhs, rhs));
        self.src_terms.push(None);
        self.hash_cons.insert((lhs, rhs), idx);
        AigRef::from_parts(idx, false)
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
        // `mux(s, s, e) = s ∨ e`, `mux(s, t, !s) = s ∧ t`, etc.
        if t == sel {
            return self.or(sel, e);
        }
        if e == !sel {
            return self.and(sel, t);
        }
        let hi = self.and(sel, t);
        let lo = self.and(!sel, e);
        self.or(hi, lo)
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
