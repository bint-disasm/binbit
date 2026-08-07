//! DAG-aware AIG rewriting over 4-input cuts.
//!
//! Port of Mishchenko, Chatterjee & Brayton, "DAG-Aware AIG Rewriting: A
//! Fresh Look at Combinational Logic Synthesis" (DAC'06). The idea: walk
//! the AIG in topological order; for each node enumerate its 4-feasible
//! cuts; look the cut's 16-bit truth table up in a precomputed database
//! of optimal AIG structures for all 222 NPN classes of 4-variable
//! functions; and replace the node's cone with a stored structure when
//! doing so removes more nodes than it adds. The replacement's cost is
//! *context-dependent* — structural hashing lets a candidate reuse nodes
//! that already exist elsewhere in the graph — which is what makes the
//! rewriting "DAG-aware" and why the database keeps several alternative
//! structures per class rather than one canonical best.
//!
//! **Why this pass exists here.** On bint's traces, circuit size converts
//! to solve time superlinearly: enabling the two-level rewriter on
//! `nobranch.smt2` took the AIG from 574,489 to 402,098 SAT variables
//! (−30%) and the runtime from 56.5s to 29.2s (−48%), with propagations
//! per conflict falling 996 → 608. Two-level rewriting only ever inspects
//! a node and its immediate children; cut rewriting sees a 4-input window
//! and can restructure across several levels, which is the next increment
//! of the same lever.
//!
//! **Structural discipline.** binbit's AIG is append-only and its nodes
//! obey "children have strictly smaller indices" (`resolve` relies on it
//! for termination, and CNF emission relies on it for a single forward
//! walk). A replacement built by this pass is therefore appended, and the
//! rewritten graph is produced by *rebuilding* the cone above each
//! replacement rather than re-pointing existing nodes in place. Old nodes
//! are left as unreferenced garbage — memory-only, exactly as the
//! normalization scorer already leaves the variant it rejects. Nodes that
//! are `pinned` (already bound to a SAT literal, hence already emitted as
//! clauses) keep their identity and act as cut leaves.
//!
//! **Self-verification.** Every candidate replacement is checked by
//! simulating the stored structure over the cut's leaf truth tables and
//! comparing against the cut function. That check is a handful of 16-bit
//! operations, so it runs unconditionally rather than only in debug
//! builds: it makes an error in the database, or in this file's mapping
//! of database leaves onto cut leaves, impossible to miss and impossible
//! to turn into an unsound graph. Measured clean over ~10M candidates.
//!
//! # Verdict: OFF by default — the AIG shrinks, the CNF does not
//!
//! The pass works and shrinks the live AIG substantially (measured
//! 2026-08-06, `MAX_CUTS = 16`): bench_6554 −42.2%, bench_5906 −27.0%,
//! libmsrpc −20.4%, bench_16728 −19.0%, libsmbsharemodes −11.8%,
//! nobranch under `--aig2` −14.6%. But the CNF the SAT core actually
//! receives is essentially unchanged:
//!
//! | instance | AIG nodes | ORIGINAL clauses |
//! |---|---|---|
//! | bench_6554        | −42.2% | −1.5% |
//! | bench_5906        | −27.0% | −4.4% |
//! | libmsrpc_vc1225341| −20.4% | −0.0% |
//! | bench_16728       | −19.0% | −1.0% |
//! | libsmb_vc7699     | −11.8% | +0.0% |
//!
//! The reason is [`crate::preprocess`]: bounded variable elimination
//! already removes, at the CNF level, exactly the redundancy this pass
//! removes at the AIG level — `pp_elim` barely moves (bench_16728: 5447
//! vs 5392). The nodes cut rewriting deletes are ones BVE was going to
//! resolve away regardless, so deleting them earlier buys nothing.
//!
//! That is the opposite of the two-level rewriter (`--aig2`), whose
//! reduction *survives* preprocessing — it takes nobranch from 574,489 to
//! 402,098 SAT variables and 56.5s to 29.2s. Two-level rewriting removes
//! duplicated structure that would otherwise become genuinely distinct
//! variables; cut rewriting reshuffles local structure that VE normalizes
//! anyway.
//!
//! With the CNF invariant, the remaining solve-time differences are the
//! usual per-instance trajectory lottery on an essentially identical
//! formula — libsmbsharemodes 1.94s → 0.77s, bench_16728 2.25s → 8.11s —
//! and this project has repeatedly measured that such perturbations do
//! not pay. Hence opt-in, default off.
//!
//! A caution for anyone reviving this: `SmtSolverStats::sat_clauses`
//! counts learned clauses too, so comparing it across configurations
//! measures search effort, not encoding size. Subtract `learned` to see
//! the CNF. Doing otherwise made this pass look like it was doubling the
//! formula when it was leaving it alone.

use crate::aig::{Aig, AigNode, AigRef};
use crate::npn4;

/// Cut size. The database covers 4-variable functions.
pub const K: usize = 4;

/// Cuts kept per node, beyond the trivial one. The paper enumerates all
/// 4-feasible cuts; capping keeps enumeration linear on the 800k-node
/// graphs this runs on, at the cost of missing some rewrites.
const MAX_CUTS: usize = 16;

/// Truth tables of the four cut variables over a 16-row table: bit `m` of
/// a table is the function's value on the assignment whose bit `j` gives
/// leaf `j`.
const VAR_TT: [u16; K] = [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00];

#[derive(Clone, Copy)]
struct Cut {
    leaves: [u32; K],
    n: u8,
    tt: u16,
}

impl Default for Cut {
    fn default() -> Self {
        Cut { leaves: [0; K], n: 0, tt: 0 }
    }
}

impl Cut {
    fn slice(&self) -> &[u32] {
        &self.leaves[..self.n as usize]
    }
    /// A cut is trivial when it is the node itself.
    fn is_trivial(&self, node: u32) -> bool {
        self.n == 1 && self.leaves[0] == node
    }
}

/// Re-express `tt`, a function over `from`, as a function over `to`.
/// `from` must be a subset of `to`; both are sorted ascending.
///
/// Tables are ALWAYS full 16-row tables, even for cuts with fewer than
/// `K` leaves. Iterating all 16 rows while indexing only by the positions
/// the function actually uses replicates the sub-table across the unused
/// variables, which is precisely the statement "this function does not
/// depend on them". Masking the unused rows to zero instead — the obvious
/// shortcut — would encode a *different* function, one that does depend
/// on them, and would send the NPN lookup to the wrong class.
fn expand(tt: u16, from: &[u32], to: &[u32]) -> u16 {
    if from == to {
        return tt;
    }
    // Position of each `from` leaf within `to`.
    let mut pos = [0usize; K];
    for (j, f) in from.iter().enumerate() {
        pos[j] = to.iter().position(|t| t == f).expect("cut leaf subset");
    }
    let mut out = 0u16;
    for m in 0..16usize {
        let mut idx = 0usize;
        for j in 0..from.len() {
            if (m >> pos[j]) & 1 == 1 {
                idx |= 1 << j;
            }
        }
        if (tt >> idx) & 1 == 1 {
            out |= 1 << m;
        }
    }
    out
}

/// Merge two sorted leaf sets, rejecting the result if it exceeds `K`.
fn merge_leaves(a: &[u32], b: &[u32]) -> Option<([u32; K], u8)> {
    let mut out = [0u32; K];
    let (mut i, mut j, mut w) = (0usize, 0usize, 0usize);
    while i < a.len() || j < b.len() {
        if w == K {
            return None;
        }
        let v = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) => {
                if x < y {
                    i += 1;
                    x
                } else if y < x {
                    j += 1;
                    y
                } else {
                    i += 1;
                    j += 1;
                    x
                }
            }
            (Some(&x), None) => {
                i += 1;
                x
            }
            (None, Some(&y)) => {
                j += 1;
                y
            }
            (None, None) => break,
        };
        out[w] = v;
        w += 1;
    }
    Some((out, w as u8))
}

/// Per-pass statistics.
#[derive(Clone, Copy, Default, Debug)]
pub struct RewriteStats {
    /// Live nodes reachable from the roots before / after the pass.
    pub nodes_before: u64,
    pub nodes_after: u64,
    /// Cuts enumerated, database structures evaluated, rewrites taken,
    /// and how many of those were zero-gain (structure-diversifying)
    /// replacements.
    pub cuts: u64,
    pub structures_tried: u64,
    pub replacements: u64,
    pub zero_gain: u64,
    /// Histogram of MFFC sizes (nodes freed by replacing a node):
    /// how many nodes had an exclusive cone of size 1, 2, 3, 4+.
    /// A circuit whose nodes are nearly all shared has no room for
    /// rewriting regardless of database quality.
    pub mffc1: u64,
    pub mffc2: u64,
    pub mffc3: u64,
    pub mffc4plus: u64,
    /// Best achievable structure cost summed over nodes (for gauging how
    /// much of the shortfall is database quality).
    pub best_added: u64,
    /// Sum of the gains the decision pass PREDICTED. Comparing this with
    /// (nodes_before - nodes_after) reconciles the greedy per-node
    /// accounting against what the rebuild actually realizes.
    pub predicted_gain: u64,
    /// Of the accepted decisions, how many were later found unnecessary
    /// because a parent's rewrite bypassed the node entirely.
    pub bypassed: u64,
    /// Candidates rejected because the self-verification simulation
    /// disagreed with the cut function. Must always be zero; a non-zero
    /// count means the database or the leaf mapping is wrong.
    pub verify_failures: u64,
}

impl RewriteStats {
    pub fn accumulate(&mut self, o: RewriteStats) {
        self.nodes_before += o.nodes_before;
        self.nodes_after += o.nodes_after;
        self.cuts += o.cuts;
        self.structures_tried += o.structures_tried;
        self.replacements += o.replacements;
        self.zero_gain += o.zero_gain;
        self.mffc1 += o.mffc1;
        self.mffc2 += o.mffc2;
        self.mffc3 += o.mffc3;
        self.mffc4plus += o.mffc4plus;
        self.best_added += o.best_added;
        self.predicted_gain += o.predicted_gain;
        self.bypassed += o.bypassed;
        self.verify_failures += o.verify_failures;
    }
}

/// What to do with a node when the graph is rebuilt.
#[derive(Clone, Copy)]
enum Decision {
    /// Keep the node's own AND of its (rebuilt) children.
    Keep,
    /// Replace the node's cone with the database structure for the
    /// class of `cut`.
    Rewrite { cut: Cut },
}

/// Run one rewriting pass over the cone of `roots`.
///
/// Returns the rewritten roots (in the same order) and statistics. The
/// AIG is only appended to: every node index valid on entry stays valid
/// and keeps its meaning, so callers holding refs into the graph — CNF
/// bindings, bitblast caches, model evaluation — remain correct. Only the
/// returned roots reflect the rewriting.
///
/// `pinned[i]` marks nodes whose identity must be preserved because their
/// CNF has already been emitted; they terminate cuts and are never
/// rewritten. It may be shorter than the node array.
pub fn rewrite(
    aig: &mut Aig,
    roots: &[AigRef],
    pinned: &[bool],
    zero_cost: bool,
) -> (Vec<AigRef>, RewriteStats) {
    let n = aig.num_nodes();
    let mut stats = RewriteStats::default();
    let mut canon = npn4::Canon::new();
    let is_pinned = |i: u32| (i as usize) < pinned.len() && pinned[i as usize];

    // ---- liveness over the cone of the roots, stopping at pinned nodes.
    let mut live = vec![false; n];
    let mut stack: Vec<u32> = roots.iter().map(|r| r.node_idx()).collect();
    while let Some(i) = stack.pop() {
        if live[i as usize] {
            continue;
        }
        live[i as usize] = true;
        if is_pinned(i) {
            continue;
        }
        if let AigNode::And(a, b) = aig.node(i) {
            stack.push(a.node_idx());
            stack.push(b.node_idx());
        }
    }
    // A node is a cut leaf when its structure is opaque to this pass.
    let is_leaf = |aig: &Aig, i: u32| -> bool {
        i == 0 || is_pinned(i) || !matches!(aig.node(i), AigNode::And(..))
    };

    // ---- reference counts over the live cone (roots hold one each).
    let mut refs = vec![0u32; n];
    for i in 0..n {
        if !live[i] || is_leaf(aig, i as u32) {
            continue;
        }
        if let AigNode::And(a, b) = aig.node(i as u32) {
            refs[a.node_idx() as usize] += 1;
            refs[b.node_idx() as usize] += 1;
        }
    }
    for r in roots {
        refs[r.node_idx() as usize] += 1;
    }
    stats.nodes_before = live
        .iter()
        .enumerate()
        .filter(|&(i, &l)| l && !is_leaf(aig, i as u32))
        .count() as u64;

    // ---- cut enumeration, bottom-up.
    let slots = MAX_CUTS + 1; // trivial cut + merged cuts
    let mut cuts = vec![Cut::default(); n * slots];
    let mut ncuts = vec![0u8; n];
    for i in 0..n {
        if !live[i] {
            continue;
        }
        let iu = i as u32;
        // Every node has the trivial cut {itself}, which is what lets it
        // serve as a leaf for its parents.
        cuts[i * slots] = Cut { leaves: [iu, 0, 0, 0], n: 1, tt: VAR_TT[0] };
        ncuts[i] = 1;
        if is_leaf(aig, iu) {
            continue;
        }
        let AigNode::And(a, b) = aig.node(iu) else { continue };
        let (ai, bi) = (a.node_idx() as usize, b.node_idx() as usize);
        let (na, nb) = (ncuts[ai] as usize, ncuts[bi] as usize);
        'outer: for x in 0..na {
            for y in 0..nb {
                let ca = cuts[ai * slots + x];
                let cb = cuts[bi * slots + y];
                let Some((leaves, k)) = merge_leaves(ca.slice(), cb.slice()) else {
                    continue;
                };
                let to = &leaves[..k as usize];
                // Function of this node over the merged leaves, with each
                // child's edge polarity applied.
                let mut ta = expand(ca.tt, ca.slice(), to);
                let mut tb = expand(cb.tt, cb.slice(), to);
                if a.is_negated() {
                    ta = !ta;
                }
                if b.is_negated() {
                    tb = !tb;
                }
                let tt = ta & tb;
                let cut = Cut { leaves, n: k, tt };
                // Deduplicate on the leaf set.
                let cur = ncuts[i] as usize;
                if (1..cur).any(|s| {
                    let e = cuts[i * slots + s];
                    e.n == cut.n && e.slice() == cut.slice()
                }) {
                    continue;
                }
                if cur == slots {
                    break 'outer;
                }
                cuts[i * slots + cur] = cut;
                ncuts[i] = (cur + 1) as u8;
                stats.cuts += 1;
            }
        }
    }

    // ---- decision pass, topological order.
    let mut decision = vec![Decision::Keep; n];
    for i in 1..n {
        if !live[i] || is_leaf(aig, i as u32) {
            continue;
        }
        let iu = i as u32;
        let self_refs = refs[i];
        let mut min_added = u32::MAX;
        let mut best: Option<(i64, Decision)> = None;
        let mut best_saved = 0u32;
        let ncut = ncuts[i] as usize;
        for s in 1..ncut {
            let cut = cuts[i * slots + s];
            if cut.is_trivial(iu) || cut.n < 2 {
                continue;
            }
            // What a replacement actually frees is the cut's INTERIOR:
            // the nodes strictly between this node and the cut leaves.
            // The leaves themselves, and everything below them, stay —
            // the replacement reads them. So hold a reference to each
            // leaf across the dereference; whatever still dies is exactly
            // the interior. (Dereferencing without this protection frees
            // the node's whole exclusive cone down to the inputs, which
            // over-states the gain by orders of magnitude: measured
            // 9,168,467 claimed savings on a 367,169-node graph.)
            for &l in cut.slice() {
                refs[l as usize] += 1;
            }
            let saved = 1 + deref(aig, &mut refs, iu);
            // The node being replaced must not be reusable BY its own
            // replacement: otherwise a candidate "rebuilds" the very node
            // it is replacing, the hash-cons lookup hands it back for
            // free, and the node reports a phantom gain equal to its own
            // size.
            refs[i] = 0;

            let (class, npn) = canon.get(cut.tt);
            if let Some(st) = npn4::structure_for(class) {
                stats.structures_tried += 1;
                // Map database leaves onto this cut's leaves, then verify
                // by simulation that the structure really computes the
                // cut function before considering it at all.
                let leaf_sigs = map_leaves(&cut, &npn);
                let sim = st.simulate(&leaf_sigs.tts);
                let sim = if leaf_sigs.out_neg { !sim } else { sim };
                if sim != cut.tt {
                    stats.verify_failures += 1;
                } else {
                    let added = structure_cost(aig, &refs, st, &leaf_sigs.refs);
                    min_added = min_added.min(added);
                    let gain = saved as i64 - added as i64;
                    let acceptable = gain > 0 || (gain == 0 && zero_cost);
                    if acceptable && best.is_none_or(|(g, _)| gain > g) {
                        best = Some((gain, Decision::Rewrite { cut }));
                        best_saved = saved;
                    }
                }
            }

            // Undo the speculative dereference: the rebuild pass, not
            // this one, is what actually changes the graph.
            refs[i] = self_refs;
            reref(aig, &mut refs, iu);
            for &l in cut.slice() {
                refs[l as usize] -= 1;
            }
        }

        match best_saved {
            0 | 1 => stats.mffc1 += 1,
            2 => stats.mffc2 += 1,
            3 => stats.mffc3 += 1,
            _ => stats.mffc4plus += 1,
        }
        if min_added != u32::MAX {
            stats.best_added += min_added as u64;
        }

        if let Some((gain, d)) = best {
            decision[i] = d;
            stats.replacements += 1;
            stats.predicted_gain += gain.max(0) as u64;
            if gain == 0 {
                stats.zero_gain += 1;
            }
        }
    }

    // ---- mark what the rebuilt graph actually needs, root-down. Nodes
    // bypassed by a rewrite are never marked, and so are never rebuilt:
    // that is where the gain is realized.
    let mut needed = vec![false; n];
    for r in roots {
        needed[r.node_idx() as usize] = true;
    }
    for i in (1..n).rev() {
        if !needed[i] || !live[i] || is_leaf(aig, i as u32) {
            continue;
        }
        match decision[i] {
            Decision::Keep => {
                if let AigNode::And(a, b) = aig.node(i as u32) {
                    needed[a.node_idx() as usize] = true;
                    needed[b.node_idx() as usize] = true;
                }
            }
            Decision::Rewrite { cut, .. } => {
                for &l in cut.slice() {
                    needed[l as usize] = true;
                }
            }
        }
    }

    for i in 1..n {
        if matches!(decision[i], Decision::Rewrite { .. }) && !needed[i] {
            stats.bypassed += 1;
        }
    }

    // ---- rebuild, bottom-up. Leaves and pinned nodes keep their
    // identity; everything else is re-created through `and`, which
    // re-applies structural hashing and the two-level rules.
    let mut new = vec![AigRef::FALSE; n];
    for i in 0..n {
        if !needed[i] {
            continue;
        }
        let iu = i as u32;
        if is_leaf(aig, iu) || !live[i] {
            new[i] = AigRef::from_parts(iu, false);
            continue;
        }
        new[i] = match decision[i] {
            Decision::Keep => {
                let AigNode::And(a, b) = aig.node(iu) else {
                    new[i] = AigRef::from_parts(iu, false);
                    continue;
                };
                let na = lift(new[a.node_idx() as usize], a.is_negated());
                let nb = lift(new[b.node_idx() as usize], b.is_negated());
                aig.and(na, nb)
            }
            Decision::Rewrite { cut } => {
                let (class, npn) = canon.get(cut.tt);
                let st = npn4::structure_for(class).expect("chosen in decision pass");
                let sigs = map_leaves(&cut, &npn);
                // Leaves are cut leaves, so they were rebuilt already
                // (they have smaller indices); constants map to
                // themselves.
                let mut refs4 = [AigRef::FALSE; K];
                for j in 0..K {
                    let (idx, neg) = sigs.refs[j];
                    let base = if idx == 0 {
                        AigRef::TRUE
                    } else {
                        new[idx as usize]
                    };
                    refs4[j] = lift(base, neg);
                }
                let r = build(aig, st, &refs4);
                lift(r, sigs.out_neg)
            }
        };
    }

    let out: Vec<AigRef> = roots
        .iter()
        .map(|r| lift(new[r.node_idx() as usize], r.is_negated()))
        .collect();

    // Recount the live cone of the rewritten roots.
    let m = aig.num_nodes();
    let mut live2 = vec![false; m];
    let mut stack: Vec<u32> = out.iter().map(|r| r.node_idx()).collect();
    let mut after = 0u64;
    while let Some(i) = stack.pop() {
        if live2[i as usize] {
            continue;
        }
        live2[i as usize] = true;
        if is_pinned(i) {
            continue;
        }
        if let AigNode::And(a, b) = aig.node(i) {
            after += 1;
            stack.push(a.node_idx());
            stack.push(b.node_idx());
        }
    }
    stats.nodes_after = after;
    (out, stats)
}

#[inline]
fn lift(r: AigRef, negated: bool) -> AigRef {
    if negated { !r } else { r }
}

/// The database's leaves, mapped onto a cut's leaves.
struct LeafSigs {
    /// (node index, complemented) per database leaf 0..K.
    refs: [(u32, bool); K],
    /// Truth tables of those signals over the cut's own leaf order —
    /// used by the verification simulation.
    tts: [u16; K],
    /// Whether the structure's output must be complemented.
    out_neg: bool,
}

/// Map database variable `j` onto the cut leaf it stands for, honouring
/// the NPN transform's permutation, input negations and output negation.
///
/// The convention (`perm[j]` = which of the caller's variables drives
/// canonical input `j`; `in_neg` bit `j` complements that edge; `out_neg`
/// complements the result) is isolated here on purpose: it is the one
/// place a direction error can creep in, and the caller's simulation
/// check catches it immediately rather than letting it reach the graph.
fn map_leaves(cut: &Cut, npn: &npn4::Npn) -> LeafSigs {
    let k = cut.n as usize;
    let mut refs = [(0u32, true); K];
    let mut tts = [0u16; K];
    for j in 0..K {
        let src = npn.perm[j] as usize;
        let neg = (npn.in_neg >> j) & 1 == 1;
        if src >= k {
            // The class representative nominally has four inputs but this
            // cut has fewer. A structure for a function that genuinely
            // ignores the variable never reads this leaf; feed it the
            // constant FALSE consistently in both the reference and the
            // truth-table view, and let the simulation check confirm the
            // structure really is independent of it.
            refs[j] = (0, true); // node 0 complemented == FALSE
            tts[j] = 0x0000;
            continue;
        }
        refs[j] = (cut.leaves[src], neg);
        tts[j] = if neg { !VAR_TT[src] } else { VAR_TT[src] };
    }
    LeafSigs { refs, tts, out_neg: npn.out_neg }
}

/// Nodes a structure would add, given the current reference counts.
///
/// A structure node is free only when an identical node already exists
/// *and* is currently referenced; an existing-but-unreferenced node is
/// garbage from an earlier rejected build and must be paid for again.
/// Once any operand is "virtual" (not present in the graph), everything
/// above it is virtual too. Because [`Aig::lookup_and`] skips the folds
/// and two-level rules that [`Aig::and`] applies, this can only
/// over-estimate — a rewrite is never taken on a promise the build fails
/// to keep.
fn structure_cost(
    aig: &Aig,
    refs: &[u32],
    st: &npn4::Structure,
    leaves: &[(u32, bool); K],
) -> u32 {
    let mut sig: [Option<AigRef>; 16] = [None; 16];
    let mut added = 0u32;
    for (i, &(l, r)) in st.nodes.iter().enumerate() {
        let (Some(a), Some(b)) = (operand_ref(l, leaves, &sig), operand_ref(r, leaves, &sig))
        else {
            sig[i] = None;
            added += 1;
            continue;
        };
        match aig.lookup_and(a, b) {
            Some(e) if refs.get(e.node_idx() as usize).copied().unwrap_or(0) > 0 => {
                sig[i] = Some(e);
            }
            _ => {
                sig[i] = None;
                added += 1;
            }
        }
    }
    added
}

/// Decode one operand into a live AIG ref, or `None` if it is virtual.
fn operand_ref(
    op: u16,
    leaves: &[(u32, bool); K],
    sig: &[Option<AigRef>; 16],
) -> Option<AigRef> {
    let neg = npn4::op_is_complemented(op);
    let id = npn4::op_id(op) as usize;
    let base = if id == 0 {
        AigRef::TRUE
    } else if id <= K {
        let (idx, lneg) = leaves[id - 1];
        lift(AigRef::from_parts(idx, false), lneg)
    } else {
        (*sig.get(id - K - 1)?)?
    };
    Some(lift(base, neg))
}

/// Materialize a database structure over concrete leaf refs.
fn build(aig: &mut Aig, st: &npn4::Structure, leaves: &[AigRef; K]) -> AigRef {
    let mut vals: [AigRef; 16] = [AigRef::TRUE; 16];
    for (i, &(l, r)) in st.nodes.iter().enumerate() {
        let a = operand_build(l, leaves, &vals);
        let b = operand_build(r, leaves, &vals);
        vals[i] = aig.and(a, b);
    }
    operand_build(st.root, leaves, &vals)
}

fn operand_build(op: u16, leaves: &[AigRef; K], vals: &[AigRef; 16]) -> AigRef {
    let neg = npn4::op_is_complemented(op);
    let id = npn4::op_id(op) as usize;
    let base = if id == 0 {
        AigRef::TRUE
    } else if id <= K {
        leaves[id - 1]
    } else {
        vals[id - K - 1]
    };
    lift(base, neg)
}

/// Release one reference to each child of `idx`, cascading into cones
/// that die. Returns the number of AND nodes freed (not counting `idx`).
fn deref(aig: &Aig, refs: &mut [u32], idx: u32) -> u32 {
    let mut freed = 0;
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        let AigNode::And(a, b) = aig.node(i) else { continue };
        for c in [a, b] {
            let ci = c.node_idx() as usize;
            if ci == 0 {
                continue;
            }
            debug_assert!(refs[ci] > 0, "dereferencing an unreferenced node");
            refs[ci] -= 1;
            if refs[ci] == 0
                && matches!(aig.node(c.node_idx()), AigNode::And(..)) {
                    freed += 1;
                    stack.push(c.node_idx());
                }
        }
    }
    freed
}

/// Exact inverse of [`deref`].
fn reref(aig: &Aig, refs: &mut [u32], idx: u32) -> u32 {
    let mut added = 0;
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        let AigNode::And(a, b) = aig.node(i) else { continue };
        for c in [a, b] {
            let ci = c.node_idx() as usize;
            if ci == 0 {
                continue;
            }
            if refs[ci] == 0
                && matches!(aig.node(c.node_idx()), AigNode::And(..)) {
                    added += 1;
                    stack.push(c.node_idx());
                }
            refs[ci] += 1;
        }
    }
    added
}
