//! Precomputed 4-input NPN class database for [`crate::aigrw`].
//!
//! There are exactly 222 NPN equivalence classes of 4-variable Boolean
//! functions (functions equal up to permuting inputs, complementing
//! inputs, and complementing the output). For each class this module
//! stores a minimal AND-tree structure, generated offline by
//! `examples/gen_npn4.rs` and verified there by simulation.
//!
//! Operand encoding (shared with the generator):
//!   bit 0 = complement flag
//!   id (bits 1..): 0 = constant TRUE, 1..=4 = leaf 0..=3,
//!                  >= 5 = node (id - 5) of the same structure.
//! Every operand of `nodes[i]` refers to the constant, a leaf, or a node
//! with a strictly smaller index, so a structure is already in
//! topological order.

mod tables;

/// Truth tables of the four canonical inputs over a 16-row table.
pub const LEAF_TT: [u16; 4] = [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00];

#[inline]
pub const fn op_id(o: u16) -> u16 {
    o >> 1
}
#[inline]
pub const fn op_is_complemented(o: u16) -> bool {
    o & 1 != 0
}

/// A precomputed AIG structure over at most four leaves.
#[derive(Clone, Copy, Debug)]
pub struct Structure {
    /// AND nodes in topological order.
    pub nodes: &'static [(u16, u16)],
    /// Operand encoding the output (so it may be complemented, a leaf, or
    /// a constant).
    pub root: u16,
}

impl Structure {
    /// Simulate with arbitrary truth tables bound to the leaves.
    /// `leaves[j]` is the table bound to structure leaf `j`.
    pub fn simulate(&self, leaves: &[u16; 4]) -> u16 {
        let mut vals = [0u16; 16];
        let value = |o: u16, vals: &[u16; 16], leaves: &[u16; 4]| -> u16 {
            let id = op_id(o) as usize;
            let base = if id == 0 {
                0xFFFF
            } else if id <= 4 {
                leaves[id - 1]
            } else {
                vals[id - 5]
            };
            if op_is_complemented(o) { !base } else { base }
        };
        for (i, &(a, b)) in self.nodes.iter().enumerate() {
            vals[i] = value(a, &vals, leaves) & value(b, &vals, leaves);
        }
        value(self.root, &vals, leaves)
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// An NPN transform. `perm[j]` is the index of the CALLER's variable that
/// drives canonical input `j`; bit `j` of `in_neg` complements that edge;
/// `out_neg` complements the structure's output. The defining identity is
/// `apply(canonical, npn) == caller_tt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Npn {
    pub perm: [u8; 4],
    pub in_neg: u8,
    pub out_neg: bool,
}

/// Rebuild the caller's function from a canonical one: exactly what the
/// rewriter does when it wires a stored structure onto a cut.
///
/// Canonical input `j` is fed with the caller's variable `perm[j]`,
/// complemented when bit `j` of `in_neg` is set; the result is
/// complemented when `out_neg`. So for a caller assignment `x`,
/// `result[x] = canonical[m] ^ out_neg` with `m[j] = x[perm[j]] ^ in_neg_j`.
pub fn apply(canonical: u16, npn: &Npn) -> u16 {
    let mut out = 0u16;
    for x in 0..16usize {
        let mut m = 0usize;
        for j in 0..4 {
            let mut bit = (x >> npn.perm[j] as usize) & 1;
            if (npn.in_neg >> j) & 1 == 1 {
                bit ^= 1;
            }
            if bit == 1 {
                m |= 1 << j;
            }
        }
        let mut v = (canonical >> m) & 1;
        if npn.out_neg {
            v ^= 1;
        }
        if v == 1 {
            out |= 1 << x;
        }
    }
    out
}

/// Inverse of [`apply`]: the canonical table that, transformed by `npn`,
/// reproduces `tt`. Used only to search for the canonical form.
fn unapply(tt: u16, npn: &Npn) -> u16 {
    let mut out = 0u16;
    for m in 0..16usize {
        let mut x = 0usize;
        for j in 0..4 {
            let mut bit = (m >> j) & 1;
            if (npn.in_neg >> j) & 1 == 1 {
                bit ^= 1;
            }
            if bit == 1 {
                x |= 1 << npn.perm[j] as usize;
            }
        }
        let mut v = (tt >> x) & 1;
        if npn.out_neg {
            v ^= 1;
        }
        if v == 1 {
            out |= 1 << m;
        }
    }
    out
}

const PERMS: [[u8; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

/// Canonical class of `tt` (the lexicographically smallest table over all
/// 768 transforms) plus a transform `npn` with `apply(canon, npn) == tt`.
///
/// Brute force over the 768 transforms. Callers memoize (see
/// [`Canon`]), so this runs once per *distinct* cut function rather than
/// once per cut.
pub fn canonicalize(tt: u16) -> (u16, Npn) {
    let mut best_tt = u16::MAX;
    let mut best = Npn { perm: [0, 1, 2, 3], in_neg: 0, out_neg: false };
    for &perm in PERMS.iter() {
        for in_neg in 0..16u8 {
            for out_neg in [false, true] {
                let npn = Npn { perm, in_neg, out_neg };
                let c = unapply(tt, &npn);
                if c < best_tt {
                    best_tt = c;
                    best = npn;
                }
            }
        }
    }
    // `apply` is an involution in the sense we need: applying the same
    // transform to the canonical table reproduces `tt`. Verified for all
    // 65536 tables by the `canonical_round_trip` test.
    (best_tt, best)
}

/// Dense memo over the 65536 possible cut functions. One of these is
/// built per rewriting pass; canonicalization is only paid once per
/// distinct function encountered.
pub struct Canon {
    memo: Vec<Option<(u16, Npn)>>,
}

impl Canon {
    pub fn new() -> Self {
        Canon { memo: vec![None; 65536] }
    }
    #[inline]
    pub fn get(&mut self, tt: u16) -> (u16, Npn) {
        if let Some(v) = self.memo[tt as usize] {
            return v;
        }
        let v = canonicalize(tt);
        self.memo[tt as usize] = Some(v);
        v
    }
}

impl Default for Canon {
    fn default() -> Self {
        Self::new()
    }
}

/// The stored structure for a canonical class, if the class is known.
pub fn structure_for(canonical_tt: u16) -> Option<&'static Structure> {
    tables::CLASSES
        .binary_search_by_key(&canonical_tt, |&(c, _)| c)
        .ok()
        .map(|i| &tables::CLASSES[i].1)
}

/// A propagation-complete irredundant CNF definition of `y ↔ class(x)`,
/// stored as implicant cubes `(pos, neg)` over the canonical inputs:
/// each `on` cube `p` is the clause `(¬p ∨ y)`, each `off` cube `q` is
/// `(¬q ∨ ¬y)`. Unit propagation on these clauses derives every literal
/// the relation forces under any partial assignment (verified
/// exhaustively by the generator over all 3^5 partial assignments) —
/// unlike an ISOP, which drops consensus primes and their propagations
/// with them. Greedily minimized offline; no runtime cover computation
/// could afford the completeness check per cut.
#[derive(Clone, Copy, Debug)]
pub struct PcCover {
    pub on: &'static [(u8, u8)],
    pub off: &'static [(u8, u8)],
}

/// The propagation-complete cover for a canonical class.
pub fn pc_cover_for(canonical_tt: u16) -> Option<&'static PcCover> {
    tables::CLASSES
        .binary_search_by_key(&canonical_tt, |&(c, _)| c)
        .ok()
        .map(|i| &tables::PC_COVERS[i])
}

/// Map an implicant cube over the CANONICAL inputs to one over the
/// caller's inputs, for a function `f == apply(canonical, npn)`.
/// Canonical input `j` reads the caller's variable `perm[j]` through an
/// `in_neg_j` inverter, so a literal on canonical input `j` becomes a
/// literal on caller variable `perm[j]` with its sign flipped by
/// `in_neg_j`. (Output negation is the *caller's* concern: it swaps the
/// roles of the on- and off-cube lists, not the cubes themselves.)
#[inline]
pub fn map_cube(cube: (u8, u8), npn: &Npn) -> (u8, u8) {
    let (mut pos, mut neg) = (0u8, 0u8);
    for j in 0..4 {
        let m = 1u8 << j;
        let flip = npn.in_neg & m != 0;
        let target = 1u8 << npn.perm[j];
        if cube.0 & m != 0 {
            if flip {
                neg |= target;
            } else {
                pos |= target;
            }
        }
        if cube.1 & m != 0 {
            if flip {
                pos |= target;
            } else {
                neg |= target;
            }
        }
    }
    (pos, neg)
}

/// Number of stored classes (must be 222).
pub fn num_classes() -> usize {
    tables::CLASSES.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_222_classes() {
        assert_eq!(num_classes(), 222, "there are 222 NPN classes of 4-var functions");
    }

    #[test]
    fn classes_are_sorted_for_binary_search() {
        let c: Vec<u16> = tables::CLASSES.iter().map(|&(c, _)| c).collect();
        let mut s = c.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(c, s, "class table must be sorted and duplicate-free");
    }

    /// Every stored structure computes exactly its class function.
    #[test]
    fn structures_compute_their_class() {
        for &(c, st) in tables::CLASSES.iter() {
            assert_eq!(st.simulate(&LEAF_TT), c, "structure for class {c:#06x} is wrong");
        }
    }

    /// Structures are well formed: operands reference only the constant,
    /// a leaf, or a strictly earlier node.
    #[test]
    fn structures_are_topological() {
        for &(c, st) in tables::CLASSES.iter() {
            for (i, &(a, b)) in st.nodes.iter().enumerate() {
                for o in [a, b, st.root] {
                    let id = op_id(o) as usize;
                    if id >= 5 {
                        assert!(
                            id - 5 < st.nodes.len(),
                            "class {c:#06x}: operand out of range"
                        );
                    }
                }
                for o in [a, b] {
                    let id = op_id(o) as usize;
                    if id >= 5 {
                        assert!(id - 5 < i, "class {c:#06x}: node {i} references a later node");
                    }
                }
            }
        }
    }

    /// The defining identity of the transform, over ALL 65536 functions.
    #[test]
    fn canonical_round_trip() {
        for f in 0..=u16::MAX {
            let (c, npn) = canonicalize(f);
            assert_eq!(apply(c, &npn), f, "round trip failed for {f:#06x}");
        }
    }

    /// Every function's canonical class is one we have a structure for.
    #[test]
    fn every_function_has_a_structure() {
        let mut canon = Canon::new();
        for f in 0..=u16::MAX {
            let (c, _) = canon.get(f);
            assert!(structure_for(c).is_some(), "no structure for class {c:#06x}");
        }
    }

    /// NPN-equivalent functions land in the same class.
    #[test]
    fn transforms_preserve_class() {
        let mut canon = Canon::new();
        for f in (0..=u16::MAX).step_by(97) {
            let base = canon.get(f).0;
            for &perm in PERMS.iter() {
                for in_neg in [0u8, 5, 15] {
                    for out_neg in [false, true] {
                        let g = apply(f, &Npn { perm, in_neg, out_neg });
                        assert_eq!(canon.get(g).0, base, "class differs under NPN transform");
                    }
                }
            }
        }
    }

    /// Truth table of an implicant cube over the canonical inputs.
    fn cube_tt(c: (u8, u8)) -> u16 {
        let mut t = 0xFFFFu16;
        for i in 0..4 {
            if c.0 & (1 << i) != 0 {
                t &= LEAF_TT[i];
            }
            if c.1 & (1 << i) != 0 {
                t &= !LEAF_TT[i];
            }
        }
        t
    }

    /// Every PC cover is a semantically exact definition: the on-cubes
    /// cover exactly the class function's on-set... on-cubes must UNION
    /// to the function (each is an implicant; PC forces full coverage),
    /// and symmetrically for off.
    #[test]
    fn pc_covers_are_exact() {
        for (i, &(c, _)) in tables::CLASSES.iter().enumerate() {
            let pc = &tables::PC_COVERS[i];
            let mut on = 0u16;
            for &cb in pc.on {
                let t = cube_tt(cb);
                assert_eq!(t & !c, 0, "class {c:#06x}: on-cube not an implicant");
                on |= t;
            }
            let mut off = 0u16;
            for &cb in pc.off {
                let t = cube_tt(cb);
                assert_eq!(t & c, 0, "class {c:#06x}: off-cube not an implicant of !f");
                off |= t;
            }
            assert_eq!(on, c, "class {c:#06x}: on-cubes don't cover the function");
            assert_eq!(off, !c, "class {c:#06x}: off-cubes don't cover the complement");
        }
    }

    /// Independent re-verification of propagation completeness (the
    /// generator checks this too; a second implementation guards against
    /// a shared bug). For every class, every partial assignment to
    /// (x, y): every literal the relation semantically forces must be
    /// derived by unit propagation on the cover, and every relation-
    /// inconsistent assignment must conflict.
    #[test]
    fn pc_covers_are_propagation_complete() {
        for (idx, &(f, _)) in tables::CLASSES.iter().enumerate() {
            let pc = &tables::PC_COVERS[idx];
            // Clause list as (pos, neg) masks over 5 vars, bit 4 = y.
            let mut clauses: Vec<(u8, u8)> = Vec::new();
            for &(p, n) in pc.on {
                clauses.push((n | 0x10, p));
            }
            for &(p, n) in pc.off {
                clauses.push((n, p | 0x10));
            }
            for code in 0..243usize {
                let (mut c, mut rho) = (code, [0u8; 5]);
                for v in rho.iter_mut() {
                    *v = [0, 1, 2][c % 3];
                    c /= 3;
                }
                // Semantic consequences over the 32 full assignments.
                let mut forced = [0u8; 5];
                let mut any = false;
                for m in 0..32usize {
                    if ((m >> 4) & 1) as u16 != (f >> (m & 15)) & 1 {
                        continue;
                    }
                    if (0..5).any(|i| {
                        rho[i] != 0 && rho[i] != if (m >> i) & 1 == 1 { 1 } else { 2 }
                    }) {
                        continue;
                    }
                    any = true;
                    for i in 0..5 {
                        let b = if (m >> i) & 1 == 1 { 1 } else { 2 };
                        forced[i] = if forced[i] == 0 || forced[i] == b { b } else { 3 };
                    }
                }
                // Unit propagation to fixpoint.
                let mut val = rho;
                let mut conflict = false;
                loop {
                    let mut changed = false;
                    for &(pos, neg) in &clauses {
                        let mut sat = false;
                        let mut free = 0;
                        let mut last = (0usize, false);
                        for i in 0..5usize {
                            let (p, n) = (pos >> i & 1 == 1, neg >> i & 1 == 1);
                            if !p && !n {
                                continue;
                            }
                            match val[i] {
                                0 => {
                                    free += 1;
                                    last = (i, p);
                                }
                                1 if p => sat = true,
                                2 if n => sat = true,
                                _ => {}
                            }
                        }
                        if sat {
                            continue;
                        }
                        if free == 0 {
                            conflict = true;
                        } else if free == 1 {
                            val[last.0] = if last.1 { 1 } else { 2 };
                            changed = true;
                        }
                    }
                    if conflict || !changed {
                        break;
                    }
                }
                if !any {
                    assert!(conflict, "class {f:#06x}: UP misses a conflict");
                    continue;
                }
                assert!(!conflict, "class {f:#06x}: UP conflicts on a consistent assignment");
                for i in 0..5 {
                    if rho[i] == 0 && forced[i] != 0 && forced[i] != 3 {
                        assert_eq!(
                            val[i], forced[i],
                            "class {f:#06x}: forced literal not propagated (rho {rho:?})"
                        );
                    }
                }
            }
        }
    }

    /// `map_cube` really transports covers: for sampled functions, the
    /// class cover mapped through the canonicalizing transform must be an
    /// exact cover of the function itself (with `out_neg` swapping the
    /// on/off roles).
    #[test]
    fn mapped_covers_are_exact() {
        let mut canon = Canon::new();
        for f in (0..=u16::MAX).step_by(31) {
            let (c, npn) = canon.get(f);
            let pc = pc_cover_for(c).expect("every class has a cover");
            let (on_src, off_src) = if npn.out_neg { (pc.off, pc.on) } else { (pc.on, pc.off) };
            let mut on = 0u16;
            for &cb in on_src {
                on |= cube_tt(map_cube(cb, &npn));
            }
            let mut off = 0u16;
            for &cb in off_src {
                off |= cube_tt(map_cube(cb, &npn));
            }
            assert_eq!(on, f, "mapped on-cover wrong for {f:#06x}");
            assert_eq!(off, !f, "mapped off-cover wrong for {f:#06x}");
        }
    }

    /// Known costs: AND2 = 1 node, XOR2 = 3, MUX = 3.
    #[test]
    fn known_structure_costs() {
        let mut canon = Canon::new();
        let a = LEAF_TT[0];
        let b = LEAF_TT[1];
        let c = LEAF_TT[2];
        for (name, tt, want) in [
            ("and2", a & b, 1usize),
            ("xor2", a ^ b, 3),
            ("mux", (c & a) | (!c & b), 3),
        ] {
            let (cl, _) = canon.get(tt);
            let st = structure_for(cl).expect("class present");
            assert_eq!(st.num_nodes(), want, "{name} should need {want} AND nodes");
        }
    }
}
