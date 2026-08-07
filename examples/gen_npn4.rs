//! One-shot generator for the 4-input NPN structure database used by
//! `src/npn4.rs` (see `src/aigrw.rs` for what consumes it).
//!
//! Emits Rust source on stdout:
//!   cargo run --release --example gen_npn4 > /tmp/npn4_tables.rs
//!
//! Method: Dijkstra over the 65536 four-variable truth tables, where the
//! cost of a function is the number of two-input AND nodes needed to
//! build it as a tree over the four leaves. Complementing an AIG edge is
//! free, so `cost(f) == cost(!f)` is maintained as an invariant and both
//! polarities of every discovered function become available operands.
//! Relaxation is `cost(x & y) <= cost(x) + cost(y) + 1` over all pairs of
//! already-final functions, processed in increasing cost order.
//!
//! Tree cost is an upper bound on the DAG cost that the consumer actually
//! pays, because binbit's `Aig::and` hash-conses: shared subexpressions
//! inside a structure collapse when it is built. So the emitted structures
//! are always realizable within their stated node count.

use std::collections::HashMap;

const LEAF_TT: [u16; 4] = [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00];

/// (on-set cubes, off-set cubes) for one class, as (pos, neg) masks.
type PcCoverPair = (Vec<(u8, u8)>, Vec<(u8, u8)>);

// ---- Propagation-complete covers (consumed by `cnfmap`) ----
//
// For each class representative f, the CNF definition of a fresh output
// y ↔ f(x) is built from clauses of the y↔f relation. All prime
// implicates of the relation make unit propagation complete for it
// (every literal the relation forces under a partial assignment is
// UP-derivable), but many primes are redundant *for propagation*. The
// generator computes the primes, then greedily drops clauses while the
// set stays propagation complete — checked exhaustively over all 3^5
// partial assignments — over several elimination orders, keeping the
// smallest survivor. Offline exhaustiveness is the point: no runtime
// cover computation can afford the PC check per cut.

/// A clause over {x0..x3, y}: pos/neg masks, bit 4 = y.
#[derive(Clone, Copy, PartialEq)]
struct Cl {
    pos: u8,
    neg: u8,
}

/// All prime implicants of a 16-row table, as (pos, neg) masks over x0..x3.
fn primes4(f: u16) -> Vec<(u8, u8)> {
    let cube_tt = |pos: u8, neg: u8| -> u16 {
        let mut t = 0xFFFFu16;
        for i in 0..4 {
            if pos & (1 << i) != 0 {
                t &= LEAF_TT[i];
            }
            if neg & (1 << i) != 0 {
                t &= !LEAF_TT[i];
            }
        }
        t
    };
    let mut out = Vec::new();
    'cand: for code in 0..81usize {
        let (mut c, mut pos, mut neg) = (code, 0u8, 0u8);
        for i in 0..4 {
            match c % 3 {
                1 => pos |= 1 << i,
                2 => neg |= 1 << i,
                _ => {}
            }
            c /= 3;
        }
        if cube_tt(pos, neg) & !f != 0 {
            continue; // not an implicant
        }
        for i in 0..4u8 {
            let m = 1 << i;
            if (pos | neg) & m != 0 && cube_tt(pos & !m, neg & !m) & !f == 0 {
                continue 'cand; // a literal can be dropped: not prime
            }
        }
        out.push((pos, neg));
    }
    out
}

/// Unit propagation over 5-var clauses from a partial assignment.
/// `val[i]`: 0 = unassigned, 1 = true, 2 = false. Returns false on conflict.
fn up(clauses: &[Cl], val: &mut [u8; 5]) -> bool {
    loop {
        let mut changed = false;
        for cl in clauses {
            let mut unassigned = None;
            let mut sat = false;
            let mut nfree = 0;
            for i in 0..5u8 {
                let (p, n) = ((cl.pos >> i) & 1 == 1, (cl.neg >> i) & 1 == 1);
                if !p && !n {
                    continue;
                }
                match val[i as usize] {
                    0 => {
                        nfree += 1;
                        unassigned = Some((i, p));
                    }
                    1 if p => sat = true,
                    2 if n => sat = true,
                    _ => {}
                }
            }
            if sat {
                continue;
            }
            match nfree {
                0 => return false,
                1 => {
                    let (i, p) = unassigned.unwrap();
                    val[i as usize] = if p { 1 } else { 2 };
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

/// Is `clauses` propagation complete for the relation y ↔ f? For every
/// partial assignment ρ: if the relation under ρ is unsatisfiable, UP
/// must conflict; otherwise every literal the relation forces must be
/// UP-derived.
fn is_pc(f: u16, clauses: &[Cl]) -> bool {
    'rho: for code in 0..243usize {
        let (mut c, mut rho) = (code, [0u8; 5]);
        for v in rho.iter_mut() {
            *v = [0, 1, 2][c % 3];
            c /= 3;
        }
        // Semantic consequences: scan the 32 full assignments in the
        // relation that extend ρ.
        let mut forced = [0u8; 5]; // 0 unknown yet, 1/2 forced value, 3 both seen
        let mut any = false;
        for m in 0..32usize {
            let y = (m >> 4) & 1;
            let fx = (f >> (m & 15)) & 1;
            if y != fx as usize {
                continue;
            }
            let mut ok = true;
            for i in 0..5 {
                if rho[i] != 0 && rho[i] != if (m >> i) & 1 == 1 { 1 } else { 2 } {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            any = true;
            for i in 0..5 {
                let b = if (m >> i) & 1 == 1 { 1 } else { 2 };
                forced[i] = if forced[i] == 0 || forced[i] == b { b } else { 3 };
            }
        }
        let mut val = rho;
        let no_conflict = up(clauses, &mut val);
        if !any {
            // Relation contradicts ρ — UP must notice.
            if no_conflict {
                return false;
            }
            continue 'rho;
        }
        if !no_conflict {
            return false; // UP conflicts where the relation is satisfiable
        }
        for i in 0..5 {
            if rho[i] == 0 && forced[i] != 0 && forced[i] != 3 && val[i] != forced[i] {
                return false; // relation forces a literal UP never derives
            }
        }
    }
    true
}

/// Apply an NPN transform to a 4-variable truth table.
/// `perm[j]` = which source variable drives canonical input `j`;
/// bit `j` of `in_neg` complements that edge; `out_neg` flips the output.
fn apply(tt: u16, perm: [u8; 4], in_neg: u8, out_neg: bool) -> u16 {
    let mut out = 0u16;
    for m in 0..16usize {
        // m is an assignment to the CANONICAL inputs; recover the
        // corresponding assignment to the source variables.
        let mut src = 0usize;
        for j in 0..4 {
            let mut bit = (m >> j) & 1;
            if (in_neg >> j) & 1 == 1 {
                bit ^= 1;
            }
            if bit == 1 {
                src |= 1 << perm[j];
            }
        }
        let mut v = (tt >> src) & 1;
        if out_neg {
            v ^= 1;
        }
        if v == 1 {
            out |= 1 << m;
        }
    }
    out
}

fn perms() -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    for a in 0..4u8 {
        for b in 0..4u8 {
            for c in 0..4u8 {
                for d in 0..4u8 {
                    if [a, b, c, d].iter().collect::<std::collections::HashSet<_>>().len() == 4 {
                        out.push([a, b, c, d]);
                    }
                }
            }
        }
    }
    out
}

/// Canonical form: the lexicographically smallest table over all 768
/// transforms, together with a transform mapping canonical -> tt.
fn canonicalize(tt: u16, ps: &[[u8; 4]]) -> (u16, [u8; 4], u8, bool) {
    let mut best = (u16::MAX, [0u8; 4], 0u8, false);
    for &p in ps {
        for in_neg in 0..16u8 {
            for out_neg in [false, true] {
                // We want canonical c with apply(c, p, in_neg, out_neg) == tt.
                // Search over transforms applied to tt directly and invert.
                let c = apply(tt, p, in_neg, out_neg);
                if c < best.0 {
                    best = (c, p, in_neg, out_neg);
                }
            }
        }
    }
    best
}

fn main() {
    let ps = perms();
    assert_eq!(ps.len(), 24);

    // ---- Dijkstra over truth tables ----
    const INF: u8 = 255;
    let mut cost = vec![INF; 65536];
    // decomposition: how each function is built
    let mut decomp: Vec<Option<(u16, u16)>> = vec![None; 65536];

    // `decomp[f]` is only ever set when it builds EXACTLY `f`; the
    // complement is reachable for free by flipping an edge, so `!f` gets
    // the same cost but no decomposition of its own. `emit` relies on
    // that asymmetry to know whether to complement.
    fn relax(
        cost: &mut [u8],
        decomp: &mut [Option<(u16, u16)>],
        f: u16,
        c: u8,
        d: Option<(u16, u16)>,
    ) {
        if c < cost[f as usize] {
            cost[f as usize] = c;
            decomp[f as usize] = d;
        }
        if c < cost[(!f) as usize] {
            cost[(!f) as usize] = c;
        }
    }

    relax(&mut cost, &mut decomp, 0x0000, 0, None);
    for &l in &LEAF_TT {
        relax(&mut cost, &mut decomp, l, 0, None);
    }

    let mut levels: Vec<Vec<u16>> = vec![Vec::new(); 26];
    for f in 0..65536usize {
        if cost[f] == 0 {
            levels[0].push(f as u16);
        }
    }

    // The famous "at most 7 AND nodes" bound is for DAGs with sharing;
    // an AND-TREE decomposition needs more, so keep going until every
    // class representative is covered.
    let max_cost = 24usize;
    for k in 1..=max_cost {
        // new functions at cost k come from pairs summing to k-1
        let mut found: Vec<(u16, u16, u16)> = Vec::new();
        for i in 0..k {
            let j = k - 1 - i;
            if j >= levels.len() {
                continue;
            }
            for &x in &levels[i] {
                for &y in &levels[j] {
                    // both polarities of each operand are free
                    for xx in [x, !x] {
                        for yy in [y, !y] {
                            let f = xx & yy;
                            if cost[f as usize] > k as u8 && cost[(!f) as usize] > k as u8 {
                                found.push((f, xx, yy));
                            }
                        }
                    }
                }
            }
        }
        for (f, a, b) in found {
            relax(&mut cost, &mut decomp, f, k as u8, Some((a, b)));
        }
        for f in 0..65536usize {
            if cost[f] == k as u8 {
                levels[k].push(f as u16);
            }
        }
        eprintln!("cost {k}: {} functions (cumulative {})", levels[k].len(),
                  cost.iter().filter(|&&c| c != INF).count());
        if cost.iter().all(|&c| c != INF) {
            break;
        }
    }
    let unreached = cost.iter().filter(|&&c| c == INF).count();
    eprintln!("unreached functions: {unreached}");

    // ---- canonical classes ----
    let mut classes: Vec<u16> = Vec::new();
    let mut canon_of = vec![0u16; 65536];
    for f in 0..65536usize {
        let (c, _, _, _) = canonicalize(f as u16, &ps);
        canon_of[f] = c;
    }
    let mut seen: HashMap<u16, ()> = HashMap::new();
    for f in 0..65536usize {
        let c = canon_of[f];
        if seen.insert(c, ()).is_none() {
            classes.push(c);
        }
    }
    classes.sort_unstable();
    eprintln!("NPN classes: {}", classes.len());

    // ---- flatten a decomposition tree into structure nodes ----
    // Operand encoding matches src/npn4.rs:
    //   bit0 = complement, id 0 = const TRUE, 1..=4 = leaf, >=5 = node.
    fn emit(
        tt: u16,
        _cost: &[u8],
        decomp: &[Option<(u16, u16)>],
        nodes: &mut Vec<(u16, u16)>,
        memo: &mut HashMap<u16, u16>,
    ) -> u16 {
        // constants
        if tt == 0xFFFF {
            return 0; // OP_TRUE
        }
        if tt == 0x0000 {
            return 1; // OP_FALSE
        }
        for (i, &l) in LEAF_TT.iter().enumerate() {
            if tt == l {
                return (i as u16 + 1) << 1;
            }
            if tt == !l {
                return ((i as u16 + 1) << 1) | 1;
            }
        }
        if let Some(&o) = memo.get(&tt) {
            return o;
        }
        // A function is stored either directly or via its complement.
        let (d, complement) = match decomp[tt as usize] {
            Some(d) => (d, false),
            None => (
                decomp[(!tt) as usize].expect("no decomposition for either polarity"),
                true,
            ),
        };
        let (a, b) = d;
        // The stored decomposition builds `tt` when !complement, else `!tt`.
        let oa = emit(a, _cost, decomp, nodes, memo);
        let ob = emit(b, _cost, decomp, nodes, memo);
        let idx = nodes.len() as u16;
        nodes.push((oa, ob));
        let mut op = (idx + 5) << 1;
        if complement {
            op ^= 1;
        }
        memo.insert(tt, op);
        op
    }

    // Verify a structure by simulation.
    fn sim(nodes: &[(u16, u16)], root: u16) -> u16 {
        let mut vals = vec![0u16; nodes.len()];
        let val = |o: u16, vals: &[u16]| -> u16 {
            let id = (o >> 1) as usize;
            let base = if id == 0 {
                0xFFFF
            } else if id <= 4 {
                LEAF_TT[id - 1]
            } else {
                vals[id - 5]
            };
            if o & 1 == 1 { !base } else { base }
        };
        for i in 0..nodes.len() {
            vals[i] = val(nodes[i].0, &vals) & val(nodes[i].1, &vals);
        }
        val(root, &vals)
    }

    // ---- propagation-complete covers per class ----
    // Relation primes, then greedy drop under the exhaustive PC check
    // over several deterministic elimination orders; smallest survivor
    // wins. `on` cubes p give clauses (¬p ∨ y), `off` cubes q give
    // (¬q ∨ ¬y).
    let mut pc_covers: Vec<PcCoverPair> = Vec::new();
    let mut pc_total = 0usize;
    let mut prime_total = 0usize;
    for (ci, &c) in classes.iter().enumerate() {
        let mut rel: Vec<Cl> = Vec::new();
        for (p, n) in primes4(c) {
            rel.push(Cl { pos: n | 0x10, neg: p }); // (¬p ∨ y)
        }
        for (p, n) in primes4(!c) {
            rel.push(Cl { pos: n, neg: p | 0x10 }); // (¬q ∨ ¬y)
        }
        assert!(is_pc(c, &rel), "all relation primes must be PC (class {c:#06x})");
        prime_total += rel.len();
        let mut best = rel.clone();
        let mut seed = 0x9E3779B9u64.wrapping_mul(ci as u64 + 1) | 1;
        for _ in 0..20 {
            let mut order: Vec<usize> = (0..rel.len()).collect();
            // Fisher-Yates with a deterministic LCG.
            for i in (1..order.len()).rev() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                order.swap(i, (seed >> 33) as usize % (i + 1));
            }
            let mut cur = rel.clone();
            for &pick in &order {
                let cand = rel[pick];
                if let Some(pos) = cur.iter().position(|&x| x == cand) {
                    let mut trial = cur.clone();
                    trial.remove(pos);
                    if is_pc(c, &trial) {
                        cur = trial;
                    }
                }
            }
            if cur.len() < best.len() {
                best = cur;
            }
        }
        assert!(is_pc(c, &best));
        pc_total += best.len();
        let mut on: Vec<(u8, u8)> = Vec::new();
        let mut off: Vec<(u8, u8)> = Vec::new();
        for cl in &best {
            if cl.pos & 0x10 != 0 {
                on.push((cl.neg & 0xF, cl.pos & 0xF)); // recover cube p
            } else {
                off.push((cl.neg & 0xF, cl.pos & 0xF));
            }
        }
        pc_covers.push((on, off));
    }
    eprintln!(
        "PC covers: {} clauses total over {} classes (relation primes: {}, avg {:.2} -> {:.2}/class)",
        pc_total,
        classes.len(),
        prime_total,
        prime_total as f64 / classes.len() as f64,
        pc_total as f64 / classes.len() as f64
    );

    let mut out = String::new();
    out.push_str("// @generated by `cargo run --release --example gen_npn4`. Do not edit.\n");
    out.push_str("// 4-input NPN class database: one minimal AND-tree structure per class.\n\n");
    out.push_str("use super::{PcCover, Structure};\n\n");

    let mut entries: Vec<(u16, String, usize)> = Vec::new();
    let mut bad = 0usize;
    for &c in &classes {
        let mut nodes: Vec<(u16, u16)> = Vec::new();
        let mut memo = HashMap::new();
        let root = emit(c, &cost, &decomp, &mut nodes, &mut memo);
        let got = sim(&nodes, root);
        if got != c {
            bad += 1;
            eprintln!("MISMATCH class {c:#06x}: structure computes {got:#06x}");
            continue;
        }
        let body = nodes
            .iter()
            .map(|(a, b)| format!("({a},{b})"))
            .collect::<Vec<_>>()
            .join(",");
        entries.push((c, format!("Structure{{nodes:&[{body}],root:{root}}}"), nodes.len()));
    }
    eprintln!("structures emitted: {} (mismatches: {bad})", entries.len());

    let mut dist: HashMap<usize, usize> = HashMap::new();
    for (_, _, n) in &entries {
        *dist.entry(*n).or_insert(0) += 1;
    }
    let mut dk: Vec<_> = dist.into_iter().collect();
    dk.sort();
    eprintln!("node-count distribution: {dk:?}");

    out.push_str(&format!(
        "pub static CLASSES: [(u16, Structure); {}] = [\n",
        entries.len()
    ));
    for (c, s, _) in &entries {
        out.push_str(&format!("    ({c}, {s}),\n"));
    }
    out.push_str("];\n\n");

    // PC covers, aligned with CLASSES by index (same sorted class keys).
    assert_eq!(pc_covers.len(), entries.len());
    out.push_str("/// Propagation-complete irredundant covers, aligned with `CLASSES`.\n");
    out.push_str(&format!(
        "pub static PC_COVERS: [PcCover; {}] = [\n",
        pc_covers.len()
    ));
    for (on, off) in &pc_covers {
        let fmt = |v: &Vec<(u8, u8)>| {
            v.iter()
                .map(|(p, n)| format!("({p},{n})"))
                .collect::<Vec<_>>()
                .join(",")
        };
        out.push_str(&format!(
            "    PcCover{{on:&[{}],off:&[{}]}},\n",
            fmt(on),
            fmt(off)
        ));
    }
    out.push_str("];\n");
    println!("{out}");
}
