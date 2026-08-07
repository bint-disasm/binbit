//! Study: over ALL 4-variable functions, compare per polarity
//!   (a) ISOP cube count (what cnfmap emits today),
//!   (b) exact minimum SOP cover size (best possible clause count),
//!   (c) complete prime-implicant count (GAC emission cost).
//! Decides how much quality headroom npn4-backed cover tables have.

use binbit::cnfmap::{isop_cover, prime_cover, Cube};

/// Truth table of a cube over 4 vars in the 16-row space.
fn cube_tt4(c: Cube) -> u16 {
    const M: [u16; 4] = [0xAAAA, 0xCCCC, 0xF0F0, 0xFF00];
    let mut t = 0xFFFFu16;
    for i in 0..4 {
        if c.pos & (1 << i) != 0 {
            t &= M[i];
        }
        if c.neg & (1 << i) != 0 {
            t &= !M[i];
        }
    }
    t
}

/// Exact minimum number of primes covering `f`, by branch and bound on
/// the least-covered minterm.
fn min_cover(f: u16, primes: &[Cube]) -> u32 {
    if f == 0 {
        return 0;
    }
    let tabs: Vec<u16> = primes.iter().map(|&c| cube_tt4(c)).collect();
    let mut best = u32::MAX;
    fn go(uncovered: u16, tabs: &[u16], depth: u32, best: &mut u32) {
        if uncovered == 0 {
            *best = (*best).min(depth);
            return;
        }
        if depth + 1 >= *best {
            return; // can't beat the incumbent
        }
        // Branch on the minterm with the fewest covering primes.
        let mut pick = 0u16;
        let mut nopts = usize::MAX;
        let mut rest = uncovered;
        while rest != 0 {
            let m = rest & rest.wrapping_neg();
            rest &= rest - 1;
            let n = tabs.iter().filter(|&&t| t & m != 0).count();
            if n < nopts {
                nopts = n;
                pick = m;
            }
        }
        for (i, &t) in tabs.iter().enumerate() {
            if t & pick != 0 {
                let _ = i;
                go(uncovered & !t, tabs, depth + 1, best);
            }
        }
    }
    go(f, &tabs, 0, &mut best);
    best
}

fn main() {
    let mut scratch: Vec<Cube> = Vec::new();
    let mut isop_total = 0u64;
    let mut min_total = 0u64;
    let mut prime_total = 0u64;
    let mut gap_fns = 0u64; // functions where ISOP > exact min
    let mut hole_fns = 0u64; // functions where primes > ISOP (propagation holes)
    let mut max_gap = 0u32;
    let mut max_primes = 0usize;
    for f in 0..=u16::MAX {
        let f64pad = {
            // replicate 16-row table into the 64-row Tt space
            let x = f as u64;
            x | x << 16 | x << 32 | x << 48
        };
        let isop_n = if isop_cover(f64pad, 4, &mut scratch).is_some() {
            scratch.len() as u32
        } else {
            u32::MAX
        };
        prime_cover(f64pad, 4, &mut scratch);
        let primes = scratch.clone();
        let pn = primes.len();
        let mn = min_cover(f, &primes);
        assert!(isop_n >= mn, "ISOP below exact minimum?! f={f:04x}");
        assert!(pn as u32 >= isop_n, "fewer primes than ISOP cubes?! f={f:04x}");
        isop_total += isop_n as u64;
        min_total += mn as u64;
        prime_total += pn as u64;
        if isop_n > mn {
            gap_fns += 1;
            max_gap = max_gap.max(isop_n - mn);
        }
        if pn as u32 > isop_n {
            hole_fns += 1;
        }
        max_primes = max_primes.max(pn);
    }
    let n = 65536f64;
    println!("per-polarity averages over all 65,536 4-var functions:");
    println!("  ISOP cubes : {:.3}", isop_total as f64 / n);
    println!("  exact min  : {:.3}", min_total as f64 / n);
    println!("  all primes : {:.3}", prime_total as f64 / n);
    println!("functions where ISOP > min : {gap_fns} ({:.2}%), max gap {max_gap}",
        gap_fns as f64 / n * 100.0);
    println!("functions with propagation holes (primes > ISOP): {hole_fns} ({:.2}%)",
        hole_fns as f64 / n * 100.0);
    println!("max primes for one polarity: {max_primes}");
}
