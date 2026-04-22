//! Randomized correctness tests for every BV operation.
//!
//! Strategy: for each op, generate random inputs, compute the expected
//! result with Rust's native `u128`/`i128` arithmetic, and verify the
//! solver agrees. This catches bitblaster, constant-folding, and rewrite
//! bugs that specific hand-written tests would miss.
//!
//! Coverage/cost tradeoff: lighter operations (add, bitwise, compare) are
//! tested at all widths including 128; heavy ones (mul, udiv, urem, sdiv,
//! srem, smod) are bounded to smaller widths, because full symbolic
//! 64-bit mul/div bitblasts to tens of thousands of gates per formula and
//! the fuzz runs 20+ of them.
//!
//! Seeds are printed on failure so any discovered bug is reproducible.

use binbit::{BvTerm, SmtResult, SmtSolver};

/// Tiny deterministic PRNG — a linear-congruential generator.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_u128(&mut self) -> u128 {
        let hi = self.next_u64() as u128;
        let lo = self.next_u64() as u128;
        (hi << 64) | lo
    }
    fn uniform_bv(&mut self, width: u32) -> u128 {
        self.next_u128() & mask(width)
    }
    fn uniform_u32(&mut self, bound: u32) -> u32 {
        (self.next_u64() % bound as u64) as u32
    }
}

#[inline]
fn mask(width: u32) -> u128 {
    if width == 128 { u128::MAX } else { (1u128 << width) - 1 }
}

fn to_signed(v: u128, width: u32) -> i128 {
    if width == 0 || width >= 128 {
        return v as i128;
    }
    let m = mask(width);
    let v = v & m;
    if v & (1 << (width - 1)) != 0 {
        (v | !m) as i128
    } else {
        v as i128
    }
}

/// Binary op correctness check — tests ONLY the symbolic (bitblaster) path.
/// The constant-folding path is covered by the existing unit tests.
/// Only one solve per iteration (SAT); UNSAT corroboration is handled by
/// a handful of targeted tests elsewhere.
fn check_bin_bv<F, G>(
    name: &str,
    width: u32,
    iterations: u32,
    seed: u64,
    mut gen_inputs: impl FnMut(&mut Rng) -> (u128, u128),
    bv_op: F,
    native: G,
) where
    F: Fn(&mut SmtSolver, BvTerm, BvTerm) -> BvTerm,
    G: Fn(u128, u128, u32) -> u128,
{
    let mut rng = Rng::new(seed);
    for iter in 0..iterations {
        let (a, b) = gen_inputs(&mut rng);
        let expected = native(a, b, width) & mask(width);

        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let y = s.bv_var(width);
        let ca = s.bv_const(a, width);
        let cb = s.bv_const(b, width);
        let eq_x = s.bv_eq(x, ca);
        let eq_y = s.bv_eq(y, cb);
        s.assert(eq_x);
        s.assert(eq_y);
        let result = bv_op(&mut s, x, y);
        let ce = s.bv_const(expected, width);
        let eq = s.bv_eq(result, ce);
        s.assert(eq);
        assert_eq!(
            s.solve(),
            SmtResult::Sat,
            "op={} width={} iter={} seed={}: {}({:#x}, {:#x}) should equal {:#x}",
            name,
            width,
            iter,
            seed,
            name,
            a,
            b,
            expected
        );
    }
}

fn check_bin_bool<F, G>(
    name: &str,
    width: u32,
    iterations: u32,
    seed: u64,
    mut gen_inputs: impl FnMut(&mut Rng) -> (u128, u128),
    bv_op: F,
    native: G,
) where
    F: Fn(&mut SmtSolver, BvTerm, BvTerm) -> binbit::BoolTerm,
    G: Fn(u128, u128, u32) -> bool,
{
    let mut rng = Rng::new(seed);
    for iter in 0..iterations {
        let (a, b) = gen_inputs(&mut rng);
        let expected = native(a, b, width);

        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let y = s.bv_var(width);
        let ca = s.bv_const(a, width);
        let cb = s.bv_const(b, width);
        let eq_x = s.bv_eq(x, ca);
        let eq_y = s.bv_eq(y, cb);
        s.assert(eq_x);
        s.assert(eq_y);
        let result = bv_op(&mut s, x, y);
        // Assert `result == expected`.
        let assertion = if expected {
            result
        } else {
            s.bool_not(result)
        };
        s.assert(assertion);
        assert_eq!(
            s.solve(),
            SmtResult::Sat,
            "op={} width={} iter={} seed={}: {}({:#x}, {:#x}) should be {}",
            name,
            width,
            iter,
            seed,
            name,
            a,
            b,
            expected
        );
    }
}

fn check_unary_bv<F, G>(
    name: &str,
    width: u32,
    iterations: u32,
    seed: u64,
    bv_op: F,
    native: G,
) where
    F: Fn(&mut SmtSolver, BvTerm) -> BvTerm,
    G: Fn(u128, u32) -> u128,
{
    let mut rng = Rng::new(seed);
    for iter in 0..iterations {
        let a = rng.uniform_bv(width);
        let expected = native(a, width) & mask(width);

        let mut s = SmtSolver::new();
        let x = s.bv_var(width);
        let ca = s.bv_const(a, width);
        let eq_x = s.bv_eq(x, ca);
        s.assert(eq_x);
        let result = bv_op(&mut s, x);
        let ce = s.bv_const(expected, width);
        let eq = s.bv_eq(result, ce);
        s.assert(eq);
        assert_eq!(
            s.solve(),
            SmtResult::Sat,
            "op={} width={} iter={} seed={}: {}({:#x}) should equal {:#x}",
            name,
            width,
            iter,
            seed,
            name,
            a,
            expected
        );
    }
}

// ============================================================
// Arithmetic — cheap ops fuzzed broadly; heavy ops narrowly.
// ============================================================

#[test]
fn fuzz_add() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 65, 96, 128] {
        check_bin_bv(
            "add", w, 15, 0xADD00 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_add(a, b),
            |a, b, _| a.wrapping_add(b),
        );
    }
}

#[test]
fn fuzz_sub() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bv(
            "sub", w, 15, 0x5ABFE ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_sub(a, b),
            |a, b, _| a.wrapping_sub(b),
        );
    }
}

#[test]
fn fuzz_neg() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_unary_bv(
            "neg", w, 15, 0x1E9 ^ w as u64,
            |s, a| s.bv_neg(a),
            |a, _| 0u128.wrapping_sub(a),
        );
    }
}

#[test]
fn fuzz_mul() {
    // Symbolic mul is O(N²) gates — fuzz narrowly to stay fast.
    for &w in &[1u32, 4, 8] {
        check_bin_bv(
            "mul", w, 8, 0x3EDD7 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_mul(a, b),
            |a, b, _| a.wrapping_mul(b),
        );
    }
}

#[test]
fn fuzz_udiv() {
    // Restoring division is O(N²) iterations × ~N gates each. Keep narrow.
    for &w in &[1u32, 4, 8] {
        check_bin_bv(
            "udiv", w, 5, 0x0D17 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_udiv(a, b),
            |a, b, w| {
                let b_m = b & mask(w);
                if b_m == 0 { mask(w) } else { (a & mask(w)) / b_m }
            },
        );
    }
}

#[test]
fn fuzz_urem() {
    for &w in &[1u32, 4, 8] {
        check_bin_bv(
            "urem", w, 5, 0x8EE11 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_urem(a, b),
            |a, b, w| {
                let b_m = b & mask(w);
                if b_m == 0 { a & mask(w) } else { (a & mask(w)) % b_m }
            },
        );
    }
}

#[test]
fn fuzz_sdiv() {
    for &w in &[2u32, 4, 8] {
        check_bin_bv(
            "sdiv", w, 5, 0x50D14 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_sdiv(a, b),
            |a, b, w| {
                let sa = to_signed(a, w);
                let sb = to_signed(b, w);
                let r = if sb == 0 {
                    if sa < 0 { 1 } else { -1i128 }
                } else {
                    sa.wrapping_div(sb)
                };
                r as u128
            },
        );
    }
}

#[test]
fn fuzz_srem() {
    for &w in &[2u32, 4, 8] {
        check_bin_bv(
            "srem", w, 5, 0x5E20B ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_srem(a, b),
            |a, b, w| {
                let sa = to_signed(a, w);
                let sb = to_signed(b, w);
                let r = if sb == 0 { sa } else { sa.wrapping_rem(sb) };
                r as u128
            },
        );
    }
}

#[test]
fn fuzz_smod() {
    for &w in &[2u32, 4, 8] {
        check_bin_bv(
            "smod", w, 5, 0x53AA1 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_smod(a, b),
            |a, b, w| {
                let sa = to_signed(a, w);
                let sb = to_signed(b, w);
                let r = if sb == 0 {
                    sa
                } else {
                    let r = sa.wrapping_rem(sb);
                    if r == 0 {
                        0
                    } else if (r < 0) != (sb < 0) {
                        r.wrapping_add(sb)
                    } else {
                        r
                    }
                };
                r as u128
            },
        );
    }
}

// ============================================================
// Bitwise + shifts — cheap per-bit, fuzz widely.
// ============================================================

#[test]
fn fuzz_bitwise_and() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bv(
            "and", w, 15, 0xA5D ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_and(a, b),
            |a, b, _| a & b,
        );
    }
}

#[test]
fn fuzz_bitwise_or() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bv(
            "or", w, 15, 0x087 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_or(a, b),
            |a, b, _| a | b,
        );
    }
}

#[test]
fn fuzz_bitwise_xor() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bv(
            "xor", w, 15, 0x704 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_xor(a, b),
            |a, b, _| a ^ b,
        );
    }
}

#[test]
fn fuzz_bitwise_not() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_unary_bv(
            "not", w, 15, 0x107 ^ w as u64,
            |s, a| s.bv_not(a),
            |a, _| !a,
        );
    }
}

#[test]
fn fuzz_shl() {
    // Symbolic shift creates a barrel shifter — modest cost, reasonable widths.
    for &w in &[2u32, 4, 8, 16, 32] {
        check_bin_bv(
            "shl", w, 10, 0x58410 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_shl(a, b),
            |a, b, w| {
                let amt = b & mask(w);
                if amt >= w as u128 { 0 } else { a << amt }
            },
        );
    }
}

#[test]
fn fuzz_lshr() {
    for &w in &[2u32, 4, 8, 16, 32] {
        check_bin_bv(
            "lshr", w, 10, 0x5DB8 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_lshr(a, b),
            |a, b, w| {
                let amt = b & mask(w);
                if amt >= w as u128 { 0 } else { (a & mask(w)) >> amt }
            },
        );
    }
}

#[test]
fn fuzz_ashr() {
    for &w in &[2u32, 4, 8, 16, 32] {
        check_bin_bv(
            "ashr", w, 10, 0xA581 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_ashr(a, b),
            |a, b, w| {
                let amt = (b & mask(w)).min(w as u128) as u32;
                let sa = to_signed(a, w);
                let shifted = sa >> amt.min(127);
                (shifted as u128) & mask(w)
            },
        );
    }
}

// ============================================================
// Comparisons
// ============================================================

#[test]
fn fuzz_ult() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bool(
            "ult", w, 15, 0x017 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_ult(a, b),
            |a, b, w| (a & mask(w)) < (b & mask(w)),
        );
    }
}

#[test]
fn fuzz_ule() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bool(
            "ule", w, 15, 0x0B1 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_ule(a, b),
            |a, b, w| (a & mask(w)) <= (b & mask(w)),
        );
    }
}

#[test]
fn fuzz_slt() {
    for &w in &[2u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bool(
            "slt", w, 15, 0x517 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_slt(a, b),
            |a, b, w| to_signed(a, w) < to_signed(b, w),
        );
    }
}

#[test]
fn fuzz_sle() {
    for &w in &[2u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bool(
            "sle", w, 15, 0x51E ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_sle(a, b),
            |a, b, w| to_signed(a, w) <= to_signed(b, w),
        );
    }
}

#[test]
fn fuzz_eq() {
    for &w in &[1u32, 4, 8, 16, 32, 64, 128] {
        check_bin_bool(
            "eq", w, 15, 0xE9 ^ w as u64,
            |r| {
                if r.next_u64() & 1 == 0 {
                    let a = r.uniform_bv(w);
                    (a, a)
                } else {
                    (r.uniform_bv(w), r.uniform_bv(w))
                }
            },
            |s, a, b| s.bv_eq(a, b),
            |a, b, w| (a & mask(w)) == (b & mask(w)),
        );
    }
}

// ============================================================
// Structural ops + round-trips
// ============================================================

#[test]
fn fuzz_extract() {
    let mut rng = Rng::new(0xE2A3);
    for _ in 0..50 {
        let w = 8 + rng.uniform_u32(24);
        let low = rng.uniform_u32(w);
        let high = low + rng.uniform_u32(w - low);
        let a = rng.uniform_bv(w);
        let expected = (a >> low) & mask(high - low + 1);

        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let ca = s.bv_const(a, w);
        let eq_x = s.bv_eq(x, ca);
        s.assert(eq_x);
        let ext = s.bv_extract(x, high, low);
        let ce = s.bv_const(expected, high - low + 1);
        let eq = s.bv_eq(ext, ce);
        s.assert(eq);
        assert_eq!(
            s.solve(),
            SmtResult::Sat,
            "extract w={} high={} low={} a={:#x} expected={:#x}",
            w, high, low, a, expected
        );
    }
}

#[test]
fn fuzz_concat_extract_round_trip() {
    let mut rng = Rng::new(0xC077);
    for _ in 0..30 {
        let w = 4 + rng.uniform_u32(28);
        let k = 1 + rng.uniform_u32(w - 1);

        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let hi = s.bv_extract(x, w - 1, k);
        let lo = s.bv_extract(x, k - 1, 0);
        let rebuilt = s.bv_concat(hi, lo);
        let differ = s.bv_ne(x, rebuilt);
        s.assert(differ);
        assert_eq!(
            s.solve(),
            SmtResult::Unsat,
            "concat/extract round-trip failed at w={} k={}",
            w, k
        );
    }
}

#[test]
fn fuzz_zero_extend_round_trip() {
    let mut rng = Rng::new(0x7E10);
    for _ in 0..30 {
        let w = 4 + rng.uniform_u32(28);
        let max_extra = 32.min(128 - w);
        if max_extra < 2 { continue; }
        let n = 1 + rng.uniform_u32(max_extra - 1);

        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let ext = s.bv_zero_extend(x, n);
        let lo = s.bv_extract(ext, w - 1, 0);
        let low_match = s.bv_ne(lo, x);
        s.assert(low_match);
        assert_eq!(s.solve(), SmtResult::Unsat, "zero_extend low-match w={} n={}", w, n);

        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let ext = s.bv_zero_extend(x, n);
        let hi = s.bv_extract(ext, w + n - 1, w);
        let zero = s.bv_const(0, n);
        let hi_nonzero = s.bv_ne(hi, zero);
        s.assert(hi_nonzero);
        assert_eq!(s.solve(), SmtResult::Unsat, "zero_extend hi-zero w={} n={}", w, n);
    }
}

#[test]
fn fuzz_sign_extend_round_trip() {
    let mut rng = Rng::new(0x516E);
    for _ in 0..30 {
        let w = 2 + rng.uniform_u32(30);
        let max_extra = 32.min(128 - w);
        if max_extra < 2 { continue; }
        let n = 1 + rng.uniform_u32(max_extra - 1);

        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let ext = s.bv_sign_extend(x, n);
        let lo = s.bv_extract(ext, w - 1, 0);
        let low_match = s.bv_ne(lo, x);
        s.assert(low_match);
        assert_eq!(s.solve(), SmtResult::Unsat, "sign_extend low-match w={} n={}", w, n);

        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let ext = s.bv_sign_extend(x, n);
        let msb = s.bv_extract(x, w - 1, w - 1);
        let mut broadcast = msb;
        for _ in 1..n {
            broadcast = s.bv_concat(msb, broadcast);
        }
        let hi = s.bv_extract(ext, w + n - 1, w);
        let hi_bad = s.bv_ne(hi, broadcast);
        s.assert(hi_bad);
        assert_eq!(s.solve(), SmtResult::Unsat, "sign_extend hi-broadcast w={} n={}", w, n);
    }
}

// ============================================================
// Overflow predicates
// ============================================================

#[test]
fn fuzz_uadd_overflow() {
    for &w in &[2u32, 8, 16, 32, 64] {
        check_bin_bool(
            "uaddo", w, 15, 0xEAD0 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_uadd_overflow(a, b),
            |a, b, w| {
                let m = mask(w);
                (a & m).checked_add(b & m).map(|v| v > m).unwrap_or(true)
            },
        );
    }
}

#[test]
fn fuzz_sadd_overflow() {
    for &w in &[2u32, 8, 16, 32, 64] {
        check_bin_bool(
            "saddo", w, 15, 0x5AD0 ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_sadd_overflow(a, b),
            |a, b, w| {
                let sa = to_signed(a, w);
                let sb = to_signed(b, w);
                if w >= 128 {
                    sa.checked_add(sb).is_none()
                } else {
                    let sum = sa.wrapping_add(sb);
                    let min = -(1i128 << (w - 1));
                    let max = (1i128 << (w - 1)) - 1;
                    sum < min || sum > max
                }
            },
        );
    }
}

#[test]
fn fuzz_umul_overflow() {
    // The bitblaster for umul_overflow runs a 2N-bit multiplication
    // internally — keep widths small.
    for &w in &[2u32, 4, 8] {
        check_bin_bool(
            "umulo", w, 5, 0x00EE ^ w as u64,
            |r| (r.uniform_bv(w), r.uniform_bv(w)),
            |s, a, b| s.bv_umul_overflow(a, b),
            |a, b, w| {
                let m = mask(w);
                (a & m).checked_mul(b & m).map(|v| v > m).unwrap_or(true)
            },
        );
    }
}

// ============================================================
// Algebraic identities
// ============================================================

fn prove_identity(desc: &str, width: u32, build: impl FnOnce(&mut SmtSolver) -> binbit::BoolTerm) {
    let mut s = SmtSolver::new();
    let neg_prop = {
        let prop = build(&mut s);
        s.bool_not(prop)
    };
    s.assert(neg_prop);
    assert_eq!(
        s.solve(),
        SmtResult::Unsat,
        "identity failed: {} at width {}",
        desc,
        width
    );
}

#[test]
fn identity_add_commutative() {
    for &w in &[4u32, 8, 16, 32] {
        prove_identity("a + b == b + a", w, |s| {
            let a = s.bv_var(w);
            let b = s.bv_var(w);
            let ab = s.bv_add(a, b);
            let ba = s.bv_add(b, a);
            s.bv_eq(ab, ba)
        });
    }
}

#[test]
fn identity_and_commutative() {
    for &w in &[4u32, 8, 16, 32, 64] {
        prove_identity("a & b == b & a", w, |s| {
            let a = s.bv_var(w);
            let b = s.bv_var(w);
            let ab = s.bv_and(a, b);
            let ba = s.bv_and(b, a);
            s.bv_eq(ab, ba)
        });
    }
}

#[test]
fn identity_mul_distributes_over_add_small() {
    // Keep widths small — proving this over all (x, a, b) at width 8 takes
    // appreciable time; 4 bits is enough to catch structural bugs.
    for &w in &[2u32, 4] {
        prove_identity("x * (a + b) == x*a + x*b", w, |s| {
            let x = s.bv_var(w);
            let a = s.bv_var(w);
            let b = s.bv_var(w);
            let apb = s.bv_add(a, b);
            let left = s.bv_mul(x, apb);
            let xa = s.bv_mul(x, a);
            let xb = s.bv_mul(x, b);
            let right = s.bv_add(xa, xb);
            s.bv_eq(left, right)
        });
    }
}

#[test]
fn identity_sub_eq_add_negation() {
    for &w in &[4u32, 8, 16, 32] {
        prove_identity("a - b == a + (-b)", w, |s| {
            let a = s.bv_var(w);
            let b = s.bv_var(w);
            let sub = s.bv_sub(a, b);
            let neg_b = s.bv_neg(b);
            let add_neg = s.bv_add(a, neg_b);
            s.bv_eq(sub, add_neg)
        });
    }
}

#[test]
fn identity_de_morgan() {
    for &w in &[4u32, 16, 32, 64] {
        prove_identity("~(a & b) == ~a | ~b", w, |s| {
            let a = s.bv_var(w);
            let b = s.bv_var(w);
            let and_ab = s.bv_and(a, b);
            let not_and = s.bv_not(and_ab);
            let not_a = s.bv_not(a);
            let not_b = s.bv_not(b);
            let or_nots = s.bv_or(not_a, not_b);
            s.bv_eq(not_and, or_nots)
        });
    }
}

#[test]
fn identity_shift_composition() {
    // (x << a) << b == x << (a + b) when a + b (as integers) < width.
    // We must constrain the mathematical sum, not the BV-wrapped sum —
    // otherwise cases like a=250, b=10 wrap to 4 and the guard spuriously
    // fires while `x << 250` has already cleared all bits.
    for &w in &[8u32] {
        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let a = s.bv_var(w);
        let b = s.bv_var(w);
        // Extend by enough bits that a + b cannot wrap.
        let ext = w + 4;
        let a_ext = s.bv_zero_extend(a, 4);
        let b_ext = s.bv_zero_extend(b, 4);
        let sum_ext = s.bv_add(a_ext, b_ext);
        let width_c_ext = s.bv_const(w as u128, ext);
        let in_range = s.bv_ult(sum_ext, width_c_ext);

        // Under that guard, the non-extended sum agrees with the math sum.
        let a_plus_b = s.bv_add(a, b);
        let inner = s.bv_shl(x, a);
        let left = s.bv_shl(inner, b);
        let right = s.bv_shl(x, a_plus_b);
        let eq = s.bv_eq(left, right);
        let neq = s.bool_not(eq);
        let counter = s.bool_and(in_range, neq);
        s.assert(counter);
        assert_eq!(s.solve(), SmtResult::Unsat, "shift composition failed at width {}", w);
    }
}

// ============================================================
// Targeted edge cases
// ============================================================

#[test]
fn edge_width_1_ops_are_sane() {
    let mut s = SmtSolver::new();
    let zero = s.bv_const(0, 1);
    let one = s.bv_const(1, 1);
    let sum = s.bv_add(one, one);
    let eq0 = s.bv_eq(sum, zero);
    s.assert(eq0);
    assert_eq!(s.solve(), SmtResult::Sat);

    let mut s = SmtSolver::new();
    let one = s.bv_const(1, 1);
    let neg = s.bv_neg(one);
    let eq = s.bv_eq(neg, one);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn edge_int_min_negation_wraps() {
    for &w in &[4u32, 8, 16, 32, 64, 128] {
        let mut s = SmtSolver::new();
        let int_min = s.bv_const(1u128 << (w - 1), w);
        let neg = s.bv_neg(int_min);
        let eq = s.bv_eq(neg, int_min);
        s.assert(eq);
        assert_eq!(s.solve(), SmtResult::Sat, "INT_MIN negation at w={}", w);
    }
}

#[test]
fn edge_sdiv_int_min_by_minus_one_wraps() {
    for &w in &[8u32, 16] {
        let mut s = SmtSolver::new();
        let int_min = s.bv_const(1u128 << (w - 1), w);
        let neg_one = s.bv_const(mask(w), w);
        let q = s.bv_sdiv(int_min, neg_one);
        let eq = s.bv_eq(q, int_min);
        s.assert(eq);
        assert_eq!(s.solve(), SmtResult::Sat, "INT_MIN/-1 at w={}", w);
    }
}

#[test]
fn edge_u128_boundary_const_folding() {
    let mut s = SmtSolver::new();
    let a = s.bv_const(u64::MAX as u128, 65);
    let one = s.bv_const(1, 65);
    let sum = s.bv_add(a, one);
    let expected = s.bv_const(1u128 << 64, 65);
    let eq = s.bv_eq(sum, expected);
    s.assert(eq);
    assert_eq!(s.solve(), SmtResult::Sat);
}

#[test]
fn edge_udiv_by_zero_default_value() {
    for &w in &[4u32, 8, 16] {
        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let zero = s.bv_const(0, w);
        let q = s.bv_udiv(x, zero);
        let all_ones = s.bv_const(mask(w), w);
        let eq = s.bv_eq(q, all_ones);
        let ne = s.bool_not(eq);
        s.assert(ne);
        assert_eq!(s.solve(), SmtResult::Unsat, "udiv by 0 at w={}", w);
    }
}

#[test]
fn edge_urem_by_zero_returns_dividend() {
    for &w in &[4u32, 8, 16] {
        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let zero = s.bv_const(0, w);
        let r = s.bv_urem(x, zero);
        let eq = s.bv_eq(r, x);
        let ne = s.bool_not(eq);
        s.assert(ne);
        assert_eq!(s.solve(), SmtResult::Unsat, "urem by 0 at w={}", w);
    }
}
// ============================================================
// Magic-number division — triggered by (symbolic / const) and
// (symbolic % const) patterns.
// ============================================================

#[test]
fn fuzz_udiv_magic_number_path() {
    let divisors = [3u128, 5, 6, 7, 9, 10, 11, 17, 33, 100, 255, 511, 1000];
    let mut rng = Rng::new(0xD12);
    for &w in &[4u32, 8, 16, 32] {
        for &d in divisors.iter() {
            if d >= (1u128 << w) {
                continue;
            }
            for _ in 0..5 {
                let a = rng.uniform_bv(w);
                let expected = (a & mask(w)) / d;
                let mut s = SmtSolver::new();
                let x = s.bv_var(w);
                let ca = s.bv_const(a, w);
                let eq_x = s.bv_eq(x, ca);
                s.assert(eq_x);
                let d_const = s.bv_const(d, w);
                // This call takes the magic-number path in the builder.
                let q = s.bv_udiv(x, d_const);
                let ce = s.bv_const(expected, w);
                let eq = s.bv_eq(q, ce);
                s.assert(eq);
                assert_eq!(
                    s.solve(),
                    SmtResult::Sat,
                    "magic udiv: w={} d={} x={:#x} expected={:#x}",
                    w, d, a, expected
                );
            }
        }
    }
}

#[test]
fn fuzz_urem_magic_number_path() {
    let divisors = [3u128, 5, 7, 10, 100, 255, 1000];
    let mut rng = Rng::new(0xE4D);
    for &w in &[4u32, 8, 16, 32] {
        for &d in divisors.iter() {
            if d >= (1u128 << w) {
                continue;
            }
            for _ in 0..5 {
                let a = rng.uniform_bv(w);
                let expected = (a & mask(w)) % d;
                let mut s = SmtSolver::new();
                let x = s.bv_var(w);
                let ca = s.bv_const(a, w);
                let eq_x = s.bv_eq(x, ca);
                s.assert(eq_x);
                let d_const = s.bv_const(d, w);
                let r = s.bv_urem(x, d_const);
                let ce = s.bv_const(expected, w);
                let eq = s.bv_eq(r, ce);
                s.assert(eq);
                assert_eq!(
                    s.solve(),
                    SmtResult::Sat,
                    "magic urem: w={} d={} x={:#x} expected={:#x}",
                    w, d, a, expected
                );
            }
        }
    }
}

#[test]
fn magic_udiv_preserves_identity() {
    // For any x and any non-zero non-power-of-2 constant d < 2^w:
    //   x == d * (x / d) + (x % d).
    // Prove it for a few representative divisors at small widths by
    // asserting the negation is UNSAT.
    for &w in &[4u32, 8] {
        for &d in &[3u128, 5, 7, 10] {
            if d >= (1u128 << w) {
                continue;
            }
            let mut s = SmtSolver::new();
            let x = s.bv_var(w);
            let d_const = s.bv_const(d, w);
            let q = s.bv_udiv(x, d_const);
            let r = s.bv_urem(x, d_const);
            let dq = s.bv_mul(d_const, q);
            let sum = s.bv_add(dq, r);
            let ne = s.bv_ne(x, sum);
            s.assert(ne);
            assert_eq!(
                s.solve(),
                SmtResult::Unsat,
                "magic identity failed: w={} d={}",
                w, d
            );
        }
    }
}

#[test]
fn edge_large_shift_amount_clamps() {
    for &w in &[8u32, 16, 32] {
        let mut s = SmtSolver::new();
        let x = s.bv_var(w);
        let big = s.bv_const(mask(w), w);
        let z = s.bv_shl(x, big);
        let zero = s.bv_const(0, w);
        let eq = s.bv_eq(z, zero);
        let ne = s.bool_not(eq);
        s.assert(ne);
        assert_eq!(s.solve(), SmtResult::Unsat, "shl overflow at w={}", w);
    }
}
