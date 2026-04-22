# binbit

Bit-precise SMT solver for QF_BV, intended as the backend for a symbolic-execution framework. CDCL SAT core + eager bitblasting + a handful of symbex-shaped preprocessing passes (SSA-style equality substitution, variable union-find aliasing, state-merge φ-nodes, constant-bound chain collapse).

All API access is via `binbit::SmtSolver`. Terms live in a hash-consed DAG; references are opaque `BvTerm` / `BoolTerm` handles. Everything below is stable enough to wire symbex against.

---

## Setting up

```rust
use binbit::{SmtSolver, SmtResult};

let mut s = SmtSolver::new();
```

Build up constraints, then `s.solve()` → `SmtResult::Sat | SmtResult::Unsat`. After `Sat`, read out the model.

`SmtSolver::new()` is cheap. For exploring distinct sub-problems you can either spin up fresh solvers (common in parallel path exploration) or use `push`/`pop` inside one solver to retract transient assertions (cheaper if the base formula is reused).

---

## Widths

Bitvectors are width-parameterised. Valid widths: `1..=65_536`. Values with width ≤ 128 use an inline `u128`; wider values spill into a limb pool transparently.

```rust
let x32  = s.bv_var(32);         // free 32-bit variable
let c    = s.bv_const(0x1234, 32);
let wide = s.bv_const_wide(&[1u64, 0, 0, 0], 256);   // 256-bit constant
```

Models for wide values come back as limbs:

```rust
let limbs: Vec<u64> = s.get_bv_value_limbs(x);
```

### Concretization

`try_bv_const_value(t) -> Option<u128>` is the non-panicking "did this term
fold to a constant?" check — returns `Some(value)` when the term is (or
collapsed to) a `BvOp::Const` of width ≤ 128, else `None`. Symbolic-
execution front-ends use this where the pcode-style executor needs a
concrete address or branch target: if the concretization probe returns
`Some`, use it directly; otherwise fall back to SAT-solving for a model.
Bits-known folding is transparent — a term that only became a constant
through mask/zero-extend propagation still returns `Some`.

`bv_width(t) -> u32` returns the width, without going through the DAG
accessors.

---

## Term construction

All builders live on `SmtSolver` and return handles (`BvTerm` or `BoolTerm`). Calls are pure DAG construction — nothing is bitblasted until `solve()`. You can freely discard a handle; the node stays in the DAG and gets GC-insensitive hash-cons dedup.

### Bitvector constructors

| Method                                   | Semantics                                          |
|------------------------------------------|----------------------------------------------------|
| `bv_var(width)`                          | Fresh symbolic BV of `width` bits                  |
| `bv_const(value: u128, width)`           | Literal (width ≤ 128, masked to width)             |
| `bv_const_wide(&[u64], width)`           | Literal for any width, little-endian limbs        |

### Bitwise

`bv_not`, `bv_and`, `bv_or`, `bv_xor` — operand widths must match.

### Arithmetic

`bv_add`, `bv_sub`, `bv_neg`, `bv_mul`, `bv_udiv`, `bv_urem`, `bv_sdiv`, `bv_srem`, `bv_smod`.

Division by zero follows SMT-LIB: `bvudiv(x, 0) = all-ones`, `bvurem(x, 0) = x`.

### Shifts

`bv_shl`, `bv_lshr` (logical), `bv_ashr` (arithmetic, sign-fills).

`bv_rotate_left(x, shift: u32)` and `bv_rotate_right(x, shift: u32)` take a
static shift amount; they synthesize `(x << k) | (x >>L (w-k))` (or the
mirror) with `k = shift mod width`. Use these rather than re-synthesizing in
the caller — the constructor folds correctly when `x` is a constant.

### Structural

`bv_extract(x, high, low)` — slice bits `[low..=high]`, output width `high - low + 1`.
`bv_concat(hi, lo)` — `hi` in high bits, `lo` in low bits.
`bv_zero_extend(x, n)`, `bv_sign_extend(x, n)` — widen by `n` bits.

### Comparisons (returning `BoolTerm`)

`bv_eq`, `bv_ne`, `bv_ult`, `bv_ule`, `bv_ugt`, `bv_uge`, `bv_slt`, `bv_sle`, `bv_sgt`, `bv_sge`.

### Overflow predicates

`bv_uadd_overflow`, `bv_sadd_overflow`, `bv_usub_overflow`, `bv_ssub_overflow`, `bv_umul_overflow`, `bv_smul_overflow`, `bv_neg_overflow`, `bv_sdiv_overflow`. Each returns a `BoolTerm` that's true iff the operation overflows at the given width.

### Conditional

```rust
let r = s.bv_ite(cond_bool, then_bv, else_bv);
```

Constant conditions fold; `bv_ite(c, x, x)` collapses; nested selector-aliased ITEs collapse; `ite(c, ite(d, x, y), ite(d, x, z))` factors through — see the state-merge section for the `bv_select` n-way form.

### Booleans

`bool_true`, `bool_false`, `bool_var`, `bool_not`, `bool_and`, `bool_or`, `bool_implies`.

---

## State merging — the n-way select

Symbolic execution produces big branch trees that need to merge back to keep the state tree tractable. When merging N paths with mutually-exclusive path conditions `pc_1..pc_N`, each merged variable `v` looks like:

```
v = pc_1 ? v_1 : pc_2 ? v_2 : … : pc_N ? v_N : default
```

You *could* build this as a left-skewed chain of `bv_ite`. **Don't.** Use `bv_select` + `assert_mutually_exclusive`:

```rust
let pcs:   Vec<BoolTerm> = /* path conditions */;
let vals:  Vec<BvTerm>   = /* per-path values of `v` */;
let dflt:  BvTerm        = /* value on the fall-through path */;

s.assert_mutually_exclusive(&pcs);    // emit O(N²) exclusion clauses
let v = s.bv_select(&pcs, &vals, dflt);
```

`bv_select` has first-match semantics (earliest `pcs[i]` that's true wins; `default` if none). It simplifies on construction:

- Drops pairs with a constant-`false` selector.
- Short-circuits to `vals[i]` on the first constant-`true` selector (rest shadowed).
- Drops pairs where `vals[i] == default` — output-indistinguishable from fall-through.
- Returns `default` outright when nothing remains.
- Collapses to the common value when all values are structurally equal.

**Call `assert_mutually_exclusive` once per set of shared selectors**, not per merged variable. The exclusion clauses are O(N²) binary clauses; with them in place, SAT propagation collapses each Select chain into a single decision (picking the live branch forces the other selectors false unit-propagation-style).

### What `bv_select` buys you over binary ITE chains

- **Simpler DAG** — one node carries the whole merge; the frontend doesn't have to manage chain construction.
- **Default-value dedup** at construction time — variables unchanged by most paths shed their branches before bitblasting.
- **Exclusion clauses are amortized** across every Select sharing the selectors (one call, unbounded reuse).
- **Semantic simplifications** fire at the DAG level where they're cheap, not at bitblast time where the chain is already expanded.

A 3-way merge with one selector forced:

```rust
let default = s.bv_var(32);
let v0 = s.bv_var(32);
let v1 = s.bv_var(32);
let v2 = s.bv_var(32);
let sels = [s.bool_var(), s.bool_var(), s.bool_var()];
s.assert_mutually_exclusive(&sels);
let out = s.bv_select(&sels, &[v0, v1, v2], default);

s.assert(sels[1]);       // force the middle path
// `out` is now pinned to `v1` by unit propagation; `sels[0]` and `sels[2]`
// are forced false by the exclusion clauses at no extra decision cost.
```

---

## Assertions and solving

```rust
s.assert(bool_term);
let r = s.solve();
```

`assert` queues the term (bitblasting is deferred to `solve` so preprocessing passes run first). `solve` returns `Sat` or `Unsat`.

### Assumption-based solving

```rust
let r = s.solve_under_assumptions(&[pc1, neg_pc2, ...]);
```

Each assumption is a `BoolTerm`. The solver commits to each being true for this call only; the state isn't retained between calls, so you can cheaply explore "formula ∧ these assumptions" without mutating the base.

Typical symbex use: each path's current path condition goes in as assumptions; the base formula holds the permanent state (memory writes, loop invariants, etc.).

### Bounded solving

Two variants, one sharing the same `Some(Sat) | Some(Unsat) | None` shape.
`None` means the solve was interrupted; any `Some(_)` is a real answer.
The solver is left in a consistent state on `None`, so retry with a larger
budget / different assumptions works without bookkeeping.

```rust
// Deterministic: stop after N SAT conflicts.
match s.solve_under_assumptions_bounded(&asmps, max_conflicts) {
    Some(SmtResult::Sat)   => { /* feasible — extract model */ }
    Some(SmtResult::Unsat) => { /* infeasible — prune */ }
    None                   => { /* budget exhausted — treat as unknown */ }
}

// Wall-clock: stop after Duration has elapsed.
match s.solve_under_assumptions_timed(&asmps, Duration::from_millis(250)) {
    Some(SmtResult::Sat)   => { /* … */ }
    Some(SmtResult::Unsat) => { /* … */ }
    None                   => { /* timed out */ }
}
```

Use conflict budgets for tight, deterministic probes (e.g., Fraig-style
equivalence checks where you want ~16 conflicts and out). Use wall-clock
timeouts for user-facing "spend at most X ms on this" queries.

Pick the right tool:
- **`solve_under_assumptions_bounded`** — deterministic, reproducible
  across runs. Every conflict costs one arithmetic check (~1ns). `max_conflicts
  = 0` is the explicit "unbounded" sentinel (identical to
  `solve_under_assumptions`).
- **`solve_under_assumptions_timed`** — non-deterministic (clock-dependent).
  The deadline is checked every 256 conflicts via a cheap bitmask test;
  worst-case overshoot is "however long 256 conflicts take", typically a
  few milliseconds. Sub-millisecond budgets aren't honoured at this
  granularity; use the conflict-budget variant for those.

A `Some(Unsat)` from either variant is a genuine proof — the budget or
timeout only ever converts "still searching" into `None`, never relaxes
soundness.

### Push / pop scoping

```rust
s.push();
s.assert(transient_constraint);
let r = s.solve();
s.pop();                     // transient_constraint goes away
```

Every assertion made inside a scope is guarded by an activation literal that `pop` retracts. Nested scopes are fine. Popping also discards any *pending* (not-yet-flushed) assertions for that scope; flushed clauses become vacuous when the activation lit is forced false.

### Named assertions + UNSAT cores

```rust
s.assert_named("path_pc_3", p3);
// ... more asserts ...
if let SmtResult::Unsat = s.solve() {
    let core_names: Vec<&str> = s.unsat_core_names();
    // core_names is the subset of named asserts that participated in UNSAT
}
```

Useful for blaming which path condition made the conjunction unreachable.

---

## Reading models

```rust
assert_eq!(s.solve(), SmtResult::Sat);
assert!(s.has_model());           // sanity

let v: bool  = s.get_bool_value(bool_term);
let w: u64   = s.get_bv_value(bv_term);          // truncates for w > 64
let w: u128  = s.get_bv_value_u128(bv_term);     // supports w ≤ 128
let w: Vec<u64> = s.get_bv_value_limbs(bv_term); // supports any width
```

`has_model()` returns false if the last state-changing call (assert, push, pop) happened after the last `solve()`, or if the last solve was Unsat. Don't trust model reads without checking.

---

## Variable aliasing (union-find)

Symbex often produces constraints like `(assert (= X Y))` where `X` and `Y` are both free vars; merging them via union-find saves the bit-biconditional clauses and reduces the SAT variable count.

```rust
if s.alias_bv_vars(x, y) {
    // x and y now share SAT literals; no equality clause needed.
}
// For Bool vars:
s.alias_bool_vars(p, q);
```

Returns `false` if either term isn't a bare `BvVar` / `BoolVar` or if either side has already been bitblasted (the alias must be installed before first use). The SMT-LIB frontend does this automatically for top-level `(= atom atom)` assertions; Rust-API callers need to call it explicitly.

---

## Incremental patterns for symbex

Typical path-exploration loop:

```rust
let base = /* permanent program state constraints */;
let mut s = SmtSolver::new();
s.assert(base);

for path in explore_paths() {
    // Is this path's constraint satisfiable given the base?
    match s.solve_under_assumptions(&path.pc_lits) {
        SmtResult::Sat   => { /* path feasible — extract a test input, fork */ }
        SmtResult::Unsat => { /* infeasible — prune */ }
    }
}
```

The SAT solver keeps learned clauses, VSIDS activities, and LBD history across `solve_under_assumptions` calls. Feeding related queries in sequence is dramatically faster than spinning up a fresh solver per query.

For exploratory forks that don't pollute the base, use `push`/`pop` to layer scope-local assertions.

---

## Preprocessing passes

These fire automatically at `solve` time; you don't need to invoke them, but knowing what they do helps you write constraints that hit the fast paths:

- **Bits-known abstract interpretation** — every term carries a `(known_ones, known_zeros)` 3-valued mask computed at construction, cascading through `not`/`and`/`or`/`xor`, `add`/`sub`/`neg`, shifts by constants, `extract`/`concat`/`zero_extend`/`sign_extend`, `ite`/`select`, and constants. When all bits are determined the term folds to a constant at build time (no SAT clauses emitted at all). `bv_eq(x, c)` short-circuits to false when x's forced bits conflict with c; `bv_ult`/`bv_ule` short-circuit on disjoint bits-known intervals.
- **SSA substitution** — `assert(bool_eq(X, expr))` with `X` a fresh `BoolTerm` var rewrites X to expr in place. (SMT-LIB frontend only; Rust API callers should build terms directly rather than via intermediate vars.)
- **Variable union-find** — `alias_bv_vars` / `alias_bool_vars`, callable explicitly.
- **Top-level equality direct-emit** — `assert(bv_eq(x, y))` becomes 2N direct biconditional clauses per bit instead of Tseitin gates.
- **Gate hash-consing** — structurally identical `mk_and` / `mk_or` / `mk_xor` / `mk_mux` gates reuse SAT literals.
- **ITE chain folding** — nested ITEs with selector aliasing collapse; common-branch factoring hoists shared sub-branches.
- **Constant-bound chain collapse** — `and(a ∧ (bvslt k_1 v) ∧ (bvslt k_2 v) ∧ …)` keeps only the tightest bound per variable.
- **BV1-as-Bool substitution** — `assert(bool_eq(bool_eq(X_1bit, bv1_1), rhs))` substitutes X throughout (SMT-LIB frontend only).

Bits-known is the most-leveraged of these for symbex workloads because symbex code tends to produce lots of `zero_extend` + `bv_and(x, mask)` + shift operations — exactly the shapes where masks propagate furthest. You'll see it fold away significant portions of the formula before any SAT work.

---

## Configuration knobs

```rust
s.set_ite_branching_hints(on: bool);   // default: on
```

When on, each bitblasted ITE bumps its selector's VSIDS priority. Empirically a large win on deep ITE trees (symbex memory reads, state merges). Leave it on unless profiling says otherwise.

---

## SMT-LIB interface

```rust
let output: String = binbit::run_script(&smt_lib_text)?;
```

Accepts QF_BV SMT-LIB 2.6 scripts. Use when debugging with external benchmarks; for the symbex pipeline, the Rust API is the right entry point.

---

## What to build against

For the symbex framework integration, the APIs that matter most are:

1. **Fresh solver + persistent base formula + `solve_under_assumptions` for path queries.** Single solver, many queries. Don't spin up a new `SmtSolver` per path; you'll throw away all the learned clause memory. Use `solve_under_assumptions_bounded` when you want to put a ceiling on branch-feasibility probes.

2. **`bv_select` + `assert_mutually_exclusive` for state merging.** Build n-way merges natively. Don't synthesize chained `bv_ite`.

3. **`alias_bv_vars` / `alias_bool_vars` when you have equality constraints between fresh vars.** Cheap to call; skips gate emission entirely when it fires.

4. **`try_bv_const_value` for concretization probes.** Cheaper than spinning up a SAT call when the term has already folded to a constant through bits-known / constant-folding.

5. **`push` / `pop` for transient exploration**, named assertions + `unsat_core_names` for blame.

Everything else is term-building — same shape as any BV-flavoured SMT API.
