//! Repeat a full SMT-LIB script run N times in one process so `sample`
//! can profile front-end-bound files whose single-shot runtime is tens
//! of milliseconds (parse + term build + flush; the SAT search runs too,
//! but on these files it is a rounding error).
//!
//!     cargo run --release --example parse_bench -- <file.smt2> [reps]

fn main() {
    // Same big-stack thread as src/main.rs — deep let-nesting recursion.
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(run)
        .expect("spawn")
        .join()
        .expect("join");
}

fn run() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: parse_bench <file.smt2> [reps]");
    let reps: usize = args
        .next()
        .map(|s| s.parse().expect("reps must be a number"))
        .unwrap_or(50);
    let input = std::fs::read_to_string(&path).expect("read input");
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let mut solver = binbit::smt::SmtSolver::new();
        let out = binbit::run_script_with(&mut solver, &input).expect("script runs");
        std::hint::black_box(out);
    }
    let dt = t0.elapsed();
    println!(
        "{} reps of {} in {:.3}s ({:.1} ms/rep)",
        reps,
        path,
        dt.as_secs_f64(),
        dt.as_secs_f64() * 1e3 / reps as f64
    );
}
