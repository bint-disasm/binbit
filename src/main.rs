use std::io::Read;
use std::time::Instant;

use binbit::{LBool, Lit, SolveResult, Solver, Var, dimacs};

// Swap the global allocator for mimalloc — the solver allocates heavily in a
// few places (clause arena growth, watch lists, learned-clause Vecs) and
// mimalloc is generally faster than the system default for this workload.
// WASM targets can't link mimalloc's C backend, so fall back to the default
// allocator there.
#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn solve_one(input: &str) -> Result<(SolveResult, Solver), String> {
    let (nvars, clauses) = dimacs::parse(input)?;
    let mut solver = Solver::new();
    solver.reserve(nvars, clauses.len());
    for _ in 0..nvars {
        solver.new_var();
    }
    for c in &clauses {
        let lits: Vec<Lit> = c
            .iter()
            .map(|&n| Lit::new(Var(n.unsigned_abs() - 1), n < 0))
            .collect();
        if !solver.add_clause(lits) {
            return Ok((SolveResult::Unsat, solver));
        }
    }
    let result = solver.solve();
    Ok((result, solver))
}

fn main() {
    // Term construction, bitblasting and the SMT-LIB parser all recurse on
    // expression depth, and real symbex traces contain single assertions
    // with 10⁵+ nested ops — enough to blow the default 8MB main stack.
    // Run the actual work on a thread with a large reservation (virtual
    // memory only; pages are committed on touch).
    let child = std::thread::Builder::new()
        .stack_size(1 << 30) // 1 GiB reserve
        .spawn(real_main)
        .expect("failed to spawn main thread");
    let code = child.join().unwrap_or(2);
    if code != 0 {
        std::process::exit(code);
    }
}

fn real_main() -> i32 {
    let args: Vec<String> = std::env::args().collect();

    // --smt <file.smt2> : run an SMT-LIB 2 script through the BV solver.
    // Reads the full file, dispatches commands, and prints the output to
    // stdout. Exit code 0 on success, 2 on parse / runtime error.
    if args.len() >= 2 && args[1] == "--smt" {
        let mut want_stats = false;
        let mut path: Option<&str> = None;
        for a in &args[2..] {
            if a == "--stats" {
                want_stats = true;
            } else if path.is_none() {
                path = Some(a.as_str());
            }
        }
        let input = match path {
            Some(p) => match std::fs::read_to_string(p) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error reading {}: {}", p, e);
                    std::process::exit(2);
                }
            },
            None => {
                let mut s = String::new();
                if std::io::stdin().read_to_string(&mut s).is_err() {
                    eprintln!("error reading stdin");
                    std::process::exit(2);
                }
                s
            }
        };
        let t0 = Instant::now();
        let mut solver = binbit::SmtSolver::new();
        match binbit::run_script_with(&mut solver, &input) {
            Ok(out) => {
                print!("{}", out);
                eprintln!("c smt elapsed : {:.3}s", t0.elapsed().as_secs_f64());
                if want_stats {
                    let s = solver.sat_stats();
                    eprintln!("c sat_vars    : {}", s.sat_vars);
                    eprintln!("c sat_clauses : {}", s.sat_clauses);
                    eprintln!("c conflicts   : {}", s.conflicts);
                    eprintln!("c decisions   : {}", s.decisions);
                    eprintln!("c restarts    : {}", s.restarts);
                    eprintln!("c learned     : {}", s.learned);
                    eprintln!("c propagations: {}", s.propagations);
                    eprintln!("c bv_var_total: {}", s.bv_var_total);
                    eprintln!("c bv_aliased  : {}", s.bv_aliased);
                    eprintln!("c bool_aliased: {}", s.bool_aliased);
                    eprintln!("c bv_nodes    : {}", s.bv_nodes_total);
                    eprintln!("c bv_blasted  : {}", s.bv_vars_bitblasted);
                    eprintln!("c pp_subst    : {}", s.pp_substituted);
                    eprintln!("c pp_elim     : {}", s.pp_eliminated);
                    eprintln!("c pp_subsumed : {}", s.pp_subsumed);
                }
            }
            Err(e) => {
                eprintln!("smt-lib error: {}", e);
                std::process::exit(2);
            }
        }
        return 0;
    }

    // --batch <dir-or-files...> : solve many instances in one process, print
    // only aggregate timing. Used for benchmarking without fork/exec overhead.
    if args.len() >= 2 && args[1] == "--batch" {
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for a in &args[2..] {
            let p = std::path::Path::new(a);
            if p.is_dir() {
                for entry in std::fs::read_dir(p).expect("read dir") {
                    let entry = entry.expect("dir entry");
                    let path = entry.path();
                    if path.extension().map(|e| e == "cnf").unwrap_or(false) {
                        files.push(path);
                    }
                }
            } else {
                files.push(p.to_path_buf());
            }
        }
        files.sort();

        let mut sat = 0u64;
        let mut unsat = 0u64;
        let mut conflicts = 0u64;
        let mut decisions = 0u64;
        let mut propagations = 0u64;
        let t0 = Instant::now();
        for f in &files {
            let input = std::fs::read_to_string(f).expect("read cnf");
            let (res, solver) = solve_one(&input).expect("solve");
            match res {
                SolveResult::Sat => sat += 1,
                SolveResult::Unsat => unsat += 1,
            }
            conflicts += solver.stats_conflicts;
            decisions += solver.stats_decisions;
            propagations += solver.stats_propagations;
        }
        let elapsed = t0.elapsed();
        eprintln!("c batch of {} instances", files.len());
        eprintln!("c SAT={} UNSAT={}", sat, unsat);
        eprintln!("c total conflicts    : {}", conflicts);
        eprintln!("c total decisions    : {}", decisions);
        eprintln!("c total propagations : {}", propagations);
        eprintln!("c total cpu time     : {:.3}s", elapsed.as_secs_f64());
        eprintln!(
            "c mean per-instance  : {:.3}ms",
            elapsed.as_secs_f64() * 1000.0 / files.len().max(1) as f64
        );
        return 0;
    }

    // Single-instance mode: read from a file if given, otherwise stdin.
    let input = if args.len() >= 2 {
        match std::fs::read_to_string(&args[1]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error reading {}: {}", args[1], e);
                std::process::exit(2);
            }
        }
    } else {
        let mut s = String::new();
        if std::io::stdin().read_to_string(&mut s).is_err() {
            eprintln!("error reading stdin");
            std::process::exit(2);
        }
        s
    };

    let t0 = Instant::now();
    let (result, solver) = match solve_one(&input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parse error: {}", e);
            std::process::exit(2);
        }
    };
    let elapsed = t0.elapsed();

    match result {
        SolveResult::Sat => {
            println!("s SATISFIABLE");
            print!("v ");
            for v in 0..solver.num_vars() {
                let val = solver.value_of_var(Var(v as u32));
                let sign = if val == LBool::False { -1 } else { 1 };
                print!("{} ", sign * (v as i32 + 1));
            }
            println!("0");
        }
        SolveResult::Unsat => {
            println!("s UNSATISFIABLE");
        }
    }

    eprintln!("c variables   : {}", solver.num_vars());
    eprintln!("c clauses     : {}", solver.num_clauses());
    eprintln!("c learnts     : {}", solver.num_learnts());
    eprintln!("c conflicts   : {}", solver.stats_conflicts);
    eprintln!("c decisions   : {}", solver.stats_decisions);
    eprintln!("c propagations: {}", solver.stats_propagations);
    eprintln!("c restarts    : {}", solver.stats_restarts);
    eprintln!("c learned     : {}", solver.stats_learned);
    eprintln!("c deleted     : {}", solver.stats_deleted);
    eprintln!("c reductions  : {}", solver.stats_reductions);
    eprintln!("c min removed : {}", solver.stats_min_removed);
    eprintln!("c cpu time    : {:.3}s", elapsed.as_secs_f64());
    0
}
