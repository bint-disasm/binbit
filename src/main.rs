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
        // Preprocessing ablation flags (all passes on by default). Let us
        // bisect which pass helps or hurts a given instance without a
        // recompile: --no-norm --no-subst --no-gauss --no-bve, or
        // --no-preprocess to turn off everything at once.
        let (mut norm, mut subst, mut gauss, mut bve) = (true, true, true, true);
        // FRAIG feasibility probe: after the script runs, random-simulate
        // the accumulated AIG and report semantic-redundancy candidates.
        let mut want_fraig_diag = false;
        // FRAIG sweep: prove + merge equivalent AIG nodes before CNF
        // emission. Off by default (changes search trajectory).
        let mut fraig = false;
        // Clause vivification in the SAT core. Off by default (per-
        // instance trajectory lottery on the symbex corpus).
        let mut vivify = false;
        // Two-level AIG rewriting (Brummayer-Biere) in the bitblaster.
        // Off by default (changes search trajectory).
        let mut aig2 = false;
        // Sharing-aware variant: safe build rules + parent-count-gated
        // post-build substitution pass.
        let mut aig2_post = false;
        // Cut-based CNF technology mapping at materialization. Off by
        // default (see `SmtSolver::cnf_mapping`); `--cnfmap` maps at Fast
        // effort, `--cnfmap-full` spends more mapping time for better
        // covers on dense arithmetic.
        let mut cnfmap = false;
        let mut cnfmap_full = false;
        for a in &args[2..] {
            match a.as_str() {
                "--stats" => want_stats = true,
                "--fraig-diag" => want_fraig_diag = true,
                "--fraig" => fraig = true,
                "--vivify" => vivify = true,
                "--aig2" => aig2 = true,
                "--cnfmap" => cnfmap = true,
                "--no-cnfmap" => cnfmap = false,
                "--cnfmap-full" => {
                    cnfmap = true;
                    cnfmap_full = true;
                }
                "--aig2-post" => aig2_post = true,
                "--no-norm" => norm = false,
                "--no-subst" => subst = false,
                "--no-gauss" => gauss = false,
                "--no-bve" => bve = false,
                "--no-preprocess" => {
                    norm = false;
                    subst = false;
                    gauss = false;
                    bve = false;
                }
                _ if path.is_none() => path = Some(a.as_str()),
                _ => {}
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
        solver.set_normalization(norm);
        solver.set_substitution(subst);
        solver.set_gaussian(gauss);
        solver.set_bve(bve);
        solver.set_fraig(fraig);
        solver.set_vivification(vivify);
        solver.set_aig_two_level(aig2);
        solver.set_cnf_mapping(cnfmap);
        solver.set_cnf_mapping_effort(cnfmap_full);
        if aig2_post {
            solver.set_aig_two_level_post(true);
        }
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
                    eprintln!("c reused_lvls : {}", s.reused_levels);
                    eprintln!(
                        "c vivify      : checked {} strengthened {} deleted {} units {}",
                        s.viv_checked, s.viv_strengthened, s.viv_deleted, s.viv_units
                    );
                    eprintln!(
                        "c phase_times : front {:.3}s emit {:.3}s preprocess {:.3}s sat {:.3}s",
                        s.time_front, s.time_emit, s.time_preprocess, s.time_sat
                    );
                    eprintln!("c learned     : {}", s.learned);
                    eprintln!("c propagations: {}", s.propagations);
                    eprintln!("c reductions  : {}", s.reductions);
                    eprintln!("c gcs         : {}", s.gcs);
                    eprintln!("c bv_var_total: {}", s.bv_var_total);
                    eprintln!("c bv_aliased  : {}", s.bv_aliased);
                    eprintln!("c bool_aliased: {}", s.bool_aliased);
                    eprintln!("c bv_nodes    : {}", s.bv_nodes_total);
                    eprintln!("c bv_blasted  : {}", s.bv_vars_bitblasted);
                    eprintln!("c pp_subst    : {}", s.pp_substituted);
                    eprintln!("c pp_elim     : {}", s.pp_eliminated);
                    eprintln!("c pp_subsumed : {}", s.pp_subsumed);
                    eprintln!("c pp_remat    : {}", s.pp_remat);
                    let (ag, xg, mg) = solver.gate_mix();
                    eprintln!("c gates_and   : {}", ag);
                    eprintln!("c gates_xor   : {}", xg);
                    eprintln!("c gates_mux   : {}", mg);
                    if aig2_post {
                        let ps = solver.aig2_post_report();
                        eprintln!(
                            "c aig2_post   : applied={} blocked={} folds={} passes={}",
                            ps.subst_applied, ps.blocked, ps.folds, ps.passes
                        );
                    }
                    if aig2 || aig2_post {
                        let rw = solver.aig_rw_counts();
                        eprintln!(
                            "c aig2_rules  : contra={} subsume={} idem2={} resol={} subst={} idem4={}",
                            rw[0], rw[1], rw[2], rw[3], rw[4], rw[5]
                        );
                    }
                }
                if fraig {
                    let (fs, ft) = solver.fraig_report();
                    eprintln!("c fraig_cand  : {}", fs.candidates);
                    eprintln!("c fraig_proven: {}", fs.proven);
                    eprintln!("c fraig_disprv: {}", fs.disproven);
                    eprintln!("c fraig_skip  : {}", fs.skipped);
                    eprintln!("c fraig_cexpr : {}", fs.cex_pruned);
                    eprintln!("c fraig_query : {}", fs.queries);
                    eprintln!("c fraig_time  : {:.3}s", ft.as_secs_f64());
                }
                if want_fraig_diag {
                    let d = solver.fraig_diagnostic();
                    eprintln!("c aig_nodes   : {}", d.num_nodes);
                    eprintln!("c aig_ands    : {}", d.num_and);
                    eprintln!("c sim_const   : {}", d.sim_const);
                    eprintln!("c sim_classes : {}", d.classes);
                    eprintln!("c sim_redund  : {}", d.redundant);
                    eprintln!("c sim_maxclass: {}", d.largest_class);
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
    eprintln!("c gcs         : {}", solver.stats_gcs);
    eprintln!("c min removed : {}", solver.stats_min_removed);
    eprintln!("c cpu time    : {:.3}s", elapsed.as_secs_f64());
    0
}
