// Minimal DIMACS CNF parser.
// Lines starting with 'c' are comments. A 'p cnf <vars> <clauses>' header
// is accepted but not required — we also track the max variable seen.
// Clauses are whitespace-separated signed integers terminated by 0.

pub fn parse(input: &str) -> Result<(usize, Vec<Vec<i32>>), String> {
    let mut clauses: Vec<Vec<i32>> = Vec::new();
    let mut current: Vec<i32> = Vec::new();
    let mut declared_vars = 0usize;
    let mut max_var = 0usize;

    for (line_no, raw) in input.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('c') {
            continue;
        }
        // SATLIB convention: '%' on its own line marks end-of-CNF; anything
        // after it (typically a lone trailing "0") is ignored.
        if line.starts_with('%') {
            break;
        }
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 && parts[1] == "cnf" {
                declared_vars = parts[2].parse().map_err(|_| {
                    format!("line {}: bad var count in header", line_no + 1)
                })?;
            }
            continue;
        }
        for tok in line.split_whitespace() {
            let n: i32 = tok
                .parse()
                .map_err(|_| format!("line {}: expected integer, got {:?}", line_no + 1, tok))?;
            if n == 0 {
                clauses.push(std::mem::take(&mut current));
            } else {
                let v = n.unsigned_abs() as usize;
                if v > max_var {
                    max_var = v;
                }
                current.push(n);
            }
        }
    }
    if !current.is_empty() {
        // Trailing clause without terminating 0 — accept it.
        clauses.push(current);
    }

    Ok((declared_vars.max(max_var), clauses))
}
