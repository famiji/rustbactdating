//! rustbact — Rust reimplementation of BactDating's core MCMC (arc model),
//! with incremental likelihood updates, multi-chain parallelism and
//! checkpoint/resume.

mod dist;
mod likelihood;
mod mcmc;
mod multichain;
mod roottip;
mod tree;

use mcmc::{Checkpoint, Config};
use std::io::Write;
use tree::parse_newick;

fn read_dates(path: &str) -> Result<Vec<(String, f64)>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 && line.starts_with("tip") {
            continue;
        }
        let mut it = line.split('\t');
        let tip = match it.next() {
            Some(t) => t.trim(),
            None => continue,
        };
        let raw = match it.next() {
            Some(v) => v.trim(),
            None => continue,
        };
        // R writes missing dates as "NA"; keep them as NaN (they stay unconstrained)
        let d = if raw.eq_ignore_ascii_case("NA") || raw.is_empty() {
            f64::NAN
        } else {
            raw.parse::<f64>().map_err(|e| format!("bad date '{}': {}", raw, e))?
        };
        out.push((tip.to_string(), d));
    }
    Ok(out)
}

struct Args {
    tree: String,
    dates: String,
    nb_its: usize,
    thin: usize,
    chains: usize,
    seed: u64,
    ckpt_out: Option<String>,
    resume: Vec<Option<String>>,
    model: String,
    /// multiply branch lengths by this factor (converts per-site to substitutions)
    scale_lengths: Option<f64>,
    /// only report the branch-length sum and exit
    check_lengths: bool,
}

fn parse_args() -> Result<Args, String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        return Err("usage: rustbact <tree.nwk> <dates.tsv> [nbIts] [thin]\n  \
             --chains N         parallel chains (default 1)\n  \
             --seed N           base seed (chain i uses seed+i)\n  \
             --model NAME       arc|carc|poisson|negbin|strictgamma|relaxedgamma (default arc)\n  \
             --ckpt-out PATH    write checkpoint(s); with --chains N writes PATH.0..PATH.N-1\n  \
             --resume PATH[,PATH...]  resume chain(s) from checkpoint(s)\n  \
             --scale-lengths L  multiply branch lengths by L (use the number of\n  \
             \x20                    alignment sites to convert per-site -> substitutions)\n  \
             --check-lengths    print the total branch length and exit"
            .to_string());
    }
let mut args = Args {
        tree: "".to_string(),
        dates: "".to_string(),
        nb_its: 10_000,
        thin: 10,
        chains: 1,
        seed: 42,
        ckpt_out: None,
        resume: Vec::new(),
        model: "arc".to_string(),
        scale_lengths: None,
        check_lengths: false,
    };

    // Positionals must come first: tree, dates, [nbIts], [thin].
    // A token starting with '-' or any token past the leading positionals is
    // a flag (flags may consume the following token as their value).
    let tokens: Vec<String> = a[1..].to_vec();
    let mut i = 0usize;
    // collect leading positionals (non-flag tokens)
    let mut pos: Vec<&str> = Vec::new();
    while i < tokens.len() && !tokens[i].starts_with('-') {
        pos.push(&tokens[i]);
        i += 1;
    }
    if pos.len() >= 1 {
        args.tree = pos[0].to_string();
    }
    if pos.len() >= 2 {
        args.dates = pos[1].to_string();
    }
    if pos.len() >= 3 {
        args.nb_its = pos[2].parse().unwrap_or(10_000);
    }
    if pos.len() >= 4 {
        args.thin = pos[3].parse().unwrap_or(10);
    }
    if args.tree.is_empty() || args.dates.is_empty() {
        return Err("error: tree and dates file are required".to_string());
    }

    // Now flags
    while i < tokens.len() {
        match tokens[i].as_str() {
            "--chains" => {
                if let Some(v) = tokens.get(i + 1).and_then(|t| t.parse::<usize>().ok()) {
                    args.chains = v.max(1);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--seed" => {
                if let Some(v) = tokens.get(i + 1).and_then(|t| t.parse::<u64>().ok()) {
                    args.seed = v;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--model" => {
                if let Some(v) = tokens.get(i + 1) {
                    args.model = v.to_string();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--ckpt-out" => {
                if let Some(v) = tokens.get(i + 1) {
                    args.ckpt_out = Some(v.to_string());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--scale-lengths" => {
                if let Some(v) = tokens.get(i + 1).and_then(|t| t.parse::<f64>().ok()) {
                    args.scale_lengths = Some(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--check-lengths" => {
                args.check_lengths = true;
                i += 1;
            }
            "--resume" => {
                if let Some(v) = tokens.get(i + 1) {
                    for p in v.split(',') {
                        let p = p.trim();
                        if p.is_empty() || p.eq_ignore_ascii_case("none") {
                            args.resume.push(None);
                        } else {
                            args.resume.push(Some(p.to_string()));
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    Ok(args)


}

fn load_ckpt(path: &str) -> Checkpoint {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read checkpoint {}: {}", path, e));
    bincode::deserialize(&bytes).unwrap_or_else(|e| panic!("deserialize {}: {}", path, e))
}

fn write_ckpt(path: &str, cp: &Checkpoint) {
    let bytes = bincode::serialize(cp).expect("serialize checkpoint");
    let mut f = std::fs::File::create(path).expect("create checkpoint");
    f.write_all(&bytes).expect("write checkpoint");
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    };

    let t0 = std::time::Instant::now();

    let nwk = std::fs::read_to_string(&args.tree).expect("read tree");
    let mut tree = parse_newick(&nwk).expect("parse newick");

    let raw_total = tree.total_length();
    if let Some(f) = args.scale_lengths {
        if !(f.is_finite() && f > 0.0) {
            eprintln!("error: --scale-lengths needs a positive number");
            std::process::exit(1);
        }
        tree.scale_lengths(f);
        eprintln!(
            "tree: tips={} nnode={} total_len={:.3} (scaled x{} from {:.6})",
            tree.ntip,
            tree.nnode,
            tree.total_length(),
            f,
            raw_total
        );
    } else {
        eprintln!(
            "tree: tips={} nnode={} total_len={:.3}",
            tree.ntip,
            tree.nnode,
            tree.total_length()
        );
    }

    if args.check_lengths {
        // BactDating expects branch lengths in substitutions, not per site.
        // A total near 1 means the tree is almost certainly per-site.
        let t = tree.total_length();
        eprintln!("total branch length: {:.6}", t);
        if t < 5.0 {
            eprintln!(
                "warning: this looks like substitutions per site, not substitutions.\n\
                 Pass --scale-lengths L with L = the number of alignment sites used\n\
                 to build the tree, or the likelihood will be meaningless."
            );
            std::process::exit(2);
        }
        eprintln!("ok: branch lengths look like substitution counts");
        return;
    }

    let raw_dates = read_dates(&args.dates).expect("read dates");
    let mut dates = vec![f64::NAN; tree.ntip];
    let mut matched = 0;
    for (tip, d) in &raw_dates {
        if let Some(idx) = tree.tip_index(tip) {
            dates[idx - 1] = *d;
            matched += 1;
        }
    }
    let n_dated = dates.iter().filter(|d| d.is_finite()).count();
    eprintln!(
        "dates matched: {}/{} ({} with actual dates)",
        matched, tree.ntip, n_dated
    );

    let cfg = Config {
        nb_its: args.nb_its,
        thin: args.thin,
        model: args.model.clone(),
        ..Default::default()
    };

    // ---- resume checkpoints ----
    let mut resume: Vec<Option<Checkpoint>> = Vec::with_capacity(args.chains);
    for i in 0..args.chains {
        let path = args.resume.get(i).cloned().flatten();
        resume.push(path.map(|p| load_ckpt(&p)));
    }

    eprintln!(
        "running {} chain(s): nbIts={} thin={} model={}",
        args.chains, args.nb_its, args.thin, args.model
    );

    let cps = multichain::run_chains(&tree, &dates, &cfg, args.seed, args.chains, Some(resume));

    eprintln!("all chains done in {:.2}s", t0.elapsed().as_secs_f64());

    // ---- write checkpoints ----
    if let Some(base) = &args.ckpt_out {
        if args.chains == 1 {
            write_ckpt(base, &cps[0]);
            eprintln!("checkpoint written: {}", base);
        } else {
            for (i, cp) in cps.iter().enumerate() {
                let p = format!("{}.{}", base, i);
                write_ckpt(&p, cp);
            }
            eprintln!("checkpoints written: {}.0 .. {}.{}", base, base, args.chains - 1);
        }
    }

    // ---- summary ----
    let total_rows: usize = cps.iter().map(|c| c.state.record.len()).sum();
    eprintln!(
        "\n=== summary ({} chain(s), {} samples total) ===",
        args.chains, total_rows
    );
    eprintln!(
        "{:<8} {:>12} {:>12} {:>23} {:>11} {:>11} {:>8}",
        "param", "mean", "sd", "95% HPD", "ESS/chain", "ESS total", "R-hat"
    );
    for p in ["mu", "sigma", "alpha"] {
        if let Some(s) = multichain::summarize(&cps, p) {
            eprintln!(
                "{:<8} {:>12.4} {:>12.4} {:>10.4}..{:<11.4} {:>11.1} {:>11.1} {:>8.4}",
                s.name, s.mean, s.sd, s.hpd_lo, s.hpd_hi, s.ess_per_chain, s.ess_total, s.r_hat
            );
        }
    }
}
