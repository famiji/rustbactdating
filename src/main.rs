//! rustbact — Rust reimplementation of BactDating's core MCMC (arc model),
//! with incremental likelihood updates, multi-chain parallelism and
//! checkpoint/resume.

mod dist;
mod export;
mod likelihood;
mod loaders;
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
    /// search resolution for initial rooting of an unrooted tree (0 = skip)
    mtry: usize,
    /// write dating results with this prefix (dated tree, node dates, summary)
    out_prefix: Option<String>,
    /// load a Gubbins run by prefix (replaces tree/dates tree input)
    gubbins_prefix: Option<String>,
    /// load a ClonalFrameML run by prefix
    cfml_prefix: Option<String>,
    /// use the recombination information from the loader
    use_rec: bool,
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
             --check-lengths    print the total branch length and exit\n  \
             --mtry N           search resolution when rooting an unrooted tree\n  \
             \x20                    (default 5, 0 disables initial rooting)\n  \
             --out-prefix P     write results: P.dated_tree.nwk, P.node_dates.tsv,\n  \
             \x20                    P.summary.txt\n  \
             --gubbins PREFIX   load tree + unrec from a Gubbins run (PREFIX.final_tree.tre,\n  \
             \x20                    PREFIX.per_branch_statistics.csv, ...); tree positional\n  \
             \x20                    is then optional\n  \
             --cfml PREFIX      load tree + unrec from a ClonalFrameML run\n  \
             --use-rec          use the loaded unrec values in the likelihood"
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
        mtry: 5,
        out_prefix: None,
        gubbins_prefix: None,
        cfml_prefix: None,
        use_rec: false,
    };

    // Positionals must come first: tree, dates, [nbIts], [thin].
    // A token starting with '-' or any token past the leading positionals is
    // a flag (flags may consume the following token as their value).
    let tokens: Vec<String> = a[1..].to_vec();
    // Flags that consume the following token as their value. Everything else
    // (leading, or not swallowed by a flag) is positional.
    const VALUE_FLAGS: [&str; 7] = [
        "--chains",
        "--seed",
        "--model",
        "--ckpt-out",
        "--resume",
        "--scale-lengths",
        "--out-prefix",
    ];
    const VALUE_FLAGS2: [&str; 3] = ["--gubbins", "--cfml", "--mtry"];
    let mut pos: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        let t = tokens[i].as_str();
        if VALUE_FLAGS.contains(&t) || VALUE_FLAGS2.contains(&t) {
            i += 2; // skip flag and its value
        } else if t.starts_with('-') {
            i += 1; // boolean flag
        } else {
            pos.push(tokens[i].clone());
            i += 1;
        }
    }
    let pos: Vec<&str> = pos.iter().map(|s| s.as_str()).collect();
    let mut i = 0usize;
    // advance i to the first flag (start of the flag section)
    while i < tokens.len() && !tokens[i].starts_with('-') {
        i += 1;
    }
    let has_loader = a.iter().any(|t| t == "--gubbins" || t == "--cfml");
    if has_loader {
        // with a loader the tree comes from the run output, so a single
        // positional is the dates file
        if pos.len() >= 1 {
            args.dates = pos[0].to_string();
        }
    } else {
        if pos.len() >= 1 {
            args.tree = pos[0].to_string();
        }
        if pos.len() >= 2 {
            args.dates = pos[1].to_string();
        }
    }
    if pos.len() >= 3 {
        args.nb_its = pos[2].parse().unwrap_or(10_000);
    }
    if pos.len() >= 4 {
        args.thin = pos[3].parse().unwrap_or(10);
    }
    if args.dates.is_empty() {
        return Err("error: a dates file is required (see --help)".to_string());
    }
    if !has_loader && args.tree.is_empty() {
        return Err(
            "error: a tree is required: pass a newick file, or --gubbins/--cfml PREFIX".to_string(),
        );
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
            "--mtry" => {
                if let Some(v) = tokens.get(i + 1).and_then(|t| t.parse::<usize>().ok()) {
                    args.mtry = v;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--out-prefix" => {
                if let Some(v) = tokens.get(i + 1) {
                    args.out_prefix = Some(v.to_string());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--gubbins" => {
                if let Some(v) = tokens.get(i + 1) {
                    args.gubbins_prefix = Some(v.to_string());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--cfml" => {
                if let Some(v) = tokens.get(i + 1) {
                    args.cfml_prefix = Some(v.to_string());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--use-rec" => {
                args.use_rec = true;
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

    // ---- tree: from a loader, or from the positional newick file ----
    let mut loaded_unrec: Option<Vec<f64>> = None;
    let mut tree: tree::Tree;
    if let Some(prefix) = &args.gubbins_prefix {
        match loaders::load_gubbins(prefix) {
            Ok(l) => {
                eprintln!(
                    "loaded Gubbins run '{}': tips={} nnode={} total_len={:.3}",
                    prefix,
                    l.tree.ntip,
                    l.tree.nnode,
                    l.tree.total_length()
                );
                tree = l.tree;
                loaded_unrec = Some(l.unrec);
            }
            Err(e) => {
                eprintln!("error loading Gubbins output: {}", e);
                std::process::exit(1);
            }
        }
    } else if let Some(prefix) = &args.cfml_prefix {
        match loaders::load_clonalframeml(prefix) {
            Ok(l) => {
                eprintln!(
                    "loaded ClonalFrameML run '{}': tips={} nnode={} total_len={:.3} (lengths scaled to substitutions)",
                    prefix,
                    l.tree.ntip,
                    l.tree.nnode,
                    l.tree.total_length()
                );
                tree = l.tree;
                loaded_unrec = Some(l.unrec);
            }
            Err(e) => {
                eprintln!("error loading ClonalFrameML output: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        if args.tree.is_empty() {
            eprintln!("error: no tree given (pass a newick file, or --gubbins/--cfml PREFIX)");
            std::process::exit(1);
        }
        // (loader branches handle the tree themselves)
        let nwk = std::fs::read_to_string(&args.tree).expect("read tree");
        tree = parse_newick(&nwk).expect("parse newick");
    }
    if args.dates.is_empty() {
        eprintln!("error: a dates file is required");
        std::process::exit(1);
    }

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

    // An unrooted tree (trifurcating root) cannot be handled by the root moves,
    // which require a bifurcating root. Pick the root that maximises the
    // date/root-to-tip correlation, as BactDating's initRoot does.
    if !tree.is_rooted() {
        if args.mtry == 0 {
            eprintln!(
                "warning: tree is unrooted and --mtry 0 was given; dating will proceed \
                 with the root fixed at the trifurcation."
            );
        } else {
            eprintln!(
                "tree is unrooted: searching for the root that maximises the \
                 date/root-to-tip correlation (mtry={})...",
                args.mtry
            );
            let t1 = std::time::Instant::now();
            tree = tree.init_root(&dates, args.mtry);
            let corr = roottip::root_to_tip_cor(&tree, &dates);
            eprintln!(
                "rooted at the best of the candidates in {:.1}s (cor={:.4}, R2={:.4})",
                t1.elapsed().as_secs_f64(),
                corr,
                corr * corr
            );
        }
    }

    let cfg = Config {
        nb_its: args.nb_its,
        thin: args.thin,
        model: args.model.clone(),
        check_loglik: std::env::var("RUSTBACT_CHECK_LOGIK").is_ok(),
        use_rec: args.use_rec && loaded_unrec.is_some(),
        ..Default::default()
    };
    if args.use_rec && loaded_unrec.is_none() {
        eprintln!(
            "warning: --use-rec needs --gubbins/--cfml (no unrec information available); \
             proceeding with useRec = FALSE"
        );
    }
    if let Some(u) = &loaded_unrec {
        let n_rec = u.iter().filter(|v| **v < 1.0).count();
        let min_u = u.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_u = u.iter().sum::<f64>() / u.len().max(1) as f64;
        if args.use_rec {
            eprintln!(
                "using recombination information (useRec = TRUE): {} of {} branches recombined, min unrec={:.4}, mean unrec={:.4}",
                n_rec,
                u.len(),
                min_u,
                mean_u
            );
        } else {
            eprintln!(
                "unrec loaded but not used ({} of {} branches recombined; pass --use-rec to apply)",
                n_rec,
                u.len()
            );
        }
    }

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

    let cps = multichain::run_chains(
        &tree,
        &dates,
        &cfg,
        args.seed,
        args.chains,
        Some(resume),
        loaded_unrec.clone(),
    );

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
    // root-move acceptance (only meaningful when updateRoot is on)
    let (ra, rt, sa): (usize, usize, usize) = cps.iter().fold((0, 0, 0), |(a, t, s), c| {
        (a + c.state.root_branch_accepted, t + c.state.root_branch_tried, s + c.state.root_slide_accepted)
    });
    if rt > 0 || sa > 0 {
        let sk_tip: usize = cps.iter().map(|c| c.state.root_skip_tip).sum();
        let sk_ch: usize = cps.iter().map(|c| c.state.root_skip_children).sum();
        let lastmh = cps.first().map(|c| c.state.root_last_mh).unwrap_or(f64::NAN);
        eprintln!(
            "root moves: branch {}/{} accepted ({:.1}%), slide {} accepted | skipped: tip-child {}, non-binary {} | last mh={:.3}",
            ra,
            rt,
            if rt > 0 { ra as f64 / rt as f64 * 100.0 } else { 0.0 },
            sa,
            sk_tip,
            sk_ch,
            lastmh
        );
    }

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

    // ---- dating results ----
    if let Some(prefix) = &args.out_prefix {
        let sum = export::summarize(&cps, &tree, &dates);
        eprintln!(
            "\nroot date: {:.2} (95% CI {:.2}..{:.2})",
            sum.root_date, sum.root_ci.0, sum.root_ci.1
        );
        eprintln!(
            "DIC: {:.1}   ({} nodes with CIs)",
            sum.dic,
            sum.node_ci.iter().filter(|(a, _)| a.is_finite()).count()
        );

        let tree_path = format!("{}.dated_tree.nwk", prefix);
        let nodes_path = format!("{}.node_dates.tsv", prefix);
        let summary_path = format!("{}.summary.txt", prefix);
        if let Err(e) = export::write_dated_tree(&tree, &sum, &tree_path) {
            eprintln!("error writing {}: {}", tree_path, e);
        }
        if let Err(e) = export::write_node_dates(&tree, &sum, &nodes_path) {
            eprintln!("error writing {}: {}", nodes_path, e);
        }

        let mut txt = String::new();
        txt.push_str(&format!("model\t{}\n", args.model));
        txt.push_str(&format!("chains\t{}\n", args.chains));
        txt.push_str(&format!("iterations\t{}\n", args.nb_its));
        txt.push_str(&format!("tips\t{}\n", tree.ntip));
        txt.push_str("\nparameter\tmean\tsd\tci_lo\tci_hi\tess\n");
        for p in &sum.params {
            txt.push_str(&format!(
                "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.1}\n",
                p.name, p.mean, p.sd, p.ci.0, p.ci.1, p.ess
            ));
        }
        txt.push_str(&format!(
            "\nroot_date\t{:.4}\nroot_date_ci_lo\t{:.4}\nroot_date_ci_hi\t{:.4}\ndic\t{:.2}\n",
            sum.root_date, sum.root_ci.0, sum.root_ci.1, sum.dic
        ));
        if let Err(e) = std::fs::write(&summary_path, txt) {
            eprintln!("error writing {}: {}", summary_path, e);
        } else {
            eprintln!(
                "wrote {}\n      {}\n      {}",
                tree_path, nodes_path, summary_path
            );
        }
    }
}
