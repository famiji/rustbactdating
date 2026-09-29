//! Export dating results: a time-scaled tree, per-node credible intervals,
//! the root date, and DIC.
//!
//! The MCMC record stores every node's date at every thinned iteration, so the
//! posterior mean and quantiles per node come straight out of the record
//! without any extra likelihood evaluations — the only additional cost is the
//! DIC, which needs one likelihood evaluation at the posterior mean.

use crate::likelihood::{full_loglik, Tab};
use crate::mcmc::Checkpoint;
use crate::tree::Tree;

pub struct DatingSummary {
    pub params: Vec<ParamRow>,
    pub root_date: f64,
    pub root_ci: (f64, f64),
    pub dic: f64,
    /// posterior mean date per node id (1-based); tips are their sampling dates
    pub node_mean: Vec<f64>,
    /// 95% credible interval per node id (1-based)
    pub node_ci: Vec<(f64, f64)>,
}

pub struct ParamRow {
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    pub ci: (f64, f64),
    pub ess: f64,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Mean and quantiles over the second half of the record (discard burn-in).
fn summarize_series(vals: &[f64]) -> (f64, f64, (f64, f64)) {
    let n = vals.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, (f64::NAN, f64::NAN));
    }
    let half = n / 2;
    let tail = &vals[half..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let var = if tail.len() > 1 {
        tail.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (tail.len() as f64 - 1.0)
    } else {
        0.0
    };
    let mut sorted = tail.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (mean, var.sqrt(), (quantile(&sorted, 0.025), quantile(&sorted, 0.975)))
}

/// Build the summary from a set of chains.
pub fn summarize(cps: &[Checkpoint], tree: &Tree, dates: &[f64]) -> DatingSummary {
    let cols = &cps[0].state.record_cols;
    let col = |name: &str| cols.iter().position(|c| c == name);
    let nrow = tree.total_nodes();

    // pooled posterior samples per parameter, and per node date
    let mut param_vals: Vec<(String, Vec<f64>)> = Vec::new();
    for p in ["mu", "sigma", "alpha"] {
        let ci = match col(p) {
            Some(i) => i,
            None => continue,
        };
        let mut v = Vec::new();
        for cp in cps {
            let half = cp.state.record.len() / 2;
            for row in cp.state.record[half..].iter() {
                if let Some(x) = row.get(ci) {
                    if x.is_finite() {
                        v.push(*x);
                    }
                }
            }
        }
        param_vals.push((p.to_string(), v));
    }

    // node dates: one column per node, "date{id}"
    let mut node_samples: Vec<Vec<f64>> = vec![Vec::new(); nrow + 1];
    for cp in cps {
        let half = cp.state.record.len() / 2;
        for row in cp.state.record[half..].iter() {
            for node in 1..=nrow {
                if let Some(i) = col(&format!("date{}", node)) {
                    if let Some(x) = row.get(i) {
                        if x.is_finite() {
                            node_samples[node].push(*x);
                        }
                    }
                }
            }
        }
    }

    let mut params = Vec::new();
    for (name, vals) in &param_vals {
        let (mean, sd, ci) = summarize_series(vals);
        let ess = crate::multichain::ess(vals);
        params.push(ParamRow {
            name: name.clone(),
            mean,
            sd,
            ci,
            ess,
        });
    }

    let root_row = (1..=nrow)
        .find(|&n| cps[0].state.father[n - 1] == 0)
        .unwrap_or(tree.root);

    let node_mean: Vec<f64> = (0..=nrow)
        .map(|n| {
            if n == 0 {
                f64::NAN
            } else if node_samples[n].is_empty() {
                f64::NAN
            } else {
                let v = &node_samples[n];
                v.iter().sum::<f64>() / v.len() as f64
            }
        })
        .collect();
    let node_ci: Vec<(f64, f64)> = (0..=nrow)
        .map(|n| {
            if n == 0 || node_samples[n].is_empty() {
                (f64::NAN, f64::NAN)
            } else {
                let mut s = node_samples[n].clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                (quantile(&s, 0.025), quantile(&s, 0.975))
            }
        })
        .collect();

    let root_date = node_mean[root_row];
    let root_ci = node_ci[root_row];

    // DIC: -2*loglik(posterior mean) + 2*var(loglik)
    let dic = dic_from_record(cps, tree, dates, &node_mean, &param_vals);

    DatingSummary {
        params,
        root_date,
        root_ci,
        dic,
        node_mean,
        node_ci,
    }
}

fn dic_from_record(
    cps: &[Checkpoint],
    _tree: &Tree,
    dates: &[f64],
    node_mean: &[f64],
    params: &[(String, Vec<f64>)],
) -> f64 {
    let cp = &cps[0];
    let st = &cp.state;
    let nrow = st.date.len();
    let ntip = cp.ntip;

    let mean_of = |name: &str| -> f64 {
        params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.iter().sum::<f64>() / v.len().max(1) as f64)
            .unwrap_or(f64::NAN)
    };
    let mu = mean_of("mu");
    let sigma = mean_of("sigma");

    // build a tab at the posterior mean dates and evaluate the likelihood
    let mut date = node_mean[1..=nrow].to_vec();
    for t in 1..=ntip {
        if dates[t - 1].is_finite() {
            date[t - 1] = dates[t - 1];
        }
    }
    let tab = Tab::new(
        ntip,
        nrow - ntip,
        st.subs.clone(),
        date,
        st.father.clone(),
        st.unrec.clone(),
        0.1,
        cp.root_idx,
    );
    let l_at_mean = full_loglik(&cp.cfg.model, &tab, mu, sigma);

    // var(-2 loglik) over the recorded samples
    let ci = cp.state.record_cols.iter().position(|c| c == "likelihood");
    let var_l = match ci {
        Some(i) => {
            let vals: Vec<f64> = cps
                .iter()
                .flat_map(|c| {
                    let half = c.state.record.len() / 2;
                    c.state.record[half..]
                        .iter()
                        .filter_map(|r| r.get(i).copied())
                        .filter(|v| v.is_finite())
                })
                .collect();
            if vals.len() > 1 {
                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (vals.len() as f64 - 1.0);
                2.0 * v
            } else {
                0.0
            }
        }
        None => 0.0,
    };
    -2.0 * l_at_mean + var_l
}

/// Write the time-scaled tree (branch lengths in time units, tip labels kept).
pub fn write_dated_tree(tree: &Tree, summary: &DatingSummary, path: &str) -> std::io::Result<()> {
    fn rec(
        tree: &Tree,
        summary: &DatingSummary,
        node: usize,
        out: &mut String,
    ) {
        if !tree.children[node].is_empty() {
            out.push('(');
            for (i, &c) in tree.children[node].iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                rec(tree, summary, c, out);
            }
            out.push(')');
        } else {
            out.push_str(&tree.tip_labels[node - 1]);
        }
        if node != tree.root {
            // elapsed time along the branch: child date minus parent date.
            // Dates are calendar years, so the parent (older) has the smaller
            // value and the difference is positive.
            let dt = summary.node_mean[node] - summary.node_mean[tree.parent[node]];
            out.push(':');
            out.push_str(&format!("{:.6}", if dt.is_finite() { dt } else { 0.0 }));
        }
    }
    let mut s = String::new();
    rec(tree, summary, tree.root, &mut s);
    s.push_str(";\n");
    std::fs::write(path, s)
}

/// Per-node posterior dates with credible intervals.
pub fn write_node_dates(
    tree: &Tree,
    summary: &DatingSummary,
    path: &str,
) -> std::io::Result<()> {
    let mut out = String::from("node\tlabel\ttip\tmean_date\tci_lo\tci_hi\n");
    for node in 1..=tree.total_nodes() {
        let label = if node <= tree.ntip {
            tree.tip_labels[node - 1].clone()
        } else {
            format!("node{}", node - tree.ntip)
        };
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\n",
            node,
            label,
            node <= tree.ntip,
            summary.node_mean[node],
            summary.node_ci[node].0,
            summary.node_ci[node].1
        ));
    }
    std::fs::write(path, out)
}
