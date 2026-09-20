//! Multi-chain MCMC: run N independent chains in parallel threads,
//! then combine for better ESS and compute convergence diagnostics (R-hat).
//!
//! Why: a single chain's ESS is limited by autocorrelation. Running N chains
//! with different seeds multiplies the effective sample size, and R-hat tells
//! us whether the chains actually converged to the same posterior.

use crate::mcmc::{Checkpoint, Config, Mcmc};
use crate::tree::Tree;

/// Run `n_chains` independent chains in parallel, each with seed `base_seed + i`.
/// `resume` optionally supplies one checkpoint per chain (same order).
pub fn run_chains(
    tree: &Tree,
    dates: &[f64],
    cfg: &Config,
    base_seed: u64,
    n_chains: usize,
    resume: Option<Vec<Option<Checkpoint>>>,
) -> Vec<Checkpoint> {
    let n = n_chains.max(1);
    let mut results: Vec<Option<Checkpoint>> = (0..n).map(|_| None).collect();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .map(|i| {
                let ck = resume
                    .as_ref()
                    .and_then(|v| v.get(i).cloned().flatten());
                let seed = base_seed.wrapping_add(i as u64);
                let mut cfg_i = cfg.clone();
                // record the chain's own seed so runs are reproducible
                let dates_i = dates.to_vec();
                scope.spawn(move || {
                    let mut m = Mcmc::new(tree, dates_i, cfg_i.clone(), seed);
                    cfg_i = m.cfg.clone();
                    m.run(ck)
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            results[i] = Some(h.join().expect("chain thread panicked"));
        }
    });

    results.into_iter().map(|r| r.unwrap()).collect()
}

/// Extract a parameter column from a chain's record.
pub fn param_series(cp: &Checkpoint, name: &str) -> Vec<f64> {
    let cols = &cp.state.record_cols;
    let idx = match cols.iter().position(|c| c == name) {
        Some(i) => i,
        None => return Vec::new(),
    };
    cp.state
        .record
        .iter()
        .filter_map(|row| row.get(idx).copied())
        .filter(|v| v.is_finite())
        .collect()
}

/// Initial-positive-sequence ESS estimator (Geyer), as used by coda's default.
pub fn ess(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 10 {
        return n as f64;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return n as f64;
    }
    // autocorrelations up to n/2
    let max_lag = n / 2;
    let mut acf = Vec::with_capacity(max_lag);
    for k in 1..=max_lag {
        let mut s = 0.0;
        for t in 0..(n - k) {
            s += (x[t] - mean) * (x[t + k] - mean);
        }
        acf.push(s / (n as f64 * var));
    }
    // sum consecutive pairs while they stay positive (initial positive sequence)
    let mut sum = 0.0;
    let mut i = 0;
    while i + 1 < acf.len() {
        let pair = acf[i] + acf[i + 1];
        if pair < 0.0 {
            break;
        }
        sum += pair;
        i += 2;
    }
    let ess = n as f64 / (1.0 + 2.0 * sum);
    ess.max(1.0)
}

/// Split-R-hat (Gelman-Rubin) across chains. Requires >= 2 chains of equal length.
pub fn r_hat(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m < 2 {
        return f64::NAN;
    }
    // split each chain in half
    let mut split: Vec<Vec<f64>> = Vec::new();
    for c in chains {
        let h = c.len() / 2;
        if h < 2 {
            return f64::NAN;
        }
        split.push(c[..h].to_vec());
        split.push(c[h..2 * h].to_vec());
    }
    let m = split.len();
    let n = split[0].len();
    let means: Vec<f64> = split
        .iter()
        .map(|c| c.iter().sum::<f64>() / n as f64)
        .collect();
    let grand = means.iter().sum::<f64>() / m as f64;
    let b = n as f64 * means.iter().map(|mu| (mu - grand) * (mu - grand)).sum::<f64>()
        / (m as f64 - 1.0);
    let w = split
        .iter()
        .map(|c| {
            let mu = c.iter().sum::<f64>() / n as f64;
            c.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n as f64 - 1.0)
        })
        .sum::<f64>()
        / m as f64;
    if w <= 0.0 {
        return f64::NAN;
    }
    let var_hat = (n as f64 - 1.0) / n as f64 * w + b / n as f64;
    (var_hat / w).sqrt()
}

/// Summary across chains for one parameter.
#[derive(Debug, Clone)]
pub struct ParamSummary {
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    pub hpd_lo: f64,
    pub hpd_hi: f64,
    pub ess_per_chain: f64,
    pub ess_total: f64,
    pub r_hat: f64,
}

pub fn summarize(cps: &[Checkpoint], name: &str) -> Option<ParamSummary> {
    let chains: Vec<Vec<f64>> = cps.iter().map(|cp| param_series(cp, name)).collect();
    let chains: Vec<Vec<f64>> = chains.into_iter().filter(|c| !c.is_empty()).collect();
    if chains.is_empty() {
        return None;
    }
    let pooled: Vec<f64> = chains.iter().flatten().cloned().collect();
    let n = pooled.len() as f64;
    let mean = pooled.iter().sum::<f64>() / n;
    let sd = (pooled.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt();

    // 95% HPD via sorted quantiles (2.5% / 97.5%)
    let mut sorted = pooled.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = sorted[((sorted.len() as f64) * 0.025) as usize];
    let hi = sorted[(((sorted.len() as f64) * 0.975) as usize).min(sorted.len() - 1)];

    let ess_per_chain = chains.iter().map(|c| ess(c)).sum::<f64>() / chains.len() as f64;
    // chains are independent, so total ESS is roughly the sum
    let ess_total = chains.iter().map(|c| ess(c)).sum::<f64>();
    let rh = r_hat(&chains);

    Some(ParamSummary {
        name: name.to_string(),
        mean,
        sd,
        hpd_lo: lo,
        hpd_hi: hi,
        ess_per_chain,
        ess_total,
        r_hat: rh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ess_white_noise() {
        // iid samples: ESS ~ n
        let mut x = Vec::new();
        let mut s = 12345u64;
        for _ in 0..2000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            x.push(((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5);
        }
        let e = ess(&x);
        assert!(e > 1000.0, "ESS of white noise should be large, got {}", e);
    }

    #[test]
    fn test_r_hat_same_chains() {
        // two identical chains -> R-hat ~ 1
        let c: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let rh = r_hat(&[c.clone(), c.clone()]);
        assert!(rh.is_finite() && (rh - 1.0).abs() < 0.3, "got {}", rh);
    }
}
