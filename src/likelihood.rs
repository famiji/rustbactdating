//! Likelihood functions, ported from BactDating's src/probs.cpp.
//!
//! Table layout (row i, 0-based, corresponds to node id i+1):
//!   subs[i]   : observed substitutions on the branch above node i+1
//!   date[i]   : node date
//!   father[i] : father node id (0 for the root)
//!   unrec[i]  : unrecombined proportion (only when ncol == 5)
//!
//! The R implementation recomputes the likelihood over the whole table on every
//! MH move. Here we additionally expose `branch_loglik` so the MCMC can do
//! incremental updates: changing one node's date only affects the branch above
//! it and the branches above its direct children.

use crate::dist::*;

#[derive(Debug, Clone)]
pub struct Tab {
    pub nrow: usize,
    pub ncol: usize,
    /// substitutions above each node (index i => node id i+1)
    pub subs: Vec<f64>,
    /// node dates
    pub date: Vec<f64>,
    /// father node id (0 = none/root)
    pub father: Vec<usize>,
    /// unrecombined proportion
    pub unrec: Vec<f64>,
    pub minbralen: f64,
    pub ntip: usize,
    /// 0-based row index of the root
    pub root_idx: usize,
}

impl Tab {
    pub fn new(
        ntip: usize,
        nnode: usize,
        subs: Vec<f64>,
        date: Vec<f64>,
        father: Vec<usize>,
        unrec: Vec<f64>,
        minbralen: f64,
        root_idx: usize,
    ) -> Self {
        let nrow = ntip + nnode;
        Tab { nrow, ncol: 4, subs, date, father, unrec, minbralen, ntip, root_idx }
    }

    /// Duration of the branch above node `i` (0-based). Root has no branch.
    #[inline]
    pub fn duration(&self, i: usize) -> f64 {
        if i == self.root_idx || self.father[i] == 0 {
            return 0.0;
        }
        self.date[i] - self.date[self.father[i] - 1]
    }

    /// Effective (unrecombined-scaled) duration and unrec factor for node `i`.
    #[inline]
    fn unrec_duration(&self, i: usize) -> (f64, f64) {
        let unrec = if self.ncol == 5 { self.unrec[i] } else { 1.0 };
        (unrec, unrec * self.duration(i))
    }
}

// ---------------- Per-branch log-likelihood (incremental building block) ----------------

/// Log-likelihood contribution of the branch above node `i` (0-based).
pub fn branch_loglik(model: &str, tab: &Tab, i: usize, mu: f64, sigma: f64) -> f64 {
    if i == tab.root_idx {
        return 0.0;
    }
    let (unrec, l) = tab.unrec_duration(i);
    let obs = unrec * tab.subs[i];
    match model {
        "arc" => {
            let size = mu * l / sigma;
            let prob = 1.0 - sigma / (1.0 + sigma);
            dnbinom(obs.round(), size, prob, true)
        }
        "carc" => {
            let shape = mu * l / (1.0 + sigma);
            let scale = 1.0 + sigma;
            if obs <= tab.minbralen {
                pgamma(tab.minbralen, shape, scale, true, true)
            } else {
                dgamma(obs, shape, scale, true)
            }
        }
        "poisson" => dpois(obs.round(), unrec * mu * tab.duration(i), true),
        "negbin" => {
            let k = mu * mu / sigma / sigma;
            let theta = sigma * sigma / mu;
            dnbinom(obs.round(), k, 1.0 / (1.0 + theta * l), true)
        }
        "strictgamma" => {
            let shape = unrec * mu * tab.duration(i);
            if obs <= tab.minbralen {
                pgamma(tab.minbralen, shape, 1.0, true, true)
            } else {
                dgamma(obs, shape, 1.0, true)
            }
        }
        "relaxedgamma" => {
            let ratevar = sigma * sigma;
            let shape = l * mu * mu / (mu + l * ratevar);
            let scale = 1.0 + l * ratevar / mu;
            if obs <= tab.minbralen {
                pgamma(tab.minbralen, shape, scale, true, true)
            } else {
                dgamma(obs, shape, scale, true)
            }
        }
        _ => 0.0,
    }
}

/// Full log-likelihood over all branches (reference semantics, used for
/// initialisation and for validating the incremental version).
pub fn full_loglik(model: &str, tab: &Tab, mu: f64, sigma: f64) -> f64 {
    let mut p = 0.0;
    for i in 0..tab.nrow {
        p += branch_loglik(model, tab, i, mu, sigma);
    }
    p
}

// ---------------- Coalescent prior ----------------

/// coalpriorC: coalescent prior on leaf and internal node dates.
/// Both slices must be sorted in DECREASING order.
pub fn coal_prior(leaves: &[f64], nodes: &[f64], alpha: f64) -> f64 {
    let n = leaves.len();
    if n < 2 {
        return 0.0;
    }
    let mut p = -alpha.ln() * (n as f64 - 1.0);
    let mut i1 = 1usize;
    let mut i2 = 0usize;
    let mut k = 1.0f64;
    let mut prev = leaves[0];
    for _ in 1..(n + n - 1) {
        if i1 < n && (i2 >= nodes.len() || leaves[i1] > nodes[i2]) {
            p -= k * (k - 1.0) / (2.0 * alpha) * (prev - leaves[i1]);
            prev = leaves[i1];
            i1 += 1;
            k += 1.0;
        } else if i2 < nodes.len() {
            p -= k * (k - 1.0) / (2.0 * alpha) * (prev - nodes[i2]);
            prev = nodes[i2];
            i2 += 1;
            k -= 1.0;
        } else {
            break;
        }
    }
    p
}

/// Sort descending.
pub fn sort_desc(v: &mut [f64]) {
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
}

/// changeinorderedvec: move the element equal to `old` to value `new`,
/// keeping the slice sorted in descending order.
pub fn change_in_ordered_vec(vec: &mut [f64], old: f64, new: f64) {
    let mut i = 0usize;
    while i < vec.len() && vec[i] != old {
        i += 1;
    }
    if i >= vec.len() {
        return;
    }
    vec[i] = new;
    loop {
        if i > 0 && vec[i - 1] < vec[i] {
            vec[i] = vec[i - 1];
            vec[i - 1] = new;
            i -= 1;
            continue;
        }
        if i < vec.len() - 1 && vec[i + 1] > vec[i] {
            vec[i] = vec[i + 1];
            vec[i + 1] = new;
            i += 1;
            continue;
        }
        break;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_in_ordered_vec() {
        let mut v = vec![5.0, 3.0, 1.0];
        change_in_ordered_vec(&mut v, 3.0, 4.0);
        assert_eq!(v, vec![5.0, 4.0, 1.0]);
        change_in_ordered_vec(&mut v, 4.0, 0.0);
        assert_eq!(v, vec![5.0, 1.0, 0.0]);
    }

    #[test]
    fn test_coal_prior_runs() {
        let leaves = vec![2010.0, 2005.0, 2000.0];
        let nodes = vec![1998.0, 1995.0];
        let p = coal_prior(&leaves, &nodes, 5.0);
        assert!(p.is_finite());
    }

    #[test]
    fn test_root_has_no_branch() {
        // 3 tips: ((1:1,2:2):3,3:4)  -> root has 0 contribution
        let subs = vec![1.0, 2.0, 4.0, 3.0, 0.0];
        let date = vec![2000.0, 2001.0, 2002.0, 1995.0, 1990.0];
        let father = vec![4, 4, 5, 5, 0];
        let tab = Tab::new(3, 2, subs, date, father, vec![1.0; 5], 0.1, 4);
        assert_eq!(branch_loglik("arc", &tab, 4, 1.0, 1.0), 0.0);
    }

    #[test]
    fn test_incremental_matches_full() {
        // Build a small 4-tip tree, then verify that the incremental update
        // (recompute only branches touching a node) equals the full recompute.
        let ntip = 4usize;
        let nnode = 3usize;
        let nrow = ntip + nnode;
        // tree: ((1:1,2:2):1,(3:1,4:1):1);
        // ids: tips 1..4, internal 5,6,7 (root=5)
        let subs = vec![1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let date = vec![2010.0, 2009.0, 2008.0, 2007.0, 2005.0, 2006.0, 2000.0];
        let father = vec![5, 5, 6, 6, 7, 7, 0];
        let unrec = vec![1.0; nrow];
        let root_idx = 6usize; // node 7
        let mut tab = Tab::new(ntip, nnode, subs, date, father, unrec, 0.1, root_idx);

        let mu = 0.5;
        let sigma = 0.3;
        let full_before = full_loglik("arc", &tab, mu, sigma);

        // change date of internal node 5 (index 4) — affects branch above 5 and above 1,2
        let node = 5usize;
        let old = tab.date[node - 1];
        tab.date[node - 1] = old + 0.7;

        // full recompute after
        let full_after = full_loglik("arc", &tab, mu, sigma);

        // incremental: only branches above node 5 and above its children (1,2)
        tab.date[node - 1] = old;
        let local_before = branch_loglik("arc", &tab, node - 1, mu, sigma)
            + branch_loglik("arc", &tab, 0, mu, sigma)
            + branch_loglik("arc", &tab, 1, mu, sigma);
        tab.date[node - 1] = old + 0.7;
        let local_after = branch_loglik("arc", &tab, node - 1, mu, sigma)
            + branch_loglik("arc", &tab, 0, mu, sigma)
            + branch_loglik("arc", &tab, 1, mu, sigma);

        let incremental = full_before - local_before + local_after;
        assert!(
            (incremental - full_after).abs() < 1e-10,
            "incremental={} full={}",
            incremental,
            full_after
        );
    }
    #[test]
    fn test_coal_prior_matches_r() {
        // R: coalpriorC(leaves=sort(dates), nodes=seq(1993.6652,2009,length.out=198) desc, alpha=4)
        //    = -5990.607366
        let mut leaves = vec![2000.0; 200];
        for (i, v) in leaves.iter_mut().enumerate() {
            *v = 2000.0 + i as f64 * 10.0 / 199.0;
        }
        sort_desc(&mut leaves);
        let mut nodes: Vec<f64> = (0..198)
            .map(|i| 1993.6652 + (2009.0 - 1993.6652) * i as f64 / 197.0)
            .collect();
        sort_desc(&mut nodes);
        let p = coal_prior(&leaves, &nodes, 4.0);
        eprintln!("coal_prior(198 nodes) = {:.6}", p);

        // KNOWN DIFFERENCE: R's coalpriorC reads nodes[n-1] out of bounds (its array
        // has n-1 elements but the loop needs n), producing -5990.607366 from garbage
        // memory. Our bounds-checked version returns the mathematically correct value.
        // Proof: give R an extra node and it agrees with us exactly.
        //   R: coalpriorC(leaves, c(nodes, 1993.6652), 4) == -5492.191066
        assert!((p - (-5492.191066)).abs() < 1e-3, "got {}", p);

        // with an explicit (n)th node appended, both implementations agree:
        let mut nodes_full = nodes.clone();
        nodes_full.push(1993.6652);
        sort_desc(&mut nodes_full);
        let p_full = coal_prior(&leaves, &nodes_full, 4.0);
        assert!((p_full - (-5492.191066)).abs() < 1e-3, "got {}", p_full);
    }

}