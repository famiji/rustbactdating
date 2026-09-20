//! MCMC for BactDating (arc model by default), with incremental likelihood
//! updates and checkpoint/resume support.

use crate::dist::*;
use crate::likelihood::*;
use crate::roottip;
use crate::tree::Tree;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub nb_its: usize,
    pub thin: usize,
    pub min_bralen: f64,
    pub update_root: bool,
    pub update_mu: bool,
    pub update_alpha: bool,
    pub update_sigma: bool,
    pub model: String,
    pub use_coal_prior: bool,
    pub acceptance_target: f64,
    pub tune: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            nb_its: 10_000,
            thin: 10,
            min_bralen: 0.1,
            update_root: true,
            update_mu: true,
            update_alpha: true,
            update_sigma: true,
            model: "arc".to_string(),
            use_coal_prior: true,
            acceptance_target: 0.234,
            tune: true,
        }
    }
}

/// Full MCMC state, sufficient to resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub date: Vec<f64>,
    pub father: Vec<usize>,
    pub subs: Vec<f64>,
    pub unrec: Vec<f64>,
    pub mu: f64,
    pub sigma: f64,
    pub alpha: f64,
    pub loglik: f64,
    pub logprior: f64,
    pub sd_mu: f64,
    pub sd_sigma: f64,
    pub sd_dates: f64,
    pub n_done: usize,
    pub record: Vec<Vec<f64>>,
    pub record_cols: Vec<String>,
    /// indices (0-based tip rows) of tips with missing/interval dates
    pub mis_dates: Vec<usize>,
    /// date range lower/upper bound for each tip (0-based)
    pub range_lo: Vec<f64>,
    pub range_hi: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub cfg: Config,
    pub state: State,
    pub rng_state: Vec<u8>,
    pub ntip: usize,
    pub nnode: usize,
    pub root_idx: usize,
    pub tip_labels: Vec<String>,
}

pub struct Mcmc<'a> {
    pub tree: &'a Tree,
    pub dates: Vec<f64>,
    pub cfg: Config,
    pub rng: ChaCha8Rng,
    /// children[node] = direct children (node ids)
    children: Vec<Vec<usize>>,
    /// tips with missing (interval) dates, 0-based row indices
    mis_dates: Vec<usize>,
    /// per-tip date bounds (for missing dates these are [min_date, max_date])
    range_lo: Vec<f64>,
    range_hi: Vec<f64>,
}

impl<'a> Mcmc<'a> {
    pub fn new(tree: &'a Tree, dates: Vec<f64>, cfg: Config, seed: u64) -> Self {
        let nrow = tree.ntip + tree.nnode;
        let mut children = vec![Vec::new(); nrow + 1];
        for node in 1..=nrow {
            let p = tree.parent[node];
            if p > 0 {
                children[p].push(node);
            }
        }
        // Missing dates get the full observed range as their interval, and are
        // initialised to its midpoint (mirrors BactDating's rangedate handling).
        let observed: Vec<f64> = dates.iter().cloned().filter(|d| d.is_finite()).collect();
        let (dmin, dmax) = if observed.is_empty() {
            (0.0, 1.0)
        } else {
            (
                observed.iter().cloned().fold(f64::INFINITY, f64::min),
                observed.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let n = tree.ntip;
        let mut range_lo = vec![dmin; n];
        let mut range_hi = vec![dmax; n];
        let mut mis_dates = Vec::new();
        let mut fixed_dates = dates.clone();
        for i in 0..n {
            if !dates[i].is_finite() {
                mis_dates.push(i);
                range_lo[i] = dmin;
                range_hi[i] = dmax;
                fixed_dates[i] = 0.5 * (dmin + dmax);
            } else {
                range_lo[i] = dates[i];
                range_hi[i] = dates[i];
            }
        }
        Mcmc {
            tree,
            dates: fixed_dates,
            cfg,
            rng: ChaCha8Rng::seed_from_u64(seed),
            children,
            mis_dates,
            range_lo,
            range_hi,
        }
    }

    #[inline]
    fn logprior(&self, leaves_desc: &[f64], nodes_desc: &[f64], alpha: f64) -> f64 {
        if self.cfg.use_coal_prior {
            coal_prior(leaves_desc, nodes_desc, alpha)
        } else {
            0.0
        }
    }

    #[inline]
    fn loglik(&self, tab: &Tab, mu: f64, sigma: f64) -> f64 {
        full_loglik(&self.cfg.model, tab, mu, sigma)
    }

    /// Local log-likelihood for the branch above `node` and above its children.
    #[inline]
    fn local_loglik(&self, tab: &Tab, node: usize, mu: f64, sigma: f64) -> f64 {
        let i = node - 1;
        let mut acc = branch_loglik(&self.cfg.model, tab, i, mu, sigma);
        for &c in &self.children[node] {
            acc += branch_loglik(&self.cfg.model, tab, c - 1, mu, sigma);
        }
        acc
    }

    /// Sample alpha via the same inverse-gamma Gibbs move as R's bactdate.
    ///
    /// R does: s = sort(tab[,3], decreasing=T, index.return=T); k = cumsum(2*(s$ix<=n)-1);
    /// i.e. it sorts the FULL date column (all nodes incl. root) and marks leaves by the
    /// ORIGINAL row index (<= ntip). We replicate that exactly to avoid tie-handling drift.
    fn sample_alpha(&mut self, tab: &Tab, n: usize) -> f64 {
        let nrow = tab.nrow;
        // (date, original_row_index) sorted by date descending
        let mut idx: Vec<usize> = (0..nrow).collect();
        idx.sort_by(|&a, &b| {
            tab.date[b]
                .partial_cmp(&tab.date[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut su = 0.0f64;
        let mut k = 0.0f64;
        for i in 0..nrow.saturating_sub(1) {
            // k = cumsum(2*(is_leaf) - 1), is_leaf = (original row index + 1 <= n)
            k += if idx[i] + 1 <= n { 1.0 } else { -1.0 };
            let dif = tab.date[idx[i]] - tab.date[idx[i + 1]];
            su += k * (k - 1.0) * dif;
        }
        let shape = n as f64 + 0.001 - 1.0;
        let scale = 2000.0 / (su * 1000.0 + 2.0);
        if std::env::var("RUSTBACT_DEBUG").is_ok() {
            eprintln!("[alpha] nrow={} su={:.6} shape={:.4} scale={:.6}", nrow, su, shape, scale);
        }
        let g = rgamma(&mut self.rng, shape, scale);
        if g > 0.0 {
            1.0 / g
        } else {
            1e6
        }
    }

    /// Build the initial table from the tree and dates.
    fn init_tab(&self) -> (Tab, f64, f64) {
        let n = self.tree.ntip;
        let nnode = self.tree.nnode;
        let nrow = n + nnode;
        let root_idx = self.tree.root - 1;

        let mut subs = vec![0.0f64; nrow];
        let mut date = vec![0.0f64; nrow];
        let mut father = vec![0usize; nrow];
        let unrec = vec![1.0f64; nrow];
        for node in 1..=nrow {
            let i = node - 1;
            subs[i] = self.tree.edge_length[node];
            father[i] = self.tree.parent[node];
            if node <= n {
                date[i] = self.dates[node - 1];
            }
        }

        let mu = {
            let r = roottip::root_to_tip_rate(self.tree, &self.dates);
            if r.is_finite() && r > 0.0 {
                r
            } else {
                1.0
            }
        };

        let mut tab = Tab::new(n, nnode, subs, date, father, unrec, self.cfg.min_bralen, root_idx);

        // internal node dates in postorder (children before parents)
        for &node in &self.tree.postorder() {
            if node <= n {
                continue;
            }
            let i = node - 1;
            let ch = &self.children[node];
            if ch.is_empty() {
                continue;
            }
            // R: tab[i,3] = min(min(children_dates) - 0.01,
            //                   mean(children_dates - children_subs/mu))
            let mut min_date = f64::INFINITY;
            let mut sum_adj = 0.0;
            for &c in ch {
                let ci = c - 1;
                min_date = min_date.min(tab.date[ci]);
                sum_adj += tab.date[ci] - tab.subs[ci] / mu;
            }
            let mean_adj = sum_adj / ch.len() as f64;
            tab.date[i] = (min_date - 0.01).min(mean_adj);
        }
        // root
        {
            let ch = &self.children[self.tree.root];
            let mut m1 = f64::INFINITY;
            for &c in ch {
                let ci = c - 1;
                m1 = m1.min(tab.date[ci] - tab.subs[ci] / mu);
            }
            tab.date[root_idx] = m1;
        }

        (tab, mu, mu) // sigma initialised to mu, as in R
    }

    pub fn run(&mut self, resume: Option<Checkpoint>) -> Checkpoint {
        let n = self.tree.ntip;
        let nnode = self.tree.nnode;
        let nrow = n + nnode;
        let root_idx = self.tree.root - 1;

        let mut leaves_desc: Vec<f64> = self.dates.clone();
        sort_desc(&mut leaves_desc);

        let mut st: State = match resume {
            Some(cp) => {
                self.rng = ChaCha8Rng::from_seed({
                    let mut b = [0u8; 32];
                    for (i, v) in cp.rng_state.iter().take(32).enumerate() {
                        b[i] = *v;
                    }
                    b
                });
                cp.state
            }
            None => {
                let (tab, mu, sigma) = self.init_tab();
                let mut nodes_desc: Vec<f64> = (n..nrow).map(|i| tab.date[i]).collect();
                sort_desc(&mut nodes_desc);
                let alpha = self.sample_alpha(&tab, n);
                let loglik = self.loglik(&tab, mu, sigma);
                let logprior = self.logprior(&leaves_desc, &nodes_desc, alpha);
                eprintln!(
                    "[init] mu={:.6} sigma={:.6} alpha={:.6} loglik={:.6} logprior={:.6}",
                    mu, sigma, alpha, loglik, logprior
                );
                eprintln!(
                    "[init] root_date={:.4} dmin={:.4} dmax={:.4}",
                    tab.date[root_idx],
                    tab.date.iter().cloned().fold(f64::INFINITY, f64::min),
                    tab.date.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                );
                let init_height = {
                    let mx = tab.date.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mn = tab.date.iter().cloned().fold(f64::INFINITY, f64::min);
                    mx - mn
                };
                State {
                    date: tab.date.clone(),
                    father: tab.father.clone(),
                    subs: tab.subs.clone(),
                    unrec: tab.unrec.clone(),
                    mu,
                    sigma,
                    alpha,
                    loglik,
                    logprior,
                    sd_mu: 0.1 * mu,
                    sd_sigma: 0.1 * sigma,
                    sd_dates: 0.05 * if init_height > 0.0 { init_height } else { 1.0 },
                    n_done: 0,
                    record: Vec::new(),
                    record_cols: Vec::new(),
                    mis_dates: self.mis_dates.clone(),
                    range_lo: self.range_lo.clone(),
                    range_hi: self.range_hi.clone(),
                }
            }
        };

        // working table
        let mut tab = Tab::new(
            n,
            nnode,
            st.subs.clone(),
            st.date.clone(),
            st.father.clone(),
            st.unrec.clone(),
            self.cfg.min_bralen,
            root_idx,
        );

        let mut mu = st.mu;
        let mut sigma = st.sigma;
        let mut alpha = st.alpha;
        let mut loglik = st.loglik;
        let mut logprior = st.logprior;

        let mut nodes_desc: Vec<f64> = (n..nrow).map(|i| tab.date[i]).collect();
        sort_desc(&mut nodes_desc);

        let thin = self.cfg.thin.max(1);
        let delta = 1.0 / self.cfg.acceptance_target / (1.0 - self.cfg.acceptance_target);

        let mut record = st.record.clone();
        if st.record_cols.is_empty() {
            let mut cols: Vec<String> = (1..=nrow).map(|i| format!("date{}", i)).collect();
            cols.extend((1..=nrow).map(|i| format!("father{}", i)));
            cols.extend((1..=nrow).map(|i| format!("subs{}", i)));
            cols.extend(
                ["likelihood", "mu", "sigma", "alpha", "prior", "root"]
                    .iter()
                    .map(|s| s.to_string()),
            );
            st.record_cols = cols;
        }

        // internal nodes (0-based rows), excluding root handled by root moves
        let internal_nodes: Vec<usize> = (n + 1..=nrow).filter(|&node| node - 1 != root_idx).collect();

        let start = st.n_done;
        for it in (start + 1)..=(self.cfg.nb_its) {
            if it % thin == 0 {
                let mut row: Vec<f64> = Vec::with_capacity(nrow * 3 + 6);
                row.extend_from_slice(&tab.date);
                row.extend(tab.father.iter().map(|&x| x as f64));
                row.extend_from_slice(&tab.subs);
                row.push(loglik);
                row.push(mu);
                row.push(sigma);
                row.push(alpha);
                row.push(logprior);
                row.push(0.0); // root edge marker (kept for column compatibility)
                record.push(row);
            }

            // ---- mu ----
            if self.cfg.update_mu {
                let mu2 = rnorm(&mut self.rng, mu, st.sd_mu).abs();
                if mu2 > 0.0 {
                    let l2 = self.loglik(&tab, mu2, sigma);
                    let mh = l2 - loglik + dgamma(mu2, 1e-3, 1e3, true) - dgamma(mu, 1e-3, 1e3, true);
                    if runif(&mut self.rng).ln() < mh {
                        loglik = l2;
                        mu = mu2;
                    }
                    if self.cfg.tune {
                        st.sd_mu *= (delta / (it as f64 + 1.0)
                            * (mh.exp().min(1.0) - self.cfg.acceptance_target))
                            .exp();
                    }
                }
            }

            // ---- sigma ----
            if self.cfg.update_sigma && sigma > 0.0 {
                let sigma2 = rnorm(&mut self.rng, sigma, st.sd_sigma).abs();
                if sigma2 > 0.0 {
                    let l2 = self.loglik(&tab, mu, sigma2);
                    let mh = l2 - loglik + dgamma(sigma2, 1e-3, 1e3, true)
                        - dgamma(sigma, 1e-3, 1e3, true);
                    if runif(&mut self.rng).ln() < mh {
                        loglik = l2;
                        sigma = sigma2;
                    }
                    if self.cfg.tune {
                        st.sd_sigma *= (delta / (it as f64 + 1.0)
                            * (mh.exp().min(1.0) - self.cfg.acceptance_target))
                            .exp();
                    }
                }
            }

            // ---- alpha (Gibbs) ----
            if self.cfg.update_alpha {
                alpha = self.sample_alpha(&tab, n);
                logprior = self.logprior(&leaves_desc, &nodes_desc, alpha);
            }

            // ---- internal dates (incremental likelihood) ----
            for &node in &internal_nodes {
                let i = node - 1;
                let old = tab.date[i];
                let new = old + rnorm(&mut self.rng, 0.0, st.sd_dates);
                let f = tab.father[i];
                let father_date = if f > 0 { tab.date[f - 1] } else { f64::INFINITY };
                let min_child = {
                    let ch = &self.children[node];
                    if ch.is_empty() {
                        f64::NEG_INFINITY
                    } else {
                        ch.iter().map(|&c| tab.date[c - 1]).fold(f64::INFINITY, f64::min)
                    }
                };
                // R semantics: an invalid move sets mh = -Inf but STILL updates the
                // tuning factor (min(1, exp(-Inf)) = 0). Skipping the tuning update
                // would make sd_dates drift differently from the reference.
                let mut mh = f64::NEG_INFINITY;
                if !(new > father_date || new < min_child) {
                    let l_before = self.local_loglik(&tab, node, mu, sigma);
                    tab.date[i] = new;
                    let l_after = self.local_loglik(&tab, node, mu, sigma);
                    let l2 = loglik - l_before + l_after;
                    change_in_ordered_vec(&mut nodes_desc, old, new);
                    let p2 = self.logprior(&leaves_desc, &nodes_desc, alpha);
                    mh = l2 - loglik + p2 - logprior;
                    if runif(&mut self.rng).ln() < mh {
                        loglik = l2;
                        logprior = p2;
                    } else {
                        tab.date[i] = old;
                        change_in_ordered_vec(&mut nodes_desc, new, old);
                    }
                }
                if self.cfg.tune {
                    let ii = (it - 1) as f64 * (nrow - n) as f64 + (node - n) as f64;
                    st.sd_dates *= (delta / (ii + 1.0)
                        * (mh.exp().min(1.0) - self.cfg.acceptance_target))
                        .exp();
                }
            }

            // ---- update missing (interval) leaf dates ----
            for &mi in &self.mis_dates.clone() {
                let i = mi; // 0-based tip row
                let old = tab.date[i];
                let width = self.range_hi[mi] - self.range_lo[mi];
                let new = old + rnorm(&mut self.rng, 0.0, width * 0.05);
                if new > self.range_hi[mi] || new < self.range_lo[mi] {
                    continue;
                }
                let f = tab.father[i];
                if f > 0 && new < tab.date[f - 1] {
                    continue;
                }
                // likelihood of the branch above this tip only
                let l_before = branch_loglik(&self.cfg.model, &tab, i, mu, sigma);
                tab.date[i] = new;
                let l_after = branch_loglik(&self.cfg.model, &tab, i, mu, sigma);
                let l2 = loglik - l_before + l_after;
                change_in_ordered_vec(&mut leaves_desc, old, new);
                let p2 = self.logprior(&leaves_desc, &nodes_desc, alpha);
                if runif(&mut self.rng).ln() < l2 - loglik + p2 - logprior {
                    loglik = l2;
                    logprior = p2;
                } else {
                    tab.date[i] = old;
                    change_in_ordered_vec(&mut leaves_desc, new, old);
                }
            }
        }

        st.mu = mu;
        st.sigma = sigma;
        st.alpha = alpha;
        st.loglik = loglik;
        st.logprior = logprior;
        st.date = tab.date.clone();
        st.subs = tab.subs.clone();
        st.father = tab.father.clone();
        st.unrec = tab.unrec.clone();
        st.n_done = self.cfg.nb_its;
        st.record = record;

        Checkpoint {
            cfg: self.cfg.clone(),
            state: st,
            rng_state: bincode::serialize(&self.rng).unwrap_or_default(),
            ntip: n,
            nnode,
            root_idx,
            tip_labels: self.tree.tip_labels.clone(),
        }
    }
}
