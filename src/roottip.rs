//! Root-to-tip regression and optimal rooting (port of BactDating's initRoot / roottip).

use crate::tree::Tree;

/// Simple root-to-tip rate estimate: linear regression of root-to-tip distance on date,
/// using the current rooting of the tree.
pub fn root_to_tip_rate(tree: &Tree, dates: &[f64]) -> f64 {
    let depths = tree.depths();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for tip in 1..=tree.ntip {
        let d = dates[tip - 1];
        if d.is_finite() {
            xs.push(d);
            ys.push(depths[tip]);
        }
    }
    if xs.len() < 3 {
        return 1.0;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..xs.len() {
        sxy += (xs[i] - mx) * (ys[i] - my);
        sxx += (xs[i] - mx) * (xs[i] - mx);
    }
    if sxx == 0.0 {
        return 1.0;
    }
    let slope = sxy / sxx;
    if slope > 0.0 && slope.is_finite() {
        slope
    } else {
        1.0
    }
}

/// Correlation between root-to-tip distance and date.
pub fn root_to_tip_cor(tree: &Tree, dates: &[f64]) -> f64 {
    let depths = tree.depths();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for tip in 1..=tree.ntip {
        let d = dates[tip - 1];
        if d.is_finite() {
            xs.push(d);
            ys.push(depths[tip]);
        }
    }
    let n = xs.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..xs.len() {
        sxy += (xs[i] - mx) * (ys[i] - my);
        sxx += (xs[i] - mx) * (xs[i] - mx);
        syy += (ys[i] - my) * (ys[i] - my);
    }
    if sxx == 0.0 || syy == 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Leaf root-to-tip distances (vector indexed by tip node id).
pub fn leaf_dates(tree: &Tree) -> Vec<f64> {
    let depths = tree.depths();
    (1..=tree.ntip).map(|t| depths[t]).collect()
}
