//! Loaders for Gubbins and ClonalFrameML output, mirroring BactDating's
//! `loadGubbins()` and `loadCFML()`.
//!
//! Both attach a per-branch `unrec` value — the fraction of the branch that is
//! *not* attributed to recombination — which the likelihood uses when
//! `useRec` is on, and which also matters for the dated tree because
//! recombination-inflated branches would otherwise be read as extra time.

use crate::tree::{parse_newick, Tree};

#[derive(Debug)]
pub struct LoadedTree {
    pub tree: Tree,
    /// 1-based node id -> unrec for the branch above it
    pub unrec: Vec<f64>,
}

impl LoadedTree {
    /// Attach the unrec values to a tree in the layout the likelihood expects
    /// (row i of the table corresponds to node id i+1).
    pub fn unrec_for_node(&self, node: usize) -> f64 {
        self.unrec.get(node).copied().unwrap_or(1.0)
    }
}

fn read_to_string(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))
}

/// Map a node name (tip label or internal node label) to a 1-based node id.
fn name_to_id(tree: &Tree, name: &str) -> Option<usize> {
    tree.tip_index(name)
}

/// Load Gubbins output given the run prefix, e.g. `gubbins_run` for files
/// `gubbins_run.final_tree.tre`, `gubbins_run.node_labelled.final_tree.tre`,
/// `gubbins_run.per_branch_statistics.csv`.
///
/// Mirrors `loadGubbins()`: `unrec = 1 - (bases in recombinations) / genome_length`,
/// with the column indices depending on the Gubbins version (11 vs 13 columns).
pub fn load_gubbins(prefix: &str) -> Result<LoadedTree, String> {
    let tree_path = format!("{}.final_tree.tre", prefix);
    let tree = parse_newick(&read_to_string(&tree_path)?)
        .map_err(|e| format!("{}: {}", tree_path, e))?;

    // node labels: either <prefix>.node_labelled.tre or .node_labelled.final_tree.tre
    let mut internal_labels: Vec<String> = Vec::new();
    for cand in [
        format!("{}.node_labelled.tre", prefix),
        format!("{}.node_labelled.final_tree.tre", prefix),
    ] {
        if let Ok(txt) = read_to_string(&cand) {
            if let Ok(t) = parse_newick(&txt) {
                // the labelled tree carries the names as internal labels
                internal_labels = extract_internal_labels(&txt, t.ntip);
                break;
            }
        }
    }

    // per-branch statistics
    let stats_path = format!("{}.per_branch_statistics.csv", prefix);
    let stats = read_to_string(&stats_path)?;
    let mut lines = stats.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    let ncol = header.len();
    let col = |name: &str| header.iter().position(|h| h.trim() == name);
    let i_node = col("Node").ok_or("per_branch_statistics.csv has no Node column")?;

    // BactDating: ncol==13 -> bases=col 5 (0-based), genome=col 11;
    //            ncol==11 -> bases=col 5 (0-based), genome=col 9
    let (i_bases, i_genome) = match ncol {
        13 => (5usize, 11usize),
        11 => (5usize, 9usize),
        _ => return Err(format!("unsupported per_branch_statistics.csv with {} columns", ncol)),
    };

    let mut unrec: Vec<f64> = vec![1.0; tree.total_nodes() + 1];
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() <= i_genome {
            continue;
        }
        let name = f[i_node].trim();
        let bases: f64 = match f[i_bases].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let genome: f64 = match f[i_genome].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if genome <= 0.0 {
            continue;
        }
        // Gubbins' own convention; nodes absent from the table keep 1.0
        let u = 1.0 - bases / genome;
        if let Some(id) = name_to_id(&tree, name) {
            unrec[id] = u;
        } else if let Some(k) = internal_index(&internal_labels, name) {
            unrec[tree.ntip + 1 + k] = u;
        }
    }

    Ok(LoadedTree { tree, unrec })
}

/// Pull internal node labels out of a labelled newick's text.
///
/// Internal labels sit between a closing paren and the next delimiter, e.g.
/// `)Node3:0.1`. Labels are collected in the order they appear, which matches
/// the order nodes are created when parsing (and `parse_newick` assigns internal
/// ids in that same order).
fn extract_internal_labels(nwk: &str, ntip: usize) -> Vec<String> {
    let chars: Vec<char> = nwk.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == ')' {
            let start = i + 1;
            let mut j = start;
            while j < chars.len() {
                let c = chars[j];
                if c == ':' || c == ',' || c == ')' || c == ';' {
                    break;
                }
                j += 1;
            }
            let label: String = chars[start..j].iter().collect();
            let label = label.trim().trim_matches(|c| c == '\'' || c == '"').to_string();
            if !label.is_empty() {
                out.push(label);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    let _ = ntip;
    out
}

fn internal_index(labels: &[String], name: &str) -> Option<usize> {
    labels.iter().position(|l| l == name)
}

/// Load ClonalFrameML output given the run prefix, e.g. `run` for
/// `run.labelled_tree.newick`, `run.importation_status.txt`,
/// `run.em.txt`, `run.position_cross_reference.txt`.
///
/// Mirrors `loadCFML()`: branch lengths are scaled by the number of sites
/// (per-site -> substitutions), and `unrec = 1 - (imported bases)/L`.
pub fn load_clonalframeml(prefix: &str) -> Result<LoadedTree, String> {
    let tree_path = format!("{}.labelled_tree.newick", prefix);
    let mut tree = parse_newick(&read_to_string(&tree_path)?)
        .map_err(|e| format!("{}: {}", tree_path, e))?;

    // L = number of sites, from the comma-separated position file
    let pcr_path = format!("{}.position_cross_reference.txt", prefix);
    let pcr = read_to_string(&pcr_path)?;
    let l_sites = pcr.trim().split(',').count() as f64;
    if l_sites <= 0.0 {
        return Err(format!("{}: could not read a site count", pcr_path));
    }
    tree.scale_lengths(l_sites);

    // imported bases per node
    let imp_path = format!("{}.importation_status.txt", prefix);
    let imp = read_to_string(&imp_path)?;
    let mut imported: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (i, line) in imp.lines().enumerate() {
        if i == 0 {
            continue; // header: Node, Beg, End
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 3 {
            continue;
        }
        let node = f[0].trim().to_string();
        let beg: f64 = match f[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let end: f64 = match f[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        *imported.entry(node).or_insert(0.0) += end - beg + 1.0;
    }

    let total = tree.total_nodes();
    let mut unrec = vec![1.0f64; total + 1];
    for node in 1..=total {
        let name = if node <= tree.ntip {
            tree.tip_labels[node - 1].clone()
        } else {
            format!("Node{}", node - tree.ntip)
        };
        let imp_bp = imported.get(&name).copied().unwrap_or(0.0);
        let u = (l_sites - imp_bp) / l_sites;
        unrec[node] = u.clamp(0.0, 1.0);
    }

    Ok(LoadedTree { tree, unrec })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_internal_labels() {
        let nwk = "((A:1,B:2)Node1:3,(C:4,D:5)Node2:6)Node3;";
        let labels = extract_internal_labels(nwk, 4);
        assert_eq!(labels, vec!["Node1", "Node2", "Node3"]);
    }

    #[test]
    fn test_extract_internal_labels_unnamed() {
        let nwk = "((A:1,B:2):3,C:4);";
        let labels = extract_internal_labels(nwk, 3);
        assert!(labels.is_empty(), "got {:?}", labels);
    }
}
