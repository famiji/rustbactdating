//! Phylogenetic tree: Newick parsing, node indexing, postorder traversal.
//!
//! Node numbering follows the R/ape convention used by BactDating:
//!   1..=ntip  = tips (leaves),  ntip+1..=ntip+nnode = internal nodes
//! The root is conventionally node ntip+1.



#[derive(Debug, Clone)]
pub struct Tree {
    pub ntip: usize,
    pub nnode: usize,
    pub tip_labels: Vec<String>,
    /// parent[child] = parent node id (index 1..=ntip+nnode; index 0 unused)
    pub parent: Vec<usize>,
    /// edge_length[child] = branch length above that node
    pub edge_length: Vec<f64>,
    /// children[parent] = list of child node ids
    pub children: Vec<Vec<usize>>,
    pub root: usize,
}

impl Tree {
    pub fn total_nodes(&self) -> usize {
        self.ntip + self.nnode
    }

    /// Build from edges (parent, child, length).
    pub fn from_edges(tip_labels: Vec<String>, edges: Vec<(usize, usize, f64)>, root: usize) -> Self {
        let ntip = tip_labels.len();
        let max_id = edges
            .iter()
            .map(|(p, c, _)| (*p).max(*c))
            .max()
            .unwrap_or(ntip + 1)
            .max(root);
        let nnode = max_id - ntip;
        let total = ntip + nnode;
        let mut parent = vec![0usize; total + 1];
        let mut edge_length = vec![0.0f64; total + 1];
        let mut children = vec![Vec::new(); total + 1];
        for (p, c, l) in edges {
            parent[c] = p;
            edge_length[c] = l;
            children[p].push(c);
        }
        Tree { ntip, nnode, tip_labels, parent, edge_length, children, root }
    }

    /// Postorder node list (children before parents), starting from root.
    pub fn postorder(&self) -> Vec<usize> {
        let mut stack = vec![self.root];
        let mut tmp = Vec::with_capacity(self.total_nodes());
        while let Some(u) = stack.pop() {
            tmp.push(u);
            for &c in &self.children[u] {
                stack.push(c);
            }
        }
        tmp.reverse();
        tmp
    }

    /// Preorder (root first).
    pub fn preorder(&self) -> Vec<usize> {
        let mut stack = vec![self.root];
        let mut out = Vec::with_capacity(self.total_nodes());
        while let Some(u) = stack.pop() {
            out.push(u);
            for &c in &self.children[u] {
                stack.push(c);
            }
        }
        out
    }

    pub fn is_tip(&self, node: usize) -> bool {
        node <= self.ntip
    }

    /// Sum of all branch lengths (total substitutions in the tree).
    pub fn total_length(&self) -> f64 {
        self.edge_length.iter().sum()
    }

    /// Index of a tip by label (1-based node id returned).
    pub fn tip_index(&self, label: &str) -> Option<usize> {
        self.tip_labels.iter().position(|l| l == label).map(|i| i + 1)
    }

    /// Total number of edges.
    pub fn n_edges(&self) -> usize {
        self.total_nodes() - 1
    }

    /// Multiply every branch length by `factor`.
    ///
    /// Use this to convert a tree whose lengths are in substitutions per site
    /// into one in absolute substitutions, by passing the number of sites in
    /// the alignment the tree was built from.
    pub fn scale_lengths(&mut self, factor: f64) {
        for l in self.edge_length.iter_mut() {
            *l *= factor;
        }
    }

    /// Depth (root-to-node distance in branch length units), computed from root.
    pub fn depths(&self) -> Vec<f64> {
        let mut depth = vec![0.0f64; self.total_nodes() + 1];
        for &u in &self.preorder() {
            for &c in &self.children[u] {
                depth[c] = depth[u] + self.edge_length[c];
            }
        }
        depth
    }
}

// ---------------- Newick parsing ----------------

struct TmpNode {
    label: Option<String>,
    len: f64,
    children: Vec<usize>,
}

pub struct NewickParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> NewickParser<'a> {
    pub fn new(s: &'a str) -> Self {
        NewickParser { bytes: s.as_bytes(), pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_subtree(&mut self, nodes: &mut Vec<TmpNode>) -> Result<usize, String> {
        self.skip_ws();
        let mut children: Vec<usize> = Vec::new();
        if self.peek() == Some(b'(') {
            self.pos += 1;
            loop {
                let c = self.parse_subtree(nodes)?;
                children.push(c);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.pos += 1;
                    }
                    Some(b')') => {
                        self.pos += 1;
                        break;
                    }
                    other => {
                        return Err(format!("newick: expected ',' or ')' at byte {}, got {:?}", self.pos, other));
                    }
                }
            }
        }
        // label: read until one of the delimiters
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b':' || b == b',' || b == b')' || b == b';' || b == b'(' || b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                break;
            }
            self.pos += 1;
        }
        let label_raw = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
        // strip surrounding quotes if present
        let label_clean = label_raw.trim_matches(|c| c == '\'' || c == '"').to_string();
        // branch length
        let mut len = 0.0f64;
        self.skip_ws();
        if self.peek() == Some(b':') {
            self.pos += 1;
            let s2 = self.pos;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'+' || b == b'e' || b == b'E' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            let txt = std::str::from_utf8(&self.bytes[s2..self.pos]).map_err(|e| e.to_string())?;
            if !txt.is_empty() {
                len = txt.parse::<f64>().map_err(|e| format!("newick length '{}': {}", txt, e))?;
            }
        }
        let id = nodes.len();
        nodes.push(TmpNode {
            label: if label_clean.is_empty() { None } else { Some(label_clean) },
            len,
            children,
        });
        Ok(id)
    }

    /// Parse into (tip_labels, edges, root_final_id).
    pub fn parse(&mut self) -> Result<(Vec<String>, Vec<(usize, usize, f64)>, usize), String> {
        let mut nodes: Vec<TmpNode> = Vec::new();
        let root_tmp = self.parse_subtree(&mut nodes)?;
        self.skip_ws();
        if self.peek() == Some(b';') {
            self.pos += 1;
        }

        // Assign final ids: tips get 1..=ntip in order of first appearance, then internals.
        let mut final_id = vec![0usize; nodes.len()];
        let mut tip_labels: Vec<String> = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            if n.children.is_empty() {
                let lbl = n.label.clone().unwrap_or_else(|| format!("tip{}", tip_labels.len() + 1));
                tip_labels.push(lbl);
                final_id[i] = tip_labels.len();
            }
        }
        let ntip = tip_labels.len();
        let mut int_count = 0usize;
        for (i, n) in nodes.iter().enumerate() {
            if !n.children.is_empty() {
                int_count += 1;
                final_id[i] = ntip + int_count;
            }
        }

        // parent map
        let mut parent_of: Vec<Option<usize>> = vec![None; nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            for &c in &n.children {
                parent_of[c] = Some(i);
            }
        }
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            if i == root_tmp {
                continue;
            }
            if let Some(p) = parent_of[i] {
                edges.push((final_id[p], final_id[i], n.len));
            }
        }
        Ok((tip_labels, edges, final_id[root_tmp]))
    }
}

pub fn parse_newick(s: &str) -> Result<Tree, String> {
    let (labels, edges, root) = NewickParser::new(s).parse()?;
    Ok(Tree::from_edges(labels, edges, root))
}

/// Serialize tree back to Newick (branch lengths only, no node labels).
pub fn write_newick(tree: &Tree) -> String {
    fn rec(tree: &Tree, node: usize, out: &mut String) {
        if !tree.children[node].is_empty() {
            out.push('(');
            for (i, &c) in tree.children[node].iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                rec(tree, c, out);
            }
            out.push(')');
        } else {
            out.push_str(&tree.tip_labels[node - 1]);
        }
        if node != tree.root {
            out.push(':');
            out.push_str(&format!("{}", tree.edge_length[node]));
        }
    }
    let mut s = String::new();
    rec(tree, tree.root, &mut s);
    s.push(';');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_simple() {
        let t = parse_newick("((A:1,B:2):3,C:4);").unwrap();
        assert_eq!(t.ntip, 3);
        assert!((t.total_length() - 10.0).abs() < 1e-9);
        assert_eq!(t.tip_index("A"), Some(1));
        assert_eq!(t.tip_index("C"), Some(3));
    }

    #[test]
    fn test_postorder_children_first() {
        let t = parse_newick("((A:1,B:2):3,C:4);").unwrap();
        let po = t.postorder();
        let pos: HashMap<usize, usize> = po.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        for u in 1..=t.total_nodes() {
            for &c in &t.children[u] {
                assert!(pos[&c] < pos[&u], "child must come before parent in postorder");
            }
        }
    }
}
