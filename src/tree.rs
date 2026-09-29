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

    /// True when the root has exactly two children, i.e. the tree is rooted.
    /// A trifurcating root means the tree is unrooted.
    pub fn is_rooted(&self) -> bool {
        self.children[self.root].len() == 2
    }

    /// Adjacency list: for each node, its neighbours and the branch length.
    pub fn adjacency(&self) -> Vec<Vec<(usize, f64)>> {
        let total = self.total_nodes();
        let mut adj = vec![Vec::new(); total + 1];
        for node in 1..=total {
            let p = self.parent[node];
            if p > 0 {
                adj[p].push((node, self.edge_length[node]));
                adj[node].push((p, self.edge_length[node]));
            }
        }
        adj
    }

    /// Distances from `src` to every node, without crossing the edge (src, blocked).
    /// `dist[0]` is unused; `dist[t]` for t <= ntip is the root-to-tip distance
    /// that would result from rooting along that edge.
    /// Distance from `u` to every node, plus which side of the edge (u, v) each
    /// node lies on: +1 for u's side (including u), -1 for v's side (including v).
    /// The edge (u, v) itself is not crossed in either direction.
    fn dist_and_side(
        &self,
        adj: &[Vec<(usize, f64)>],
        u: usize,
        v: usize,
    ) -> (Vec<f64>, Vec<i8>) {
        let total = self.total_nodes();
        let cut = adj[u]
            .iter()
            .find(|(w, _)| *w == v)
            .map(|(_, l)| *l)
            .unwrap_or(0.0);
        let mut dist = vec![f64::NAN; total + 1];
        let mut side = vec![0i8; total + 1];

        // u's side: distances measured from u
        dist[u] = 0.0;
        side[u] = 1;
        let mut stack = vec![u];
        while let Some(w) = stack.pop() {
            for &(z, l) in &adj[w] {
                if z == v && w == u {
                    continue; // do not cross the cut edge
                }
                if !dist[z].is_nan() {
                    continue;
                }
                dist[z] = dist[w] + l;
                side[z] = 1;
                stack.push(z);
            }
        }

        // v's side: distances still measured from u (so add the cut length)
        dist[v] = cut;
        side[v] = -1;
        let mut stack = vec![v];
        while let Some(w) = stack.pop() {
            for &(z, l) in &adj[w] {
                if z == u && w == v {
                    continue;
                }
                if !dist[z].is_nan() {
                    continue;
                }
                dist[z] = dist[w] + l;
                side[z] = -1;
                stack.push(z);
            }
        }
        (dist, side)
    }

    /// Re-root the tree in the middle of edge (u, v): the new root's children are
    /// `u` (branch `x * L`) and `v` (branch `(1 - x) * L`).
    ///
    /// Tips keep their ids so that caller-supplied dates stay aligned. Nodes are
    /// renumbered compactly, which also drops the old root if it ends up
    /// unifurcating.
    pub fn reroot_at_edge(&self, adj: &[Vec<(usize, f64)>], u: usize, v: usize, x: f64) -> Tree {
        let total = self.total_nodes();
        let ntip = self.ntip;
        let split_len = adj[u]
            .iter()
            .find(|(w, _)| *w == v)
            .map(|(_, l)| *l)
            .unwrap_or(0.0);

        let tmp_root = total + 1;
        let mut parent = vec![0usize; total + 2];
        let mut blen = vec![0.0f64; total + 2];
        parent[u] = tmp_root;
        blen[u] = x * split_len;
        parent[v] = tmp_root;
        blen[v] = (1.0 - x) * split_len;

        // orient the two sides away from the cut edge
        let mut visited = vec![false; total + 2];
        visited[tmp_root] = true;
        visited[u] = true;
        visited[v] = true;
        for &start in &[u, v] {
            let mut stack = vec![start];
            while let Some(w) = stack.pop() {
                for &(z, l) in &adj[w] {
                    if visited[z] {
                        continue;
                    }
                    visited[z] = true;
                    parent[z] = w;
                    blen[z] = l;
                    stack.push(z);
                }
            }
        }

        // renumber: tips keep 1..=ntip, internals get ntip+1, ntip+2, ... in BFS order
        let mut children_tmp: Vec<Vec<usize>> = vec![Vec::new(); total + 2];
        for w in 1..=total {
            if parent[w] > 0 {
                children_tmp[parent[w]].push(w);
            }
        }
        let mut id_map = vec![0usize; total + 2];
        for t in 1..=ntip {
            id_map[t] = t;
        }
        let mut next_internal = ntip + 1;
        let mut queue = std::collections::VecDeque::from([tmp_root]);
        id_map[tmp_root] = next_internal;
        next_internal += 1;
        let mut order = vec![tmp_root];
        while let Some(w) = queue.pop_front() {
            for &c in &children_tmp[w] {
                // tips keep their original ids so caller dates stay aligned;
                // only internal nodes get freshly assigned numbers
                if c > ntip {
                    id_map[c] = next_internal;
                    next_internal += 1;
                }
                order.push(c);
                queue.push_back(c);
            }
        }

        let mut edges = Vec::with_capacity(order.len());
        for &w in &order {
            if w == tmp_root {
                continue;
            }
            edges.push((id_map[parent[w]], id_map[w], blen[w]));
        }
        Tree::from_edges(self.tip_labels.clone(), edges, id_map[tmp_root])
    }

    /// Pick the root that maximises the correlation between sampling dates and
    /// root-to-tip distances (BactDating's `initRoot`).
    ///
    /// For a candidate root at fraction `x` along edge (u, v) with length `L`,
    /// every tip's root-to-tip distance is `d_u(t) + s(t) * x * L`, where
    /// `d_u(t)` is the distance from `u` and `s(t)` is +1 on u's side and -1 on
    /// v's side. The correlation is therefore a ratio of quadratics in `x`, so
    /// the per-edge cost is one O(n) distance pass plus cheap arithmetic per
    /// candidate position.
    pub fn init_root(&self, dates: &[f64], mtry: usize) -> Tree {
        let total = self.total_nodes();
        let ntip = self.ntip;
        let adj = self.adjacency();
        let mean_edge = if self.n_edges() > 0 {
            self.total_length() / self.n_edges() as f64
        } else {
            1.0
        };

        // only dated tips contribute to the correlation
        let mut idx: Vec<usize> = Vec::new();
        let mut dv: Vec<f64> = Vec::new();
        for t in 1..=ntip {
            if dates[t - 1].is_finite() {
                idx.push(t);
                dv.push(dates[t - 1]);
            }
        }
        if idx.len() < 3 {
            return self.clone();
        }
        let n = idx.len() as f64;
        let mean_d = dv.iter().sum::<f64>() / n;
        let a: Vec<f64> = dv.iter().map(|d| d - mean_d).collect();
        let s_dd: f64 = a.iter().map(|v| v * v).sum();

        let mut best: Option<(f64, Tree)> = None;

        for u in 1..=total {
            for &(v, l) in &adj[u] {
                if v < u {
                    continue;
                }
                let (du, sd) = self.dist_and_side(&adj, u, v);
                // b_t = d_u(t) shifted to zero mean; c_t = side(t)*L shifted likewise
                let mut b = Vec::with_capacity(idx.len());
                let mut c = Vec::with_capacity(idx.len());
                for &t in &idx {
                    b.push(du[t]);
                    c.push(sd[t] as f64 * l);
                }
                let mean_b = b.iter().sum::<f64>() / n;
                let mean_c = c.iter().sum::<f64>() / n;
                for x in b.iter_mut() {
                    *x -= mean_b;
                }
                for x in c.iter_mut() {
                    *x -= mean_c;
                }
                let s_ab: f64 = a.iter().zip(&b).map(|(p, q)| p * q).sum();
                let s_ac: f64 = a.iter().zip(&c).map(|(p, q)| p * q).sum();
                let s_bb: f64 = b.iter().map(|q| q * q).sum();
                let s_bc: f64 = b.iter().zip(&c).map(|(p, q)| p * q).sum();
                let s_cc: f64 = c.iter().map(|q| q * q).sum();

                let attempts = if mean_edge > 0.0 {
                    ((mtry as f64 * l / mean_edge).ceil() as usize).max(1)
                } else {
                    1
                };
                for step in 1..=attempts {
                    let x = step as f64 / (attempts + 1) as f64;
                    let cov = s_ab + x * s_ac;
                    let var = s_bb + 2.0 * x * s_bc + x * x * s_cc;
                    if var <= 0.0 || s_dd <= 0.0 {
                        continue;
                    }
                    let corr = cov / (s_dd.sqrt() * var.sqrt());
                    if corr.is_finite() && best.as_ref().map_or(true, |(bc, _)| corr > *bc) {
                        best = Some((corr, self.reroot_at_edge(&adj, u, v, x)));
                    }
                }
            }
        }

        match best {
            Some((_, t)) => t,
            None => self.clone(),
        }
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
    #[test]
    fn test_reroot_preserves_tips_and_total_length() {
        // unrooted tree: ((A:1,B:2):3,C:4,D:5);  (root has 3 children)
        let t = parse_newick("((A:1,B:2):3,C:4,D:5);").unwrap();
        assert_eq!(t.ntip, 4);
        assert!(!t.is_rooted(), "trifurcating root means unrooted");
        let adj = t.adjacency();
        // re-root in the middle of the edge to C
        let c = t.tip_index("C").unwrap();
        let parent_of_c = t.parent[c];
        let r = t.reroot_at_edge(&adj, parent_of_c, c, 0.5);
        assert_eq!(r.tip_labels, t.tip_labels, "tip order must be preserved");
        assert!(
            (r.total_length() - t.total_length()).abs() < 1e-9,
            "total length {} vs {}",
            r.total_length(),
            t.total_length()
        );
        assert!(r.is_rooted(), "re-rooted tree must be rooted");
        // no node may be childless-but-internal, and exactly one root
        let roots = (1..=r.total_nodes())
            .filter(|&n| r.parent[n] == 0)
            .count();
        assert_eq!(roots, 1, "exactly one root expected");
        for n in 1..=r.total_nodes() {
            if !r.is_tip(n) {
                assert!(!r.children[n].is_empty(), "internal node {} has no children", n);
            }
        }
    }

    #[test]
    fn test_init_root_prefers_temporal_signal() {
        // Build a tree whose true root sits deep in the past, then check that
        // init_root recovers a rooting at least as good as the input's.
        // ((A:1,B:1):1,(C:1,D:1):1);  dates increase with distance from root
        let t = parse_newick("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);").unwrap();
        let dates = vec![2000.0, 2000.0, 2010.0, 2010.0]; // A,B old; C,D young
        let corr_before = crate::roottip::root_to_tip_cor(&t, &dates);
        let r = t.init_root(&dates, 5);
        let corr_after = crate::roottip::root_to_tip_cor(&r, &dates);
        assert!(
            corr_after >= corr_before - 1e-9,
            "init_root should not make the correlation worse: {} -> {}",
            corr_before,
            corr_after
        );
        assert_eq!(r.tip_labels, t.tip_labels);
    }

}