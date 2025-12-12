use crate::coords::Coord;
use crate::ir::{Node, Relation};
use crate::program::Program;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult<'a> {
    pub node: &'a Node,
    #[serde(default)]
    pub distance: f64,
    #[serde(default)]
    pub path: Vec<String>,
}

pub struct Space<'a> {
    program: &'a Program,
    by_major: HashMap<u32, Vec<&'a Node>>,
    outgoing: HashMap<String, Vec<&'a Relation>>,
    incoming: HashMap<String, Vec<&'a Relation>>,
}

impl<'a> Space<'a> {
    pub fn new(program: &'a Program) -> Self {
        let mut by_major: HashMap<u32, Vec<&Node>> = HashMap::new();
        for node in program.nodes.values() {
            by_major.entry(node.coords[0]).or_default().push(node);
        }

        let mut outgoing: HashMap<String, Vec<&Relation>> = HashMap::new();
        let mut incoming: HashMap<String, Vec<&Relation>> = HashMap::new();
        for rel in &program.relations {
            outgoing.entry(rel.subject.clone()).or_default().push(rel);
            incoming.entry(rel.object.clone()).or_default().push(rel);
        }

        Self {
            program,
            by_major,
            outgoing,
            incoming,
        }
    }

    pub fn get(&self, node_id: &str) -> Option<&'a Node> {
        self.program.nodes.get(node_id)
    }

    pub fn all_nodes(&self) -> impl Iterator<Item = &'a Node> {
        self.program.nodes.values()
    }

    pub fn query(
        &'a self,
        major: Option<u32>,
        type_: Option<u32>,
        subtype: Option<u32>,
        label_contains: Option<&str>,
        value_gt: Option<f64>,
        value_lt: Option<f64>,
    ) -> Vec<&'a Node> {
        let candidates: Vec<&Node> = if let Some(m) = major {
            self.by_major.get(&m).map(|v| v.clone()).unwrap_or_default()
        } else {
            self.program.nodes.values().collect()
        };

        let mut results = Vec::new();
        for node in candidates {
            if let Some(t) = type_ {
                if node.coords[1] != t {
                    continue;
                }
            }
            if let Some(st) = subtype {
                if node.coords[2] != st {
                    continue;
                }
            }
            if let Some(substr) = label_contains {
                if !node.label.to_lowercase().contains(&substr.to_lowercase()) {
                    continue;
                }
            }

            let numeric = node.value.as_ref().and_then(|v| extract_numeric(v));
            if let Some(gt) = value_gt {
                if numeric.map(|n| n <= gt).unwrap_or(true) {
                    continue;
                }
            }
            if let Some(lt) = value_lt {
                if numeric.map(|n| n >= lt).unwrap_or(true) {
                    continue;
                }
            }

            results.push(node);
        }
        results
    }

    pub fn find(
        &'a self,
        major: Option<u32>,
        type_: Option<u32>,
        subtype: Option<u32>,
        label_contains: Option<&str>,
        value_gt: Option<f64>,
        value_lt: Option<f64>,
    ) -> Vec<&'a Node> {
        self.query(major, type_, subtype, label_contains, value_gt, value_lt)
    }

    pub fn first(
        &'a self,
        major: Option<u32>,
        type_: Option<u32>,
        subtype: Option<u32>,
        label_contains: Option<&str>,
        value_gt: Option<f64>,
        value_lt: Option<f64>,
    ) -> Option<&'a Node> {
        self.query(major, type_, subtype, label_contains, value_gt, value_lt)
            .into_iter()
            .next()
    }

    pub fn neighbors_from_coord(
        &'a self,
        coord: &Coord,
        radius: f64,
        weights: [f64; 4],
    ) -> Vec<QueryResult<'a>> {
        let mut results = Vec::new();
        for node in self.program.nodes.values() {
            let other = Coord::from_list(&node.coords).expect("stored coords valid");
            let distance = coord.distance(&other, weights);
            if distance <= radius {
                results.push(QueryResult {
                    node,
                    distance,
                    path: Vec::new(),
                });
            }
        }
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    pub fn neighbors(
        &'a self,
        node_id: &str,
        radius: f64,
        weights: [f64; 4],
    ) -> Vec<QueryResult<'a>> {
        if let Some(node) = self.get(node_id) {
            let coord = Coord::from_list(&node.coords).expect("stored coords valid");
            self.neighbors_from_coord(&coord, radius, weights)
        } else {
            Vec::new()
        }
    }

    pub fn outgoing(&'a self, node_id: &str) -> impl Iterator<Item = &'a Relation> {
        self.outgoing
            .get(node_id)
            .into_iter()
            .flat_map(|rels| rels.iter().copied())
    }

    pub fn incoming(&'a self, node_id: &str) -> impl Iterator<Item = &'a Relation> {
        self.incoming
            .get(node_id)
            .into_iter()
            .flat_map(|rels| rels.iter().copied())
    }

    pub fn traverse(
        &'a self,
        start_id: &str,
        predicate: Option<&str>,
        max_depth: usize,
        direction: &str,
    ) -> Vec<QueryResult<'a>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier = vec![(start_id.to_string(), vec![start_id.to_string()], 0usize)];
        let mut results = Vec::new();

        while let Some((node_id, path, depth)) = frontier.pop() {
            if !visited.insert(node_id.clone()) {
                continue;
            }

            if node_id != start_id {
                if let Some(node) = self.get(&node_id) {
                    results.push(QueryResult {
                        node,
                        distance: 0.0,
                        path: path.clone(),
                    });
                }
            }

            if depth >= max_depth {
                continue;
            }

            if matches!(direction, "out" | "both") {
                for rel in self.outgoing.get(&node_id).into_iter().flatten() {
                    if predicate.map(|p| p == rel.predicate).unwrap_or(true) {
                        let mut next_path = path.clone();
                        next_path.push(rel.object.clone());
                        frontier.push((rel.object.clone(), next_path, depth + 1));
                    }
                }
            }

            if matches!(direction, "in" | "both") {
                for rel in self.incoming.get(&node_id).into_iter().flatten() {
                    if predicate.map(|p| p == rel.predicate).unwrap_or(true) {
                        let mut next_path = path.clone();
                        next_path.push(rel.subject.clone());
                        frontier.push((rel.subject.clone(), next_path, depth + 1));
                    }
                }
            }
        }

        results
    }

    pub fn path(&'a self, start_id: &str, end_id: &str, max_depth: usize) -> Option<Vec<String>> {
        if start_id == end_id {
            return Some(vec![start_id.to_string()]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier = vec![(start_id.to_string(), vec![start_id.to_string()])];

        for _ in 0..max_depth {
            let mut next_frontier = Vec::new();
            for (node_id, path) in frontier {
                if !visited.insert(node_id.clone()) {
                    continue;
                }

                for rel in self.outgoing.get(&node_id).into_iter().flatten() {
                    if rel.object == end_id {
                        let mut new_path = path.clone();
                        new_path.push(end_id.to_string());
                        return Some(new_path);
                    }
                    if !visited.contains(&rel.object) {
                        let mut new_path = path.clone();
                        new_path.push(rel.object.clone());
                        next_frontier.push((rel.object.clone(), new_path));
                    }
                }

                for rel in self.incoming.get(&node_id).into_iter().flatten() {
                    if rel.subject == end_id {
                        let mut new_path = path.clone();
                        new_path.push(end_id.to_string());
                        return Some(new_path);
                    }
                    if !visited.contains(&rel.subject) {
                        let mut new_path = path.clone();
                        new_path.push(rel.subject.clone());
                        next_frontier.push((rel.subject.clone(), new_path));
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        None
    }

    pub fn count(&self, major: Option<u32>) -> usize {
        if let Some(m) = major {
            self.by_major.get(&m).map(|v| v.len()).unwrap_or(0)
        } else {
            self.program.nodes.len()
        }
    }

    pub fn sum_values(
        &'a self,
        major: Option<u32>,
        type_: Option<u32>,
        label_contains: Option<&str>,
    ) -> f64 {
        let mut total = 0.0;
        for node in self.query(major, type_, None, label_contains, None, None) {
            if let Some(val) = node.value.as_ref().and_then(|v| extract_numeric(v)) {
                total += val;
            }
        }
        total
    }

    pub fn distinct_labels(&'a self, major: Option<u32>) -> HashSet<String> {
        self.query(major, None, None, None, None, None)
            .into_iter()
            .map(|n| n.label.clone())
            .collect()
    }
}

fn extract_numeric(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(num) => num.as_f64(),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}
