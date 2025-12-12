use crate::coords::Coord;
use crate::ir::Relation;
use crate::program::Program;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub node_id: String,
    pub label: String,
    pub value: Option<serde_json::Value>,
    pub distance: f64,
}

pub struct Space<'a> {
    program: &'a Program,
    by_major: HashMap<u32, Vec<String>>, // node ids
    outgoing: HashMap<String, Vec<&'a Relation>>,
    incoming: HashMap<String, Vec<&'a Relation>>,
}

impl<'a> Space<'a> {
    pub fn new(program: &'a Program) -> Self {
        let mut by_major: HashMap<u32, Vec<String>> = HashMap::new();
        for (id, node) in &program.nodes {
            by_major.entry(node.coords[0]).or_default().push(id.clone());
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

    pub fn query(
        &'a self,
        major: Option<u32>,
        type_: Option<u32>,
        subtype: Option<u32>,
        label_contains: Option<&str>,
    ) -> Vec<&'a crate::ir::Node> {
        let candidates: Vec<&String> = if let Some(m) = major {
            self.by_major.get(&m).map(|v| v.iter().collect()).unwrap_or_default()
        } else {
            self.program.nodes.keys().collect()
        };

        let mut results = Vec::new();
        for id in candidates {
            if let Some(node) = self.program.nodes.get(id) {
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
                results.push(node);
            }
        }
        results
    }

    pub fn neighbors(&'a self, coord: &Coord, radius: f64) -> Vec<QueryResult> {
        let mut results = Vec::new();
        for (id, node) in &self.program.nodes {
            let other = Coord::from_list(&node.coords).expect("stored coords valid");
            let distance = coord.distance(&other, [1.0, 0.5, 0.3, 0.1]);
            if distance <= radius {
                results.push(QueryResult {
                    node_id: id.clone(),
                    label: node.label.clone(),
                    value: node.value.clone(),
                    distance,
                });
            }
        }
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    pub fn relation_paths(&'a self, start: &str, predicate: Option<&str>, max_depth: usize) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut frontier = vec![(start.to_string(), vec![start.to_string()], 0usize)];
        let mut visited = std::collections::HashSet::new();

        while let Some((node_id, path, depth)) = frontier.pop() {
            if depth >= max_depth {
                continue;
            }
            if !visited.insert(node_id.clone()) {
                continue;
            }
            for rel in self.outgoing.get(&node_id).into_iter().flatten() {
                if predicate.map(|p| p == rel.predicate).unwrap_or(true) {
                    let mut new_path = path.clone();
                    new_path.push(rel.object.clone());
                    paths.push(new_path.clone());
                    frontier.push((rel.object.clone(), new_path, depth + 1));
                }
            }
            for rel in self.incoming.get(&node_id).into_iter().flatten() {
                if predicate.map(|p| p == rel.predicate).unwrap_or(true) {
                    let mut new_path = path.clone();
                    new_path.push(rel.subject.clone());
                    paths.push(new_path.clone());
                    frontier.push((rel.subject.clone(), new_path, depth + 1));
                }
            }
        }
        paths
    }

    pub fn count(&self, major: Option<u32>) -> usize {
        if let Some(m) = major {
            self.by_major.get(&m).map(|v| v.len()).unwrap_or(0)
        } else {
            self.program.nodes.len()
        }
    }
}
