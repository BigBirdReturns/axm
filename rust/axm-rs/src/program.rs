use crate::coords::{coordinate_schema_json, IR_SCHEMA_VERSION};
use crate::ir::{Fork, Node, Provenance, Relation, SourceInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counts {
    pub nodes: usize,
    pub relations: usize,
    pub forks: usize,
    pub provenance: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ChunkIndexEntry {
    Map {
        nodes: Vec<String>,
        #[serde(default)]
        content_hash: Option<String>,
    },
    Legacy(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramManifest {
    pub axm_version: String,
    pub ir_schema_version: String,
    pub created_at: String,
    pub source: serde_json::Value,
    pub coordinate_system: serde_json::Value,
    pub counts: Counts,
    #[serde(default)]
    pub stats: HashMap<String, i64>,
    #[serde(default)]
    pub chunk_index: HashMap<String, ChunkIndexEntry>,
    #[serde(default)]
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub source: SourceInfo,
    pub nodes: HashMap<String, Node>,
    pub relations: Vec<Relation>,
    pub forks: Vec<Fork>,
    pub provenance: HashMap<String, Provenance>,
    pub chunk_index: HashMap<String, Vec<String>>,
    pub chunk_hashes: HashMap<String, String>,
    pub created_at: String,
    pub version: String,
    pub stats: HashMap<String, i64>,
}

impl Program {
    pub fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();

        let mut node_ids: Vec<&String> = self.nodes.keys().collect();
        node_ids.sort();
        for id in node_ids {
            if let Some(node) = self.nodes.get(id) {
                let value = serde_json::to_value(node).unwrap();
                let json = serde_json::to_string(&canonicalize(&value)).unwrap();
                hasher.update(json.as_bytes());
            }
        }

        let mut rels = self.relations.clone();
        rels.sort_by_key(|r| r.sort_key());
        for rel in rels {
            let value = serde_json::to_value(&rel).unwrap();
            let json = serde_json::to_string(&canonicalize(&value)).unwrap();
            hasher.update(json.as_bytes());
        }

        let digest = hex::encode(hasher.finalize());
        digest.chars().take(16).collect()
    }

    pub fn manifest(&self) -> ProgramManifest {
        let chunk_index = self
            .chunk_index
            .iter()
            .map(|(chunk_id, nodes)| {
                let content_hash = self.chunk_hashes.get(chunk_id).cloned();
                (
                    chunk_id.clone(),
                    ChunkIndexEntry::Map {
                        nodes: nodes.clone(),
                        content_hash,
                    },
                )
            })
            .collect();

        ProgramManifest {
            axm_version: self.version.clone(),
            ir_schema_version: IR_SCHEMA_VERSION.to_string(),
            created_at: self.created_at.clone(),
            source: self.source.manifest(),
            coordinate_system: coordinate_schema_json(),
            counts: Counts {
                nodes: self.nodes.len(),
                relations: self.relations.len(),
                forks: self.forks.len(),
                provenance: self.provenance.len(),
            },
            stats: self.stats.clone(),
            chunk_index,
            content_hash: Some(self.content_hash()),
        }
    }

    pub fn load_dir(path: impl AsRef<Path>) -> Result<Self, String> {
        let base = path.as_ref();
        let manifest_path = base.join("manifest.json");
        let manifest_file = File::open(&manifest_path)
            .map_err(|_| format!("Missing manifest.json in {}", base.display()))?;
        let manifest: ProgramManifest = serde_json::from_reader(manifest_file)
            .map_err(|e| format!("Invalid manifest: {}", e))?;

        let source: SourceInfo = serde_json::from_value(manifest.source.clone())
            .map_err(|e| format!("Invalid source: {}", e))?;

        let nodes = read_nodes(base.join("nodes.jsonl"))?;
        let relations = read_relations(base.join("relations.jsonl"))?;
        let forks = read_forks(base.join("forks.jsonl"))?;
        let provenance = read_provenance(base.join("provenance.jsonl"))?;
        let (chunk_index, chunk_hashes) = normalize_chunk_index(&manifest.chunk_index);

        Ok(Self {
            source,
            nodes,
            relations,
            forks,
            provenance,
            chunk_index,
            chunk_hashes,
            created_at: manifest.created_at,
            version: manifest.axm_version,
            stats: manifest.stats,
        })
    }

    pub fn load_zip_bytes(bytes: &[u8]) -> Result<Self, String> {
        let reader = std::io::Cursor::new(bytes);
        let mut archive =
            ZipArchive::new(reader).map_err(|e| format!("Invalid zip archive: {}", e))?;
        let mut manifest_file = archive
            .by_name("manifest.json")
            .map_err(|_| "manifest.json missing from archive")?;
        let mut manifest_str = String::new();
        manifest_file
            .read_to_string(&mut manifest_str)
            .map_err(|e| format!("Failed to read manifest: {}", e))?;
        let manifest: ProgramManifest =
            serde_json::from_str(&manifest_str).map_err(|e| format!("Invalid manifest: {}", e))?;
        let source: SourceInfo = serde_json::from_value(manifest.source.clone())
            .map_err(|e| format!("Invalid source: {}", e))?;

        let nodes = read_nodes_from_zip(&mut archive, "nodes.jsonl")?;
        let relations = read_relations_from_zip(&mut archive, "relations.jsonl")?;
        let forks = read_forks_from_zip(&mut archive, "forks.jsonl")?;
        let provenance = read_provenance_from_zip(&mut archive, "provenance.jsonl")?;
        let (chunk_index, chunk_hashes) = normalize_chunk_index(&manifest.chunk_index);

        Ok(Self {
            source,
            nodes,
            relations,
            forks,
            provenance,
            chunk_index,
            chunk_hashes,
            created_at: manifest.created_at,
            version: manifest.axm_version,
            stats: manifest.stats,
        })
    }
}

fn canonicalize(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut ordered = BTreeMap::new();
            for (k, v) in map {
                ordered.insert(k.clone(), canonicalize(v));
            }
            let mut new_map = serde_json::Map::new();
            for (k, v) in ordered {
                new_map.insert(k, v);
            }
            serde_json::Value::Object(new_map)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize).collect())
        }
        other => other.clone(),
    }
}

fn normalize_chunk_index(
    raw: &HashMap<String, ChunkIndexEntry>,
) -> (HashMap<String, Vec<String>>, HashMap<String, String>) {
    let mut index = HashMap::new();
    let mut hashes = HashMap::new();
    for (chunk_id, entry) in raw {
        match entry {
            ChunkIndexEntry::Legacy(nodes) => {
                index.insert(chunk_id.clone(), nodes.clone());
            }
            ChunkIndexEntry::Map {
                nodes,
                content_hash,
            } => {
                index.insert(chunk_id.clone(), nodes.clone());
                if let Some(hash) = content_hash {
                    hashes.insert(chunk_id.clone(), hash.clone());
                }
            }
        }
    }
    (index, hashes)
}

fn read_nodes(path: PathBuf) -> Result<HashMap<String, Node>, String> {
    let file =
        File::open(&path).map_err(|_| format!("Missing nodes.jsonl in {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut nodes = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Failed reading nodes: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }
        let node: Node = serde_json::from_str(&line)
            .map_err(|e| format!("Invalid node at line {}: {}", idx + 1, e))?;
        let node = node.validated().map_err(|e| e.to_string())?;
        nodes.insert(node.id.clone(), node);
    }
    Ok(nodes)
}

fn read_relations(path: PathBuf) -> Result<Vec<Relation>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(&path).map_err(|e| format!("Unable to read relations: {}", e))?;
    let reader = BufReader::new(file);
    let mut rels = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Failed reading relations: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }
        let rel: Relation = serde_json::from_str(&line)
            .map_err(|e| format!("Invalid relation at line {}: {}", idx + 1, e))?;
        rels.push(rel.validated().map_err(|e| e.to_string())?);
    }
    Ok(rels)
}

fn read_forks(path: PathBuf) -> Result<Vec<Fork>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(&path).map_err(|e| format!("Unable to read forks: {}", e))?;
    let reader = BufReader::new(file);
    let mut forks = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed reading forks: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }
        let fork: Fork = serde_json::from_str(&line).map_err(|e| format!("Invalid fork: {}", e))?;
        forks.push(fork);
    }
    Ok(forks)
}

fn read_provenance(path: PathBuf) -> Result<HashMap<String, Provenance>, String> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let file = File::open(&path).map_err(|e| format!("Unable to read provenance: {}", e))?;
    let reader = BufReader::new(file);
    let mut provs = HashMap::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed reading provenance: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }
        let prov: Provenance =
            serde_json::from_str(&line).map_err(|e| format!("Invalid provenance: {}", e))?;
        provs.insert(prov.prov_id.clone(), prov);
    }
    Ok(provs)
}

fn read_nodes_from_zip<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<HashMap<String, Node>, String> {
    let mut file = archive
        .by_name(name)
        .map_err(|_| format!("{} missing", name))?;
    let mut data = String::new();
    file.read_to_string(&mut data)
        .map_err(|e| format!("Failed reading {}: {}", name, e))?;
    let mut nodes = HashMap::new();
    for (idx, line) in data.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let node: Node = serde_json::from_str(line)
            .map_err(|e| format!("Invalid node at line {}: {}", idx + 1, e))?;
        let node = node.validated().map_err(|e| e.to_string())?;
        nodes.insert(node.id.clone(), node);
    }
    Ok(nodes)
}

fn read_relations_from_zip<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<Vec<Relation>, String> {
    let mut rels = Vec::new();
    if let Ok(mut file) = archive.by_name(name) {
        let mut data = String::new();
        file.read_to_string(&mut data)
            .map_err(|e| format!("Failed reading {}: {}", name, e))?;
        for (idx, line) in data.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let rel: Relation = serde_json::from_str(line)
                .map_err(|e| format!("Invalid relation at line {}: {}", idx + 1, e))?;
            rels.push(rel.validated().map_err(|e| e.to_string())?);
        }
    }
    Ok(rels)
}

fn read_forks_from_zip<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<Vec<Fork>, String> {
    let mut forks = Vec::new();
    if let Ok(mut file) = archive.by_name(name) {
        let mut data = String::new();
        file.read_to_string(&mut data)
            .map_err(|e| format!("Failed reading {}: {}", name, e))?;
        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let fork: Fork =
                serde_json::from_str(line).map_err(|e| format!("Invalid fork: {}", e))?;
            forks.push(fork);
        }
    }
    Ok(forks)
}

fn read_provenance_from_zip<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<HashMap<String, Provenance>, String> {
    let mut provs = HashMap::new();
    if let Ok(mut file) = archive.by_name(name) {
        let mut data = String::new();
        file.read_to_string(&mut data)
            .map_err(|e| format!("Failed reading {}: {}", name, e))?;
        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let prov: Provenance =
                serde_json::from_str(line).map_err(|e| format!("Invalid provenance: {}", e))?;
            provs.insert(prov.prov_id.clone(), prov);
        }
    }
    Ok(provs)
}
