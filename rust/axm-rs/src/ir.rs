use crate::coords::Coord;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IrError {
    #[error("missing field: {0}")]
    MissingField(&'static str),
    #[error("invalid field: {0}")]
    InvalidField(String),
}

fn hash_prefix(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    base64::encode(&digest)[0..16].to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Provenance {
    pub prov_id: String,
    pub chunk_id: String,
    pub extractor: String,
    pub timestamp: String,
    #[serde(default)]
    pub tier: i32,
    #[serde(default = "one_f64")]
    pub confidence: f64,
    #[serde(default)]
    pub source_span: Option<SourceSpan>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceSpan {
    pub start: Option<i64>,
    pub end: Option<i64>,
}

fn one_f64() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub coords: [u32; 4],
    pub prov_id: String,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub content_hash: Option<String>,
}

impl Node {
    pub fn from_coord(coord: Coord, label: String, prov_id: String, value: Option<serde_json::Value>, unit: Option<String>) -> Self {
        let coords = coord.to_list();
        Self {
            id: coord.to_id(),
            label,
            coords,
            prov_id,
            value,
            unit,
            metadata: HashMap::new(),
            content_hash: None,
        }
    }

    pub fn coord(&self) -> Coord {
        Coord::from_list(&self.coords).expect("stored coords are valid")
    }

    pub fn compute_content_hash(&self) -> String {
        let value = self.value.as_ref().map(|v| v.to_string()).unwrap_or_default();
        let unit = self.unit.clone().unwrap_or_default();
        hash_prefix(&format!("{}|{}|{}", self.label, value, unit))
    }

    pub fn validated(mut self) -> Result<Self, IrError> {
        if self.label.is_empty() {
            return Err(IrError::MissingField("label"));
        }
        if self.prov_id.is_empty() {
            return Err(IrError::MissingField("prov_id"));
        }
        if self.coords.len() != 4 {
            return Err(IrError::InvalidField("coords".into()));
        }
        let _ = Coord::from_list(&self.coords).map_err(|e| IrError::InvalidField(e))?;
        if self.content_hash.is_none() {
            self.content_hash = Some(self.compute_content_hash());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Relation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub prov_id: String,
    #[serde(default = "one_f64")]
    pub confidence: f64,
}

impl Relation {
    pub fn validated(self) -> Result<Self, IrError> {
        if self.subject == self.object {
            return Err(IrError::InvalidField("Self-loops not allowed".into()));
        }
        if self.subject.is_empty() || self.object.is_empty() {
            return Err(IrError::MissingField("subject/object"));
        }
        if self.predicate.is_empty() {
            return Err(IrError::MissingField("predicate"));
        }
        Ok(self)
    }

    pub fn sort_key(&self) -> (String, String, String) {
        (self.subject.clone(), self.predicate.clone(), self.object.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForkOption {
    pub node_id: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fork {
    pub fork_id: String,
    pub options: Vec<ForkOption>,
    pub prov_id: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceInfo {
    pub uri: String,
    pub hash: String,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
}

impl SourceInfo {
    pub fn manifest(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("uri".into(), serde_json::Value::String(self.uri.clone()));
        map.insert("hash".into(), serde_json::Value::String(self.hash.clone()));
        if let Some(size) = self.size_bytes {
            map.insert("size_bytes".into(), serde_json::Value::Number(size.into()));
        }
        if let Some(mt) = &self.media_type {
            map.insert("media_type".into(), serde_json::Value::String(mt.clone()));
        }
        if let Some(title) = &self.title {
            map.insert("title".into(), serde_json::Value::String(title.clone()));
        }
        serde_json::Value::Object(map)
    }
}

pub fn manifest_coordinate_schema() -> serde_json::Value {
    serde_json::json!({
        "version": crate::coords::IR_SCHEMA_VERSION,
        "dimensions": 4,
    })
}
