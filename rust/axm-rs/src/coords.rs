use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;

pub const IR_SCHEMA_VERSION: &str = "0.5";

pub fn coordinate_schema_json() -> serde_json::Value {
    json!({
        "version": IR_SCHEMA_VERSION,
        "dimensions": 4,
        "axes": [
            {
                "index": 0,
                "name": "major",
                "description": "Major semantic category",
                "range": [1, 8]
            },
            {
                "index": 1,
                "name": "type",
                "description": "Type within major category",
                "range": [0, 99]
            },
            {
                "index": 2,
                "name": "subtype",
                "description": "Subtype refinement",
                "range": [0, 99]
            },
            {
                "index": 3,
                "name": "instance",
                "description": "Unique instance counter",
                "range": [0, 9999]
            },
        ],
        "major_categories": {
            1: {"name": "entity", "description": "Who or what: organizations, people, products"},
            2: {"name": "action", "description": "What happened: events, transactions"},
            3: {"name": "property", "description": "Attributes, features, states"},
            4: {"name": "relation", "description": "Reserved for edge representation"},
            5: {"name": "location", "description": "Where: places, addresses, regions"},
            6: {"name": "time", "description": "When: dates, periods, timestamps"},
            7: {"name": "quantity", "description": "How much: numbers, metrics"},
            8: {"name": "abstract", "description": "Claims, beliefs, narratives, concepts"},
        },
        "id_format": "MM-TT-SS-IIII"
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Major {
    Entity = 1,
    Action = 2,
    Property = 3,
    Relation = 4,
    Location = 5,
    Time = 6,
    Quantity = 7,
    Abstract = 8,
}

impl TryFrom<i32> for Major {
    type Error = String;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Major::Entity),
            2 => Ok(Major::Action),
            3 => Ok(Major::Property),
            4 => Ok(Major::Relation),
            5 => Ok(Major::Location),
            6 => Ok(Major::Time),
            7 => Ok(Major::Quantity),
            8 => Ok(Major::Abstract),
            _ => Err(format!("Major must be 1-8, got {}", value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coord {
    pub major: u32,
    pub type_: u32,
    pub subtype: u32,
    pub instance: u32,
}

impl Coord {
    pub fn new(major: u32, type_: u32, subtype: u32, instance: u32) -> Result<Self, String> {
        let coord = Self {
            major,
            type_,
            subtype,
            instance,
        };
        coord.validate()?;
        Ok(coord)
    }

    pub fn to_list(&self) -> [u32; 4] {
        [self.major, self.type_, self.subtype, self.instance]
    }

    pub fn from_list(values: &[u32]) -> Result<Self, String> {
        if values.len() != 4 {
            return Err(format!("Coords must have 4 elements, got {}", values.len()));
        }
        Self::new(values[0], values[1], values[2], values[3])
    }

    pub fn from_id(id: &str) -> Result<Self, String> {
        let parts: Vec<&str> = id.split('-').collect();
        if parts.len() != 4 {
            return Err(format!("Invalid coord ID format: {}", id));
        }
        let numbers: Result<Vec<u32>, _> = parts.iter().map(|p| p.parse::<u32>()).collect();
        match numbers {
            Ok(nums) => Self::from_list(&nums),
            Err(_) => Err(format!("Invalid coord ID: {}", id)),
        }
    }

    pub fn to_id(&self) -> String {
        format!(
            "{:02}-{:02}-{:02}-{:04}",
            self.major, self.type_, self.subtype, self.instance
        )
    }

    pub fn distance(&self, other: &Coord, weights: [f64; 4]) -> f64 {
        let a = self.to_list();
        let b = other.to_list();
        let mut sum = 0.0;
        for i in 0..4 {
            let diff = a[i] as f64 - b[i] as f64;
            sum += weights[i] * diff * diff;
        }
        sum.sqrt()
    }

    fn validate(&self) -> Result<(), String> {
        if !(1..=8).contains(&self.major) {
            return Err(format!("Major must be 1-8, got {}", self.major));
        }
        if self.type_ > 99 {
            return Err(format!("Type must be 0-99, got {}", self.type_));
        }
        if self.subtype > 99 {
            return Err(format!("Subtype must be 0-99, got {}", self.subtype));
        }
        if self.instance > 9999 {
            return Err(format!("Instance must be 0-9999, got {}", self.instance));
        }
        Ok(())
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_round_trip() {
        let c = Coord::new(1, 1, 1, 1).unwrap();
        assert_eq!(c.to_id(), "01-01-01-0001");
        let parsed = Coord::from_id(&c.to_id()).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn distance_close() {
        let c1 = Coord::new(1, 1, 1, 1).unwrap();
        let c2 = Coord::new(1, 1, 1, 2).unwrap();
        let d = c1.distance(&c2, [1.0, 0.5, 0.3, 0.1]);
        assert!(d > 0.0 && d < 1.0);
    }
}
