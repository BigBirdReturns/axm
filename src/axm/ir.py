"""
AXM IR Primitives - Nodes, Relations, Provenance (v0.4)

Immutable, serializable, content-addressable data structures.
All from_dict() methods validate input strictly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .coords import Coord, COORDINATE_SCHEMA, IR_SCHEMA_VERSION


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _require(d: Dict, field: str, expected_type: type = None) -> Any:
    """Require field exists and optionally check type."""
    if field not in d:
        raise ValueError(f"Missing required field: {field}")
    value = d[field]
    if expected_type and not isinstance(value, expected_type):
        raise TypeError(f"Field {field} must be {expected_type.__name__}, got {type(value).__name__}")
    return value


def _validate_coords(coords: Any) -> List[int]:
    """Validate coordinate list."""
    if not isinstance(coords, (list, tuple)):
        raise TypeError(f"Coords must be list, got {type(coords)}")
    if len(coords) != 4:
        raise ValueError(f"Coords must have 4 elements, got {len(coords)}")
    if not all(isinstance(c, int) for c in coords):
        raise TypeError(f"Coords must be integers: {coords}")
    if not (1 <= coords[0] <= 8):
        raise ValueError(f"Major must be 1-8, got {coords[0]}")
    if not (0 <= coords[1] <= 99):
        raise ValueError(f"Type must be 0-99, got {coords[1]}")
    if not (0 <= coords[2] <= 99):
        raise ValueError(f"Subtype must be 0-99, got {coords[2]}")
    if not (0 <= coords[3] <= 9999):
        raise ValueError(f"Instance must be 0-9999, got {coords[3]}")
    return list(coords)


# =============================================================================
# PROVENANCE
# =============================================================================

@dataclass(frozen=True)
class Provenance:
    """
    Immutable record of how a node was extracted.
    Full audit trail for compliance.
    """
    prov_id: str
    chunk_id: str
    extractor: str
    timestamp: str
    tier: int = 0
    confidence: float = 1.0
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    model: Optional[str] = None
    prompt_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "prov_id": self.prov_id,
            "chunk_id": self.chunk_id,
            "extractor": self.extractor,
            "timestamp": self.timestamp,
            "tier": self.tier,
            "confidence": self.confidence,
        }
        if self.source_start is not None:
            d["source_span"] = {"start": self.source_start, "end": self.source_end}
        if self.model:
            d["model"] = self.model
        if self.prompt_hash:
            d["prompt_hash"] = self.prompt_hash
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Provenance":
        """Deserialize with validation."""
        _require(d, "prov_id", str)
        _require(d, "chunk_id", str)
        _require(d, "extractor", str)
        _require(d, "timestamp", str)
        
        span = d.get("source_span", {})
        
        return cls(
            prov_id=d["prov_id"],
            chunk_id=d["chunk_id"],
            extractor=d["extractor"],
            timestamp=d["timestamp"],
            tier=d.get("tier", 0),
            confidence=d.get("confidence", 1.0),
            source_start=span.get("start"),
            source_end=span.get("end"),
            model=d.get("model"),
            prompt_hash=d.get("prompt_hash"),
        )


# =============================================================================
# NODE
# =============================================================================

@dataclass(frozen=True)
class Node:
    """
    Immutable semantic node.
    Content-addressable: same (label, value, unit) = same content_hash.
    """
    coord: Coord
    label: str
    prov_id: str
    value: Any = None
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure metadata is a dict (frozen dataclass workaround)
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @property
    def id(self) -> str:
        """Canonical ID from coordinates."""
        return self.coord.to_id()
    
    @property
    def content_hash(self) -> str:
        """Hash of semantic content for deduplication."""
        content = f"{self.label}|{self.value}|{self.unit}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def contextual_hash(self, chunk_id: str) -> str:
        """Hash including chunk context - use for Tier 1+ to preserve duplicates in different contexts."""
        content = f"{chunk_id}|{self.label}|{self.value}|{self.unit}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = {
            "id": self.id,
            "label": self.label,
            "coords": self.coord.to_list(),
            "prov_id": self.prov_id,
            "content_hash": self.content_hash,
        }
        if self.value is not None:
            d["value"] = self.value
        if self.unit is not None:
            d["unit"] = self.unit
        if self.metadata:
            d["metadata"] = self.metadata
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        """Deserialize with strict validation."""
        _require(d, "label", str)
        _require(d, "prov_id", str)
        
        # Coords can be in "coords" or derived from "id"
        if "coords" in d:
            coords = _validate_coords(d["coords"])
            coord = Coord.from_list(coords)
        elif "id" in d:
            coord = Coord.from_id(d["id"])
        else:
            raise ValueError("Node must have 'coords' or 'id'")
        
        return cls(
            coord=coord,
            label=d["label"],
            prov_id=d["prov_id"],
            value=d.get("value"),
            unit=d.get("unit"),
            metadata=d.get("metadata", {}),
        )
    
    def with_coord(self, new_coord: Coord) -> "Node":
        """Create copy with different coordinates."""
        return Node(
            coord=new_coord,
            label=self.label,
            prov_id=self.prov_id,
            value=self.value,
            unit=self.unit,
            metadata=self.metadata,
        )


# =============================================================================
# RELATION
# =============================================================================

PREDICATES = {
    "HAS_VALUE", "HAS_UNIT",
    "FOR_PERIOD", "OCCURRED_AT", "BEFORE", "AFTER",
    "REPORTED_BY", "AUTHORED_BY", "OWNED_BY", "EMPLOYED_BY", "LOCATED_IN",
    "SUPPORTS", "CONTRADICTS", "DERIVED_FROM", "SAME_AS", "PART_OF",
    "EXTRACTED_FROM", "CITED_IN",
}


@dataclass(frozen=True)
class Relation:
    """Immutable edge between two nodes."""
    subject_id: str
    predicate: str
    object_id: str
    prov_id: str
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.subject_id == self.object_id:
            raise ValueError("Self-loops not allowed")
        if not self.subject_id or not self.object_id:
            raise ValueError("Subject and object IDs required")
        if not self.predicate:
            raise ValueError("Predicate required")
    
    @property
    def sort_key(self) -> Tuple[str, str, str]:
        return (self.subject_id, self.predicate, self.object_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject_id,
            "predicate": self.predicate,
            "object": self.object_id,
            "prov_id": self.prov_id,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Relation":
        """Deserialize with validation."""
        _require(d, "subject", str)
        _require(d, "predicate", str)
        _require(d, "object", str)
        _require(d, "prov_id", str)
        
        return cls(
            subject_id=d["subject"],
            predicate=d["predicate"],
            object_id=d["object"],
            prov_id=d["prov_id"],
            confidence=d.get("confidence", 1.0),
        )


# =============================================================================
# FORK
# =============================================================================

@dataclass(frozen=True)
class ForkOption:
    """One possible interpretation."""
    node_id: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "confidence": self.confidence}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ForkOption":
        _require(d, "node_id", str)
        _require(d, "confidence", (int, float))
        return cls(node_id=d["node_id"], confidence=float(d["confidence"]))


@dataclass(frozen=True)
class Fork:
    """Explicit ambiguity in extraction."""
    fork_id: str
    options: Tuple[ForkOption, ...]
    prov_id: str
    description: str = ""
    
    def best(self) -> Optional[str]:
        if not self.options:
            return None
        return max(self.options, key=lambda o: o.confidence).node_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fork_id": self.fork_id,
            "options": [o.to_dict() for o in self.options],
            "prov_id": self.prov_id,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fork":
        _require(d, "fork_id", str)
        _require(d, "options", list)
        _require(d, "prov_id", str)
        
        return cls(
            fork_id=d["fork_id"],
            options=tuple(ForkOption.from_dict(o) for o in d["options"]),
            prov_id=d["prov_id"],
            description=d.get("description", ""),
        )


# =============================================================================
# SOURCE INFO
# =============================================================================

@dataclass(frozen=True)
class SourceInfo:
    """Metadata about the source document."""
    uri: str
    hash_sha256: str
    size_bytes: Optional[int] = None
    media_type: Optional[str] = None
    title: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"uri": self.uri, "hash": self.hash_sha256}
        if self.size_bytes:
            d["size_bytes"] = self.size_bytes
        if self.media_type:
            d["media_type"] = self.media_type
        if self.title:
            d["title"] = self.title
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceInfo":
        _require(d, "uri", str)
        _require(d, "hash", str)
        
        return cls(
            uri=d["uri"],
            hash_sha256=d["hash"],
            size_bytes=d.get("size_bytes"),
            media_type=d.get("media_type"),
            title=d.get("title"),
        )
