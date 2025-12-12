"""
AXM Program - The Compiled Semantic Artifact (v0.4)

Programs are:
- Deterministic (same input → same output)
- Git-diffable (sorted JSONL)
- Round-trip safe (write → load → write = identical)
- Validated on load
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from .coords import Coord, CoordAssigner, COORDINATE_SCHEMA, IR_SCHEMA_VERSION
from .ir import Node, Relation, Fork, ForkOption, Provenance, SourceInfo


# =============================================================================
# PROGRAM BUILDER
# =============================================================================

class ProgramBuilder:
    """
    Builder for creating AXM programs.
    
    Provides O(1) lookups via indices. No direct access to internal state.
    """
    
    VERSION = "0.5.3"
    
    def __init__(self, source: SourceInfo):
        self.source = source
        self.created_at = datetime.utcnow().isoformat() + "Z"
        
        # Storage
        self._nodes: Dict[str, Node] = {}
        self._relations: List[Relation] = []
        self._forks: List[Fork] = []
        self._provenance: Dict[str, Provenance] = {}
        
        # Indices for O(1) lookup
        self._content_hashes: Dict[str, str] = {}  # content_hash -> node_id
        self._label_index: Dict[str, str] = {}      # label -> node_id (first occurrence)
        
        # Chunk tracking for incremental compilation
        self._chunk_nodes: Dict[str, List[str]] = {}  # chunk_id -> [node_ids]
        self._chunk_hashes: Dict[str, str] = {}       # chunk_id -> content_hash
        
        # Coordinate assigner
        self.coords = CoordAssigner()
        
        # Stats
        self._stats = {
            "tier_0": 0, "tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0,
            "deduplicated": 0,
        }
    
    # =========================================================================
    # PUBLIC API - Use these instead of accessing _nodes directly
    # =========================================================================
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID. O(1)."""
        return self._nodes.get(node_id)
    
    def find_by_content_hash(self, content_hash: str) -> Optional[str]:
        """Find node_id by content hash. O(1)."""
        return self._content_hashes.get(content_hash)
    
    def find_by_label(self, label: str) -> Optional[str]:
        """Find first node_id with exact label. O(1)."""
        return self._label_index.get(label)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists. O(1)."""
        return node_id in self._nodes
    
    def node_count(self) -> int:
        """Get node count."""
        return len(self._nodes)
    
    def register_chunk_hash(self, chunk_id: str, content_hash: str) -> None:
        """Register a chunk's content hash for incremental compilation."""
        self._chunk_hashes[chunk_id] = content_hash
    
    # =========================================================================
    # MUTATION
    # =========================================================================
    
    def add_provenance(self, prov: Provenance) -> None:
        """Register a provenance record."""
        self._provenance[prov.prov_id] = prov
        tier_key = f"tier_{prov.tier}"
        if tier_key in self._stats:
            self._stats[tier_key] += 1
    
    def add_node(self, node: Node, deduplicate: bool = True, chunk_id: str = None) -> str:
        """
        Add a node. Returns node_id (may differ if deduplicated).
        """
        if node.prov_id not in self._provenance:
            raise ValueError(f"Node references unknown provenance: {node.prov_id}")
        
        # Deduplication
        if deduplicate:
            existing_id = self._content_hashes.get(node.content_hash)
            if existing_id:
                self._stats["deduplicated"] += 1
                return existing_id
        
        # ID collision check
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        
        # Add to storage and indices
        self._nodes[node.id] = node
        self._content_hashes[node.content_hash] = node.id
        
        if node.label not in self._label_index:
            self._label_index[node.label] = node.id
        
        # Track chunk association
        if chunk_id:
            if chunk_id not in self._chunk_nodes:
                self._chunk_nodes[chunk_id] = []
            self._chunk_nodes[chunk_id].append(node.id)
        
        return node.id
    
    def add_relation(self, rel: Relation) -> None:
        """Add a relation. Validates references."""
        if rel.subject_id not in self._nodes:
            raise ValueError(f"Relation subject not found: {rel.subject_id}")
        if rel.object_id not in self._nodes:
            raise ValueError(f"Relation object not found: {rel.object_id}")
        if rel.prov_id not in self._provenance:
            raise ValueError(f"Relation references unknown provenance: {rel.prov_id}")
        
        self._relations.append(rel)
    
    def add_fork(self, fork: Fork) -> None:
        """Add a fork. Validates references."""
        for opt in fork.options:
            if opt.node_id not in self._nodes:
                raise ValueError(f"Fork references unknown node: {opt.node_id}")
        
        self._forks.append(fork)
    
    def link(self, subject_id: str, predicate: str, object_id: str,
             prov_id: str, confidence: float = 1.0) -> None:
        """Convenience method to create and add a relation."""
        rel = Relation(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            prov_id=prov_id,
            confidence=confidence,
        )
        self.add_relation(rel)
    
    def build(self) -> "Program":
        """Build the immutable Program."""
        return Program(
            source=self.source,
            nodes=dict(self._nodes),
            relations=list(self._relations),
            forks=list(self._forks),
            provenance=dict(self._provenance),
            stats=dict(self._stats),
            chunk_index=dict(self._chunk_nodes),
            chunk_hashes=dict(self._chunk_hashes),
            created_at=self.created_at,
            version=self.VERSION,
        )


# =============================================================================
# PROGRAM (IMMUTABLE)
# =============================================================================

@dataclass(frozen=True)
class Program:
    """Immutable compiled program."""
    source: SourceInfo
    nodes: Dict[str, Node]
    relations: List[Relation]
    forks: List[Fork]
    provenance: Dict[str, Provenance]
    stats: Dict[str, int]
    chunk_index: Dict[str, List[str]]
    chunk_hashes: Dict[str, str]  # chunk_id -> content_hash for incremental
    created_at: str
    version: str
    
    @property
    def content_hash(self) -> str:
        """Deterministic hash of all content."""
        hasher = hashlib.sha256()
        
        for node_id in sorted(self.nodes.keys()):
            hasher.update(json.dumps(self.nodes[node_id].to_dict(), sort_keys=True).encode())
        
        for rel in sorted(self.relations, key=lambda r: r.sort_key):
            hasher.update(json.dumps(rel.to_dict(), sort_keys=True).encode())
        
        return hasher.hexdigest()[:16]
    
    def manifest(self) -> Dict[str, Any]:
        """Generate manifest for writing."""
        return {
            "axm_version": self.version,
            "ir_schema_version": IR_SCHEMA_VERSION,
            "created_at": self.created_at,
            "source": self.source.to_dict(),
            "coordinate_system": COORDINATE_SCHEMA,
            "counts": {
                "nodes": len(self.nodes),
                "relations": len(self.relations),
                "forks": len(self.forks),
                "provenance": len(self.provenance),
            },
            "stats": self.stats,
            "chunk_index": {
                chunk_id: {
                    "nodes": node_ids,
                    "content_hash": self.chunk_hashes.get(chunk_id, ""),
                }
                for chunk_id, node_ids in self.chunk_index.items()
            },
            "content_hash": self.content_hash,
        }
    
    def write(self, path: str) -> None:
        """Write program to disk as .axm directory."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        
        # Manifest
        with open(out / "manifest.json", "w") as f:
            json.dump(self.manifest(), f, indent=2, sort_keys=True)
        
        # Nodes (sorted by ID)
        with open(out / "nodes.jsonl", "w") as f:
            for node_id in sorted(self.nodes.keys()):
                f.write(json.dumps(self.nodes[node_id].to_dict(), sort_keys=True) + "\n")
        
        # Relations (sorted)
        with open(out / "relations.jsonl", "w") as f:
            for rel in sorted(self.relations, key=lambda r: r.sort_key):
                f.write(json.dumps(rel.to_dict(), sort_keys=True) + "\n")
        
        # Forks (if any)
        if self.forks:
            with open(out / "forks.jsonl", "w") as f:
                for fork in sorted(self.forks, key=lambda f: f.fork_id):
                    f.write(json.dumps(fork.to_dict(), sort_keys=True) + "\n")
        
        # Provenance (sorted)
        with open(out / "provenance.jsonl", "w") as f:
            for prov_id in sorted(self.provenance.keys()):
                f.write(json.dumps(self.provenance[prov_id].to_dict(), sort_keys=True) + "\n")
        
        print(f"✓ Written to {path}")
        print(f"  {len(self.nodes)} nodes, {len(self.relations)} relations")
    
    @classmethod
    def load(cls, path: str) -> "Program":
        """Load program from disk with validation."""
        p = Path(path)
        
        if not p.is_dir():
            raise ValueError(f"Not a directory: {path}")
        
        # Manifest
        manifest_path = p / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Missing manifest.json in {path}")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Validate schema version
        ir_version = manifest.get("ir_schema_version", "unknown")
        if ir_version != IR_SCHEMA_VERSION:
            # Could add migration logic here in future
            pass  # For now, accept any version
        
        # Source
        source = SourceInfo.from_dict(manifest["source"])
        
        # Nodes (with validation)
        nodes = {}
        nodes_path = p / "nodes.jsonl"
        if not nodes_path.exists():
            raise ValueError(f"Missing nodes.jsonl in {path}")
        
        line_num = 0
        with open(nodes_path) as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    node = Node.from_dict(data)
                    nodes[node.id] = node
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    raise ValueError(f"Invalid node at line {line_num}: {e}") from e
        
        # Relations (with validation)
        relations = []
        relations_path = p / "relations.jsonl"
        if relations_path.exists():
            line_num = 0
            with open(relations_path) as f:
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        rel = Relation.from_dict(data)
                        relations.append(rel)
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        raise ValueError(f"Invalid relation at line {line_num}: {e}") from e
        
        # Forks
        forks = []
        forks_path = p / "forks.jsonl"
        if forks_path.exists():
            with open(forks_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        forks.append(Fork.from_dict(json.loads(line)))
        
        # Provenance
        provenance = {}
        prov_path = p / "provenance.jsonl"
        if prov_path.exists():
            with open(prov_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prov = Provenance.from_dict(json.loads(line))
                        provenance[prov.prov_id] = prov
        
        # Chunk index and hashes
        chunk_index = {}
        chunk_hashes = {}
        for chunk_id, info in manifest.get("chunk_index", {}).items():
            if isinstance(info, dict):
                chunk_index[chunk_id] = info.get("nodes", [])
                if "content_hash" in info:
                    chunk_hashes[chunk_id] = info["content_hash"]
            else:
                # Legacy format: info is just the node list
                chunk_index[chunk_id] = info if isinstance(info, list) else []
        
        return cls(
            source=source,
            nodes=nodes,
            relations=relations,
            forks=forks,
            provenance=provenance,
            stats=manifest.get("stats", {}),
            chunk_index=chunk_index,
            chunk_hashes=chunk_hashes,
            created_at=manifest["created_at"],
            version=manifest.get("axm_version", "unknown"),
        )


# =============================================================================
# CONVENIENCE
# =============================================================================

def load(path: str) -> Program:
    """Load a compiled .axm program."""
    return Program.load(path)
