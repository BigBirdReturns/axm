"""
AXM Space - Query Engine (v0.4)

Queryable view of a compiled Program.
All queries are deterministic. No LLM. Pure geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from .coords import Coord, Major
from .ir import Node, Relation, Fork
from .program import Program


@dataclass
class QueryResult:
    """Result of a query with optional path information."""
    node: Node
    distance: float = 0.0
    path: List[str] = None
    
    def __post_init__(self):
        if self.path is None:
            self.path = []


class Space:
    """
    Queryable semantic space.
    
    Usage:
        space = Space(program)
        for node in space.query(major=Major.QUANTITY):
            print(node.label, node.value)
    """
    
    def __init__(self, program: Program):
        self.program = program
        
        # Index by major category
        self._by_major: Dict[int, List[Node]] = {}
        for node in program.nodes.values():
            major = node.coord.major
            if major not in self._by_major:
                self._by_major[major] = []
            self._by_major[major].append(node)
        
        # Index relations
        self._outgoing: Dict[str, List[Relation]] = {}
        self._incoming: Dict[str, List[Relation]] = {}
        for rel in program.relations:
            if rel.subject_id not in self._outgoing:
                self._outgoing[rel.subject_id] = []
            self._outgoing[rel.subject_id].append(rel)
            
            if rel.object_id not in self._incoming:
                self._incoming[rel.object_id] = []
            self._incoming[rel.object_id].append(rel)
    
    # =========================================================================
    # BASIC QUERIES
    # =========================================================================
    
    def get(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.program.nodes.get(node_id)
    
    def all_nodes(self) -> Iterator[Node]:
        """Iterate all nodes."""
        return iter(self.program.nodes.values())
    
    def query(
        self,
        major: Optional[int] = None,
        type_: Optional[int] = None,
        subtype: Optional[int] = None,
        label_contains: Optional[str] = None,
        value_gt: Optional[float] = None,
        value_lt: Optional[float] = None,
    ) -> Iterator[Node]:
        """Query nodes by criteria. All criteria are AND-ed."""
        if major is not None:
            candidates = self._by_major.get(major, [])
        else:
            candidates = self.program.nodes.values()
        
        for node in candidates:
            if type_ is not None and node.coord.type_ != type_:
                continue
            if subtype is not None and node.coord.subtype != subtype:
                continue
            if label_contains is not None:
                if label_contains.lower() not in node.label.lower():
                    continue
            if node.value is not None:
                try:
                    val = float(node.value)
                    if value_gt is not None and val <= value_gt:
                        continue
                    if value_lt is not None and val >= value_lt:
                        continue
                except (ValueError, TypeError):
                    if value_gt is not None or value_lt is not None:
                        continue
            elif value_gt is not None or value_lt is not None:
                continue
            
            yield node
    
    def find(self, **kwargs) -> List[Node]:
        """Return list instead of iterator."""
        return list(self.query(**kwargs))
    
    def first(self, **kwargs) -> Optional[Node]:
        """Return first match or None."""
        for node in self.query(**kwargs):
            return node
        return None
    
    # =========================================================================
    # COORDINATE QUERIES
    # =========================================================================
    
    def by_prefix(self, prefix: str) -> Iterator[Node]:
        """Find nodes whose ID starts with prefix."""
        for node_id, node in self.program.nodes.items():
            if node_id.startswith(prefix):
                yield node
    
    def neighbors(
        self,
        node_or_id: str | Node | Coord,
        radius: float = 1.0,
        weights: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.1),
    ) -> Iterator[QueryResult]:
        """Find nodes within semantic distance."""
        if isinstance(node_or_id, str):
            ref_node = self.get(node_or_id)
            if not ref_node:
                return
            ref_coord = ref_node.coord
        elif isinstance(node_or_id, Node):
            ref_coord = node_or_id.coord
        else:
            ref_coord = node_or_id
        
        results = []
        for node in self.program.nodes.values():
            dist = ref_coord.distance(node.coord, weights)
            if dist <= radius:
                results.append(QueryResult(node=node, distance=dist))
        
        results.sort(key=lambda r: r.distance)
        yield from results
    
    # =========================================================================
    # GRAPH TRAVERSAL
    # =========================================================================
    
    def outgoing(self, node_id: str) -> Iterator[Relation]:
        """Get outgoing relations from a node."""
        yield from self._outgoing.get(node_id, [])
    
    def incoming(self, node_id: str) -> Iterator[Relation]:
        """Get incoming relations to a node."""
        yield from self._incoming.get(node_id, [])
    
    def traverse(
        self,
        start_id: str,
        predicate: Optional[str] = None,
        max_depth: int = 3,
        direction: str = "out",
    ) -> Iterator[QueryResult]:
        """Traverse graph from starting node."""
        visited: Set[str] = set()
        frontier = [(start_id, [start_id], 0)]
        
        while frontier:
            node_id, path, depth = frontier.pop(0)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            node = self.get(node_id)
            if node and node_id != start_id:
                yield QueryResult(node=node, path=path)
            
            if depth >= max_depth:
                continue
            
            if direction in ("out", "both"):
                for rel in self._outgoing.get(node_id, []):
                    if predicate is None or rel.predicate == predicate:
                        frontier.append((rel.object_id, path + [rel.object_id], depth + 1))
            
            if direction in ("in", "both"):
                for rel in self._incoming.get(node_id, []):
                    if predicate is None or rel.predicate == predicate:
                        frontier.append((rel.subject_id, path + [rel.subject_id], depth + 1))
    
    def path(self, start_id: str, end_id: str, max_depth: int = 5) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        if start_id == end_id:
            return [start_id]
        
        visited: Set[str] = set()
        frontier = [(start_id, [start_id])]
        
        for _ in range(max_depth):
            next_frontier = []
            
            for node_id, p in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                for rel in self._outgoing.get(node_id, []):
                    if rel.object_id == end_id:
                        return p + [end_id]
                    if rel.object_id not in visited:
                        next_frontier.append((rel.object_id, p + [rel.object_id]))
                
                for rel in self._incoming.get(node_id, []):
                    if rel.subject_id == end_id:
                        return p + [end_id]
                    if rel.subject_id not in visited:
                        next_frontier.append((rel.subject_id, p + [rel.subject_id]))
            
            frontier = next_frontier
            if not frontier:
                break
        
        return None
    
    # =========================================================================
    # AGGREGATION
    # =========================================================================
    
    def count(self, major: Optional[int] = None) -> int:
        if major is None:
            return len(self.program.nodes)
        return len(self._by_major.get(major, []))
    
    def sum_values(self, major: Optional[int] = None, type_: Optional[int] = None, label_contains: Optional[str] = None) -> float:
        total = 0.0
        for node in self.query(major=major, type_=type_, label_contains=label_contains):
            if node.value is not None:
                try:
                    total += float(node.value)
                except (ValueError, TypeError):
                    pass
        return total
    
    def distinct_labels(self, major: Optional[int] = None) -> Set[str]:
        return {node.label for node in self.query(major=major)}


def query(program: Program) -> Space:
    """Create a queryable space from a program."""
    return Space(program)
