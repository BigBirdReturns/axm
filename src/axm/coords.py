"""
AXM Coordinate System - The Foundation (v0.4)

This module defines the semantic coordinate space. It is the ONLY place 
where coordinate semantics are defined.

SCHEMA VERSION: 0.4 (frozen)

Coordinates are 4-dimensional addresses in semantic space:
    [major, type, subtype, instance]

This is NOT embeddings. This is structural addressing.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Tuple


# =============================================================================
# SCHEMA VERSION - FROZEN
# =============================================================================

IR_SCHEMA_VERSION = "0.5"


# =============================================================================
# MAJOR CATEGORIES (Axis 0) - FROZEN
# =============================================================================

class Major(IntEnum):
    """The 8 major semantic categories. FROZEN in v0.4."""
    ENTITY = 1      # Who or what
    ACTION = 2      # What happened
    PROPERTY = 3    # Attributes
    RELATION = 4    # Reserved for edges
    LOCATION = 5    # Where
    TIME = 6        # When
    QUANTITY = 7    # How much
    ABSTRACT = 8    # Claims, concepts


# =============================================================================
# TYPE ENUMS (Axis 1)
# =============================================================================

class EntityType(IntEnum):
    ORGANIZATION = 1
    PERSON = 2
    PRODUCT = 3
    SERVICE = 4
    DOCUMENT = 5
    SYSTEM = 6


class ActionType(IntEnum):
    EVENT = 1
    TRANSACTION = 2
    PROCESS = 3
    ANNOUNCEMENT = 4
    DECISION = 5


class PropertyType(IntEnum):
    ATTRIBUTE = 1
    STATE = 2
    FEATURE = 3
    STATUS = 4


class LocationType(IntEnum):
    ADDRESS = 1
    CITY = 2
    REGION = 3
    COUNTRY = 4
    COORDINATE = 5


class TimeType(IntEnum):
    DATE = 1
    PERIOD = 2
    TIMESTAMP = 3
    FISCAL_PERIOD = 4


class QuantityType(IntEnum):
    FINANCIAL = 1
    RATIO = 2
    COUNT = 3
    MEASURE = 4


class AbstractType(IntEnum):
    CLAIM = 1
    OPINION = 2
    NARRATIVE = 3
    CONCEPT = 4


# =============================================================================
# COORDINATE SCHEMA - FROZEN
# =============================================================================

COORDINATE_SCHEMA = {
    "version": IR_SCHEMA_VERSION,
    "dimensions": 4,
    "axes": [
        {
            "index": 0,
            "name": "major",
            "description": "Major semantic category",
            "range": [1, 8],
        },
        {
            "index": 1,
            "name": "type",
            "description": "Type within major category",
            "range": [0, 99],
        },
        {
            "index": 2,
            "name": "subtype",
            "description": "Subtype refinement",
            "range": [0, 99],
        },
        {
            "index": 3,
            "name": "instance",
            "description": "Unique instance counter",
            "range": [0, 9999],
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
    "id_format": "MM-TT-SS-IIII",
}


# =============================================================================
# COORDINATE CLASS
# =============================================================================

@dataclass(frozen=True, order=True)
class Coord:
    """
    A semantic coordinate - an address in meaning space.
    Immutable and orderable for deterministic sorting.
    """
    major: int
    type_: int
    subtype: int
    instance: int
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self) -> None:
        """Validate coordinate ranges."""
        if not isinstance(self.major, int):
            raise TypeError(f"Major must be int, got {type(self.major)}")
        if not isinstance(self.type_, int):
            raise TypeError(f"Type must be int, got {type(self.type_)}")
        if not isinstance(self.subtype, int):
            raise TypeError(f"Subtype must be int, got {type(self.subtype)}")
        if not isinstance(self.instance, int):
            raise TypeError(f"Instance must be int, got {type(self.instance)}")
        
        if not (1 <= self.major <= 8):
            raise ValueError(f"Major must be 1-8, got {self.major}")
        if not (0 <= self.type_ <= 99):
            raise ValueError(f"Type must be 0-99, got {self.type_}")
        if not (0 <= self.subtype <= 99):
            raise ValueError(f"Subtype must be 0-99, got {self.subtype}")
        if not (0 <= self.instance <= 9999):
            raise ValueError(f"Instance must be 0-9999, got {self.instance}")
    
    def to_id(self) -> str:
        """Format as canonical ID string: MM-TT-SS-IIII"""
        return f"{self.major:02d}-{self.type_:02d}-{self.subtype:02d}-{self.instance:04d}"
    
    def to_list(self) -> List[int]:
        """Convert to list form."""
        return [self.major, self.type_, self.subtype, self.instance]
    
    @classmethod
    def from_id(cls, id_str: str) -> "Coord":
        """Parse from ID string with validation."""
        if not isinstance(id_str, str):
            raise TypeError(f"ID must be string, got {type(id_str)}")
        
        parts = id_str.split("-")
        if len(parts) != 4:
            raise ValueError(f"Invalid coord ID format: {id_str}")
        
        try:
            return cls(
                major=int(parts[0]),
                type_=int(parts[1]),
                subtype=int(parts[2]),
                instance=int(parts[3]),
            )
        except ValueError as e:
            raise ValueError(f"Invalid coord ID: {id_str}") from e
    
    @classmethod
    def from_list(cls, coords: List[int]) -> "Coord":
        """Create from list form with validation."""
        if not isinstance(coords, (list, tuple)):
            raise TypeError(f"Coords must be list/tuple, got {type(coords)}")
        if len(coords) != 4:
            raise ValueError(f"Coords must have 4 elements, got {len(coords)}")
        
        return cls(coords[0], coords[1], coords[2], coords[3])
    
    def distance(self, other: "Coord", weights: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.1)) -> float:
        """Compute weighted Euclidean distance."""
        return math.sqrt(sum(
            w * (a - b) ** 2
            for w, a, b in zip(weights, self.to_list(), other.to_list())
        ))
    
    def same_major(self, other: "Coord") -> bool:
        return self.major == other.major
    
    def same_type(self, other: "Coord") -> bool:
        return self.major == other.major and self.type_ == other.type_


# =============================================================================
# COORDINATE ASSIGNER
# =============================================================================

class CoordAssigner:
    """
    Assigns coordinates to semantic extractions.
    Thread-safe via internal locking.
    """
    
    def __init__(self):
        self._counters: Dict[Tuple[int, int, int], int] = {}
        self._lock = threading.Lock()
    
    def next(self, major: int, type_: int = 1, subtype: int = 1) -> Coord:
        """Get next coordinate for given major/type/subtype."""
        with self._lock:
            key = (major, type_, subtype)
            instance = self._counters.get(key, 0) + 1
            self._counters[key] = instance
            return Coord(major, type_, subtype, instance)
    
    def next_entity(self, type_: EntityType = EntityType.ORGANIZATION) -> Coord:
        return self.next(Major.ENTITY, type_)
    
    def next_quantity(self, type_: QuantityType = QuantityType.FINANCIAL) -> Coord:
        return self.next(Major.QUANTITY, type_)
    
    def next_time(self, type_: TimeType = TimeType.DATE) -> Coord:
        return self.next(Major.TIME, type_)
    
    def next_abstract(self, type_: AbstractType = AbstractType.CLAIM) -> Coord:
        return self.next(Major.ABSTRACT, type_)
    
    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
    
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {f"{k[0]}-{k[1]}-{k[2]}": v for k, v in self._counters.items()}
