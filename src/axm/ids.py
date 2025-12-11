"""
AXM ID Generator - Thread-safe centralized ID generation

Replaces scattered class-level counters with a single thread-safe generator.
"""

from __future__ import annotations

import threading
from typing import Dict


class IDGenerator:
    """
    Thread-safe ID generator with namespaced counters.
    
    Usage:
        ids = IDGenerator()
        ids.next("chunk")   # "chunk_000001"
        ids.next("chunk")   # "chunk_000002"
        ids.next("ext")     # "ext_000001"
    """
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def next(self, prefix: str, width: int = 6) -> str:
        """Get next ID for prefix."""
        with self._lock:
            count = self._counters.get(prefix, 0) + 1
            self._counters[prefix] = count
            return f"{prefix}_{count:0{width}d}"
    
    def current(self, prefix: str) -> int:
        """Get current counter value (without incrementing)."""
        with self._lock:
            return self._counters.get(prefix, 0)
    
    def reset(self, prefix: str = None) -> None:
        """Reset counter(s). If prefix is None, reset all."""
        with self._lock:
            if prefix is None:
                self._counters.clear()
            elif prefix in self._counters:
                del self._counters[prefix]
    
    def stats(self) -> Dict[str, int]:
        """Get all counter values."""
        with self._lock:
            return dict(self._counters)


# Global instance for convenience
# Each Compiler instance can also create its own
_global_ids = IDGenerator()


def next_id(prefix: str) -> str:
    """Get next ID from global generator."""
    return _global_ids.next(prefix)


def reset_ids() -> None:
    """Reset global generator."""
    _global_ids.reset()
