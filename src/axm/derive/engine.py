
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from axm.program import Program
from .strategies import STRATEGIES

@dataclass(frozen=True)
class DeriveResult:
    program: Program
    kind: str
    stats: Dict[str, Any]

class DerivationEngine:
    """Run derivation passes that transform a Program into a new Program."""

    def __init__(self, program: Program):
        self.program = program

    def run(self, kind: str, **kwargs) -> DeriveResult:
        if kind not in STRATEGIES:
            allowed = ", ".join(sorted(STRATEGIES.keys()))
            raise ValueError(f"Unknown derivation kind: {kind}. Allowed: {allowed}")
        new_program, stats = STRATEGIES[kind].run(self.program, **kwargs)
        return DeriveResult(program=new_program, kind=kind, stats=stats)
