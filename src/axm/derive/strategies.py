
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .temporal import temporal_alignment_pass
from .confidence import confidence_summary_pass

@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    run: Callable

STRATEGIES: Dict[str, Strategy] = {
    "temporal": Strategy(
        name="temporal",
        description="Attach TemporalAlignment edges by chunk using detected time reference nodes.",
        run=temporal_alignment_pass,
    ),
    "confidence": Strategy(
        name="confidence",
        description="Compute per-chunk confidence summary nodes and derivations (mean provenance confidence).",
        run=confidence_summary_pass,
    ),
}
