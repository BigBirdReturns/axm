
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Tuple, Optional, List

from axm.coords import Coord
from axm.ir import TemporalAlignment, Provenance
from axm.program import Program

_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
]

def _parse_dt(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    if not isinstance(value, str):
        value = str(value)
    s = value.strip()
    if not s:
        return None

    # Fast ISO date check
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s + "T00:00:00Z"

    for fmt in _DATE_PATTERNS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return None

def _looks_like_time_node(label: str, value) -> bool:
    lab = (label or "").lower()
    if any(k in lab for k in ["date", "time", "timestamp", "as of", "period", "quarter", "year"]):
        if _parse_dt(value) is not None:
            return True
    # If label does not give it away, still accept strict date-like values
    if _parse_dt(value) is not None and isinstance(value, str) and len(value.strip()) <= 25:
        return True
    return False

def temporal_alignment_pass(program: Program, operator: str = "at") -> Tuple[Program, Dict]:
    """Create TemporalAlignment edges per chunk by attaching nodes to a detected reference time node."""
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    prov_id = _make_prov_id(program, "derive:temporal", now)

    prov = Provenance(
        prov_id=prov_id,
        chunk_id="__DERIVE__",
        extractor="derive:temporal",
        timestamp=now,
        tier=2,
        confidence=1.0,
        source_start=None,
        source_end=None,
        model=None,
        prompt_hash=None,
    )

    # Copy immutable structures
    provenance = dict(program.provenance)
    provenance[prov_id] = prov

    alignments: List[TemporalAlignment] = list(program.temporal_alignments)

    created = 0
    chunks_with_ref = 0

    for chunk_id, node_ids in program.chunk_index.items():
        ref_id = None
        # Find first time-like node in chunk
        for nid in node_ids:
            n = program.nodes.get(nid)
            if n and _looks_like_time_node(n.label, n.value):
                ref_id = nid
                break
        if not ref_id:
            continue
        chunks_with_ref += 1

        for nid in node_ids:
            if nid == ref_id:
                continue
            n = program.nodes.get(nid)
            if not n:
                continue
            # Do not align other time nodes to avoid noisy duplicates
            if _looks_like_time_node(n.label, n.value):
                continue

            # Confidence is bounded by both nodes' provenance confidence if present
            subj_prov = provenance.get(n.prov_id)
            ref_prov = provenance.get(program.nodes[ref_id].prov_id)
            conf = 1.0
            if subj_prov is not None:
                conf = min(conf, getattr(subj_prov, "confidence", 1.0))
            if ref_prov is not None:
                conf = min(conf, getattr(ref_prov, "confidence", 1.0))

            alignments.append(
                TemporalAlignment(
                    subject_id=nid,
                    reference_id=ref_id,
                    operator=operator,
                    prov_id=prov_id,
                    offset=None,
                    confidence=conf,
                )
            )
            created += 1

    new_program = Program(
        source=program.source,
        nodes=dict(program.nodes),
        relations=list(program.relations),
        derivations=list(program.derivations),
        temporal_alignments=alignments,
        forks=list(program.forks),
        provenance=provenance,
        stats=dict(program.stats),
        chunk_index=dict(program.chunk_index),
        chunk_hashes=dict(program.chunk_hashes),
        created_at=program.created_at,
    )

    stats = {"created_alignments": created, "chunks_with_reference": chunks_with_ref, "prov_id": prov_id}
    return new_program, stats

def _make_prov_id(program: Program, extractor: str, ts: str) -> str:
    h = (program.version + "|" + extractor + "|" + ts).encode("utf-8")
    import hashlib
    return hashlib.sha256(h).hexdigest()[:16]
