
from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple, List

from axm.coords import Coord
from axm.ir import Node, Derivation, Provenance
from axm.program import Program

def confidence_summary_pass(program: Program, major: int = 99, type_: int = 90, subtype: int = 1) -> Tuple[Program, Dict]:
    """Create per-chunk confidence summary nodes and derivations."""
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    prov_id = _make_prov_id(program, "derive:confidence", now)

    prov = Provenance(
        prov_id=prov_id,
        chunk_id="__DERIVE__",
        extractor="derive:confidence",
        timestamp=now,
        tier=2,
        confidence=1.0,
        source_start=None,
        source_end=None,
        model=None,
        prompt_hash=None,
    )

    provenance = dict(program.provenance)
    provenance[prov_id] = prov

    nodes = dict(program.nodes)
    derivations = list(program.derivations)

    created_nodes = 0
    created_derivations = 0

    instance = 1

    for chunk_id, node_ids in program.chunk_index.items():
        confs: List[float] = []
        for nid in node_ids:
            n = nodes.get(nid)
            if not n:
                continue
            p = provenance.get(n.prov_id)
            if p is None:
                continue
            confs.append(float(getattr(p, "confidence", 1.0)))
        if not confs:
            continue

        mean_conf = sum(confs) / float(len(confs))

        coord = Coord(major=major, type_=type_, subtype=subtype, instance=instance)
        instance += 1

        summary = Node(
            coord=coord,
            label="chunk.confidence.mean",
            prov_id=prov_id,
            value=round(mean_conf, 6),
            unit=None,
            metadata={"chunk_id": chunk_id, "n": len(confs)},
        )

        nodes[summary.id] = summary
        created_nodes += 1

        derivations.append(
            Derivation(
                result_id=summary.id,
                operands=tuple(node_ids),
                operator="mean_provenance_confidence",
                prov_id=prov_id,
                confidence=1.0,
            )
        )
        created_derivations += 1

    new_program = Program(
        source=program.source,
        nodes=nodes,
        relations=list(program.relations),
        derivations=derivations,
        temporal_alignments=list(program.temporal_alignments),
        forks=list(program.forks),
        provenance=provenance,
        stats=dict(program.stats),
        chunk_index=dict(program.chunk_index),
        chunk_hashes=dict(program.chunk_hashes),
        created_at=program.created_at,
    )

    stats = {"created_nodes": created_nodes, "created_derivations": created_derivations, "prov_id": prov_id}
    return new_program, stats

def _make_prov_id(program: Program, extractor: str, ts: str) -> str:
    import hashlib
    h = (program.version + "|" + extractor + "|" + ts).encode("utf-8")
    return hashlib.sha256(h).hexdigest()[:16]
