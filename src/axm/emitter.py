"""
AXM Emitter - Convert Extractions to IR (v0.4)

Creates nodes and relations using ProgramBuilder's public API.
No direct access to internal state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .coords import Coord, Major, QuantityType, AbstractType
from .ir import Node, Relation, Provenance, SourceInfo
from .program import ProgramBuilder, Program
from .lexer import Chunk
from .parser import Extraction, LLMRequest


ENTITY_TO_COORD = {
    "Organization": (Major.ENTITY, 1, 1),
    "Person": (Major.ENTITY, 2, 1),
    "Product": (Major.ENTITY, 3, 1),
    "Event": (Major.ACTION, 1, 1),
    "Transaction": (Major.ACTION, 2, 1),
    "KeyValue": (Major.PROPERTY, 1, 1),
    "ListItem": (Major.PROPERTY, 2, 1),
    "Date": (Major.TIME, 1, 1),
    "Period": (Major.TIME, 2, 1),
    "Money": (Major.QUANTITY, 1, 1),
    "Percentage": (Major.QUANTITY, 2, 1),
    "Count": (Major.QUANTITY, 3, 1),
    "Claim": (Major.ABSTRACT, 1, 1),
    "ClaimType": (Major.ABSTRACT, 1, 2),
}


class Emitter:
    """Converts extractions to Program IR using public API."""
    
    def __init__(self, source: SourceInfo):
        self.source = source
    
    def emit(
        self,
        chunks: List[Chunk],
        extractions: List[Extraction],
        llm_responses: Optional[Dict[str, List[Dict]]] = None,
    ) -> Program:
        builder = ProgramBuilder(self.source)
        ext_to_node: Dict[str, str] = {}
        chunk_prov: Dict[str, str] = {}
        
        for chunk in chunks:
            # Register chunk hash for incremental compilation
            builder.register_chunk_hash(chunk.chunk_id, chunk.content_hash)
            
            prov_id = f"prov_{chunk.chunk_id}"
            prov = Provenance(
                prov_id=prov_id,
                chunk_id=chunk.chunk_id,
                extractor=f"tier{chunk.tier}:{chunk.chunk_type.name.lower()}",
                timestamp=datetime.utcnow().isoformat() + "Z",
                tier=chunk.tier,
                confidence=chunk.confidence,
                source_start=chunk.start,
                source_end=chunk.end,
            )
            builder.add_provenance(prov)
            chunk_prov[chunk.chunk_id] = prov_id
        
        for ext in extractions:
            node = self._extraction_to_node(ext, chunk_prov, builder)
            node_id = builder.add_node(node, deduplicate=True, chunk_id=ext.chunk_id)
            ext_to_node[ext.ext_id] = node_id
        
        for ext in extractions:
            self._emit_relations(ext, ext_to_node, chunk_prov, builder)
        
        if llm_responses:
            self._emit_llm_responses(llm_responses, chunk_prov, builder)
        
        return builder.build()
    
    def _extraction_to_node(self, ext: Extraction, chunk_prov: Dict[str, str], builder: ProgramBuilder) -> Node:
        mapping = ENTITY_TO_COORD.get(ext.entity_type)
        if mapping:
            major, type_, subtype = mapping
        else:
            major, type_, subtype = ext.major, ext.type_, ext.subtype
        
        coord = builder.coords.next(major, type_, subtype)
        prov_id = chunk_prov.get(ext.chunk_id, "prov_unknown")
        
        return Node(
            coord=coord,
            label=ext.label,
            prov_id=prov_id,
            value=ext.value,
            unit=ext.unit,
            metadata={"entity_type": ext.entity_type, "extractor": ext.extractor},
        )
    
    def _emit_relations(self, ext: Extraction, ext_to_node: Dict[str, str], chunk_prov: Dict[str, str], builder: ProgramBuilder) -> None:
        subject_id = ext_to_node.get(ext.ext_id)
        if not subject_id:
            return
        
        prov_id = chunk_prov.get(ext.chunk_id, "prov_unknown")
        
        for predicate, target_ext_id in ext.relations:
            object_id = ext_to_node.get(target_ext_id)
            if object_id and object_id != subject_id:
                builder.link(subject_id, predicate, object_id, prov_id, ext.confidence)
    
    def _emit_llm_responses(self, llm_responses: Dict[str, List[Dict]], chunk_prov: Dict[str, str], builder: ProgramBuilder) -> None:
        for req_id, responses in llm_responses.items():
            for i, resp in enumerate(responses):
                prov_id = f"prov_llm_{req_id}_{i}"
                prov = Provenance(
                    prov_id=prov_id,
                    chunk_id=req_id,
                    extractor="tier3:llm",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    tier=3,
                    confidence=resp.get("confidence", 0.7),
                )
                builder.add_provenance(prov)
                
                subject = resp.get("subject", "claim")
                obj = resp.get("object", "")
                label = f"{subject}: {obj}"
                
                coord = builder.coords.next(Major.ABSTRACT, AbstractType.CLAIM, 1)
                node = Node(
                    coord=coord,
                    label=label,
                    prov_id=prov_id,
                    metadata={"entity_type": "Claim", "llm_response": resp},
                )
                
                claim_node_id = builder.add_node(node, deduplicate=True, chunk_id=req_id)
                
                if obj:
                    existing_id = builder.find_by_label(obj)
                    
                    if existing_id:
                        value_node_id = existing_id
                    else:
                        value_coord = builder.coords.next(Major.QUANTITY, 1, 1)
                        value_node = Node(
                            coord=value_coord,
                            label=obj,
                            prov_id=prov_id,
                            value=self._parse_value(obj),
                            unit=self._parse_unit(obj),
                        )
                        value_node_id = builder.add_node(value_node, deduplicate=True, chunk_id=req_id)
                    
                    if value_node_id != claim_node_id:
                        builder.link(claim_node_id, "HAS_VALUE", value_node_id, prov_id)
    
    def _parse_value(self, s: str) -> Optional[float]:
        if not s:
            return None
        
        cleaned = s.replace('$', '').replace(',', '').replace('%', '').strip()
        multiplier = 1
        
        if 'billion' in cleaned.lower() or cleaned.endswith('B'):
            multiplier = 1e9
            cleaned = cleaned.lower().replace('billion', '').replace('b', '').strip()
        elif 'million' in cleaned.lower() or cleaned.endswith('M'):
            multiplier = 1e6
            cleaned = cleaned.lower().replace('million', '').replace('m', '').strip()
        
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None
    
    def _parse_unit(self, s: str) -> Optional[str]:
        if '$' in s:
            return "USD"
        if '%' in s:
            return "%"
        return None
