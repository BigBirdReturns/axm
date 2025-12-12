"""
AXM Compiler - The Complete Pipeline (v0.5)

Features:
- REAL incremental compilation (not just API)
- File size and path validation
- Configurable executors
- Chunk-level change detection
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .coords import COORDINATE_SCHEMA, IR_SCHEMA_VERSION, Major
from .ir import SourceInfo, Node, Relation, Provenance
from .program import Program, ProgramBuilder
from .space import Space
from .lexer import Lexer, Chunk
from .parser import Parser, LLMRequest, Extraction
from .emitter import Emitter
from .executor import MockExecutor, RetryExecutor, LLMResult, get_executor


# =============================================================================
# LIMITS
# =============================================================================

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB default
MAX_PATH_LENGTH = 4096


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    """Compiler configuration."""
    max_tier: int = 3
    executor: Optional[Callable] = None
    deduplicate: bool = True
    max_file_size: int = MAX_FILE_SIZE
    
    @classmethod
    def default(cls) -> "Config":
        return cls(executor=RetryExecutor(MockExecutor()))
    
    @classmethod
    def no_llm(cls) -> "Config":
        return cls(max_tier=2, executor=None)
    
    @classmethod
    def with_ollama(cls, model: str = "llama3") -> "Config":
        from .executor import OllamaExecutor
        return cls(executor=RetryExecutor(OllamaExecutor(model=model)))
    
    @classmethod
    def with_anthropic(cls, model: str = "claude-sonnet-4-20250514") -> "Config":
        from .executor import AnthropicExecutor
        return cls(executor=RetryExecutor(AnthropicExecutor(model=model)))


# =============================================================================
# INCREMENTAL COMPILATION STATE
# =============================================================================

@dataclass
class ChunkState:
    """State of a chunk for incremental comparison."""
    chunk_id: str
    content_hash: str
    node_ids: List[str]
    relation_keys: List[Tuple[str, str, str]]  # (subj, pred, obj)
    provenance_ids: List[str]


@dataclass 
class IncrementalPlan:
    """Plan for incremental recompilation."""
    reuse_chunks: List[str]      # Chunks to copy from previous
    recompile_chunks: List[str]  # Chunks that need fresh extraction
    remove_chunks: List[str]     # Chunks no longer present
    
    # Stats
    reused_nodes: int = 0
    recompiled_nodes: int = 0
    removed_nodes: int = 0
    
    @property
    def efficiency(self) -> float:
        """Percentage of work avoided."""
        total = self.reused_nodes + self.recompiled_nodes + self.removed_nodes
        if total == 0:
            return 1.0
        return self.reused_nodes / total


# =============================================================================
# COMPILER
# =============================================================================

class Compiler:
    """The AXM Semantic Compiler with incremental support."""
    
    VERSION = "0.5.3"
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self.lexer = Lexer()
        self.parser = Parser(max_tier=self.config.max_tier)
    
    def compile(self, source: str, source_type: str = "auto") -> Program:
        """Compile a document into a Program."""
        content, source_info = self._load_source(source, source_type)
        chunks = self.lexer.lex(content)
        extractions, llm_requests = self.parser.parse(
            chunks, generate_llm=self.config.executor is not None
        )
        llm_responses = self._execute_llm(llm_requests)
        emitter = Emitter(source_info)
        return emitter.emit(chunks, extractions, llm_responses or None)
    
    def compile_incremental(
        self,
        source: str,
        previous: Program,
        source_type: str = "auto",
    ) -> Tuple[Program, IncrementalPlan]:
        """
        Recompile only changed chunks.
        
        Returns (new_program, plan) where plan shows what was reused/recompiled.
        """
        content, source_info = self._load_source(source, source_type)
        
        # Lex new document
        new_chunks = self.lexer.lex(content)
        
        # Build hash maps
        prev_chunk_hashes = self._get_chunk_hashes(previous)
        new_chunk_hashes = {c.chunk_id: c.content_hash for c in new_chunks}
        
        # Compute incremental plan
        plan = self._compute_plan(new_chunks, previous, prev_chunk_hashes)
        
        # Build new program
        builder = ProgramBuilder(source_info)
        
        # 1. Copy unchanged chunks from previous
        self._copy_unchanged(builder, previous, plan.reuse_chunks, prev_chunk_hashes)
        
        # 2. Recompile changed chunks
        chunks_to_compile = [c for c in new_chunks if c.chunk_id in plan.recompile_chunks]
        
        if chunks_to_compile:
            extractions, llm_requests = self.parser.parse(
                chunks_to_compile, 
                generate_llm=self.config.executor is not None
            )
            llm_responses = self._execute_llm(llm_requests)
            
            # Emit new nodes
            self._emit_to_builder(builder, chunks_to_compile, extractions, llm_responses)
            plan.recompiled_nodes = builder.node_count() - plan.reused_nodes
        
        program = builder.build()
        return program, plan
    
    def _compute_plan(
        self,
        new_chunks: List[Chunk],
        previous: Program,
        prev_hashes: Dict[str, str],
    ) -> IncrementalPlan:
        """Determine what to reuse, recompile, remove."""
        
        # Map content_hash -> chunk_id for previous
        prev_hash_to_chunk: Dict[str, str] = {h: cid for cid, h in prev_hashes.items()}
        
        # Map content_hash -> chunk for new
        new_hash_to_chunk: Dict[str, Chunk] = {c.content_hash: c for c in new_chunks}
        
        reuse = []
        recompile = []
        
        # Check each new chunk
        matched_prev_chunks: Set[str] = set()
        
        for chunk in new_chunks:
            if chunk.content_hash in prev_hash_to_chunk:
                # This content existed before - reuse
                prev_chunk_id = prev_hash_to_chunk[chunk.content_hash]
                reuse.append(prev_chunk_id)
                matched_prev_chunks.add(prev_chunk_id)
            else:
                # New or changed content - recompile
                recompile.append(chunk.chunk_id)
        
        # Find removed chunks (in previous but not matched)
        all_prev_chunks = set(prev_hashes.keys())
        remove = list(all_prev_chunks - matched_prev_chunks)
        
        # Count nodes
        reused_nodes = sum(
            len(previous.chunk_index.get(cid, []))
            for cid in reuse
        )
        removed_nodes = sum(
            len(previous.chunk_index.get(cid, []))
            for cid in remove
        )
        
        return IncrementalPlan(
            reuse_chunks=reuse,
            recompile_chunks=recompile,
            remove_chunks=remove,
            reused_nodes=reused_nodes,
            removed_nodes=removed_nodes,
        )
    
    def _get_chunk_hashes(self, program: Program) -> Dict[str, str]:
        """Extract chunk content hashes from program."""
        return dict(program.chunk_hashes)
    
    def _copy_unchanged(
        self,
        builder: ProgramBuilder,
        previous: Program,
        chunk_ids: List[str],
        prev_hashes: Dict[str, str],
    ) -> None:
        """Copy nodes and relations from unchanged chunks."""
        
        # Collect node IDs to copy
        node_ids_to_copy: Set[str] = set()
        for chunk_id in chunk_ids:
            node_ids = previous.chunk_index.get(chunk_id, [])
            node_ids_to_copy.update(node_ids)
            
            # Register chunk hash
            if chunk_id in prev_hashes:
                builder.register_chunk_hash(chunk_id, prev_hashes[chunk_id])
        
        # Copy provenance first
        for prov_id, prov in previous.provenance.items():
            if prov.chunk_id in chunk_ids:
                builder.add_provenance(prov)
        
        # Copy nodes and track chunk association
        # Also update coord assigner to avoid collisions
        for chunk_id in chunk_ids:
            for node_id in previous.chunk_index.get(chunk_id, []):
                node = previous.nodes.get(node_id)
                if node:
                    try:
                        # Update coord assigner to know this coord is used
                        c = node.coord
                        key = (c.major, c.type_, c.subtype)
                        with builder.coords._lock:
                            current = builder.coords._counters.get(key, 0)
                            if c.instance > current:
                                builder.coords._counters[key] = c.instance
                        
                        builder.add_node(node, deduplicate=False, chunk_id=chunk_id)
                    except ValueError:
                        pass
        
        # Copy relations where both endpoints are copied
        for rel in previous.relations:
            if rel.subject_id in node_ids_to_copy and rel.object_id in node_ids_to_copy:
                try:
                    builder.add_relation(rel)
                except ValueError:
                    pass
    
    def _emit_to_builder(
        self,
        builder: ProgramBuilder,
        chunks: List[Chunk],
        extractions: List[Extraction],
        llm_responses: Optional[Dict[str, List[Dict]]],
    ) -> None:
        """Emit extractions directly to an existing builder."""
        from .emitter import ENTITY_TO_COORD
        
        ext_to_node: Dict[str, str] = {}
        chunk_prov: Dict[str, str] = {}
        
        # Create provenance for each chunk and register hash
        for chunk in chunks:
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
        
        # Emit nodes from extractions
        for ext in extractions:
            mapping = ENTITY_TO_COORD.get(ext.entity_type)
            if mapping:
                major, type_, subtype = mapping
            else:
                major, type_, subtype = ext.major, ext.type_, ext.subtype
            
            coord = builder.coords.next(major, type_, subtype)
            prov_id = chunk_prov.get(ext.chunk_id, "prov_unknown")
            
            node = Node(
                coord=coord,
                label=ext.label,
                prov_id=prov_id,
                value=ext.value,
                unit=ext.unit,
                metadata={"entity_type": ext.entity_type, "extractor": ext.extractor},
            )
            node_id = builder.add_node(node, deduplicate=True, chunk_id=ext.chunk_id)
            ext_to_node[ext.ext_id] = node_id
        
        # Emit relations
        for ext in extractions:
            subject_id = ext_to_node.get(ext.ext_id)
            if not subject_id:
                continue
            prov_id = chunk_prov.get(ext.chunk_id, "prov_unknown")
            for predicate, target_ext_id in ext.relations:
                object_id = ext_to_node.get(target_ext_id)
                if object_id and object_id != subject_id:
                    builder.link(subject_id, predicate, object_id, prov_id, ext.confidence)
        
        # Handle LLM responses
        if llm_responses:
            self._emit_llm_to_builder(builder, llm_responses, chunk_prov)
    
    def _emit_llm_to_builder(
        self,
        builder: ProgramBuilder,
        llm_responses: Dict[str, List[Dict]],
        chunk_prov: Dict[str, str],
    ) -> None:
        """Emit LLM responses to builder."""
        from .coords import AbstractType
        
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
    
    def _load_source(self, source: str, source_type: str) -> Tuple[str, SourceInfo]:
        """Load content and create SourceInfo."""
        if source_type == "auto":
            if len(source) < 500 and '\n' not in source:
                try:
                    if Path(source).exists():
                        source_type = "file"
                    else:
                        source_type = "string"
                except (OSError, ValueError):
                    source_type = "string"
            else:
                source_type = "string"
        
        if source_type == "file":
            self._validate_path(source)
            content = Path(source).read_text()
            uri = str(Path(source).resolve())
            hash_ = self._hash_file(source)
            size = Path(source).stat().st_size
        else:
            content = source
            uri = f"string://{hashlib.md5(content.encode()).hexdigest()[:8]}"
            hash_ = hashlib.sha256(content.encode()).hexdigest()
            size = len(content)
        
        source_info = SourceInfo(uri=uri, hash_sha256=hash_, size_bytes=size)
        return content, source_info
    
    def _execute_llm(self, llm_requests: List[LLMRequest]) -> Optional[Dict[str, List[Dict]]]:
        """Execute LLM requests."""
        if not llm_requests or not self.config.executor:
            return None
        
        llm_responses = {}
        for req in llm_requests:
            try:
                result = self.config.executor(req)
                if hasattr(result, 'success'):
                    if result.success:
                        llm_responses[req.req_id] = result.data
                else:
                    llm_responses[req.req_id] = result
            except Exception as e:
                print(f"LLM request failed: {e}")
        
        return llm_responses if llm_responses else None
    
    def _validate_path(self, path: str) -> None:
        """Validate source file before loading."""
        if len(path) > MAX_PATH_LENGTH:
            raise ValueError(f"Path too long: {len(path)} > {MAX_PATH_LENGTH}")
        
        p = Path(path).resolve()
        
        if not p.is_file():
            raise ValueError(f"Not a regular file: {path}")
        
        try:
            p.relative_to(Path.cwd())
        except ValueError:
            if ".." in str(p):
                raise ValueError(f"Suspicious path: {path}")
        
        size = p.stat().st_size
        if size > self.config.max_file_size:
            raise ValueError(f"File too large: {size} > {self.config.max_file_size}")
    
    def _hash_file(self, path: str) -> str:
        """Compute file hash."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compile(source: str, config: Optional[Config] = None) -> Program:
    """Compile a document."""
    return Compiler(config).compile(source)


def compile_incremental(
    source: str,
    previous: Program,
    config: Optional[Config] = None,
) -> Tuple[Program, IncrementalPlan]:
    """Compile with incremental reuse."""
    return Compiler(config).compile_incremental(source, previous)


def load(path: str) -> Program:
    """Load a compiled program."""
    return Program.load(path)


def query(program: Program) -> Space:
    """Create queryable space from program."""
    return Space(program)
