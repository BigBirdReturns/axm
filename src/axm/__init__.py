"""
AXM - Semantic Compiler Foundation (v0.5.2)

Compile documents once. Query forever. No LLM at runtime.
Now with unified intake: structured OR unstructured → same coordinate space.

Quick start:
    from axm import compile, load, query
    
    # Compile
    program = compile("document.txt")
    program.write("output.axm")
    
    # Query (no LLM)
    program = load("output.axm")
    space = query(program)
    
    for node in space.query(major=7):  # All quantities
        print(node.label, node.value)

Universal intake (NEW):
    from axm import compile_universal, merge_programs
    
    # Works with any source type
    p1 = compile_universal("report.pdf")       # Unstructured → compiler
    p2 = compile_universal("financials.xbrl")  # Structured → adapter
    
    # Merge into single space
    merged = merge_programs([p1, p2])
    
    # Query unified knowledge
    space = query(merged)
"""

__version__ = "0.5.3"

# Core types
from .coords import (
    Coord,
    CoordAssigner,
    Major,
    EntityType,
    ActionType,
    QuantityType,
    TimeType,
    AbstractType,
    COORDINATE_SCHEMA,
    IR_SCHEMA_VERSION,
)

from .ir import (
    Node,
    Relation,
    Derivation,
    TemporalAlignment,
    Fork,
    ForkOption,
    Provenance,
    SourceInfo,
    PREDICATES,
)

from .program import (
    Program,
    ProgramBuilder,
    load,
)

from .space import (
    Space,
    QueryResult,
    query,
)

# Compiler
from .compiler import (
    Compiler,
    Config,
    IncrementalPlan,
    compile,
    compile_incremental,
)

# IDs
from .ids import IDGenerator, next_id, reset_ids

# Components
from .lexer import Lexer, Chunk, ChunkType
from .parser import Parser, Extraction, LLMRequest
from .emitter import Emitter
from .executor import (
    MockExecutor,
    OllamaExecutor,
    AnthropicExecutor,
    OpenAIExecutor,
    RetryExecutor,
    LLMResult,
    get_executor,
)

# Chat
from .chat import Chat, chat_repl

# Intake (universal routing)
from .intake import (
    compile_universal,
    merge_programs,
    detect_source,
    list_adapters,
    avg_confidence,
    SourceType,
    ProcessingPath,
    Detector,
    Router,
    AdapterRegistry,
    BaseAdapter,
    ExtractedEntity,
)

__all__ = [
    "__version__",
    "IR_SCHEMA_VERSION",
    
    # Main API
    "compile",
    "compile_incremental",
    "compile_universal",  # NEW
    "merge_programs",     # NEW
    "load", 
    "query",
    "Chat",
    
    # Config
    "Compiler",
    "Config",
    "IncrementalPlan",
    
    # Core types
    "Coord",
    "Node",
    "Relation",
    "Derivation",
    "TemporalAlignment",
    "Program",
    "Space",
    
    # Enums
    "Major",
    "EntityType",
    "QuantityType",
    "TimeType",
    "AbstractType",
    "SourceType",      # NEW
    "ProcessingPath",  # NEW
    
    # Advanced
    "ProgramBuilder",
    "QueryResult",
    "Lexer",
    "Parser",
    "Emitter",
    "IDGenerator",
    
    # Intake (NEW)
    "Detector",
    "Router",
    "AdapterRegistry",
    "BaseAdapter",
    "detect_source",
    "list_adapters",
    "avg_confidence",
    
    # Executors
    "MockExecutor",
    "OllamaExecutor",
    "RetryExecutor",
    "LLMResult",
    "get_executor",
]
