"""
AXM - Semantic Compiler Foundation (v0.5)

Compile documents once. Query forever. No LLM at runtime.
Now with REAL incremental compilation.

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

Incremental compilation:
    from axm import compile, compile_incremental, load
    
    # First compile
    v1 = compile("document_v1.txt")
    v1.write("output.axm")
    
    # Later, after edits (only recompiles changed chunks)
    v1 = load("output.axm")
    v2, plan = compile_incremental("document_v2.txt", v1)
    print(f"Reused {plan.reused_nodes} nodes, recompiled {plan.recompiled_nodes}")
"""

__version__ = "0.5.0"

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

__all__ = [
    "__version__",
    "IR_SCHEMA_VERSION",
    
    # Main API
    "compile",
    "compile_incremental",
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
    "Program",
    "Space",
    
    # Enums
    "Major",
    "EntityType",
    "QuantityType",
    "TimeType",
    "AbstractType",
    
    # Advanced
    "ProgramBuilder",
    "QueryResult",
    "Lexer",
    "Parser",
    "Emitter",
    "IDGenerator",
    
    # Executors
    "MockExecutor",
    "OllamaExecutor",
    "RetryExecutor",
    "LLMResult",
    "get_executor",
]
