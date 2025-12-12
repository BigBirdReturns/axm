# AXM - Semantic Compiler

**Compile once. Query forever. No LLM at runtime.**

AXM transforms documents into queryable semantic programs. Like a traditional compiler transforms source code into executables, AXM transforms text into structured knowledge graphs that can be queried without any neural network.

## Why AXM?

| | RAG | AXM |
|---|-----|-----|
| Query cost | LLM every time | Zero (coordinate lookup) |
| Determinism | Non-deterministic | Deterministic |
| Auditability | Black box | Full provenance |
| Air-gap ready | Needs network | Works offline |
| Diff-able | Can't diff | Git-diff semantics |
| Incremental | Reprocess everything | Only changed content |

## Installation

```bash
pip install axm
```

Or from source:

```bash
git clone https://github.com/sandhuconsulting/axm.git
cd axm
pip install -e .
```

## Quick Start

```python
from axm import compile, query, Major

# Compile a document (LLM runs once)
program = compile("quarterly_report.txt")
program.write("report.axm")

# Query forever (no LLM)
space = query(program)

# Find all financial quantities
for node in space.query(major=Major.QUANTITY):
    print(f"{node.label}: {node.value} {node.unit}")

# Find revenue-related claims
for node in space.query(label_contains="revenue"):
    print(f"{node.id}: {node.label}")
```

## Universal Intake (v0.5.2)

AXM auto-routes sources to the optimal processing path:

```python
from axm import compile_universal, merge_programs, query

# Structured data → Adapter path (confidence 1.0)
xbrl = compile_universal("financials.xbrl")    # XBRL → direct extraction
cal = compile_universal("events.ics")          # iCal → direct extraction

# Unstructured data → Compiler path (confidence varies)  
text = compile_universal("earnings_call.txt")  # Text → lexer → parser

# Merge into unified space
merged = merge_programs([xbrl, cal, text])

# Query by confidence
space = query(merged)
for node in space.all_nodes():
    prov = merged.provenance.get(node.prov_id)
    print(f"{node.label}: confidence {prov.confidence:.0%}")
```

**Two paths, one destination:**
```
STRUCTURED (schema exists)          UNSTRUCTURED (text soup)
     XBRL, FHIR, iCal                 PDF, Word, transcripts
          ↓                                   ↓
     Adapter path                      Compiler path
     Confidence: 1.0                   Confidence: varies
          ↓                                   ↓
          └───────────→ MERGED SPACE ←────────┘
```

**Available Adapters:** XBRL, iCalendar, RSS/Atom

## Incremental Compilation

When documents change, AXM only reprocesses changed sections:

```python
from axm import compile, compile_incremental, load

# First compile
v1 = compile("report_v1.txt")
v1.write("report.axm")

# After edits, compile incrementally
v1 = load("report.axm")
v2, plan = compile_incremental("report_v2.txt", v1)

print(f"Reused {plan.reused_nodes} nodes")
print(f"Efficiency: {plan.efficiency:.0%} work avoided")
```

## CLI

```bash
# Compile
axm compile document.txt -o output.axm

# Incremental compile
axm compile document_v2.txt --previous output.axm -o output_v2.axm

# Query
axm query output.axm --major 7  # All quantities
axm query output.axm --label revenue

# Inspect
axm inspect output.axm
axm stats output.axm

# Compare versions
axm diff v1.axm v2.axm

# Interactive exploration
axm repl output.axm
```

## Rust/WASM Runtime

- **Library**: `rust/axm-rs` mirrors the Python query engine (coords, IR validation, semantic queries).
- **Bindings**: `wasm_bindgen` exports `WasmProgram` so browsers can load zipped `.axm` artifacts and run queries offline.
- **Demo**: `web/index.html` + `web/app.js` show a load-and-query flow entirely in the browser (no server round-trips).

Build the WebAssembly package and run the static demo:

```bash
cd web
npm install
npm run build:wasm   # outputs web/pkg/* from the Rust crate
npm run serve         # open http://localhost:8080 and load a zipped .axm
```

The same Rust engine is available on the CLI:

```bash
cargo run --manifest-path rust/axm-rs/Cargo.toml -- summary path/to/program.axm
cargo run --manifest-path rust/axm-rs/Cargo.toml -- query path/to/program.axm 7
```

## How It Works

### 1. Multi-Tier Extraction

AXM uses a tiered approach to minimize LLM usage:

- **Tier 0**: Structured data (JSON, XML) - no LLM needed
- **Tier 1**: Pattern extraction (money, dates, percentages, entities) - regex
- **Tier 2**: Keyword classification (growth, risk, sentiment) - no LLM
- **Tier 3-4**: Complex claims and relations - LLM (compile-time only)

### 2. Coordinate System

Every extracted fact gets a 4D coordinate:

```
[Major, Type, Subtype, Instance]
  │      │      │        └── Unique counter
  │      │      └── Refinement
  │      └── Category type
  └── 1-8: Entity, Action, Property, Relation, Location, Time, Quantity, Abstract
```

Example: `07-01-01-0042` = Quantity / Financial / Default / Instance 42

### 3. Git-Diffable Output

Programs are stored as sorted JSONL:

```
output.axm/
├── manifest.json     # Metadata, schema, counts
├── nodes.jsonl       # Extracted facts
├── relations.jsonl   # Connections between facts
└── provenance.jsonl  # Audit trail
```

## Configuration

```python
from axm import Config, compile

# No LLM (Tier 0-2 only)
program = compile(doc, Config.no_llm())

# Local LLM via Ollama
program = compile(doc, Config.with_ollama("llama3"))

# Claude API
program = compile(doc, Config.with_anthropic())
```

## What Gets Extracted

**Tier 1 (No LLM):**
- Money amounts: `$2.87 billion`, `$500M`
- Percentages: `25%`, `18 percent`
- Dates: `December 31, 2024`, `Q4 2024`
- Organizations: `Palantir Technologies Inc.`
- People: `CEO Alex Karp`
- Locations: `Denver, Colorado`
- Durations: `5 years`, `3 quarters`
- Ratios: `2.5x`, `3:1`

**Tier 2 (No LLM):**
- Claim types: GROWTH, DECLINE, RISK, MOAT, GUIDANCE
- Sentiment: POSITIVE, NEGATIVE, NEUTRAL

**Tier 3-4 (LLM at compile-time):**
- Complex financial claims
- Subject-predicate-object relations
- Cross-sentence reasoning

## Use Cases

- **Financial Analysis**: Extract metrics from SEC filings, compare across quarters
- **Intelligence**: Compile reports into queryable knowledge bases
- **Due Diligence**: Track claims and their provenance
- **Compliance**: Audit trail for every extracted fact
- **Air-Gapped Systems**: Compile with LLM, deploy without

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
