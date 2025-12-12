# Changelog

All notable changes to AXM will be documented in this file.

## [0.5.3] - 2025-12-12

### Added
- Derivation primitives and temporal alignment operators in IR (`derivations.jsonl`, `alignments.jsonl`)
- Confidence propagation in the query engine with derivation and relation-aware rules
- `axm derive` CLI command for on-demand arithmetic with propagated confidence output
- Example documenting `derive` usage and new helper methods

### Changed
- Program manifests include derivation/alignment counts and persist new sections on write/load
- README updated with derivation workflow and query helper examples

### Fixed
- Defensive handling for cyclic derivations during confidence propagation


## [0.5.2] - 2025-12-11

### Added
- **Universal Intake Layer** - Auto-routes sources to adapter or compiler path
- **Structured Data Adapters** - XBRL, iCalendar, RSS feed parsing at confidence 1.0
- **Source Detection** - Automatic detection of XML, JSON-LD, XBRL, iCal, etc.
- **Program Merging** - `merge_programs()` combines multiple sources into unified space
- **Container Extraction** - PDF, Word, Excel, PowerPoint text extraction (optional deps)
- `compile_universal()` - Single entry point for any source type
- `detect_source()` - Returns source type and recommended processing path
- `list_adapters()` - Lists available structured data adapters
- `avg_confidence()` - Calculate average confidence across program nodes
- New enums: `SourceType`, `ProcessingPath`
- `AdapterRegistry` for pluggable adapter system
- `BaseAdapter` class for custom adapter development
- Sample files: `sample_10k.xbrl`, `sample_calendar.ics`
- `intake_demo.py` - Demonstrates two-path architecture
- 13 new tests for intake layer (47 total)

### Changed
- README updated to document universal intake
- Module exports expanded with intake types

### Architecture
```
STRUCTURED (XBRL, FHIR, etc.)    UNSTRUCTURED (PDF, text)
         ↓                              ↓
    Adapter path                 Compiler path
    Confidence: 1.0              Confidence: varies
         ↓                              ↓
         └──────→ MERGED SPACE ←────────┘
```

## [0.5.1] - 2025-12-11

### Fixed
- **Schema version mismatch** - IR_SCHEMA_VERSION now correctly says "0.5"
- **Negation handling** - "no growth" no longer triggers GROWTH claim, properly flips to DECLINE
- **Table detection** - Now handles Markdown alignment syntax (`:---:`, `:---|`)
- **Citation detection** - References sections properly classified as CITATION chunks

### Added
- `ChunkType.CITATION` for bibliography/reference entries
- `Node.contextual_hash()` method for chunk-aware deduplication
- Context-aware lexer that tracks section headings
- Negation patterns in Tier 2 claim extraction
- 4 new tests (negation, citations, tables)

### Changed
- Lexer now maintains section context for smarter classification
- Chunk dataclass has new `context` field for parent heading info
- 34 tests total (was 30)

## [0.5.0] - 2025-12-11

### Added
- **Real incremental compilation** - Only recompile changed chunks, reuse unchanged nodes
- **Expanded Tier 0-1 extractors** - Organizations, people, locations, durations, ratios
- **Tier 2 sentiment extraction** - Positive/negative/neutral classification
- **Chunk content hashing** - SHA256 hashes stored in manifest for change detection
- **CLI improvements** - `inspect`, `stats` commands, `--previous` for incremental
- **IncrementalPlan** - Reports reused/recompiled/removed nodes and efficiency
- LICENSE file (MIT)
- CHANGELOG.md

### Changed
- ProgramBuilder tracks chunk hashes for incremental compilation
- Program.chunk_hashes stored in manifest.json
- Parser runs entity extraction on all prose chunks
- CLI repl has more commands (out, in, neighbors)

### Fixed
- Coordinate conflicts when copying unchanged nodes in incremental mode
- Version numbers updated consistently across modules

## [0.4.0] - 2025-12-11

### Added
- Thread-safe IDGenerator with namespaced counters
- IR validation on load (rejects malformed JSONL)
- Schema versioning (IR_SCHEMA_VERSION = "0.4")
- RetryExecutor with exponential backoff
- File size and path validation
- Chunk tracking in Program (chunk_index)
- REPL for interactive exploration
- 27 passing tests

### Changed
- ProgramBuilder uses O(1) indices for deduplication
- Emitter uses public API only (no _nodes access)
- Program.load() validates all fields

## [0.3.0] - 2025-12-11

Initial foundation release.

### Added
- Coordinate system (4D semantic addressing)
- IR primitives (Node, Relation, Fork, Provenance)
- Multi-tier extraction (Tier 0-4)
- Lexer, Parser, Emitter, Compiler pipeline
- Git-diffable JSONL output format
- Space query engine
- MockExecutor, OllamaExecutor, AnthropicExecutor
- Basic CLI
