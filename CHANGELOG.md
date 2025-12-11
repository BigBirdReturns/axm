# Changelog

All notable changes to AXM will be documented in this file.

## [0.5.0] - 2024-12-11

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

## [0.4.0] - 2024-12-11

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

## [0.3.0] - 2024-12-11

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
