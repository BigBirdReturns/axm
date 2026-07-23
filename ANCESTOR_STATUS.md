# Executable Ancestor Status

Status date: 2026-07-23

This repository is the executable pre-split AXM ancestor. It is preserved for
lineage, comparison, and bounded replay; it is not the current AXM product or
the authority for current shard formats.

## Pinned witness

Source baseline: `184ebee37782d3eece4f73451762912c18e85b84`

The following witness passed on Windows with Python 3.12:

```text
python -m pytest -q --basetemp S:\Temp\axm-ancestor-pytest-20260723-execute
54 passed, 48 deprecation warnings
```

The no-LLM compile/query path also passed:

```python
from axm import compile, query, Config

program = compile(
    "Revenue was $5 million in 2025.",
    config=Config.no_llm(),
)
labels = [node.label for node in query(program).query()]
assert labels == ["$5 million"]
```

The deprecation warnings are retained ancestor residue, not evidence of a
failed witness.

## Mechanism ownership audit

| Ancestor mechanism | Current ownership and evidence | Disposition |
| --- | --- | --- |
| XBRL intake | `axm-core/forge/axm_forge/ingestion/extractors.py`; live extraction produced one tier-zero fact | Already harvested |
| iCalendar intake | Same Forge extractor surface; live extraction produced scheduled and location facts | Already harvested |
| RSS/Atom intake | Same Forge extractor surface; live extraction produced tier-zero publication facts | Already harvested |
| Semantic coordinates | Current Forge `derivation/coords.py` explicitly derives a local cache from the frozen ancestor scheme | Already harvested under current boundaries |
| Temporal metadata | Forge annotates candidates, Genesis emits `temporal@1`, and Spectra mounts the extension | Superseded by the current sealed-shard path |
| Resumable work | Current Tier 3 extraction has progress markers and resumable batches; its focused suite passed 30 tests | Current mechanism retained |
| Chunk-hash semantic recompilation | The ancestor reuses unchanged compiled chunks; no equivalent current end-to-end requirement failed during this audit | Hold as ancestor evidence; do not transplant speculatively |
| Mutable numeric/temporal derivation engine | Current AXM records claims and versioned extensions through Forge, Genesis, and Spectra rather than mutating the old `Program` model | Do not revive wholesale |

The one verified current gap was permanent regression coverage for structured
intake. XBRL, iCalendar, RSS, and Atom witnesses were added to current
`axm-core` and merged through PR #26 at
`968df05` after both GitHub CI checks passed.

The current Genesis-to-Spectra mount witness also passed (`3 passed, 2 skipped`)
with the repository's declared dependencies isolated on `S:` and the sibling
Genesis source on `PYTHONPATH`.

## Custody rule

- Preserve the private GitHub remote and full history.
- Keep this checkout executable as an ancestor.
- Do not add new product work here.
- Evaluate any future salvage against a failing current AXM path first.
- Current compiler, signature, identity, and verification authority remains
  `axm-genesis`; current orchestration, Forge, and Spectra authority remains
  `axm-core`.
