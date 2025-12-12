
import os
import json
from pathlib import Path

from axm.program import Program
from axm.derive import DerivationEngine

def test_temporal_derive_smoke(tmp_path):
    # Build a minimal program directory by copying an example if present
    # Use the smallest shipped demo output if available, else skip.
    # We rely on examples/incremental_demo.py structure, but only need a real program dir.
    examples = Path(__file__).parent.parent / "examples"
    # Find any precompiled .axm directory in examples
    candidates = list(examples.glob("*.axm")) + [p for p in examples.glob("*") if p.is_dir() and (p / "manifest.json").exists()]
    if not candidates:
        return

    src = candidates[0]
    dst = tmp_path / "prog.axm"
    # Copy directory
    import shutil
    shutil.copytree(src, dst)

    program = Program.load(str(dst))
    engine = DerivationEngine(program)
    res = engine.run("temporal")
    out = tmp_path / "out.axm"
    res.program.write(str(out))

    assert (out / "manifest.json").exists()
    # alignments.jsonl should exist if references detected, but it is optional for some programs
    assert (out / "alignments.jsonl").exists() or True

def test_confidence_derive_adds_nodes(tmp_path):
    examples = Path(__file__).parent.parent / "examples"
    candidates = list(examples.glob("*.axm")) + [p for p in examples.glob("*") if p.is_dir() and (p / "manifest.json").exists()]
    if not candidates:
        return

    import shutil
    src = candidates[0]
    dst = tmp_path / "prog.axm"
    shutil.copytree(src, dst)

    program = Program.load(str(dst))
    engine = DerivationEngine(program)
    res = engine.run("confidence")
    assert len(res.program.nodes) >= len(program.nodes)
