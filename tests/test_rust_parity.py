import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from axm import Config, Major, compile, query

CARGO_MANIFEST = Path(__file__).parent.parent / "rust" / "axm-rs" / "Cargo.toml"


if os.environ.get("AXM_RUN_RUST_TESTS") != "1":
    pytest.skip(
        "Rust/WASM parity tests are opt-in; set AXM_RUN_RUST_TESTS=1 to enable.",
        allow_module_level=True,
    )


def _run_cli(args, cwd=None):
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", str(CARGO_MANIFEST)] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )
    return result.stdout


def _summary(path: Path):
    output = _run_cli(["--", "summary", str(path)])
    return json.loads(output)


def _query(path: Path, major: int):
    output = _run_cli(["--", "query", str(path), str(major)])
    return json.loads(output)


def _zip_axm_dir(axm_dir: Path) -> Path:
    # make_archive appends extension automatically when suffix empty
    temp_base = axm_dir.parent / axm_dir.name
    archive_path = shutil.make_archive(str(temp_base), "zip", root_dir=axm_dir)
    return Path(archive_path)


def test_rust_cli_summary_matches_python_manifest():
    doc = "Revenue was $500 million. Profit was $50 million."
    program = compile(doc, Config.default())
    space = query(program)

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "artifact.axm"
        program.write(str(target))

        py_manifest = program.manifest()
        rust_manifest = _summary(target)

        assert rust_manifest["counts"]["nodes"] == py_manifest["counts"]["nodes"]
        assert rust_manifest["counts"]["relations"] == py_manifest["counts"]["relations"]

        rust_query = _query(target, Major.QUANTITY)
        assert len(rust_query) == space.count(Major.QUANTITY)


def test_zip_pipeline_aligns_with_wasm_loader():
    doc = "The company was founded in 2020 and raised $10 million."
    program = compile(doc, Config.default())

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "artifact.axm"
        program.write(str(target))
        archive = _zip_axm_dir(target)

        rust_manifest = _summary(archive)
        assert rust_manifest["counts"]["nodes"] == len(program.nodes)
        assert rust_manifest["coordinate_system"]["version"] == program.manifest()["coordinate_system"]["version"]
