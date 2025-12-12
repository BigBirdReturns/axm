import argparse
import tempfile
from pathlib import Path

from axm import (
    Config,
    Coord,
    Derivation,
    Major,
    Node,
    ProgramBuilder,
    Provenance,
    Relation,
    SourceInfo,
    TemporalAlignment,
    compile,
    load,
    query,
)
from axm.cli import cmd_derive


def build_program_with_derivation():
    source = SourceInfo(uri="test://", hash_sha256="abc123")
    builder = ProgramBuilder(source)

    prov_main = Provenance("prov_main", "chunk1", "test", "2024-01-01T00:00:00Z", confidence=0.9)
    prov_support = Provenance("prov_support", "chunk1", "test", "2024-01-01T00:00:00Z", confidence=0.7)

    builder.add_provenance(prov_main)
    builder.add_provenance(prov_support)

    revenue = Node(builder.coords.next(Major.QUANTITY, 1, 1), "revenue", prov_main.prov_id, value=200)
    expense = Node(builder.coords.next(Major.QUANTITY, 1, 2), "expense", prov_support.prov_id, value=50)
    profit = Node(builder.coords.next(Major.QUANTITY, 1, 3), "profit", prov_main.prov_id, value=150)

    builder.add_node(revenue)
    builder.add_node(expense)
    builder.add_node(profit)

    builder.add_relation(Relation(revenue.id, "SUPPORTS", profit.id, prov_main.prov_id, confidence=0.85))

    deriv = Derivation(result_id=profit.id, operands=(revenue.id, expense.id), operator="SUBTRACT", prov_id=prov_main.prov_id)
    builder.add_derivation(deriv)

    alignment = TemporalAlignment(subject_id=revenue.id, reference_id=expense.id, operator="ALIGN_START", prov_id=prov_support.prov_id, offset=0)
    builder.add_temporal_alignment(alignment)

    return builder.build()


def test_ir_primitives_round_trip():
    deriv = Derivation(result_id="01", operands=("a", "b"), operator="ADD", prov_id="prov", confidence=0.8, metadata={"note": "sum"})
    as_dict = deriv.to_dict()
    restored = Derivation.from_dict(as_dict)
    assert restored == deriv

    align = TemporalAlignment(subject_id="a", reference_id="b", operator="ALIGN_START", prov_id="prov", offset=1.5, confidence=0.6)
    restored_align = TemporalAlignment.from_dict(align.to_dict())
    assert restored_align == align


def test_confidence_propagation():
    program = build_program_with_derivation()
    space = query(program)

    profit = space.first(label_contains="profit")
    conf = space.confidence(profit.id)
    assert 0 <= conf <= 1
    assert conf >= 0.6  # should propagate from operands and relation

    operands = [n.id for n in nodes_or_ids(space, ["revenue", "expense"])]
    value, combined, nodes = space.derive_numeric("subtract", operands)
    assert value == 150
    assert combined <= conf


def nodes_or_ids(space, labels):
    resolved = []
    for label in labels:
        node = space.first(label_contains=label)
        if node:
            resolved.append(node)
    return resolved


def test_temporal_alignment_lookup():
    program = build_program_with_derivation()
    space = query(program)

    revenue = space.first(label_contains="revenue")
    alignments = space.temporal_alignments(revenue.id)
    assert len(alignments) == 1
    assert alignments[0].operator == "ALIGN_START"


def test_cli_derive(capsys):
    program = build_program_with_derivation()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "program.axm"
        program.write(path)

        args = argparse.Namespace(program=str(path), operator="add", operands=["revenue", "expense"], unit=None)
        cmd_derive(args)

        out = capsys.readouterr().out
        assert "Derivation: ADD" in out
        assert "confidence" in out


def test_compile_derivation_flow():
    doc = "Revenue was $200 and expenses were $50."
    program = compile(doc, Config.no_llm())
    space = query(program)
    derived = list(space.query())
    assert len(derived) >= 0

    # Writing and loading should retain new IR sections even if empty
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "p.axm"
        program.write(path)
        loaded = load(path)
        assert loaded.derivations == []
        assert loaded.temporal_alignments == []
