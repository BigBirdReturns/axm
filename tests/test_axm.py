"""
AXM Test Suite

Run with: python -m pytest tests/ -v
Or just: python tests/test_axm.py
"""

import filecmp
import json
import tempfile
from pathlib import Path

# Import everything we need to test
from axm import (
    compile, load, query, Config,
    Coord, Node, Relation, Provenance,
    Major, QuantityType, IR_SCHEMA_VERSION,
    ProgramBuilder, SourceInfo,
    Lexer, Parser, Emitter,
    MockExecutor, RetryExecutor,
)
from axm.ids import IDGenerator


class TestCoords:
    """Test coordinate system."""
    
    def test_valid_coord(self):
        c = Coord(1, 1, 1, 1)
        assert c.major == 1
        assert c.to_id() == "01-01-01-0001"
    
    def test_invalid_major(self):
        try:
            Coord(9, 1, 1, 1)
            assert False, "Should reject major=9"
        except ValueError as e:
            assert "Major must be 1-8" in str(e)
    
    def test_invalid_type(self):
        try:
            Coord(1, 100, 1, 1)
            assert False, "Should reject type=100"
        except ValueError as e:
            assert "Type must be 0-99" in str(e)
    
    def test_from_id(self):
        c = Coord.from_id("07-02-01-0042")
        assert c.major == 7
        assert c.type_ == 2
        assert c.instance == 42
    
    def test_from_list(self):
        c = Coord.from_list([8, 1, 1, 100])
        assert c.major == 8
        assert c.instance == 100
    
    def test_distance(self):
        c1 = Coord(1, 1, 1, 1)
        c2 = Coord(1, 1, 1, 2)
        d = c1.distance(c2)
        assert d > 0
        assert d < 1  # Close coords


class TestIR:
    """Test IR primitives."""
    
    def test_node_from_dict_valid(self):
        data = {
            "id": "01-01-01-0001",
            "label": "Test Node",
            "coords": [1, 1, 1, 1],
            "prov_id": "prov_001",
            "value": 100,
        }
        node = Node.from_dict(data)
        assert node.label == "Test Node"
        assert node.value == 100
    
    def test_node_from_dict_missing_field(self):
        try:
            Node.from_dict({"label": "test"})
            assert False, "Should reject missing prov_id"
        except ValueError as e:
            assert "Missing required field" in str(e)
    
    def test_node_content_hash(self):
        n1 = Node(Coord(1, 1, 1, 1), "test", "prov", value=100)
        n2 = Node(Coord(1, 1, 1, 2), "test", "prov", value=100)  # Different coord
        assert n1.content_hash == n2.content_hash  # Same content
    
    def test_relation_no_self_loop(self):
        try:
            Relation("01-01-01-0001", "HAS_VALUE", "01-01-01-0001", "prov")
            assert False, "Should reject self-loop"
        except ValueError as e:
            assert "Self-loops not allowed" in str(e)
    
    def test_provenance_from_dict(self):
        data = {
            "prov_id": "prov_001",
            "chunk_id": "chunk_001",
            "extractor": "tier1:regex",
            "timestamp": "2024-01-01T00:00:00Z",
            "tier": 1,
        }
        prov = Provenance.from_dict(data)
        assert prov.tier == 1


class TestProgram:
    """Test Program building and loading."""
    
    def test_builder_deduplication(self):
        source = SourceInfo(uri="test://", hash_sha256="abc123")
        builder = ProgramBuilder(source)
        
        prov = Provenance("prov1", "chunk1", "test", "2024-01-01T00:00:00Z")
        builder.add_provenance(prov)
        
        n1 = Node(builder.coords.next(7, 1, 1), "revenue", "prov1", value=100)
        n2 = Node(builder.coords.next(7, 1, 1), "revenue", "prov1", value=100)  # Duplicate
        
        id1 = builder.add_node(n1, deduplicate=True)
        id2 = builder.add_node(n2, deduplicate=True)
        
        assert id1 == id2  # Should return same ID
        assert builder.node_count() == 1
    
    def test_round_trip(self):
        source = SourceInfo(uri="test://", hash_sha256="abc123")
        builder = ProgramBuilder(source)
        
        prov = Provenance("prov1", "chunk1", "test", "2024-01-01T00:00:00Z")
        builder.add_provenance(prov)
        
        node = Node(builder.coords.next(7, 1, 1), "test_node", "prov1", value=42)
        builder.add_node(node)
        
        program = builder.build()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.axm"
            program.write(str(path))
            
            loaded = load(str(path))
            assert len(loaded.nodes) == 1
            assert loaded.nodes["07-01-01-0001"].label == "test_node"


class TestCompiler:
    """Test full compilation."""
    
    def test_compile_string(self):
        doc = "Total revenue was $500 million. Growth was 25%."
        program = compile(doc, Config.default())
        
        assert len(program.nodes) > 0
        assert program.manifest()["ir_schema_version"] == IR_SCHEMA_VERSION
    
    def test_compile_no_llm(self):
        doc = "Revenue: $100 million\nExpenses: $80 million"
        program = compile(doc, Config.no_llm())
        
        assert len(program.nodes) > 0
        # Should have extracted the key-values
    
    def test_compile_round_trip_identical(self):
        doc = "Total revenue was $2.87 billion, representing 25% growth."
        program = compile(doc, Config.default())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "v1.axm"
            path2 = Path(tmpdir) / "v2.axm"
            
            program.write(str(path1))
            loaded = load(str(path1))
            loaded.write(str(path2))
            
            # Compare JSONL files
            assert filecmp.cmp(path1 / "nodes.jsonl", path2 / "nodes.jsonl")
            assert filecmp.cmp(path1 / "relations.jsonl", path2 / "relations.jsonl")


class TestSpace:
    """Test query engine."""
    
    def test_query_by_major(self):
        doc = "Revenue was $500 million. The company was founded in 2020."
        program = compile(doc, Config.default())
        space = query(program)
        
        quantities = list(space.query(major=Major.QUANTITY))
        assert len(quantities) > 0
    
    def test_query_by_label(self):
        doc = "Total revenue was $500 million."
        program = compile(doc, Config.default())
        space = query(program)
        
        # Mock extractor finds money patterns
        results = list(space.query(label_contains="million"))
        assert len(results) > 0
    
    def test_first_and_find(self):
        doc = "Revenue was $500 million. Profit was $50 million."
        program = compile(doc, Config.default())
        space = query(program)
        
        first = space.first(major=Major.QUANTITY)
        assert first is not None
        
        all_nodes = space.find(major=Major.QUANTITY)
        assert isinstance(all_nodes, list)


class TestLexer:
    """Test lexer."""
    
    def test_detect_json(self):
        lexer = Lexer()
        chunks = lexer.lex('{"key": "value"}')
        assert len(chunks) == 1
        assert chunks[0].chunk_type.name == "JSON"
        assert chunks[0].tier == 0
    
    def test_detect_kv(self):
        lexer = Lexer()
        chunks = lexer.lex("Name: John\nAge: 30")
        assert any(c.chunk_type.name == "KEY_VALUE" for c in chunks)
    
    def test_content_hash(self):
        lexer = Lexer()
        chunks = lexer.lex("Test content here.")
        assert chunks[0].content_hash is not None
        assert len(chunks[0].content_hash) == 16


class TestIDGenerator:
    """Test thread-safe ID generation."""
    
    def test_sequential(self):
        gen = IDGenerator()
        id1 = gen.next("test")
        id2 = gen.next("test")
        assert id1 == "test_000001"
        assert id2 == "test_000002"
    
    def test_namespaced(self):
        gen = IDGenerator()
        gen.next("a")
        gen.next("a")
        gen.next("b")
        
        stats = gen.stats()
        assert stats["a"] == 2
        assert stats["b"] == 1
    
    def test_reset(self):
        gen = IDGenerator()
        gen.next("test")
        gen.reset()
        id1 = gen.next("test")
        assert id1 == "test_000001"


class TestExecutor:
    """Test LLM executors."""
    
    def test_mock_executor(self):
        from axm.parser import LLMRequest
        
        executor = MockExecutor()
        request = LLMRequest(
            req_id="test",
            chunk_id="chunk",
            content="Total revenue was $500 million.",
            prompt="Extract claims",
        )
        
        results = executor(request)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "subject" in results[0]
    
    def test_retry_executor_validation(self):
        from axm.parser import LLMRequest
        
        def bad_executor(req):
            return [{"invalid": "no subject"}]  # Missing subject
        
        retry = RetryExecutor(bad_executor, max_retries=2, validate=True)
        request = LLMRequest("test", "chunk", "content", "prompt")
        
        result = retry(request)
        assert not result.success  # Should fail validation


class TestIncremental:
    """Test incremental compilation."""
    
    def test_incremental_reuse(self):
        from axm import compile, compile_incremental, Config
        
        doc_v1 = """
        Revenue was $500 million.
        
        Expenses were $400 million.
        """
        
        doc_v2 = """
        Revenue was $500 million.
        
        Expenses were $450 million.
        """
        
        config = Config.default()
        v1 = compile(doc_v1, config)
        v2, plan = compile_incremental(doc_v2, v1, config)
        
        # Should reuse some nodes
        assert plan.reused_nodes > 0 or plan.recompiled_nodes > 0
        assert plan.efficiency >= 0
    
    def test_chunk_hashes_stored(self):
        from axm import compile, Config
        import tempfile
        from pathlib import Path
        
        doc = "Revenue was $500 million."
        program = compile(doc, Config.default())
        
        # Check hashes exist
        assert len(program.chunk_hashes) > 0
        
        # Check round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.axm"
            program.write(str(path))
            
            loaded = load(str(path))
            assert len(loaded.chunk_hashes) == len(program.chunk_hashes)
    
    def test_incremental_efficiency(self):
        from axm import compile, compile_incremental, Config
        
        # Longer document
        doc_v1 = """
        Q1 Report
        
        Revenue was $100 million in Q1.
        
        Q2 Report
        
        Revenue was $150 million in Q2.
        
        Q3 Report
        
        Revenue was $200 million in Q3.
        """
        
        # Only change one paragraph
        doc_v2 = """
        Q1 Report
        
        Revenue was $100 million in Q1.
        
        Q2 Report
        
        Revenue was $175 million in Q2.
        
        Q3 Report
        
        Revenue was $200 million in Q3.
        """
        
        config = Config.default()
        v1 = compile(doc_v1, config)
        v2, plan = compile_incremental(doc_v2, v1, config)
        
        # Should have good efficiency (most chunks unchanged)
        assert plan.efficiency > 0.3  # At least 30% reused


def run_all():
    """Run all tests manually."""
    test_classes = [
        TestCoords,
        TestIR,
        TestProgram,
        TestCompiler,
        TestSpace,
        TestLexer,
        TestIDGenerator,
        TestExecutor,
        TestIncremental,
    ]
    
    passed = 0
    failed = 0
    
    for cls in test_classes:
        print(f"\n{cls.__name__}:")
        instance = cls()
        
        for name in dir(instance):
            if name.startswith("test_"):
                try:
                    getattr(instance, name)()
                    print(f"  ✓ {name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {name}: {type(e).__name__}: {e}")
                    failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all()
    sys.exit(0 if success else 1)
