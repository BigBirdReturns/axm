"""
AXM Command Line Interface (v0.5)

Commands:
    axm compile document.txt -o output.axm
    axm compile document.txt --previous v1.axm -o v2.axm  # Incremental
    axm query output.axm --major 7
    axm info output.axm
    axm inspect output.axm
    axm diff v1.axm v2.axm
    axm stats output.axm
    axm repl output.axm
"""

import argparse
import json
import sys
from pathlib import Path

from .compiler import Compiler, Config, compile, compile_incremental, load, query
from .coords import Major, IR_SCHEMA_VERSION
from .chat import Chat, chat_repl


def cmd_compile(args):
    """Compile a document."""
    if args.no_llm:
        config = Config.no_llm()
    else:
        config = Config(max_tier=args.max_tier)
    
    if args.previous:
        print(f"Incremental compile: {args.source} (previous: {args.previous})")
        previous = load(args.previous)
        program, plan = compile_incremental(args.source, previous, config)
        program.write(args.output)
        
        print(f"\n✓ Compiled to {args.output}")
        print(f"  Nodes: {len(program.nodes)}")
        print(f"  Relations: {len(program.relations)}")
        print(f"\n  Incremental stats:")
        print(f"    Reused: {plan.reused_nodes} nodes ({plan.efficiency:.0%} work avoided)")
        print(f"    Recompiled: {plan.recompiled_nodes} nodes")
        print(f"    Removed: {plan.removed_nodes} nodes")
    else:
        print(f"Compiling {args.source}...")
        program = compile(args.source, config)
        program.write(args.output)
        
        print(f"\n✓ Compiled to {args.output}")
        print(f"  Nodes: {len(program.nodes)}")
        print(f"  Relations: {len(program.relations)}")
        print(f"  Schema: {IR_SCHEMA_VERSION}")


def cmd_query(args):
    """Query a compiled program."""
    program = load(args.program)
    space = query(program)
    
    results = list(space.query(
        major=args.major,
        label_contains=args.label,
        value_gt=args.gt,
        value_lt=args.lt,
    ))[:args.limit]
    
    print(f"\nResults ({len(results)} nodes):\n")
    for node in results:
        val = f" = {node.value}" if node.value is not None else ""
        unit = f" {node.unit}" if node.unit else ""
        print(f"  {node.id}: {node.label}{val}{unit}")


def cmd_info(args):
    """Show program info."""
    program = load(args.program)
    manifest = program.manifest()
    
    print(f"\nProgram: {args.program}")
    print(f"  AXM Version: {manifest.get('axm_version', 'unknown')}")
    print(f"  IR Schema: {manifest.get('ir_schema_version', 'unknown')}")
    print(f"  Created: {manifest['created_at']}")
    print(f"  Source: {manifest['source']['uri'][:60]}...")
    print(f"  Nodes: {manifest['counts']['nodes']}")
    print(f"  Relations: {manifest['counts']['relations']}")
    print(f"  Content hash: {manifest['content_hash']}")


def cmd_inspect(args):
    """Detailed inspection of program contents."""
    program = load(args.program)
    space = query(program)
    
    print(f"\n{'='*60}")
    print(f"PROGRAM INSPECTION: {args.program}")
    print(f"{'='*60}")
    
    print("\nNodes by Category:")
    for m in range(1, 9):
        count = space.count(major=m)
        if count > 0:
            name = Major(m).name
            print(f"  {m} ({name}): {count}")
    
    print("\nSample Nodes:")
    for i, node in enumerate(space.all_nodes()):
        if i >= args.limit:
            print(f"  ... ({len(program.nodes) - args.limit} more)")
            break
        val = f" = {node.value}" if node.value is not None else ""
        unit = f" {node.unit}" if node.unit else ""
        print(f"  {node.id}: {node.label[:40]}{val}{unit}")
    
    if program.relations:
        print(f"\nRelations ({len(program.relations)}):")
        for i, rel in enumerate(program.relations):
            if i >= 10:
                print(f"  ... ({len(program.relations) - 10} more)")
                break
            print(f"  {rel.subject_id} --{rel.predicate}--> {rel.object_id}")
    
    print(f"\nChunks ({len(program.chunk_index)}):")
    for chunk_id, node_ids in list(program.chunk_index.items())[:5]:
        hash_ = program.chunk_hashes.get(chunk_id, "?")[:8]
        print(f"  {chunk_id} ({hash_}...): {len(node_ids)} nodes")


def cmd_stats(args):
    """Show detailed statistics."""
    program = load(args.program)
    space = query(program)
    
    print(f"\n{'='*60}")
    print(f"STATISTICS: {args.program}")
    print(f"{'='*60}")
    
    print("\nCounts:")
    print(f"  Nodes: {len(program.nodes)}")
    print(f"  Relations: {len(program.relations)}")
    print(f"  Provenance: {len(program.provenance)}")
    print(f"  Chunks: {len(program.chunk_index)}")
    
    print("\nBy Major Category:")
    total = len(program.nodes) or 1
    for m in range(1, 9):
        count = space.count(major=m)
        name = Major(m).name
        pct = count / total * 100
        print(f"  {m} {name:12}: {count:5} ({pct:5.1f}%)")
    
    stats = program.stats
    print("\nBy Extraction Tier:")
    for i in range(5):
        count = stats.get(f"tier_{i}", 0)
        print(f"  Tier {i}: {count}")
    
    print(f"\nDeduplication: {stats.get('deduplicated', 0)} nodes")
    
    values = []
    for node in space.query(major=Major.QUANTITY):
        if node.value is not None:
            try:
                values.append(float(node.value))
            except (ValueError, TypeError):
                pass
    
    if values:
        print(f"\nQuantity Values:")
        print(f"  Count: {len(values)}")
        print(f"  Min: {min(values):,.2f}")
        print(f"  Max: {max(values):,.2f}")
        print(f"  Sum: {sum(values):,.2f}")


def cmd_diff(args):
    """Compare two programs."""
    prog_a = load(args.a)
    prog_b = load(args.b)
    
    nodes_a = set(prog_a.nodes.keys())
    nodes_b = set(prog_b.nodes.keys())
    
    added = nodes_b - nodes_a
    removed = nodes_a - nodes_b
    common = nodes_a & nodes_b
    
    changed = [nid for nid in common 
               if prog_a.nodes[nid].content_hash != prog_b.nodes[nid].content_hash]
    unchanged = len(common) - len(changed)
    
    print(f"\n{'='*60}")
    print(f"DIFF: {args.a} → {args.b}")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Added: {len(added)} nodes")
    print(f"  Removed: {len(removed)} nodes")
    print(f"  Changed: {len(changed)} nodes")
    print(f"  Unchanged: {unchanged} nodes")
    
    if not args.quiet:
        if added:
            print(f"\n+ Added:")
            for nid in sorted(added)[:10]:
                print(f"    + {nid}: {prog_b.nodes[nid].label[:50]}")
            if len(added) > 10:
                print(f"    ... and {len(added) - 10} more")
        
        if removed:
            print(f"\n- Removed:")
            for nid in sorted(removed)[:10]:
                print(f"    - {nid}: {prog_a.nodes[nid].label[:50]}")
            if len(removed) > 10:
                print(f"    ... and {len(removed) - 10} more")


def cmd_repl(args):
    """Interactive exploration."""
    program = load(args.program)
    space = query(program)
    
    print(f"\nAXM REPL - {args.program}")
    print(f"  {len(program.nodes)} nodes, {len(program.relations)} relations")
    print("\nCommands: nodes, relations, query <text>, major <1-8>, get <id>,")
    print("          neighbors <id>, out <id>, in <id>, stats, quit\n")
    
    while True:
        try:
            line = input("axm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not line:
            continue
        
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "nodes":
                limit = int(arg) if arg else 20
                for i, n in enumerate(space.all_nodes()):
                    if i >= limit:
                        break
                    val = f" = {n.value}" if n.value is not None else ""
                    print(f"  {n.id}: {n.label[:50]}{val}")
            elif cmd == "relations":
                for i, r in enumerate(program.relations[:20]):
                    print(f"  {r.subject_id} --{r.predicate}--> {r.object_id}")
            elif cmd == "query" and arg:
                for n in list(space.query(label_contains=arg))[:20]:
                    val = f" = {n.value}" if n.value is not None else ""
                    print(f"  {n.id}: {n.label}{val}")
            elif cmd == "major" and arg:
                for n in list(space.query(major=int(arg)))[:20]:
                    val = f" = {n.value}" if n.value is not None else ""
                    print(f"  {n.id}: {n.label}{val}")
            elif cmd == "get" and arg:
                n = space.get(arg)
                if n:
                    print(f"  ID: {n.id}, Label: {n.label}")
                    print(f"  Value: {n.value}, Unit: {n.unit}")
                    print(f"  Major: {Major(n.coord.major).name}")
            elif cmd == "neighbors" and arg:
                for r in list(space.neighbors(arg, radius=2.0))[:10]:
                    print(f"  {r.node.id}: {r.node.label[:40]} (d={r.distance:.2f})")
            elif cmd == "out" and arg:
                for rel in space.outgoing(arg):
                    print(f"  --{rel.predicate}--> {rel.object_id}")
            elif cmd == "in" and arg:
                for rel in space.incoming(arg):
                    print(f"  {rel.subject_id} --{rel.predicate}-->")
            elif cmd == "stats":
                print(f"  Nodes: {len(program.nodes)}, Relations: {len(program.relations)}")
            else:
                print("Unknown command")
        except Exception as e:
            print(f"Error: {e}")


def cmd_chat(args):
    """Chat with knowledge base."""
    chat_repl(args.program, model=args.model, use_mock=args.mock)


def main():
    parser = argparse.ArgumentParser(description="AXM Semantic Compiler")
    parser.add_argument("--version", action="version", version=f"AXM 0.5.3")
    
    sub = parser.add_subparsers(dest="command")
    
    c = sub.add_parser("compile", help="Compile document")
    c.add_argument("source")
    c.add_argument("-o", "--output", default="output.axm")
    c.add_argument("--previous", help="Previous .axm for incremental")
    c.add_argument("--max-tier", type=int, default=3)
    c.add_argument("--no-llm", action="store_true")
    
    q = sub.add_parser("query", help="Query program")
    q.add_argument("program")
    q.add_argument("--major", type=int)
    q.add_argument("--label")
    q.add_argument("--gt", type=float)
    q.add_argument("--lt", type=float)
    q.add_argument("--limit", type=int, default=20)
    
    sub.add_parser("info", help="Show info").add_argument("program")
    
    ins = sub.add_parser("inspect", help="Inspect program")
    ins.add_argument("program")
    ins.add_argument("--limit", type=int, default=20)
    
    sub.add_parser("stats", help="Statistics").add_argument("program")
    
    d = sub.add_parser("diff", help="Compare programs")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("-q", "--quiet", action="store_true")
    
    sub.add_parser("repl", help="Interactive mode").add_argument("program")
    
    ch = sub.add_parser("chat", help="Chat with KB")
    ch.add_argument("program")
    ch.add_argument("--model", default="llama3", help="Ollama model")
    ch.add_argument("--mock", action="store_true", help="Use mock LLM")
    
    args = parser.parse_args()
    
    cmds = {
        "compile": cmd_compile, "query": cmd_query, "info": cmd_info,
        "inspect": cmd_inspect, "stats": cmd_stats, "diff": cmd_diff, 
        "repl": cmd_repl, "chat": cmd_chat,
    }
    
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
