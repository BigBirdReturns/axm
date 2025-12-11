#!/usr/bin/env python3
"""
AXM Example: Incremental Compilation

This example demonstrates how to use incremental compilation to
efficiently process document updates without re-extracting unchanged content.
"""

from axm import compile, compile_incremental, query, Config, Major

# Original Q3 report
doc_q3 = """
Acme Corp Q3 2024 Quarterly Report

Revenue: $500 million
Net Income: $50 million
Operating Margin: 10%

The company maintains strong market position in the widget sector.
CEO Jane Smith announced plans for expansion into European markets.
"""

# Updated Q4 report (same company info, different financials)
doc_q4 = """
Acme Corp Q4 2024 Quarterly Report

Revenue: $620 million
Net Income: $75 million
Operating Margin: 12%

The company maintains strong market position in the widget sector.
CEO Jane Smith announced plans for expansion into European markets.
"""

def main():
    print("="*60)
    print("AXM Incremental Compilation Example")
    print("="*60)
    
    # Initial compile
    config = Config.default()
    
    print("\n[1] Compiling Q3 report...")
    q3 = compile(doc_q3, config)
    print(f"    Nodes: {len(q3.nodes)}")
    print(f"    Chunks: {len(q3.chunk_index)}")
    
    # Show Q3 quantities
    space_q3 = query(q3)
    print("\n    Q3 Quantities:")
    for node in space_q3.query(major=Major.QUANTITY):
        print(f"      {node.label}: {node.value}")
    
    # Incremental compile
    print("\n[2] Incremental compile Q4 (reusing unchanged chunks)...")
    q4, plan = compile_incremental(doc_q4, q3, config)
    
    print(f"\n    Incremental Plan:")
    print(f"      Reused: {plan.reused_nodes} nodes")
    print(f"      Recompiled: {plan.recompiled_nodes} nodes")
    print(f"      Removed: {plan.removed_nodes} nodes")
    print(f"      Efficiency: {plan.efficiency:.0%} work avoided")
    
    # Show Q4 quantities
    space_q4 = query(q4)
    print("\n    Q4 Quantities:")
    for node in space_q4.query(major=Major.QUANTITY):
        print(f"      {node.label}: {node.value}")
    
    # Compare
    print("\n[3] What changed?")
    q3_values = {n.label: n.value for n in space_q3.query(major=Major.QUANTITY)}
    q4_values = {n.label: n.value for n in space_q4.query(major=Major.QUANTITY)}
    
    for label in set(q3_values) | set(q4_values):
        v3 = q3_values.get(label)
        v4 = q4_values.get(label)
        if v3 != v4:
            print(f"    {label}: {v3} → {v4}")
    
    print("\n" + "="*60)
    print("Done! Incremental compilation saved extraction work on")
    print("unchanged content like company descriptions.")
    print("="*60)


if __name__ == "__main__":
    main()
