#!/usr/bin/env python3
"""
AXM Intake Demo - Two-Path Architecture

Demonstrates:
1. Structured data (XBRL) → Adapter path → High confidence
2. Unstructured text → Compiler path → Variable confidence
3. Merging both into unified coordinate space
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

from axm import (
    compile_universal,
    merge_programs,
    query,
    avg_confidence,
    detect_source,
    list_adapters,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str) -> None:
    print(f"\n[{title}]\n")


# =============================================================================
# SAMPLE DATA
# =============================================================================

XBRL_SAMPLE = '''<?xml version="1.0"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:us-gaap="http://fasb.org/us-gaap/2020">
    <xbrli:context id="FY2024">
        <xbrli:entity>
            <xbrli:identifier scheme="http://www.sec.gov/CIK">0001234567</xbrli:identifier>
        </xbrli:entity>
        <xbrli:period>
            <xbrli:startDate>2024-01-01</xbrli:startDate>
            <xbrli:endDate>2024-12-31</xbrli:endDate>
        </xbrli:period>
    </xbrli:context>
    <xbrli:context id="Q4_2024">
        <xbrli:entity>
            <xbrli:identifier scheme="http://www.sec.gov/CIK">0001234567</xbrli:identifier>
        </xbrli:entity>
        <xbrli:period>
            <xbrli:instant>2024-12-31</xbrli:instant>
        </xbrli:period>
    </xbrli:context>
    <xbrli:unit id="USD">
        <xbrli:measure>iso4217:USD</xbrli:measure>
    </xbrli:unit>
    
    <us-gaap:Assets contextRef="Q4_2024" unitRef="USD">15000000000</us-gaap:Assets>
    <us-gaap:CashAndCashEquivalents contextRef="Q4_2024" unitRef="USD">3200000000</us-gaap:CashAndCashEquivalents>
    <us-gaap:Revenues contextRef="FY2024" unitRef="USD">12400000000</us-gaap:Revenues>
    <us-gaap:NetIncome contextRef="FY2024" unitRef="USD">1650000000</us-gaap:NetIncome>
    <us-gaap:StockholdersEquity contextRef="Q4_2024" unitRef="USD">6500000000</us-gaap:StockholdersEquity>
</xbrli:xbrl>'''


TEXT_SAMPLE = """
Acme Corp Q4 2024 Earnings Summary

Revenue grew 15% year-over-year to $12.4 billion, driven by strong performance 
in our cloud services division. Operating margins expanded to 22%, reflecting 
cost discipline and favorable product mix.

Key risks include supply chain disruptions and increasing competition in the 
enterprise software market. Management expects revenue growth of 10-12% in FY2025.

The company returned $500 million to shareholders through dividends and share 
repurchases during the quarter.
"""


ICAL_SAMPLE = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:20241215T090000Z
DTEND:20241215T100000Z
SUMMARY:Q4 Earnings Review
LOCATION:Conference Room A
END:VEVENT
BEGIN:VEVENT
DTSTART:20241216T140000Z
DTEND:20241216T150000Z
SUMMARY:Board Meeting
LOCATION:Boardroom
END:VEVENT
END:VCALENDAR"""


# =============================================================================
# DEMO
# =============================================================================

def main():
    print_header("AXM INTAKE DEMO: Two-Path Architecture")
    
    # -------------------------------------------------------------------------
    print_section("Available Adapters")
    # -------------------------------------------------------------------------
    
    adapters = list_adapters()
    print(f"Registered adapters: {', '.join(adapters)}")
    
    # -------------------------------------------------------------------------
    print_section("Source Detection")
    # -------------------------------------------------------------------------
    
    # Detect each source type
    sources = [
        ("XBRL financial data", XBRL_SAMPLE[:100] + "..."),
        ("Plain text narrative", TEXT_SAMPLE[:100] + "..."),
        ("iCalendar events", ICAL_SAMPLE[:100] + "..."),
    ]
    
    for name, preview in sources:
        if name == "XBRL financial data":
            content = XBRL_SAMPLE
        elif name == "Plain text narrative":
            content = TEXT_SAMPLE
        else:
            content = ICAL_SAMPLE
            
        source_type, path = detect_source(content)
        print(f"  {name}:")
        print(f"    Type: {source_type.name}")
        print(f"    Path: {path.name}")
        print()
    
    # -------------------------------------------------------------------------
    print_section("Compile: Structured Data (XBRL) → Adapter Path")
    # -------------------------------------------------------------------------
    
    p_xbrl = compile_universal(XBRL_SAMPLE)
    print(f"  Nodes extracted: {len(p_xbrl.nodes)}")
    print(f"  Confidence: {avg_confidence(p_xbrl):.1%}")
    print()
    print("  Sample nodes:")
    space_xbrl = query(p_xbrl)
    for node in list(space_xbrl.all_nodes())[:5]:
        value_str = f" = {node.value:,.0f}" if isinstance(node.value, (int, float)) else ""
        unit_str = f" {node.unit}" if node.unit else ""
        print(f"    • {node.label}{value_str}{unit_str}")
    
    # -------------------------------------------------------------------------
    print_section("Compile: Unstructured Text → Compiler Path")
    # -------------------------------------------------------------------------
    
    p_text = compile_universal(TEXT_SAMPLE)
    print(f"  Nodes extracted: {len(p_text.nodes)}")
    print(f"  Confidence: {avg_confidence(p_text):.1%}")
    print()
    print("  Sample nodes:")
    space_text = query(p_text)
    for node in list(space_text.all_nodes())[:5]:
        value_str = f" = {node.value}" if node.value else ""
        print(f"    • {node.label}{value_str}")
    
    # -------------------------------------------------------------------------
    print_section("Compile: Calendar Data → Adapter Path")
    # -------------------------------------------------------------------------
    
    p_ical = compile_universal(ICAL_SAMPLE)
    print(f"  Nodes extracted: {len(p_ical.nodes)}")
    print(f"  Confidence: {avg_confidence(p_ical):.1%}")
    print()
    print("  Events:")
    space_ical = query(p_ical)
    for node in space_ical.all_nodes():
        loc = node.metadata.get('location', 'N/A')
        print(f"    • {node.label} @ {loc}")
    
    # -------------------------------------------------------------------------
    print_section("Merge: Unified Coordinate Space")
    # -------------------------------------------------------------------------
    
    merged = merge_programs([p_xbrl, p_text, p_ical])
    print(f"  Total nodes: {len(merged.nodes)}")
    print(f"  From XBRL:   {len(p_xbrl.nodes)} (confidence 1.0)")
    print(f"  From text:   {len(p_text.nodes)} (confidence varies)")
    print(f"  From iCal:   {len(p_ical.nodes)} (confidence 1.0)")
    
    # -------------------------------------------------------------------------
    print_section("Query: Unified Space by Confidence")
    # -------------------------------------------------------------------------
    
    space = query(merged)
    
    # High confidence (from adapters)
    high_conf = []
    med_conf = []
    
    for node in space.all_nodes():
        prov = merged.provenance.get(node.prov_id)
        conf = prov.confidence if prov else 0.5
        if conf >= 0.95:
            high_conf.append(node)
        else:
            med_conf.append(node)
    
    print(f"  High confidence (≥95%): {len(high_conf)} nodes")
    print(f"  Medium confidence (<95%): {len(med_conf)} nodes")
    print()
    print("  High confidence nodes (from structured sources):")
    for node in high_conf[:5]:
        value_str = f" = {node.value:,.0f}" if isinstance(node.value, (int, float)) else ""
        unit_str = f" {node.unit}" if node.unit else ""
        print(f"    ✓ {node.label}{value_str}{unit_str}")
    
    # -------------------------------------------------------------------------
    print_header("THE POINT")
    # -------------------------------------------------------------------------
    
    print("""
    Two paths, one destination:
    
    STRUCTURED (schema exists)          UNSTRUCTURED (text soup)
         XBRL, FHIR, OpenAPI              PDF, Word, earnings calls
              ↓                                   ↓
         Adapter path                      Compiler path
         Confidence: 1.0                   Confidence: varies
              ↓                                   ↓
              └───────────→ MERGED SPACE ←────────┘
                              
    Query by:
    • Coordinates (what it IS)
    • Confidence (how sure are we)
    • Source (where it came from)
    
    This is how you build knowledge systems that scale.
    The schema IS the extraction when it exists.
    The compiler handles everything else.
    """)


if __name__ == "__main__":
    main()
