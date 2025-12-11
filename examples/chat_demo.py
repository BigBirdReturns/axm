#!/usr/bin/env python3
"""
AXM Chat Example - Local RAG Replacement

This example shows how to:
1. Compile documents into a knowledge base
2. Query it conversationally with a local LLM

Requirements:
- Ollama running locally (ollama serve)
- A model pulled (ollama pull llama3)

Or run with --mock to test without Ollama.
"""

from axm import compile, Config, Chat

# Sample document (could be a scanned book, PDF text, etc.)
DOCUMENT = """
The Art of War by Sun Tzu

Chapter 1: Laying Plans

Sun Tzu said: The art of war is of vital importance to the State.
It is a matter of life and death, a road either to safety or to ruin.

There are five constant factors to be considered:
1. The Moral Law - causes people to be in harmony with their ruler
2. Heaven - signifies night and day, cold and heat, seasons
3. Earth - comprises distances, danger and security, terrain
4. The Commander - stands for virtues of wisdom, sincerity, courage
5. Method and discipline - marshaling of the army, logistics

These five factors should be familiar to every general.
He who knows them will be victorious; he who knows them not will fail.

Chapter 2: Waging War

Sun Tzu said: In operations of war, where there are a thousand swift chariots,
a thousand heavy chariots, and a hundred thousand mail-clad soldiers, with
provisions enough to carry them a thousand li, the expenditure at home and
at the front will reach the total of a thousand ounces of silver per day.

When you engage in actual fighting, if victory is long in coming, men's
weapons will grow dull and their ardor will be damped. A prolonged campaign
will exhaust the state's resources.

Hence a wise general makes a point of foraging on the enemy.
One cartload of the enemy's provisions is equivalent to twenty of one's own.
"""


def main():
    print("="*60)
    print("AXM Chat - Local RAG Demo")
    print("="*60)
    
    # Compile the document
    print("\n[1] Compiling 'The Art of War' into KB...")
    program = compile(DOCUMENT, Config.no_llm())  # No LLM for extraction
    program.write("/tmp/artofwar.axm")
    print(f"    Nodes: {len(program.nodes)}")
    
    # Create chat interface
    print("\n[2] Starting chat (mock mode)...")
    print("    For real LLM answers, run: axm chat /tmp/artofwar.axm")
    
    chat = Chat("/tmp/artofwar.axm", use_mock=True)
    
    # Demo questions
    questions = [
        "What are the five constant factors?",
        "What did Sun Tzu say about prolonged campaigns?",
        "How much silver per day for war operations?",
    ]
    
    print("\n[3] Sample questions:\n")
    for q in questions:
        print(f"Q: {q}")
        facts = chat.get_facts(q)
        if facts:
            print("Retrieved facts:")
            for f in facts[:3]:
                val = f": {f.value}" if f.value else ""
                print(f"  - {f.label[:60]}{val}")
        print()
    
    print("="*60)
    print("To chat interactively with Ollama:")
    print("  1. Start Ollama: ollama serve")
    print("  2. Pull a model: ollama pull llama3")
    print("  3. Run: axm chat /tmp/artofwar.axm --model llama3")
    print("="*60)


if __name__ == "__main__":
    main()
