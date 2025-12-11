"""
AXM Chat - Conversational Knowledge Base Interface

Local RAG replacement: Query your compiled KB with natural language,
answered by a local LLM using structured facts as context.

Usage:
    from axm.chat import Chat
    
    chat = Chat("library.axm")
    answer = chat.ask("What was revenue in Q3?")
    
CLI:
    axm chat library.axm
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .program import Program, load
from .space import Space, query
from .coords import Major


# =============================================================================
# KEYWORD EXTRACTION
# =============================================================================

# Common words to ignore when extracting query keywords
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "and", "but",
    "if", "or", "because", "until", "while", "although", "though",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "about", "tell",
    "show", "give", "find", "get", "know", "think", "see", "look",
}

# Question type patterns
QUANTITY_WORDS = {"how much", "how many", "what is the", "what was the", "what's the", "amount", "total", "sum", "count"}
TIME_WORDS = {"when", "what date", "what year", "what time", "which quarter", "which year"}
ENTITY_WORDS = {"who", "which company", "which organization", "what company"}
LOCATION_WORDS = {"where", "which city", "which country", "what location"}


def extract_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from a question."""
    # Lowercase and tokenize
    words = re.findall(r'\b[a-z0-9]+\b', question.lower())
    
    # Filter stopwords and short words
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    return keywords


def detect_query_type(question: str) -> Tuple[Optional[int], List[str]]:
    """
    Detect what type of answer the question is looking for.
    Returns (major_category, additional_keywords).
    """
    q = question.lower()
    
    # Check for quantity questions
    for pattern in QUANTITY_WORDS:
        if pattern in q:
            return (Major.QUANTITY, ["revenue", "income", "profit", "cost", "margin", "growth"])
    
    # Check for time questions
    for pattern in TIME_WORDS:
        if pattern in q:
            return (Major.TIME, [])
    
    # Check for entity questions
    for pattern in ENTITY_WORDS:
        if pattern in q:
            return (Major.ENTITY, [])
    
    # Check for location questions
    for pattern in LOCATION_WORDS:
        if pattern in q:
            return (Major.LOCATION, [])
    
    return (None, [])


# =============================================================================
# CONTEXT BUILDING
# =============================================================================

@dataclass
class RetrievedFact:
    """A fact retrieved from the KB for context."""
    node_id: str
    label: str
    value: Any
    unit: Optional[str]
    category: str
    relevance: float = 1.0
    
    def format(self) -> str:
        """Format for LLM context."""
        if self.value is not None:
            unit_str = f" {self.unit}" if self.unit else ""
            return f"- {self.label}: {self.value}{unit_str}"
        return f"- {self.label}"


class ContextBuilder:
    """Builds relevant context from KB for a question."""
    
    def __init__(self, space: Space, max_facts: int = 20):
        self.space = space
        self.max_facts = max_facts
    
    def retrieve(self, question: str) -> List[RetrievedFact]:
        """Retrieve relevant facts for a question."""
        facts = []
        seen_hashes = set()
        
        # Extract keywords and detect type
        keywords = extract_keywords(question)
        major_hint, extra_keywords = detect_query_type(question)
        
        all_keywords = keywords + extra_keywords
        
        # Strategy 1: Search by major category if detected
        if major_hint is not None:
            for node in self.space.query(major=major_hint):
                if node.content_hash not in seen_hashes:
                    seen_hashes.add(node.content_hash)
                    facts.append(RetrievedFact(
                        node_id=node.id,
                        label=node.label,
                        value=node.value,
                        unit=node.unit,
                        category=Major(node.coord.major).name,
                        relevance=0.8,
                    ))
        
        # Strategy 2: Search by keywords
        for keyword in all_keywords:
            if len(keyword) < 3:
                continue
            for node in self.space.query(label_contains=keyword):
                if node.content_hash not in seen_hashes:
                    seen_hashes.add(node.content_hash)
                    facts.append(RetrievedFact(
                        node_id=node.id,
                        label=node.label,
                        value=node.value,
                        unit=node.unit,
                        category=Major(node.coord.major).name,
                        relevance=1.0,
                    ))
        
        # Sort by relevance and limit
        facts.sort(key=lambda f: -f.relevance)
        return facts[:self.max_facts]
    
    def format_context(self, facts: List[RetrievedFact]) -> str:
        """Format facts as context string for LLM."""
        if not facts:
            return "No relevant facts found in the knowledge base."
        
        # Group by category
        by_category: Dict[str, List[RetrievedFact]] = {}
        for fact in facts:
            if fact.category not in by_category:
                by_category[fact.category] = []
            by_category[fact.category].append(fact)
        
        lines = ["Knowledge base facts:"]
        for category, cat_facts in sorted(by_category.items()):
            lines.append(f"\n{category}:")
            for fact in cat_facts:
                lines.append(fact.format())
        
        return "\n".join(lines)


# =============================================================================
# LLM INTERFACE
# =============================================================================

class OllamaChat:
    """Chat completions via local Ollama."""
    
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def complete(self, prompt: str) -> str:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")
        
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.status_code} - {response.text}")
        
        return response.json().get("response", "").strip()
    
    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False


class MockChat:
    """Mock LLM for testing without Ollama."""
    
    def complete(self, prompt: str) -> str:
        # Extract the question and facts from the prompt
        if "No relevant facts" in prompt:
            return "I don't have information about that in my knowledge base."
        
        # Return a simple templated response
        return "Based on the knowledge base, I found several relevant facts. Please review the context provided."
    
    def is_available(self) -> bool:
        return True


# =============================================================================
# CHAT INTERFACE
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant answering questions based on a knowledge base.
You will be given extracted facts from documents. Use ONLY these facts to answer.
If the facts don't contain the answer, say so. Don't make up information.
Be concise and direct. Cite specific values when available."""

QUERY_TEMPLATE = """{system}

{context}

Question: {question}

Answer based only on the facts above:"""


class Chat:
    """
    Conversational interface to an AXM knowledge base.
    
    Usage:
        chat = Chat("library.axm")
        answer = chat.ask("What was total revenue?")
        
        # Or with custom model
        chat = Chat("library.axm", model="mistral")
    """
    
    def __init__(
        self,
        kb_path: str,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        max_context_facts: int = 20,
        use_mock: bool = False,
    ):
        self.program = load(kb_path)
        self.space = query(self.program)
        self.context_builder = ContextBuilder(self.space, max_facts=max_context_facts)
        
        if use_mock:
            self.llm = MockChat()
        else:
            self.llm = OllamaChat(model=model, host=host)
        
        self.history: List[Tuple[str, str, List[RetrievedFact]]] = []
    
    def ask(self, question: str, show_sources: bool = False) -> str:
        """
        Ask a question about the knowledge base.
        
        Args:
            question: Natural language question
            show_sources: If True, append source node IDs to answer
            
        Returns:
            Answer string from LLM
        """
        # Retrieve relevant facts
        facts = self.context_builder.retrieve(question)
        context = self.context_builder.format_context(facts)
        
        # Build prompt
        prompt = QUERY_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
        )
        
        # Get answer
        answer = self.llm.complete(prompt)
        
        # Track history
        self.history.append((question, answer, facts))
        
        # Optionally append sources
        if show_sources and facts:
            source_ids = [f.node_id for f in facts[:5]]
            answer += f"\n\n[Sources: {', '.join(source_ids)}]"
        
        return answer
    
    def get_facts(self, question: str) -> List[RetrievedFact]:
        """Get retrieved facts without asking LLM."""
        return self.context_builder.retrieve(question)
    
    def explain(self, question: str) -> str:
        """Show what facts would be retrieved for a question."""
        facts = self.context_builder.retrieve(question)
        
        lines = [f"Query: {question}", f"Keywords: {extract_keywords(question)}", ""]
        
        if facts:
            lines.append(f"Retrieved {len(facts)} facts:")
            for f in facts:
                lines.append(f"  [{f.node_id}] {f.label}: {f.value} ({f.category})")
        else:
            lines.append("No facts retrieved.")
        
        return "\n".join(lines)
    
    @property
    def kb_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "nodes": len(self.program.nodes),
            "relations": len(self.program.relations),
            "chunks": len(self.program.chunk_index),
            "source": self.program.source.uri,
        }


# =============================================================================
# CLI CHAT REPL
# =============================================================================

def chat_repl(kb_path: str, model: str = "llama3", use_mock: bool = False):
    """Interactive chat REPL."""
    
    print(f"\nAXM Chat - {kb_path}")
    print(f"Model: {'mock' if use_mock else model}")
    
    # Check Ollama availability
    if not use_mock:
        test_llm = OllamaChat(model=model)
        if not test_llm.is_available():
            print("\n⚠ Ollama not available. Using mock mode.")
            print("  Start Ollama with: ollama serve")
            print("  Or run with --mock flag\n")
            use_mock = True
    
    chat = Chat(kb_path, model=model, use_mock=use_mock)
    stats = chat.kb_stats
    
    print(f"KB: {stats['nodes']} nodes, {stats['relations']} relations")
    print("\nCommands:")
    print("  <question>     Ask a question")
    print("  /facts <q>     Show retrieved facts without LLM")
    print("  /explain <q>   Explain retrieval for question")
    print("  /stats         Show KB stats")
    print("  /history       Show conversation history")
    print("  /quit          Exit")
    print()
    
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not line:
            continue
        
        try:
            if line.startswith("/quit") or line.startswith("/exit"):
                print("Bye!")
                break
            
            elif line.startswith("/facts "):
                q = line[7:].strip()
                facts = chat.get_facts(q)
                print(f"\nRetrieved {len(facts)} facts:")
                for f in facts:
                    print(f"  {f.format()}")
                print()
            
            elif line.startswith("/explain "):
                q = line[9:].strip()
                print(f"\n{chat.explain(q)}\n")
            
            elif line == "/stats":
                for k, v in chat.kb_stats.items():
                    print(f"  {k}: {v}")
            
            elif line == "/history":
                if not chat.history:
                    print("  No history yet.")
                else:
                    for i, (q, a, _) in enumerate(chat.history):
                        print(f"  [{i+1}] Q: {q[:50]}...")
                        print(f"      A: {a[:50]}...")
            
            elif line.startswith("/"):
                print("Unknown command. Try /quit, /facts, /explain, /stats, /history")
            
            else:
                answer = chat.ask(line, show_sources=True)
                print(f"\naxm> {answer}\n")
        
        except Exception as e:
            print(f"Error: {e}\n")
