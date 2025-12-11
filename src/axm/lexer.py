"""
AXM Lexer - Document Chunking (v0.4)

Chunks documents and classifies by extraction tier.
Adds content_hash for incremental compilation support.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Optional

from .ids import IDGenerator


# =============================================================================
# CHUNK TYPES
# =============================================================================

class ChunkType(Enum):
    """Document chunk classification."""
    JSON = auto()
    XML = auto()
    XBRL = auto()
    TABLE = auto()
    KEY_VALUE = auto()
    LIST = auto()
    CITATION = auto()  # References, bibliography entries
    METADATA = auto()
    CODE = auto()
    PROSE = auto()


@dataclass
class Chunk:
    """A segment of source document with content hash."""
    chunk_id: str
    chunk_type: ChunkType
    content: str
    tier: int
    start: int
    end: int
    confidence: float = 1.0
    patterns: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # Parent heading, section info
    
    @property
    def length(self) -> int:
        return self.end - self.start
    
    @property
    def content_hash(self) -> str:
        """Hash for change detection in incremental compilation."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


# =============================================================================
# PATTERN DETECTION
# =============================================================================

class Patterns:
    """Detect extractable patterns in text."""
    
    MONEY = re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B|K))?', re.I)
    PERCENT = re.compile(r'[\d.]+\s*%')
    DATE = re.compile(
        r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'\s+\d{1,2},?\s+\d{4})\b', re.I
    )
    
    KV = re.compile(r'^([A-Z][A-Za-z\s]+):\s*(.+)$', re.M)
    LIST_ITEM = re.compile(r'^[\-\*]\s+(.+)$', re.M)
    NUMBERED = re.compile(r'^\d+\.\s+(.+)$', re.M)
    
    # Improved table detection - handles alignment colons
    TABLE_PIPE = re.compile(r'^\|.+\|.+\|$', re.M)
    TABLE_TAB = re.compile(r'^.+\t.+\t.+$', re.M)
    TABLE_SEPARATOR = re.compile(r'^\|?[\s:]*-+[\s:|-]*\|?$', re.M)  # |:---|:---:| style
    
    CODE_FENCE = re.compile(r'^```(\w*)\n(.*?)^```', re.M | re.S)
    
    # Citation/reference patterns
    CITATION_HEADER = re.compile(r'^(?:References|Bibliography|Works Cited|Citations)\s*$', re.I | re.M)
    CITATION_ENTRY = re.compile(r'^\[?\d+\]?\s*[A-Z]', re.M)  # [1] Author or 1. Author
    
    @classmethod
    def detect(cls, text: str) -> Dict[str, Any]:
        """Detect all patterns in text."""
        patterns = {}
        
        money = cls.MONEY.findall(text)
        if money:
            patterns["money"] = money
        
        percent = cls.PERCENT.findall(text)
        if percent:
            patterns["percent"] = percent
        
        dates = cls.DATE.findall(text)
        if dates:
            patterns["dates"] = dates
        
        kv = cls.KV.findall(text)
        if kv:
            patterns["key_value"] = [{"key": k.strip(), "value": v.strip()} for k, v in kv]
        
        items = cls.LIST_ITEM.findall(text) + cls.NUMBERED.findall(text)
        if items:
            patterns["list_items"] = items
        
        # Improved table detection
        if cls.TABLE_PIPE.search(text) or cls.TABLE_TAB.search(text):
            # Verify it's actually a table by checking for separator row
            if cls.TABLE_SEPARATOR.search(text) or cls.TABLE_TAB.search(text):
                patterns["has_table"] = True
        
        code = cls.CODE_FENCE.findall(text)
        if code:
            patterns["code"] = [{"lang": lang, "code": c} for lang, c in code]
        
        # Citation detection
        if cls.CITATION_ENTRY.search(text):
            patterns["has_citations"] = True
        
        return patterns


# =============================================================================
# LEXER
# =============================================================================

class Lexer:
    """Chunks documents and classifies by extraction tier."""
    
    def __init__(self):
        self._ids = IDGenerator()
        self._current_section: Optional[str] = None  # Track section headers
    
    def lex(self, content: str) -> List[Chunk]:
        """Lex document into typed chunks."""
        self._ids.reset()
        self._current_section = None
        
        stripped = content.strip()
        
        if stripped.startswith('{') or stripped.startswith('['):
            return [self._chunk(ChunkType.JSON, content, 0, 0, len(content))]
        
        if stripped.startswith('<?xml') or stripped.startswith('<'):
            chunk_type = ChunkType.XBRL if 'xbrl' in content.lower() else ChunkType.XML
            return [self._chunk(chunk_type, content, 0, 0, len(content))]
        
        return self._lex_text(content)
    
    def _lex_text(self, content: str) -> List[Chunk]:
        """Lex text content with section awareness."""
        chunks = []
        paragraphs = re.split(r'\n\s*\n', content)
        
        pos = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            start = content.find(para, pos)
            end = start + len(para)
            pos = end
            
            # Check if this is a section header
            if self._is_section_header(para):
                self._current_section = para.lower()
            
            chunk = self._classify_paragraph(para, start, end)
            chunks.append(chunk)
        
        return chunks
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header."""
        # Short, title-case or all caps, no punctuation at end
        if len(text) > 100:
            return False
        if text.endswith(('.', ',', ';', ':')):
            return False
        # Check for header patterns
        if re.match(r'^#+\s+', text):  # Markdown header
            return True
        if re.match(r'^[A-Z][A-Z\s]+$', text):  # ALL CAPS
            return True
        if re.match(r'^(?:References|Bibliography|Abstract|Introduction|Conclusion|Methods|Results|Discussion)\s*$', text, re.I):
            return True
        return False
    
    def _classify_paragraph(self, text: str, start: int, end: int) -> Chunk:
        """Classify a paragraph by extraction tier with context awareness."""
        patterns = Patterns.detect(text)
        words = len(text.split())
        
        # Context-aware: Citations section
        if self._current_section and 'reference' in self._current_section:
            if patterns.get("has_citations") or patterns.get("list_items"):
                return self._chunk(ChunkType.CITATION, text, 1, start, end, 
                                   patterns=patterns, context={"section": self._current_section})
        
        if patterns.get("has_table"):
            return self._chunk(ChunkType.TABLE, text, 0, start, end, patterns=patterns)
        
        if patterns.get("code"):
            return self._chunk(ChunkType.CODE, text, 0, start, end, patterns=patterns)
        
        kv = patterns.get("key_value", [])
        if kv and len(kv) * 10 > words:
            return self._chunk(ChunkType.KEY_VALUE, text, 1, start, end, patterns=patterns)
        
        items = patterns.get("list_items", [])
        if items and len(items) * 5 > words:
            return self._chunk(ChunkType.LIST, text, 1, start, end, patterns=patterns)
        
        extractable = (
            len(patterns.get("money", [])) +
            len(patterns.get("percent", [])) +
            len(patterns.get("dates", []))
        )
        if extractable > 0 and words < 20:
            return self._chunk(ChunkType.METADATA, text, 1, start, end, patterns=patterns)
        
        tier = 3 if words < 200 else 4
        return self._chunk(ChunkType.PROSE, text, tier, start, end, patterns=patterns)
    
    def _chunk(
        self,
        chunk_type: ChunkType,
        content: str,
        tier: int,
        start: int,
        end: int,
        patterns: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> Chunk:
        """Create a chunk."""
        return Chunk(
            chunk_id=self._ids.next("chunk"),
            chunk_type=chunk_type,
            content=content,
            tier=tier,
            start=start,
            end=end,
            patterns=patterns or {},
            context=context or {},
        )
    
    def summary(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Summarize lexer output."""
        by_tier = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        by_type: Dict[str, int] = {}
        chunk_hashes = {}
        
        for chunk in chunks:
            by_tier[chunk.tier] += 1
            t = chunk.chunk_type.name
            by_type[t] = by_type.get(t, 0) + 1
            chunk_hashes[chunk.chunk_id] = chunk.content_hash
        
        cheap = by_tier[0] + by_tier[1]
        expensive = by_tier[3] + by_tier[4]
        
        return {
            "chunks": len(chunks),
            "by_tier": by_tier,
            "by_type": by_type,
            "efficiency": cheap / expensive if expensive > 0 else float('inf'),
            "chunk_hashes": chunk_hashes,
        }
