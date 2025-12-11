"""
AXM LLM Executor - Handles Tier 3-4 Extraction (v0.4)

Features:
- Retry logic with exponential backoff
- Response validation
- Multiple backends (Mock, Ollama, Anthropic, OpenAI)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .parser import LLMRequest


# =============================================================================
# LLM RESULT
# =============================================================================

@dataclass
class LLMResult:
    """Result from LLM execution with metadata."""
    success: bool
    data: List[Dict[str, Any]]
    error: Optional[str] = None
    retries: int = 0
    latency_ms: float = 0.0


# =============================================================================
# RESPONSE PARSING & VALIDATION
# =============================================================================

def parse_json_response(text: str) -> List[Dict[str, Any]]:
    """Parse LLM response containing JSON."""
    # Strip markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Find JSON
    start = text.find('[')
    if start == -1:
        start = text.find('{')
        if start == -1:
            return []
        end = text.rfind('}')
        if end == -1:
            return []
        text = '[' + text[start:end+1] + ']'
    else:
        end = text.rfind(']')
        if end == -1:
            return []
        text = text[start:end+1]
    
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        return []


def validate_response(data: List[Dict[str, Any]]) -> bool:
    """Validate LLM response structure."""
    if not isinstance(data, list):
        return False
    
    for item in data:
        if not isinstance(item, dict):
            return False
        # Must have subject at minimum
        if "subject" not in item:
            return False
    
    return True


# =============================================================================
# RETRY EXECUTOR WRAPPER
# =============================================================================

class RetryExecutor:
    """
    Wraps any executor with retry logic and validation.
    
    Usage:
        base = MockExecutor()
        executor = RetryExecutor(base, max_retries=3)
        result = executor(request)
    """
    
    def __init__(
        self,
        executor: Callable[[LLMRequest], List[Dict]],
        max_retries: int = 3,
        base_delay: float = 1.0,
        validate: bool = True,
    ):
        self.executor = executor
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.validate = validate
    
    def __call__(self, request: LLMRequest) -> LLMResult:
        """Execute with retry and validation."""
        start = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                data = self.executor(request)
                
                if self.validate and not validate_response(data):
                    last_error = "Validation failed"
                    continue
                
                latency = (time.time() - start) * 1000
                return LLMResult(
                    success=True,
                    data=data,
                    retries=attempt,
                    latency_ms=latency,
                )
                
            except Exception as e:
                last_error = str(e)
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        latency = (time.time() - start) * 1000
        return LLMResult(
            success=False,
            data=[],
            error=last_error,
            retries=self.max_retries,
            latency_ms=latency,
        )


# =============================================================================
# MOCK EXECUTOR
# =============================================================================

class MockExecutor:
    """Mock LLM for testing. Uses regex to simulate claim extraction."""
    
    METRIC_PATTERNS = {
        "total_revenue": r"total revenue",
        "net_income": r"net income",
        "operating_income": r"operating income",
        "gross_profit": r"gross profit",
        "revenue": r"(?<!total )revenue",
        "expenses": r"expense",
    }
    
    def __call__(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """Execute mock extraction."""
        results = []
        content = request.content
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            
            metric = None
            for name, pattern in self.METRIC_PATTERNS.items():
                if re.search(pattern, sent_lower):
                    metric = name
                    break
            
            if not metric:
                continue
            
            money = re.search(r'\$[\d.,]+\s*(?:million|billion|M|B)?', sentence, re.I)
            if money:
                results.append({
                    "subject": metric,
                    "predicate": "has_value",
                    "object": money.group(0),
                    "confidence": 0.85,
                    "source": sentence[:100],
                })
            
            growth = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:increase|growth|grew)', sentence, re.I)
            if growth:
                results.append({
                    "subject": f"{metric}_growth",
                    "predicate": "percentage",
                    "object": f"{growth.group(1)}%",
                    "confidence": 0.8,
                })
        
        return results


# =============================================================================
# OLLAMA EXECUTOR
# =============================================================================

class OllamaExecutor:
    """Local LLM via Ollama. Recommended for production."""
    
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def __call__(self, request: LLMRequest) -> List[Dict[str, Any]]:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")
        
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": request.prompt,
                "stream": False,
            },
            timeout=60,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.status_code}")
        
        text = response.json().get("response", "")
        return parse_json_response(text)


# =============================================================================
# ANTHROPIC EXECUTOR
# =============================================================================

class AnthropicExecutor:
    """Claude API executor."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import os
            try:
                import anthropic
            except ImportError:
                raise ImportError("pip install anthropic")
            
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client
    
    def __call__(self, request: LLMRequest) -> List[Dict[str, Any]]:
        client = self._get_client()
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": request.prompt}],
        )
        
        text = response.content[0].text
        return parse_json_response(text)


# =============================================================================
# OPENAI EXECUTOR
# =============================================================================

class OpenAIExecutor:
    """OpenAI API executor."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import os
            try:
                import openai
            except ImportError:
                raise ImportError("pip install openai")
            
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = openai.OpenAI(api_key=key)
        return self._client
    
    def __call__(self, request: LLMRequest) -> List[Dict[str, Any]]:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=1024,
        )
        
        text = response.choices[0].message.content
        return parse_json_response(text)


# =============================================================================
# FACTORY
# =============================================================================

def get_executor(
    backend: str = "mock",
    with_retry: bool = True,
    **kwargs
) -> Callable[[LLMRequest], LLMResult] | Callable[[LLMRequest], List[Dict]]:
    """Get an executor by name, optionally wrapped with retry."""
    
    if backend == "mock":
        base = MockExecutor()
    elif backend == "ollama":
        base = OllamaExecutor(**kwargs)
    elif backend == "anthropic":
        base = AnthropicExecutor(**kwargs)
    elif backend == "openai":
        base = OpenAIExecutor(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    if with_retry:
        return RetryExecutor(base)
    return base
