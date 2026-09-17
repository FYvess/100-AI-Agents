"""
Lightweight classes that mimic the OpenAI SDK response shape.
Same as RAG_AI for compatibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str = "stop"


@dataclass
class Source:
    """A single retrieved chunk, attached to RAG-based responses for inspection."""
    chunk_id: int
    source: str
    doc_type: Optional[str]
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        snippet = self.text[:80].replace("\n", " ")
        return f"Source(source={self.source!r}, score={self.score:.3f}, text='{snippet}...')"


@dataclass
class ChatCompletionResponse:
    """Mimics openai.types.chat.ChatCompletion, with an extra `.sources` field."""
    id: str
    model: str
    choices: List[Choice]
    sources: List[Source] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"ChatCompletionResponse(model={self.model!r}, "
            f"content={self.choices[0].message.content[:80]!r}..., "
            f"sources={len(self.sources)})"
        )
