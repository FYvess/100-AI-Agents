"""
VLLMRAGClient: Drop-in replacement for RAGClient using VLLM for faster inference.

Uses:
  - VLLM OpenAI-compatible API for LLM generation (faster than Ollama)
  - sentence-transformers for local embeddings (faster than Ollama)
  - pgvector (Postgres) for retrieval, when use_rag=True

VLLM serves models via OpenAI-compatible API:
    vllm serve qwen2.5-7b-instruct

Usage mirrors RAGClient:

    client = VLLMRAGClient(pg_conn_string="postgresql://user:pass@localhost/ragdb")

    response = client.chat.completions.create(
        model="qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": "..."}],
        response_format={"type": "json_object"},
        use_rag=True,
        top_k=5,
    )

    print(response.choices[0].message.content)

NOTE: Uses direct HTTP (no external SDK dependencies) for privacy.
All communication is local to http://localhost:8000/v1
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from .embeddings import VLLMEmbedder
from .retriever import PgVectorRetriever
from .response_types import ChatCompletionResponse, Choice, Message, Source

try:
    import requests
except ImportError:
    requests = None


class _Completions:
    def __init__(self, parent: "VLLMRAGClient"):
        self._parent = parent

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        use_rag: bool = True,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        temperature: float = 0.2,
        **vllm_kwargs: Any,
    ) -> ChatCompletionResponse:
        parent = self._parent
        sources: List[Source] = []

        # Work on a copy so we don't mutate the caller's message list
        working_messages = [dict(m) for m in messages]

        if use_rag:
            # Retrieve context based on the last user message
            last_user_msg = next(
                (m["content"] for m in reversed(working_messages) if m["role"] == "user"),
                None,
            )
            if last_user_msg is None:
                raise ValueError("use_rag=True requires at least one user message.")

            query_embedding = parent.embedder.embed(last_user_msg)
            sources = parent.retriever.search(
                query_embedding, top_k=top_k, doc_type=doc_type
            )

            context_block = "\n\n".join(
                f"[Source: {s.source}]\n{s.text}" for s in sources
            )

            # Inject retrieved context ahead of the last user message
            rag_prefix = (
                "Use the following retrieved context to answer the question. "
                "If the context is insufficient, say so explicitly.\n\n"
                f"Context:\n{context_block}\n\n"
                "Question:"
            )
            for i in range(len(working_messages) - 1, -1, -1):
                if working_messages[i]["role"] == "user":
                    working_messages[i]["content"] = (
                        f"{rag_prefix}\n{working_messages[i]['content']}"
                    )
                    break

        content = parent._call_vllm(
            model=model,
            messages=working_messages,
            response_format=response_format,
            temperature=temperature,
            **vllm_kwargs,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            model=model,
            choices=[Choice(index=0, message=Message(role="assistant", content=content))],
            sources=sources,
        )



class _Chat:
    def __init__(self, parent: "VLLMRAGClient"):
        self.completions = _Completions(parent)


class VLLMRAGClient:
    def __init__(
        self,
        pg_conn_string: str,
        vllm_base_url: str = "http://localhost:8000/v1",
        embed_model: str = "all-MiniLM-L6-v2",
        rag_table: str = "rag_chunks",
    ):
        """
        Initialize VLLM-based RAG client (direct HTTP, no external SDK).

        Args:
            pg_conn_string: PostgreSQL connection string for pgvector storage
            vllm_base_url: VLLM OpenAI-compatible API base URL
            embed_model: Embedding model (sentence-transformers)
            rag_table: PostgreSQL table name for RAG chunks
        """
        if requests is None:
            raise ImportError("requests not installed. Run: pip install requests")

        self.vllm_base_url = vllm_base_url
        self.embedder = VLLMEmbedder(model=embed_model)
        self.retriever = PgVectorRetriever(conn_string=pg_conn_string, table=rag_table)
        self.chat = _Chat(self)

    def _call_vllm(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Call VLLM via direct HTTP (no SDK dependency)."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format and response_format.get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        payload.update(kwargs)

        try:
            response = requests.post(
                f"{self.vllm_base_url}/chat/completions",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to VLLM server at {self.vllm_base_url}. "
                "Start it with: vllm serve qwen2.5-7b-instruct"
            )
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected VLLM response format: {e}")
