"""
RAGClient: drop-in-style replacement for the OpenAI client, but backed by:
  - a local Ollama LLM for generation
  - pgvector (Postgres) for retrieval, when use_rag=True

Usage mirrors openai's client:

    client = RAGClient(pg_conn_string="postgresql://user:pass@localhost/ragdb")

    response = client.chat.completions.create(
        model="qwen2.5:7b-instruct",
        messages=[{"role": "user", "content": "..."}],
        response_format={"type": "json_object"},
        use_rag=True,        # default True; set False for plain LLM calls
        top_k=5,
        doc_type=None,       # optional filter, e.g. "schema" or "document"
    )

    print(response.choices[0].message.content)
    print(response.sources)   # [] if use_rag=False
"""

import uuid
from typing import Any, Dict, List, Optional

import requests

from .embeddings import OllamaEmbedder
from .retriever import PgVectorRetriever
from .response_types import ChatCompletionResponse, Choice, Message, Source


class _Completions:
    def __init__(self, parent: "RAGClient"):
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
        **ollama_kwargs: Any,
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

        content = parent._call_ollama(
            model=model,
            messages=working_messages,
            response_format=response_format,
            temperature=temperature,
            **ollama_kwargs,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            model=model,
            choices=[Choice(index=0, message=Message(role="assistant", content=content))],
            sources=sources,
        )


class _Chat:
    def __init__(self, parent: "RAGClient"):
        self.completions = _Completions(parent)


class RAGClient:
    def __init__(
        self,
        pg_conn_string: str,
        ollama_host: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        rag_table: str = "rag_chunks",
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.embedder = OllamaEmbedder(model=embed_model, host=ollama_host)
        self.retriever = PgVectorRetriever(conn_string=pg_conn_string, table=rag_table)
        self.chat = _Chat(self)

    def _call_ollama(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        payload.update(kwargs)

        resp = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
