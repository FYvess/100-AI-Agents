"""
Thin wrapper around Ollama's embedding endpoint.
Requires `ollama serve` running locally (default http://localhost:11434)
and the embedding model pulled, e.g.:
    ollama pull nomic-embed-text
"""

import requests
from typing import List


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a list of floats."""
        resp = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings. Ollama's /api/embeddings is single-input,
        so this loops — fine for query-time (one embedding per call), use
        your ingestion pipeline's own batching strategy for bulk loads."""
        return [self.embed(t) for t in texts]
