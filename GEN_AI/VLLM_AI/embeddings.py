"""
VLLM-based embedder using sentence-transformers.
Faster than Ollama for embedding generation.

Install: pip install sentence-transformers vllm
"""

from typing import List
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class VLLMEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        self.model = model
        self.embedder = SentenceTransformer(model)

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a list of floats."""
        embedding = self.embedder.encode(text, convert_to_tensor=False)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings efficiently in batch."""
        embeddings = self.embedder.encode(texts, convert_to_tensor=False)
        return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
