"""Quick test of VLLM_AI client structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    from VLLM_AI import VLLMRAGClient
    print("✅ VLLMRAGClient import successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test response types
try:
    from VLLM_AI.response_types import Message, Choice, Source, ChatCompletionResponse
    print("✅ Response types import successful")
except ImportError as e:
    print(f"❌ Response types import error: {e}")
    sys.exit(1)

# Test embedder (without sentence-transformers, just check structure)
try:
    from VLLM_AI.embeddings import VLLMEmbedder
    print("✅ VLLMEmbedder import successful")
except ImportError as e:
    print(f"❌ VLLMEmbedder import error: {e}")
    sys.exit(1)

# Test retriever
try:
    from VLLM_AI.retriever import PgVectorRetriever
    print("✅ PgVectorRetriever import successful")
except ImportError as e:
    print(f"❌ PgVectorRetriever import error: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All VLLM_AI components loaded successfully!")
print("="*50)

# Show structure
print("\nVLLMRAGClient initialization signature:")
print("  VLLMRAGClient(")
print("    pg_conn_string: str,")
print("    vllm_base_url: str = 'http://localhost:8000/v1',")
print("    embed_model: str = 'all-MiniLM-L6-v2',")
print("    rag_table: str = 'rag_chunks'")
print("  )")

print("\nUsage example:")
print("""
  from VLLM_AI import VLLMRAGClient
  
  client = VLLMRAGClient(
      pg_conn_string="postgresql://user:pass@localhost:5432/db"
  )
  
  response = client.chat.completions.create(
      model="qwen2.5-7b-instruct",
      messages=[{"role": "user", "content": "..."}],
      response_format={"type": "json_object"},
      use_rag=True
  )
""")
