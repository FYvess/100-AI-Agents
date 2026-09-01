import sys
from pathlib import Path

rag_ai_parent = Path(__file__).parent.parent.parent / "GEN_AI"
sys.path.insert(0, str(rag_ai_parent))

from RAG_AI import RAGClient

client = RAGClient(
    pg_conn_string="postgresql://postgres:DB_password00@localhost:5432/pg_core",
    ollama_host="http://localhost:11434",
    embed_model="nomic-embed-text",
)

# Test embedding
print("Testing embedding...")
emb = client.embedder.embed("hello world")
print(f"Embedding successful! Length: {len(emb)}")

# Test chat
print("\nTesting chat...")
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    use_rag=False,
    temperature=0.2
)
print(f"Chat response: {response.choices[0].message.content}")
