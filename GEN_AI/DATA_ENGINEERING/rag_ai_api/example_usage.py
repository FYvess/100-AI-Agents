"""
Example usage of rag_ai_api.RAGClient

Prerequisites:
    1. Ollama running locally:        ollama serve
    2. Models pulled:
         ollama pull qwen2.5:7b-instruct
         ollama pull nomic-embed-text
    3. Postgres + pgvector running, with `rag_chunks` table already populated
       by your ingestion pipeline (separate from this client).
    4. pip install psycopg2-binary requests
"""

import json
from rag_ai_api import RAGClient

client = RAGClient(
    pg_conn_string="postgresql://postgres:postgres@localhost:5432/pg_core",
    ollama_host="http://localhost:11434",
    embed_model="nomic-embed-text",
)


# ----------------------------------------------------------------------
# Case 1: Extraction — no retrieval needed, text is passed directly.
# Mirrors your original gpt-4o-mini contract-extraction example.
# ----------------------------------------------------------------------
contract_text = """
This Employment Agreement is made between Acme Corp and John Doe.
John Doe will serve as Senior Engineer with a yearly salary of $145,000,
starting on March 1, 2025. This agreement includes a non-compete clause
restricting employment with direct competitors for 12 months post-termination.
"""

prompt = f"""Given this contract text, extract the following fields: 'Employee Name',
'Yearly Salary', 'Non-Compete Clause (Y/N)', 'Start Date'. Output in the following JSON format
{{"Agreement": {{"Employee Name": "...", "Yearly Salary": "...", "Non-Compete Clause (Y/N)": "...", "Start Date": "..."}}}}

Contract text:
{contract_text}
"""

response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    use_rag=False,  # <-- plain LLM call, no pgvector retrieval
)

extracted = json.loads(response.choices[0].message.content)
print("Extraction result:")
print(json.dumps(extracted, indent=2))
print("Sources used:", response.sources)  # [] since use_rag=False


# ----------------------------------------------------------------------
# Case 2: RAG query — retrieval happens automatically inside create().
# Useful for grounded Q&A over your ingested documents/schemas.
# ----------------------------------------------------------------------
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": "What's our refund policy for enterprise clients?"}],
    use_rag=True,        # default, shown explicitly here
    top_k=5,
    doc_type=None,       # or "document" / "schema" to filter retrieval
)

print("\nRAG answer:")
print(response.choices[0].message.content)

print("\nRetrieved sources:")
for s in response.sources:
    print(" -", s)
