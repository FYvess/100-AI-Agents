# rag_ai_api

A local-first RAG (Retrieval-Augmented Generation) client that mimics the
OpenAI Python SDK interface — but runs entirely on your own hardware using
**Ollama** for generation/embeddings and **Postgres + pgvector** for
retrieval.

If you've written code like this against OpenAI:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)
content = response.choices[0].message.content
```

...this package lets you run the same shaped code locally, with retrieval
automatically injected when you want it.

---

## Why this exists

- Run LLM inference fully offline/locally (no API costs, no data leaving
  your machine).
- Ground answers in your own documents/schemas via pgvector retrieval.
- Keep using `response.choices[0].message.content` — no rewriting existing
  prompt-based code.
- Use the **same client** for two different jobs:
  - **Extraction** (`use_rag=False`): text is already in your prompt, no
    retrieval needed — e.g. pulling structured fields out of a contract.
  - **RAG querying** (`use_rag=True`, default): the question gets embedded,
    relevant chunks are retrieved from Postgres, and injected into the
    prompt automatically.

---

## Architecture

```
                       ┌───────────────────────────────┐
                       │        Ingestion Pipeline       │
                       │   (you build this separately)   │
                       ├───────────────────────────────┤
 Documents/Schemas ───▶│ Load → Chunk → Embed → Upsert   │
                       └───────────────────────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────────┐
                       │     Postgres + pgvector          │
                       │   table: rag_chunks               │
                       │   (content, embedding, source,    │
                       │    doc_type, metadata)            │
                       └───────────────────────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────────┐
                       │           RAGClient              │
                       ├───────────────────────────────┤
 Your code ───────────▶│ .chat.completions.create(...)    │
                       │   use_rag=True  → embed query,   │
                       │     retrieve top-k, inject context│
                       │   use_rag=False → straight to LLM │
                       │                                   │
                       │   Generation via Ollama /api/chat │
                       │   Embeddings via Ollama /api/embeddings│
                       └───────────────────────────────┘
```

This package handles the **query-time** side only (`RAGClient`). The
**ingestion** side (loading documents/schemas, chunking, embedding, and
upserting into `rag_chunks`) is intentionally separate — you build/own that
pipeline (see `ingestion/` placeholder).

---

## Prerequisites

1. **Ollama** installed and running:
   ```bash
   ollama serve
   ```
2. **Pull the models** used by default:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull nomic-embed-text
   ```
3. **Postgres with the `pgvector` extension** installed and running, with a
   database/table ready to receive chunks:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;

   CREATE TABLE rag_chunks (
       id BIGSERIAL PRIMARY KEY,
       content TEXT,
       embedding VECTOR(768),     -- 768 = nomic-embed-text dimension
       source TEXT,               -- filename, table name, origin, etc.
       doc_type TEXT,              -- e.g. 'document', 'schema'
       metadata JSONB,
       created_at TIMESTAMPTZ DEFAULT now()
   );

   CREATE INDEX ON rag_chunks USING hnsw (embedding vector_cosine_ops);
   ```

   This table must be **populated** by your ingestion pipeline before
   `use_rag=True` calls will return useful results.

4. **Python dependencies**:
   ```bash
   pip install psycopg2-binary requests
   ```

---

## Installation

Copy the `rag_ai_api/` folder into your project, or treat it as a local
package:

```
your_project/
├── rag_ai_api/
│   ├── __init__.py
│   ├── client.py
│   ├── retriever.py
│   ├── embeddings.py
│   ├── response_types.py
│   └── example_usage.py
└── your_script.py
```

Then import it like any local module:

```python
from rag_ai_api import RAGClient
```

---

## Usage

### 1. Create a client

```python
from rag_ai_api import RAGClient

client = RAGClient(
    pg_conn_string="postgresql://postgres:password@localhost:5432/ragdb",
    ollama_host="http://localhost:11434",   # default, can omit
    embed_model="nomic-embed-text",          # default, can omit
)
```

### 2. Extraction — no retrieval (`use_rag=False`)

Use this when you already have the text you need in the prompt (e.g.
pulling structured fields out of a contract, invoice, email, etc.). Behaves
exactly like a plain LLM call — no database or embedding calls happen.

```python
import json

contract_text = "...full contract text..."

prompt = f"""Given this contract text, extract the following fields: 'Employee Name',
'Yearly Salary', 'Non-Compete Clause (Y/N)', 'Start Date'. Output as JSON.

Contract text:
{contract_text}
"""

response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    use_rag=False,
)

data = json.loads(response.choices[0].message.content)
print(data)
print(response.sources)  # always [] when use_rag=False
```

### 3. RAG querying — retrieval on (`use_rag=True`, default)

Use this for grounded Q&A over whatever you've ingested into `rag_chunks`.
The last user message is embedded, top-k similar chunks are retrieved, and
injected into the prompt automatically.

```python
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": "What's our refund policy for enterprise clients?"}],
    use_rag=True,     # default, can be omitted
    top_k=5,           # how many chunks to retrieve, default 5
    doc_type=None,     # optional filter, e.g. "document" or "schema"
)

print(response.choices[0].message.content)

for source in response.sources:
    print(source)   # Source(source=..., score=..., text='...')
```

### 4. JSON output + retrieval together

Both flags compose — you can ask for structured JSON _and_ have it grounded
in retrieved context:

```python
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": "List the top 3 columns relevant to customer churn analysis."}],
    response_format={"type": "json_object"},
    use_rag=True,
    doc_type="schema",   # only search schema chunks
)
```

---

## API Reference

### `RAGClient(pg_conn_string, ollama_host="http://localhost:11434", embed_model="nomic-embed-text", rag_table="rag_chunks")`

| Param            | Description                            |
| ---------------- | -------------------------------------- |
| `pg_conn_string` | Postgres connection string             |
| `ollama_host`    | Base URL for the running Ollama server |
| `embed_model`    | Ollama embedding model name            |
| `rag_table`      | Table name to query for retrieval      |

### `client.chat.completions.create(...)`

| Param             | Default | Description                                                      |
| ----------------- | ------- | ---------------------------------------------------------------- |
| `model`           | —       | Ollama model name (e.g. `qwen2.5:7b-instruct`)                   |
| `messages`        | —       | OpenAI-style message list `[{"role": "user", "content": "..."}]` |
| `response_format` | `None`  | `{"type": "json_object"}` enables Ollama's JSON mode             |
| `use_rag`         | `True`  | `False` = plain LLM passthrough, no retrieval                    |
| `top_k`           | `5`     | Number of chunks to retrieve when `use_rag=True`                 |
| `doc_type`        | `None`  | Optional filter on the `doc_type` column                         |
| `temperature`     | `0.2`   | Sampling temperature                                             |

**Returns**: `ChatCompletionResponse` with:

- `.choices[0].message.content` — the model's text/JSON output
- `.sources` — list of `Source` objects (empty if `use_rag=False`)
- `.model`, `.id`, `.usage` — present for shape parity, `usage` currently unpopulated

### `Source`

| Field      | Description                               |
| ---------- | ----------------------------------------- |
| `chunk_id` | Row id in `rag_chunks`                    |
| `source`   | Origin label (filename, table name, etc.) |
| `doc_type` | Content type tag                          |
| `score`    | Cosine similarity score                   |
| `text`     | The retrieved chunk text                  |
| `metadata` | Arbitrary JSONB metadata from ingestion   |

---

## Hardware notes

Tuned for running comfortably on:

- RTX 4060 8GB VRAM
- 16GB system RAM
- Ryzen 7 5700X3D

Stick to ~7-8B parameter models at `Q4_K_M` quantization (e.g.
`qwen2.5:7b-instruct`) for the best balance of quality and speed on this
hardware. Larger models will spill into CPU RAM and slow down noticeably.

---

## What's NOT included here

- **Ingestion pipeline** (loading, chunking, embedding, upserting documents
  into `rag_chunks`) — you build this separately; this package only
  queries the table.
- **Authentication/rate limiting** — this is a local dev tool, not a hosted
  multi-user API.
- **Streaming responses** — `stream=False` is hardcoded against Ollama for
  now.

---

## Project structure

```
project/
├── CLAUDE.md                    # Claude Code project context
├── .claude/rules/                # scoped behavior rules for Claude Code
├── .gitignore
├── README.md                     # this file
├── rag_ai_api/                    # the package
│   ├── __init__.py
│   ├── client.py
│   ├── retriever.py
│   ├── embeddings.py
│   ├── response_types.py
│   └── example_usage.py
└── ingestion/                      # your ingestion pipeline goes here
```
