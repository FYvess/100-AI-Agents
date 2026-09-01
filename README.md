# 100 AI Agents

A collection of specialized AI agents built on a local-first architecture using **Ollama** for LLM inference and **PostgreSQL + pgvector** for retrieval-augmented generation (RAG).

Each agent is designed to solve a specific productivity or content creation task, from email summarization to resume optimization to LinkedIn ideation.

---

## 🚀 Quick Start

### Prerequisites

1. **Ollama** (LLM inference engine)
   - Download from [ollama.ai](https://ollama.ai)
   - Ensure it's running: `ollama serve`

2. **PostgreSQL** with pgvector extension
   - Required for RAG-enabled agents
   - Setup: See [pgvector installation](https://github.com/pgvector/pgvector#installation)

3. **Python 3.8+**

4. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. **Copy the environment template**
   ```bash
   cp .env.example .env
   ```

2. **Update `.env` with your settings**
   ```
   PG_CONN_STRING=postgresql://user:password@localhost:5432/your_db
   OLLAMA_HOST=http://localhost:11434
   EMBED_MODEL=nomic-embed-text
   CHAT_MODEL=qwen2.5:7b-instruct
   ```

3. **Pull Ollama models** (if not already installed)
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull nomic-embed-text
   ```

4. **Start Ollama**
   ```bash
   ollama serve
   ```

### Run an Agent

```bash
cd 1PRODUCTIVITY/email-summarization-agent
python agent.py
```

---

## 📁 Project Structure

```
100-ai-agents/
├── 1PRODUCTIVITY/              # Productivity & workflow agents
│   ├── DAILY_TASK_PRIO/
│   ├── email-summarization-agent/
│   ├── meeting-agenda-agent/
│   ├── daily-goal-reflection-agent/
│   ├── personal-knowledge-agent/
│   ├── smart-reminder-agent/
│   ├── habit-tracking-agent/
│   └── ...
│
├── 2CONTENT/                   # Content creation & optimization agents
│   ├── resume-optimization-agent/
│   ├── cover-letter-agent/
│   ├── blog-post-generator-agent/
│   ├── linkedin-ideation-agent/
│   ├── grammar-correction-agent/
│   ├── tone-rewriting-agent/
│   └── ...
│
├── GEN_AI/
│   ├── DATA_ENGINEERING/       # Data pipeline & RAG setup
│   ├── RAG_AI/                 # Local RAG client (Ollama + pgvector)
│   │   ├── client.py
│   │   ├── retriever.py
│   │   ├── embeddings.py
│   │   └── response_types.py
│   └── RAG_AI/
│
├── config.py                   # Centralized config loader
├── .env                        # Environment variables (secrets)
├── .env.example                # Template for .env
└── requirements.txt            # Python dependencies
```

---

## 🎯 Agent Categories

### 1PRODUCTIVITY (Workflow & Productivity)

| Agent | Purpose | Status |
|-------|---------|--------|
| **email-summarization-agent** | Extract summary, action items, deadlines from emails | ✅ |
| **meeting-agenda-agent** | Generate time-boxed meeting agendas | ✅ |
| **daily-goal-reflection-agent** | Reflect on daily goals vs. outcomes | ✅ |
| **personal-knowledge-agent** | Query personal notes via semantic search | ✅ |
| **smart-reminder-agent** | Generate priority-based reminders | ✅ |
| **DAILY_TASK_PRIO** | Daily task prioritization | ✅ |

### 2CONTENT (Content Creation & Optimization)

| Agent | Purpose | Status |
|-------|---------|--------|
| **resume-optimization-agent** | Optimize resume for job descriptions (PDF support) | ✅ |
| **cover-letter-agent** | Generate role-specific cover letters | ✅ |
| **blog-post-generator-agent** | Generate structured blog posts | ✅ |
| **linkedin-ideation-agent** | Generate LinkedIn post ideas | ✅ |
| **grammar-correction-agent** | Correct grammar while preserving tone | ✅ |
| **tone-rewriting-agent** | Rewrite text in different tones | ✅ |

---

## 🔧 Configuration

All agents share a centralized configuration file. Edit `.env` to customize:

```bash
# RAG AI Configuration
PG_CONN_STRING=postgresql://postgres:DB_password00@localhost:5432/pg_core
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=nomic-embed-text
CHAT_MODEL=qwen2.5:7b-instruct

# Agent Parameters
DEFAULT_TEMPERATURE=0.2
JSON_TEMPERATURE=0.3
```

Each agent imports from `config.py`:
```python
import config

client = RAGClient(
    pg_conn_string=config.PG_CONN_STRING,
    ollama_host=config.OLLAMA_HOST,
    embed_model=config.EMBED_MODEL,
)
```

---

## 🏗️ Architecture

### Local-First Design

- **No cloud dependencies**: All inference runs locally on your machine
- **No API costs**: Use open-source models (Ollama)
- **No data leaving your system**: Keep proprietary data private
- **Fast iteration**: No network latency

### RAG Client

The `GEN_AI/RAG_AI/client.py` provides an OpenAI-compatible interface:

```python
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": "..."}],
    response_format={"type": "json_object"},
    use_rag=False,  # Set to True for retrieval-augmented generation
    temperature=0.2
)
```

**With RAG enabled** (`use_rag=True`):
1. Query is embedded using `nomic-embed-text`
2. Top-k similar chunks retrieved from `rag_chunks` table
3. Context injected into prompt automatically
4. LLM responds grounded in your documents

---

## 📦 Dependencies

### Core
- `psycopg2-binary` - PostgreSQL driver
- `requests` - HTTP client for Ollama API
- `python-dotenv` - Environment variable management

### Optional
- `pypdf` - PDF extraction for resume-optimization-agent
- `numpy` - Vector math for semantic search

See `requirements.txt` for full list.

---

## 🚦 Running Agents

### From Command Line

```bash
cd 1PRODUCTIVITY/email-summarization-agent
python agent.py
```

### Batch Processing

Create a script to run multiple agents:

```python
import subprocess
import os

agents = [
    "1PRODUCTIVITY/email-summarization-agent",
    "2CONTENT/grammar-correction-agent",
]

for agent_dir in agents:
    print(f"\n▶ Running {agent_dir}...")
    result = subprocess.run(
        ["python", "agent.py"],
        cwd=agent_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
```

---

## 🧠 Building New Agents

### Basic Agent Template

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from GEN_AI.RAG_AI import RAGClient

client = RAGClient(
    pg_conn_string=config.PG_CONN_STRING,
    ollama_host=config.OLLAMA_HOST,
    embed_model=config.EMBED_MODEL,
)

SYSTEM_PROMPT = """
You are a [Agent Name] Agent.

Your job:
- [Task 1]
- [Task 2]

Return ONLY valid JSON with this schema:
{
  "field1": "",
  "field2": []
}
"""

def process(input_text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ],
        response_format={"type": "json_object"},
        use_rag=False,
        temperature=config.JSON_TEMPERATURE
    )
    return json.loads(response.choices[0].message.content)

def main():
    data = process("...")
    # Save outputs
    print("Done!")

if __name__ == "__main__":
    main()
```

---

## 🐛 Troubleshooting

### Ollama Connection Error
```
ConnectionRefusedError: [WinError 10061] No connection
```
**Fix:** Start Ollama: `ollama serve`

### Model Not Found
```
Error: model not found
```
**Fix:** Pull the model:
```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

### PostgreSQL Connection Error
```
psycopg2.OperationalError: could not connect to server
```
**Fix:** Ensure PostgreSQL is running and connection string in `.env` is correct.

### pypdf ImportError
```
ModuleNotFoundError: No module named 'pypdf'
```
**Fix:** Install optional dependency:
```bash
pip install pypdf
```

---

## 📊 Performance Notes

Optimized for **RTX 4060 8GB** hardware. Adjust based on your system:

- **RTX 3060 6GB or less**: Use smaller models (3-4B parameter)
- **RTX 4070+ 12GB**: Can run larger models (13B+)
- **CPU-only**: Very slow; recommend cloud alternative

**Recommended models:**
- `qwen2.5:7b-instruct` - Balanced quality/speed
- `nomic-embed-text` - Fast, accurate embeddings (768-dim)

---

## 🤝 Contributing

Contributions welcome! To add a new agent:

1. Create a new folder: `1PRODUCTIVITY/my-new-agent/` or `2CONTENT/my-new-agent/`
2. Add `agent.py` using the template above
3. Add input files (e.g., `input.txt`)
4. Test with Ollama running
5. Document in this README

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🔒 Privacy & Security

- **Local-first**: All data stays on your machine
- **No telemetry**: No tracking or analytics
- **No API keys needed**: Use Ollama locally
- **Secrets in .env**: Never commit `.env` to git (included in `.gitignore`)

---

## 🗺️ Roadmap to 100 Agents

Current: **10 agents** (across 1PRODUCTIVITY and 2CONTENT)

**Planned additions:**
- 2RESEARCH - Research & analysis agents
- 3AUTOMATION - Automation & workflow agents
- 4ANALYTICS - Data analysis & reporting agents
- 5IDEATION - Brainstorming & creative agents
- And more...

---

## 💡 Use Cases

### Productivity
- Summarize daily emails → extract action items
- Plan meetings with auto-generated agendas
- Track habits and reflect on progress
- Organize personal notes with semantic search

### Content Creation
- Optimize resume for job descriptions
- Generate tailored cover letters
- Create blog posts from outlines
- Brainstorm LinkedIn content
- Fix grammar without changing tone

### Business
- Automate document processing
- Extract structured data from unstructured text
- Generate reports and summaries
- Ideate products, features, campaigns

---

## 🔗 Resources

- **Ollama**: https://ollama.ai
- **pgvector**: https://github.com/pgvector/pgvector
- **Qwen Models**: https://huggingface.co/Qwen
- **PostgreSQL**: https://www.postgresql.org

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review agent-specific `README.md` files
3. Check `.env` configuration
4. Ensure Ollama and PostgreSQL are running

---

**Happy agent building! 🤖**
