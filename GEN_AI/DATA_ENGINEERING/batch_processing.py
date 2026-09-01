"""
Batch-parse all contract documents in contracts/ using the local RAG AI client.
Extracts structured fields from each employment contract via Ollama,
then writes results to PostgreSQL: parsed.contracts
"""

import json
import os
from pathlib import Path

import psycopg2
import psycopg2.extras

from rag_ai_api import RAGClient

# ── Config ───────────────────────────────────────────────────────────
PG_CONN = "postgresql://postgres:DB_password00@localhost:5432/pg_core"

# ── Client setup ─────────────────────────────────────────────────────
client = RAGClient(
    pg_conn_string=PG_CONN,
    ollama_host="http://localhost:11434",
    embed_model="nomic-embed-text",
)

# ── Ensure schema + table exist ──────────────────────────────────────
DDL = """
CREATE SCHEMA IF NOT EXISTS parsed;

CREATE TABLE IF NOT EXISTS parsed.contracts (
    id SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    employee_name TEXT,
    position TEXT,
    company TEXT,
    yearly_salary TEXT,
    start_date TEXT,
    vacation_days TEXT,
    non_compete TEXT,
    non_compete_duration TEXT,
    governing_law_state TEXT,
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

conn = psycopg2.connect(PG_CONN)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(DDL)
print("[OK] Schema 'parsed' and table 'parsed.contracts' ready.\n")

# ── Load contracts from contracts/ folder ────────────────────────────
contracts_dir = Path(__file__).parent / "contracts"
documents = sorted(contracts_dir.glob("*.txt"))

if not documents:
    print("No .txt files found in contracts/")
    raise SystemExit(1)

print(f"Found {len(documents)} contract(s) to parse.\n")

# ── Parse each contract and insert into Postgres ─────────────────────
INSERT_SQL = """
    INSERT INTO parsed.contracts
        (source_file, employee_name, position, company, yearly_salary,
         start_date, vacation_days, non_compete, non_compete_duration,
         governing_law_state, raw_json)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

inserted = 0

for doc_path in documents:
    contract_text = doc_path.read_text(encoding="utf-8")
    filename = doc_path.name

    print(f"Parsing {filename} ...")

    prompt = f"""Given this employment contract, extract the following fields and output as JSON:
- "Employee Name"
- "Position"
- "Company"
- "Yearly Salary"
- "Start Date"
- "Vacation Days"
- "Non-Compete Clause (Y/N)"
- "Non-Compete Duration"
- "Governing Law (State)"

Output format:
{{"Contract": {{"Employee Name": "...", "Position": "...", "Company": "...", "Yearly Salary": "...", "Start Date": "...", "Vacation Days": "...", "Non-Compete Clause (Y/N)": "...", "Non-Compete Duration": "...", "Governing Law (State)": "..."}}}}

Contract text:
{contract_text}
"""

    response = client.chat.completions.create(
        model="qwen2.5:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        use_rag=False,  # full text is in prompt, no retrieval needed
    )

    result = json.loads(response.choices[0].message.content)

    # Handle nested or flat JSON from the LLM
    data = result.get("Contract", result)

    with conn.cursor() as cur:
        cur.execute(INSERT_SQL, (
            filename,
            data.get("Employee Name"),
            data.get("Position"),
            data.get("Company"),
            data.get("Yearly Salary"),
            data.get("Start Date"),
            data.get("Vacation Days"),
            data.get("Non-Compete Clause (Y/N)"),
            data.get("Non-Compete Duration"),
            data.get("Governing Law (State)"),
            json.dumps(result),
        ))

    inserted += 1
    print(f"  [OK] Inserted into parsed.contracts\n")

conn.close()

print("=" * 60)
print(f"Done! {inserted}/{len(documents)} contracts parsed and saved to parsed.contracts")
print("=" * 60)