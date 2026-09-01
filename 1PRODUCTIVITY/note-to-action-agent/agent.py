import json
from datetime import date
# START of rag
import sys
from pathlib import Path

# Add RAG_AI to path
rag_ai_parent = Path(__file__).parent.parent.parent / "GEN_AI"
sys.path.insert(0, str(rag_ai_parent))

from RAG_AI import RAGClient # CALL THE RAG

client = RAGClient( #CONNECT TO PGVECTOR AND POSTGRESQL
    pg_conn_string="postgresql://postgres:DB_password00@localhost:5432/pg_core",
    ollama_host="http://localhost:11434",
    embed_model="nomic-embed-text",
)
# END of rag
 
SYSTEM_PROMPT = """
You are a Note-to-Action Item Agent.
 
Your job:
- Extract ONLY actionable tasks from the notes
- Ignore ideas, opinions, or decisions without actions
- Identify owner if mentioned; otherwise use "Unassigned"
- Suggest a deadline if implied; otherwise "Not specified"
- Assign priority: Low, Medium, or High
 
Return ONLY valid JSON with this schema:
 
{
  "actions": [
    {
      "action": "",
      "owner": "",
      "deadline": "",
      "priority": "",
      "source_context": ""
    }
  ]
}
"""
 
def read_notes(path="notes.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def extract_actions(notes_text):
    response = client.chat.completions.create(
        model="qwen2.5:7b-instruct", # replace using actual llm
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": notes_text}
        ],
        response_format={"type": "json_object"}, # add using response format
        use_rag=False, # True if neeeded to get the data from database
        temperature=0.2
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("actions.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("actions.txt", "w", encoding="utf-8") as f:
        f.write(f"Extracted Action Items ({date.today()})\n")
        f.write("=" * 45 + "\n\n")
 
        for i, a in enumerate(data["actions"], 1):
            f.write(f"{i}. {a['action']}\n")
            f.write(f"   Owner: {a['owner']}\n")
            f.write(f"   Deadline: {a['deadline']}\n")
            f.write(f"   Priority: {a['priority']}\n")
            f.write(f"   Source: {a['source_context']}\n\n")
 
def main():
    notes_text = read_notes()
    actions = extract_actions(notes_text)
    save_outputs(actions)
    print("Action items extracted successfully.")
    print(actions)
 
if __name__ == "__main__":
    main()