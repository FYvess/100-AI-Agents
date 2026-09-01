import json
from datetime import date
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
You are a Grammar Correction Agent.
 
Rules:
- Correct grammar, spelling, and punctuation
- Improve clarity while preserving original meaning
- Do NOT rewrite content or change tone
- Do NOT add new information
- Keep edits minimal
 
Return ONLY valid JSON with this schema:
 
{
  "corrected_text": "",
  "notes": []
}
"""
 
def read_input(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def correct_text(raw_text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_text}
        ],
        response_format={"type": "json_object"},
        use_rag=False,
        temperature=config.DEFAULT_TEMPERATURE
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("corrected.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("corrected.txt", "w", encoding="utf-8") as f:
        f.write(f"Corrected Text ({date.today()})\n")
        f.write("=" * 45 + "\n\n")
        f.write(data["corrected_text"] + "\n")
 
def main():
    raw_text = read_input()
    corrected = correct_text(raw_text)
    save_outputs(corrected)
    print("Grammar correction complete.")
    print(corrected["corrected_text"])
 
if __name__ == "__main__":
    main()