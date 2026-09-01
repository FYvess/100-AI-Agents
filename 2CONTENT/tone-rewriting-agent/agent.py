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
You are a Tone Rewriting Agent.
 
Rules:
- Rewrite the text to match the requested tone
- Preserve original meaning exactly
- Do NOT add or remove information
- Do NOT exaggerate emotion
- Keep output natural and professional
 
Return ONLY valid JSON with this schema:
 
{
  "rewritten_text": "",
  "tone_applied": ""
}
"""
 
def read_input(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def rewrite_text(text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"},
        use_rag=False,
        temperature=config.JSON_TEMPERATURE
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("rewritten.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("rewritten.txt", "w", encoding="utf-8") as f:
        f.write(f"Tone-Rewritten Text ({date.today()})\n")
        f.write("=" * 45 + "\n\n")
        f.write(data["rewritten_text"] + "\n")
 
def main():
    raw = read_input()
    rewritten = rewrite_text(raw)
    save_outputs(rewritten)
    print("Tone rewriting complete.")
    print(rewritten["rewritten_text"])
 
if __name__ == "__main__":
    main()