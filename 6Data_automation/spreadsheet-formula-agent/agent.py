import json
from datetime import date
 
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from GEN_AI.RAG_AI import RAGClient

client = RAGClient(
    pg_conn_string=config.PG_CONN_STRING,
    ollama_host=config.OLLAMA_HOST,
    embed_model=config.EMBED_MODEL,
)
 
SYSTEM_PROMPT = """
You are a Spreadsheet Formula Generator Agent.
 
Rules:
- Convert plain English to spreadsheet formulas
- Match syntax to the specified platform
- Handle conditions clearly
- Avoid unnecessary complexity
 
Return ONLY valid JSON with this schema:
 
{
  "formula": "",
  "explanation": "",
  "notes": []
}
"""
 
def read_input(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def generate_formula(prompt_text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        use_rag=False,
        temperature=0.2
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("formula_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("formula_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Spreadsheet Formula ({date.today()})\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Formula:\n{data['formula']}\n\n")
        f.write(f"Explanation:\n{data['explanation']}\n\n")
        if data["notes"]:
            f.write("Notes:\n")
            for n in data["notes"]:
                f.write(f"- {n}\n")
 
def main():
    prompt_text = read_input()
    result = generate_formula(prompt_text)
    save_outputs(result)
    print("Spreadsheet formula generated successfully.")
 
if __name__ == "__main__":
    main()