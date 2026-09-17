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
You are an SEO Keyword Expansion Agent.
 
Rules:
- Expand the seed keyword into relevant SEO terms
- Group keywords by intent and purpose
- Avoid keyword stuffing or spam
- Focus on relevance and clarity
 
Return ONLY valid JSON with this schema:
 
{
  "primary_keywords": [],
  "supporting_keywords": [],
  "long_tail_keywords": [],
  "question_keywords": []
}
"""
 
def read_input(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def expand_keywords(prompt_text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        response_format={"type": "json_object"},
        use_rag=False,
        temperature=config.JSON_TEMPERATURE
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("keywords.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("keywords.txt", "w", encoding="utf-8") as f:
        f.write(f"SEO Keyword Expansion ({date.today()})\n")
        f.write("=" * 45 + "\n\n")
 
        for section, items in data.items():
            f.write(section.replace("_", " ").title() + ":\n")
            for k in items:
                f.write(f"- {k}\n")
            f.write("\n")
 
def main():
    prompt_text = read_input()
    keywords = expand_keywords(prompt_text)
    save_outputs(keywords)
    print("SEO keyword expansion complete.")
 
if __name__ == "__main__":
    main()