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
You are a LinkedIn Post Ideation Agent.
 
Your job:
- Generate 5 distinct LinkedIn post ideas
- Each idea must be skimmable and hook-driven
- Optimize for professional audiences
- Avoid generic motivational content
- Suggest a discussion-oriented CTA
 
Return ONLY valid JSON with this schema:
 
{
  "ideas": [
    {
      "title": "",
      "hook": "",
      "core_message": "",
      "cta": "",
      "hashtags": []
    }
  ]
}
"""
 
def read_input(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def generate_ideas(prompt_text):
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
    with open("ideas.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("ideas.txt", "w", encoding="utf-8") as f:
        f.write(f"LinkedIn Post Ideas ({date.today()})\n")
        f.write("=" * 45 + "\n\n")
        for i, idea in enumerate(data["ideas"], 1):
            f.write(f"{i}. {idea['title']}\n")
            f.write(f"   Hook: {idea['hook']}\n")
            f.write(f"   Core Message: {idea['core_message']}\n")
            f.write(f"   CTA: {idea['cta']}\n")
            if idea["hashtags"]:
                f.write(f"   Hashtags: {' '.join(idea['hashtags'])}\n")
            f.write("\n")
 
def main():
    prompt_text = read_input()
    ideas = generate_ideas(prompt_text)
    save_outputs(ideas)
    print("LinkedIn post ideas generated successfully.")
    print(f"Ideas generated: {len(ideas['ideas'])}")
 
if __name__ == "__main__":
    main()