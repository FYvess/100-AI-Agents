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
You are an FAQ Generation Agent.
 
Rules:
- Generate clear, relevant FAQs
- Focus on real user concerns
- Avoid marketing language
- Keep answers concise and honest
 
Return ONLY valid JSON with this schema:
 
{
  "faqs": [
    {
      "question": "",
      "answer": ""
    }
  ]
}
"""
 
def read_input(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def generate_faqs(prompt_text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        use_rag=False,
        temperature=0.35
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("faq.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("faq.txt", "w", encoding="utf-8") as f:
        f.write(f"Frequently Asked Questions ({date.today()})\n")
        f.write("=" * 50 + "\n\n")
        for faq in data["faqs"]:
            f.write(f"Q: {faq['question']}\n")
            f.write(f"A: {faq['answer']}\n\n")
 
def main():
    prompt_text = read_input()
    faqs = generate_faqs(prompt_text)
    save_outputs(faqs)
    print("FAQ generation complete.")
 
if __name__ == "__main__":
    main()