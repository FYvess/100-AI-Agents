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
You are a Product Description Agent.
 
Rules:
- Write clear, benefit-driven product descriptions
- Map features to outcomes
- Avoid hype or unsupported claims
- Adapt tone to the target audience
 
Return ONLY valid JSON with this schema:
 
{
  "product_name": "",
  "description": "",
  "key_benefits": [],
  "ideal_for": "",
  "cta": ""
}
"""
 
def read_input(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def generate_description(prompt_text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        use_rag=False,
        temperature=0.4
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("product_description.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("product_description.txt", "w", encoding="utf-8") as f:
        f.write(f"{data['product_name']}\n")
        f.write("=" * len(data["product_name"]) + "\n\n")
        f.write(data["description"] + "\n\n")
 
        f.write("Key Benefits:\n")
        for b in data["key_benefits"]:
            f.write(f"- {b}\n")
 
        f.write(f"\nIdeal For:\n{data['ideal_for']}\n")
        f.write(f"\nCall to Action:\n{data['cta']}\n")
 
def main():
    prompt_text = read_input()
    product = generate_description(prompt_text)
    save_outputs(product)
    print("Product description generated successfully.")
 
if __name__ == "__main__":
    main()