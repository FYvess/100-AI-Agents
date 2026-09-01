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
You are a Cover Letter Writing Agent.
 
Your goals:
- Write a concise, role-specific cover letter
- Align experience with the job requirements
- Preserve factual accuracy (do not invent experience)
- Maintain a professional, human tone
- Keep length to 3–4 short paragraphs
 
Return ONLY valid JSON with this schema:
 
{
  "company": "",
  "role": "",
  "cover_letter": ""
}
"""
 
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def generate_cover_letter(resume_text, job_text):
    prompt = f"""
RESUME:
{resume_text}
 
JOB DESCRIPTION:
{job_text}
"""
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        use_rag=False,
        temperature=config.JSON_TEMPERATURE
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("cover_letter.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("cover_letter.txt", "w", encoding="utf-8") as f:
        f.write(f"Cover Letter — {data['role']} at {data['company']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(data["cover_letter"] + "\n")
 
def main():
    resume_text = read_file("resume.txt")
    job_text = read_file("job.txt")
    letter = generate_cover_letter(resume_text, job_text)
    save_outputs(letter)
    print("Cover letter generated successfully.")
 
if __name__ == "__main__":
    main()