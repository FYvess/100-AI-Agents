import json
from datetime import date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from GEN_AI.RAG_AI import RAGClient

try:
    import pypdf
except ImportError:
    pypdf = None

client = RAGClient(
    pg_conn_string=config.PG_CONN_STRING,
    ollama_host=config.OLLAMA_HOST,
    embed_model=config.EMBED_MODEL,
)
 
SYSTEM_PROMPT = """
You are a Resume Optimization Agent.

Your goals:
- Rewrite resume bullets to emphasize measurable impact
- Align content with the target role
- Preserve factual accuracy (do not invent experience)
- Keep language ATS-friendly and concise
- Extract professional summary from the resume

Return ONLY valid JSON with this exact schema:

{
  "PROFESSIONAL SUMMARY": "",
  "PROJECTS": [],
  "optimized_experience": [],
  "optimized_skills": [],
  "summary_suggestion": ""
}
"""

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_pdf(pdf_path):
    """Extract text from PDF file."""
    if pypdf is None:
        raise ImportError("pypdf not installed. Install with: pip install pypdf")
    
    text = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(text)

def read_resume(resume_path=None):
    """Read resume from PDF or fallback to txt. Auto-detect PDF if not specified."""
    if resume_path is None:
        # Find first PDF in directory
        pdf_files = list(Path(".").glob("*.pdf"))
        if pdf_files:
            resume_path = str(pdf_files[0])
        else:
            resume_path = "resume.txt"
    
    path = Path(resume_path)
    if path.suffix.lower() == ".pdf":
        return extract_pdf(resume_path)
    return read_file(resume_path)
 
def optimize_resume(resume_text, job_text):
    prompt = f"""
RESUME:
{resume_text}
 
TARGET ROLE:
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
    with open("resume_optimized.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open("resume_optimized.txt", "w", encoding="utf-8") as f:
        f.write(f"Optimized Resume ({date.today()})\n")
        f.write("=" * 50 + "\n\n")

        if data.get("PROFESSIONAL SUMMARY"):
            f.write("PROFESSIONAL SUMMARY:\n")
            f.write(data["PROFESSIONAL SUMMARY"] + "\n\n")

        if data.get("PROJECTS"):
            f.write("PROJECTS:\n")
            for proj in data["PROJECTS"]:
                f.write(f"- {proj}\n")
            f.write("\n")

        if data.get("optimized_experience"):
            f.write("OPTIMIZED EXPERIENCE:\n")
            for exp in data["optimized_experience"]:
                f.write(f"- {exp}\n")
            f.write("\n")

        if data.get("optimized_skills"):
            f.write("OPTIMIZED SKILLS:\n")
            for skill in data["optimized_skills"]:
                f.write(f"- {skill}\n")
            f.write("\n")

        if data.get("summary_suggestion"):
            f.write("SUMMARY SUGGESTION:\n")
            f.write(data["summary_suggestion"] + "\n")
 
def main():
    # Auto-detect any PDF file or fallback to txt
    resume_text = read_resume()
    job_text = read_file("job.txt")
    optimized = optimize_resume(resume_text, job_text)
    save_outputs(optimized)
    
    # Find which file was used
    pdf_files = list(Path(".").glob("*.pdf"))
    source = str(pdf_files[0]) if pdf_files else "resume.txt"
    print(f"Resume optimization complete (source: {source})")
 
if __name__ == "__main__":
    main()