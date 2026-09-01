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
You are a Daily Goal Reflection Agent.
 
Your task:
- Compare planned goals with actual outcomes
- Identify what was completed and what was missed
- Analyze reasons for misses
- Extract insights and lessons
- Provide 2–3 actionable suggestions for tomorrow
 
Return ONLY valid JSON with this schema:
 
{
  "summary": "",
  "completed_goals": [],
  "missed_goals": [],
  "insights": [],
  "lessons_learned": [],
  "tomorrow_suggestions": []
}
"""
 
def read_day(path="day.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
def reflect(day_text):
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": day_text}
        ],
        response_format={"type": "json_object"},
        use_rag=False,
        temperature=config.JSON_TEMPERATURE
    )
    return json.loads(response.choices[0].message.content)
 
def save_outputs(data):
    with open("reflection.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
    with open("reflection.txt", "w", encoding="utf-8") as f:
        f.write(f"Daily Reflection ({date.today()})\n")
        f.write("=" * 45 + "\n\n")
 
        f.write("SUMMARY:\n")
        f.write(data["summary"] + "\n\n")
 
        f.write("COMPLETED GOALS:\n")
        for g in data["completed_goals"]:
            f.write(f"- {g}\n")
 
        f.write("\nMISSED GOALS:\n")
        for g in data["missed_goals"]:
            f.write(f"- {g}\n")
 
        f.write("\nINSIGHTS:\n")
        for i in data["insights"]:
            f.write(f"- {i}\n")
 
        f.write("\nLESSONS LEARNED:\n")
        for l in data["lessons_learned"]:
            f.write(f"- {l}\n")
 
        f.write("\nSUGGESTIONS FOR TOMORROW:\n")
        for s in data["tomorrow_suggestions"]:
            f.write(f"- {s}\n")
 
def main():
    day_text = read_day()
    reflection = reflect(day_text)
    save_outputs(reflection)
    print("Daily reflection generated successfully.")
    print(reflection)
 
if __name__ == "__main__":
    main()