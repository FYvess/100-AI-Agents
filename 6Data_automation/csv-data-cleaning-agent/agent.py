import pandas as pd
import json
from datetime import date
 
# START of rag
import sys
from pathlib import Path

# Add RAG_AI to path
rag_ai_parent = Path(__file__).parent.parent.parent / "GEN_AI"
sys.path.insert(0, str(rag_ai_parent))

from RAG_AI import RAGClient # CALL THE RAG

client = RAGClient( #CONNECT TO PGVECTOR AND POSTGRESQL
    pg_conn_string="postgresql://postgres:DB_password00@localhost:5432/pg_core",
    ollama_host="http://localhost:11434",
    embed_model="nomic-embed-text",
)
# END of rag
 
def load_csv(path="raw_data.csv"):
    return pd.read_csv(path)
 
def clean_data(df):
    issues = []
 
    # Detect missing values
    missing = df.isnull().sum()
    issues.append({"missing_values": missing.to_dict()})
 
    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    issues.append({"duplicates_removed": before - len(df)})
 
    # Normalize date format
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
 
    # Validate age range
    invalid_age = df[(df["age"] < 0) | (df["age"] > 120)]
    issues.append({"invalid_age_rows": invalid_age.index.tolist()})
    df.loc[invalid_age.index, "age"] = None
 
    return df, issues
 
def save_outputs(df, issues):
    df.to_csv("cleaned_data.csv", index=False)
 
    report = {
        "date": str(date.today()),
        "issues_detected": issues
    }
 
    with open("data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
 
def main():
    df = load_csv()
    cleaned_df, issues = clean_data(df)
    save_outputs(cleaned_df, issues)
    print("CSV data cleaning completed successfully.")
 
if __name__ == "__main__":
    main()