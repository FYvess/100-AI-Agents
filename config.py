"""
Shared configuration for all agents.
Loads settings from .env file with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# RAG AI Configuration
PG_CONN_STRING = os.getenv("PG_CONN_STRING", "postgresql://postgres:DB_password00@localhost:5432/pg_core")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b-instruct")

# Agent Parameters
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
JSON_TEMPERATURE = float(os.getenv("JSON_TEMPERATURE", "0.3"))
