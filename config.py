import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "env.local")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # nomic-embed-text default
DATABASE_URL = os.getenv(
	"DATABASE_URL",
	os.getenv("DB_PATH", "postgresql://postgres:postgres@localhost:5432/rag_assistant"),
)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "3"))  # parallel embedding threads
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()   # DEBUG | INFO | WARNING | ERROR
