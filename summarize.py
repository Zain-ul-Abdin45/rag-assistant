"""
Generates a Markdown summary of an ingested PDF and saves it to summaries/.
File name pattern: summary_<stem>_YYYYMMDD.md
"""

from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import ollama

from config import CHAT_MODEL, OLLAMA_HOST
from logger import get_logger

logger = get_logger(__name__)

SUMMARY_DIR = Path(__file__).parent / "summaries"
# How much text to feed the model (chars). Keeps prompt inside context window.
_MAX_CHARS = 12_000


def _extract_text(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def _call_ollama(text: str, filename: str) -> str:
    client = ollama.Client(host=OLLAMA_HOST, timeout=180)
    context = text[:_MAX_CHARS]
    resp = client.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical document analyst. "
                    "Produce a structured Markdown summary with these sections:\n"
                    "## Overview\n## Key Topics\n## Main Findings / Contributions\n"
                    "## Methodology (if applicable)\n## Conclusions\n\n"
                    "Be concise but thorough."
                ),
            },
            {
                "role": "user",
                "content": f"Summarise this document ({filename}):\n\n{context}",
            },
        ],
        stream=False,
    )
    return resp["message"]["content"]


def generate_and_save(filepath: str, filename: str, pages: int, chunks: int) -> Path:
    """
    Extract text from *filepath*, call Ollama, write the summary .md file.
    Returns the path to the saved file.
    """
    logger.info("── summary start: %s (%d pages, %d chunks)", filename, pages, chunks)
    SUMMARY_DIR.mkdir(exist_ok=True)

    stem = Path(filename).stem
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = SUMMARY_DIR / f"summary_{stem}_{date_str}.md"

    text = _extract_text(filepath)
    summary_body = _call_ollama(text, filename)

    header = (
        f"# Summary — {filename}\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| File | `{filename}` |\n"
        f"| Date indexed | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n"
        f"| Pages | {pages} |\n"
        f"| Chunks indexed | {chunks} |\n\n"
        f"---\n\n"
    )

    out_path.write_text(header + summary_body + "\n", encoding="utf-8")
    logger.info("── summary saved: %s", out_path)
    return out_path
