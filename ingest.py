from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout

import fitz  # PyMuPDF
import ollama
from pgvector import Vector

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, EMBED_WORKERS, OLLAMA_HOST
from db import get_db
from logger import get_logger

logger = get_logger(__name__)

# Per-chunk Ollama timeout (seconds). First chunk may be slow — model cold-start.
_EMBED_TIMEOUT = 180


# ── text extraction ───────────────────────────────────────────────────────────

def extract_text(filepath: str) -> tuple[str, int]:
    logger.info("extracting text from: %s", filepath)
    doc = fitz.open(filepath)
    pages = len(doc)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    char_count = len(text)
    logger.info("extracted %d pages, %d chars", pages, char_count)
    if char_count < 500:
        logger.warning(
            "very little text found (%d chars) — PDF may be image-based or scanned. "
            "OCR is not supported; RAG quality will be limited.",
            char_count,
        )
    return text, pages


# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    logger.info("chunking text (size=%d overlap=%d)…", CHUNK_SIZE, CHUNK_OVERLAP)
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        if end < text_len:
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:          # guard against non-advancing loop
            next_start = start + CHUNK_SIZE
        start = next_start
    logger.info("produced %d chunk(s)", len(chunks))
    return chunks


# ── embedding ─────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    client = ollama.Client(host=OLLAMA_HOST, timeout=_EMBED_TIMEOUT)
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def _embed_all(chunks: list[str], on_progress=None) -> list[list[float]]:
    """
    Embed chunks using a thread pool.
    EMBED_WORKERS=1  → sequential (safe for CPU-only Ollama).
    EMBED_WORKERS>1  → concurrent (useful when Ollama has GPU parallelism).
    The first chunk is always slow (model cold-start); subsequent ones are faster.
    """
    total = len(chunks)
    if total == 0:
        logger.warning("embed called with 0 chunks — nothing to embed")
        return []

    logger.info(
        "starting embed: %d chunk(s), workers=%d, model=%s "
        "(first chunk may take 30-90 s while Ollama loads the model)",
        total, EMBED_WORKERS, EMBED_MODEL,
    )

    embeddings: list[list[float] | None] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        future_to_idx = {pool.submit(get_embedding, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                embeddings[idx] = future.result(timeout=_EMBED_TIMEOUT)
            except FutureTimeout:
                raise RuntimeError(
                    f"Ollama did not respond within {_EMBED_TIMEOUT}s for chunk {idx}. "
                    "Is the embed model pulled? Run: ollama pull nomic-embed-text"
                )
            completed += 1
            logger.debug("embedded %d/%d (chunk slot %d)", completed, total, idx)
            if on_progress:
                on_progress(completed, total)

    logger.info("all %d embeddings done", total)
    return embeddings  # type: ignore[return-value]


# ── main entry point ──────────────────────────────────────────────────────────

def ingest_pdf(filepath: str, filename: str, on_progress=None) -> dict:
    logger.info("══ ingest start ══ %s", filename)

    text, pages = extract_text(filepath)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "The PDF may be image-only or password protected."
        )

    total = len(chunks)
    embeddings = _embed_all(chunks, on_progress=on_progress)

    logger.info("writing %d chunk(s) + vectors to database…", total)
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO documents (filename, filepath, pages) VALUES (%s, %s, %s) RETURNING id",
        (filename, filepath, pages),
    )
    doc_id = cur.fetchone()["id"]

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_text, chunk_index) VALUES (%s, %s, %s) RETURNING id",
            (doc_id, chunk, i),
        )
        chunk_id = cur.fetchone()["id"]
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (%s, %s)",
            (chunk_id, Vector(embedding)),
        )

    conn.commit()
    conn.close()
    logger.info("══ ingest done ══ doc_id=%d pages=%d chunks=%d", doc_id, pages, total)
    return {"doc_id": doc_id, "chunks": total, "pages": pages}
