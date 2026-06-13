import re
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
    """
    Paragraph-first recursive chunking.

    Strategy:
    1. Split on blank lines (paragraphs). Accumulate paragraphs until the
       running buffer would exceed CHUNK_SIZE, then flush with CHUNK_OVERLAP
       carry-over.
    2. If a single paragraph exceeds CHUNK_SIZE, fall back to sentence
       splitting (regex on .!? boundaries).
    3. If a single sentence still exceeds CHUNK_SIZE, hard-split at character
       boundaries as a last resort.

    This keeps sentences and paragraphs intact instead of cutting mid-word,
    which improves both embedding quality and readability of retrieved excerpts.
    """
    logger.info("chunking text (size=%d overlap=%d)…", CHUNK_SIZE, CHUNK_OVERLAP)

    def _split_long(segment: str) -> list[str]:
        """Hard-split a segment that exceeds CHUNK_SIZE at sentence boundaries,
        then character boundaries if needed."""
        parts: list[str] = []
        sentences = re.split(r'(?<=[.!?])\s+', segment)
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 <= CHUNK_SIZE:
                buf = (buf + " " + sent).strip() if buf else sent
            else:
                if buf:
                    parts.append(buf)
                    buf = buf[-CHUNK_OVERLAP:] + " " + sent if len(buf) > CHUNK_OVERLAP else sent
                else:
                    # Single sentence longer than CHUNK_SIZE — hard character split
                    for i in range(0, len(sent), CHUNK_SIZE - CHUNK_OVERLAP):
                        parts.append(sent[i:i + CHUNK_SIZE])
                    buf = ""
        if buf:
            parts.append(buf)
        return parts

    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > CHUNK_SIZE:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_long(para))
            continue
        if len(buf) + len(para) + 2 <= CHUNK_SIZE:
            buf = (buf + "\n\n" + para).strip() if buf else para
        else:
            if buf:
                chunks.append(buf)
                # Carry overlap into next chunk for context continuity
                buf = buf[-CHUNK_OVERLAP:] + "\n\n" + para if len(buf) > CHUNK_OVERLAP else para
            else:
                buf = para

    if buf:
        chunks.append(buf)

    result = [c.strip() for c in chunks if c.strip()]
    logger.info("produced %d chunk(s)", len(result))
    return result


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
