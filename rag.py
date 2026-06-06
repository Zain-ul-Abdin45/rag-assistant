import json
import uuid
from typing import Generator

import ollama

from config import CHAT_MODEL, EMBED_MODEL, OLLAMA_HOST, TOP_K
from db import get_db
from ingest import get_embedding
from logger import get_logger

logger = get_logger(__name__)

# Cosine distance thresholds for nomic-embed-text (768-dim).
# pgvector <=> operator returns cosine distance in [0, 2].
# 0 = identical, 1 = orthogonal, 2 = opposite.
# Chunks above _SEARCH_THRESHOLD are excluded — off-topic queries return no sources.
_SEARCH_THRESHOLD = 0.7   # cosine distance; ~0.5 similarity — tune up if queries are over-filtered
_MEMORY_THRESHOLD = 0.5   # tighter for memory recall
_MEMORY_K = 3


# ── document retrieval ────────────────────────────────────────────────────────

def search(query: str, k: int = TOP_K) -> list[dict]:
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
    if count == 0:
        logger.warning("search called but chunk table is empty")
        conn.close()
        return []

    k = min(k, count)
    query_vec = get_embedding(query)

    rows = conn.execute(
        """
        SELECT c.chunk_text, d.filename, (v.embedding <=> %s::vector) AS distance
        FROM vec_chunks v
        JOIN chunks    c ON c.id  = v.chunk_id
        JOIN documents d ON d.id  = c.doc_id
        ORDER BY distance
        LIMIT %s
        """,
        (query_vec, k),
    ).fetchall()
    conn.close()

    all_rows = [dict(r) for r in rows]
    results = [r for r in all_rows if r["distance"] < _SEARCH_THRESHOLD]
    best = all_rows[0]["distance"] if all_rows else 0
    if results:
        logger.info("search: %d relevant chunk(s) — best distance=%.4f", len(results), best)
    else:
        logger.info("search: no chunks within threshold %.1f (best distance=%.4f)",
                    _SEARCH_THRESHOLD, best)
    return results


# ── conversation memory ───────────────────────────────────────────────────────

def recall_memory(query: str, k: int = _MEMORY_K) -> list[dict]:
    """Return past Q+A pairs semantically similar to query."""
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) AS count FROM vec_conversations").fetchone()["count"]
    conn.close()
    if count == 0:
        return []

    query_vec = get_embedding(query)
    conn = get_db()
    rows = conn.execute(
        """
        SELECT
            u.content  AS question,
            a.content  AS answer,
            (v.embedding <=> %s::vector) AS distance
        FROM vec_conversations v
        JOIN conversations u ON u.id = v.conversation_id
        JOIN conversations a ON a.turn_id = u.turn_id AND a.role = 'assistant'
        WHERE u.role = 'user'
        ORDER BY distance
        LIMIT %s
        """,
        (query_vec, k),
    ).fetchall()
    conn.close()

    results = [dict(r) for r in rows if r["distance"] < _MEMORY_THRESHOLD]
    if results:
        logger.info("memory recall: %d hit(s) — best distance=%.4f",
                    len(results), results[0]["distance"])
    return results


def save_turn(session_id: str, question: str, answer: str) -> None:
    """Persist a Q+A exchange and embed the user message for future recall."""
    turn_id = str(uuid.uuid4())
    conn = get_db()
    user_row = conn.execute(
        """
        INSERT INTO conversations (session_id, turn_id, role, content)
        VALUES (%s, %s, 'user', %s) RETURNING id
        """,
        (session_id, turn_id, question),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO conversations (session_id, turn_id, role, content)
        VALUES (%s, %s, 'assistant', %s)
        """,
        (session_id, turn_id, answer),
    )
    embedding = get_embedding(question)
    conn.execute(
        "INSERT INTO vec_conversations (conversation_id, embedding) VALUES (%s, %s)",
        (user_row["id"], embedding),
    )
    conn.commit()
    conn.close()


# ── prompt building ───────────────────────────────────────────────────────────

def build_messages(
    query: str,
    history: list[dict],
    sources: list[dict],
    memory: list[dict],
) -> list[dict]:
    doc_context = "\n\n---\n\n".join(
        f"[Source: {s['filename']}]\n{s['chunk_text']}" for s in sources
    )
    system = (
        "You are a helpful assistant that answers questions based on the provided document context.\n"
        "Use the context to answer accurately. If the answer is not in the context, say so clearly.\n\n"
        f"Document context:\n{doc_context}"
    )
    if memory:
        memory_text = "\n\n".join(
            f"Q: {m['question']}\nA: {m['answer']}" for m in memory
        )
        system += f"\n\nRelevant past conversation memory:\n{memory_text}"

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


# ── streaming chat ────────────────────────────────────────────────────────────

def stream_chat(
    query: str,
    history: list[dict],
    session_id: str = "",
) -> Generator[str, None, None]:
    logger.info("stream_chat start — query: %.80s…", query)

    memory = recall_memory(query) if session_id else []
    sources = search(query)

    yield json.dumps({"type": "sources", "data": sources}) + "\n"

    if not sources:
        conn = get_db()
        has_docs = conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone()
        conn.close()
        if has_docs:
            msg = ("I couldn't find content relevant to that question in your uploaded documents. "
                   "Try asking about topics covered in those files.")
        else:
            msg = "No documents uploaded yet. Drop a PDF in the sidebar to get started."
        logger.warning("no relevant sources — %s", msg[:60])
        yield json.dumps({"type": "token", "data": msg}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return

    messages = build_messages(query, history, sources, memory)
    client = ollama.Client(host=OLLAMA_HOST)

    token_count = 0
    full_response = ""
    stream = client.chat(model=CHAT_MODEL, messages=messages, stream=True)
    for chunk in stream:
        content = chunk["message"]["content"]
        if content:
            token_count += 1
            full_response += content
            yield json.dumps({"type": "token", "data": content}) + "\n"

    yield json.dumps({"type": "done"}) + "\n"
    logger.info("stream_chat done — %d tokens emitted", token_count)

    # Persist after the response is fully streamed so the next conversation
    # can recall this exchange.  Runs during generator teardown — the client
    # has already received all tokens by this point.
    if session_id and full_response:
        try:
            save_turn(session_id, query, full_response)
        except Exception as e:
            logger.warning("save_turn failed (non-fatal): %s", e)
