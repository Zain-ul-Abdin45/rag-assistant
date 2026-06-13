import json
import uuid
from typing import Generator

import ollama
from rank_bm25 import BM25Okapi

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

def _rrf(rankings: list[list[int]], n: int, k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion over multiple ranked lists of indices."""
    scores = [0.0] * n
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            scores[idx] += 1 / (k + rank + 1)
    return scores


def _rerank_for_context(chunks: list[dict]) -> list[dict]:
    """Lost-in-the-middle mitigation: place best chunks at edges, not buried centre."""
    if len(chunks) <= 2:
        return chunks
    # LLMs attend most to position 0 and the last position.
    return [chunks[0]] + chunks[2:] + [chunks[1]]


def search(query: str, k: int = TOP_K) -> list[dict]:
    """
    Hybrid BM25 + cosine vector search with RRF fusion.

    Fetches a wider candidate pool via pgvector cosine distance, then
    re-ranks using BM25 keyword scores and RRF fusion before applying the
    relevance threshold. Improves recall for exact-term queries that pure
    cosine similarity misses.
    """
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
    if count == 0:
        logger.warning("search called but chunk table is empty")
        conn.close()
        return []

    candidate_k = min(k * 4, count)
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
        (query_vec, candidate_k),
    ).fetchall()
    conn.close()

    candidates = [dict(r) for r in rows]
    if not candidates:
        return []

    # pgvector already returns rows sorted by cosine distance ascending
    vec_ranked = list(range(len(candidates)))

    # BM25 keyword ranking over the same candidate pool
    tokenized = [c["chunk_text"].lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)

    # RRF fusion — combine both signal rankings
    rrf_scores = _rrf([vec_ranked, bm25_ranked], len(candidates))
    fused_order = sorted(range(len(candidates)), key=lambda i: rrf_scores[i], reverse=True)

    # Apply cosine threshold: exclude chunks that are clearly off-topic
    results = [
        candidates[i] for i in fused_order
        if candidates[i]["distance"] < _SEARCH_THRESHOLD
    ][:k]

    best = candidates[0]["distance"]
    if results:
        logger.info(
            "search: %d relevant chunk(s) — best cosine=%.4f (hybrid BM25+vector)",
            len(results), best,
        )
    else:
        logger.info(
            "search: no chunks within threshold %.1f (best cosine=%.4f)",
            _SEARCH_THRESHOLD, best,
        )
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
    sources = _rerank_for_context(search(query))

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
