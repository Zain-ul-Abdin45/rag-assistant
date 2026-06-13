# RAG Assistant

A self-hosted, NotebookLM-style document chat. Upload PDFs, ask questions, get streamed answers grounded in your documents. No external API keys. Everything runs locally.

**Stack:** FastAPI · PostgreSQL + pgvector · Ollama · PyMuPDF · Vanilla JS

---

## How It Works — End to End

### 1. Upload a PDF

```
Browser → POST /documents (multipart PDF)
```

- File is saved to `uploads/`
- A background job starts immediately; the browser polls `GET /jobs/{id}` for progress
- **Phase 1 — Ingest:**
  1. `PyMuPDF` extracts raw text page by page
  2. `chunk_text()` splits into chunks using paragraph-first recursive chunking:
     - Split on blank lines (paragraphs) and accumulate until `CHUNK_SIZE` chars
     - If a paragraph is too long, fall back to sentence boundaries (`re` split on `.!?`)
     - If a sentence is still too long, hard-split at character boundaries
  3. Each chunk is embedded with `nomic-embed-text` via Ollama (parallel workers, `EMBED_WORKERS` threads)
  4. Chunks + vectors stored in PostgreSQL: `chunks` table (text) and `vec_chunks` table (768-dim vector)
- **Phase 2 — Summarize:**
  - `summarize.py` sends all chunk text to `llama3.2` via Ollama
  - Output is a structured Markdown file saved to `summaries/summary_{name}_{date}.md`
  - Available via the Summary button in the sidebar

### 2. Ask a Question

```
Browser → POST /chat { message, history, session_id }
```

**Step 1 — Memory recall**
- The user's question is embedded and compared against `vec_conversations` (past Q+A pairs)
- Any past exchange with cosine distance `< 0.5` is injected into the prompt as prior context
- This lets the assistant remember what was discussed earlier, even after page refresh

**Step 2 — Document retrieval (hybrid search)**
- Question is embedded with `nomic-embed-text`
- pgvector fetches `TOP_K × 4` candidate chunks using cosine distance (`<=>`)
- BM25 keyword scoring (`rank-bm25`) is run over the same candidate pool
- Both rankings are fused using **Reciprocal Rank Fusion (RRF)** — combines semantic and keyword signals
- Chunks with cosine distance `≥ 0.7` are excluded (off-topic filter)
- Top `TOP_K` chunks survive

**Step 3 — Reranking**
- **Lost-in-the-middle mitigation:** best chunk goes to position 0, second-best to the last position
- LLMs attend most to the beginning and end of their context — this ensures the most relevant evidence is at those positions

**Step 4 — Prompt assembly**
```
System:
  [doc context — chunks formatted as Source: filename\n chunk_text]
  [memory context — past Q+A pairs if recalled]
User history: [prior messages in this browser tab]
User: [current question]
```

**Step 5 — Stream response**
- Ollama streams `llama3.2` output token by token
- FastAPI returns NDJSON; each line is one of:
  - `{"type": "sources", "data": [...]}` — sent first, renders source cards in UI
  - `{"type": "token", "data": "..."}` — each word/piece as it arrives
  - `{"type": "done"}` — signals end of stream

**Step 6 — Save to memory**
- After streaming completes, the Q+A pair is saved to `conversations`
- The question is embedded and stored in `vec_conversations` for future recall

### 3. No Relevant Documents

If no chunk passes the cosine threshold:
- Sources panel is empty
- Chat shows: *"I couldn't find content relevant to that question in your uploaded documents."*
- If no documents exist at all: *"No documents uploaded yet. Drop a PDF in the sidebar to get started."*
- Chat input is disabled when the document list is empty

### 4. View Document Summary

```
Browser → GET /documents/{id}/summary
```

- Returns the Markdown summary generated during ingestion
- Rendered in a modal with a Download `.md` button

### 5. Delete a Document

```
Browser → DELETE /documents/{id}
```

- Removes the document row, all its chunks, and all vectors (CASCADE)
- Deletes the uploaded PDF file from disk

---

## Project Layout

```
config.py            all runtime settings (loaded from env.local)
db.py                schema creation, connection factory
ingest.py            PDF extraction → paragraph chunking → parallel embedding
rag.py               hybrid search (BM25+cosine+RRF), reranking, memory, streaming
summarize.py         per-document Markdown summary via Ollama
logger.py            file + console handler, respects LOG_LEVEL
main.py              FastAPI routes, lifespan startup checks, job tracking

Dockerfile           python:3.12-slim image
docker-compose.yml   local stack: app + postgres (pgvector) + ollama
.dockerignore

static/index.html    single-page UI (no framework)
uploads/             uploaded PDFs (bind-mounted in Docker)
summaries/           generated Markdown summaries (bind-mounted)
logs/                application log files (bind-mounted)

env.local            local secrets — never committed
env.local.example    template
requirements.txt     Python dependencies
system_architecture.mmd  full Mermaid architecture diagram
deployment.md        Docker + Railway deployment guide
```

---

## Database Schema

```
documents         id, filename, filepath, pages, created_at
chunks            id, doc_id→documents, chunk_text, chunk_index
vec_chunks        chunk_id→chunks, embedding vector(768)

conversations     id, session_id, turn_id, role, content, created_at
vec_conversations conversation_id→conversations (user rows), embedding vector(768)
```

- `turn_id` groups each user message with its assistant reply — used in the memory recall JOIN
- `session_id` is `crypto.randomUUID()` from the browser, persisted in `localStorage`
- Memory accumulates across page refreshes; different browsers or incognito start fresh

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the UI |
| `GET` | `/health` | Ollama connectivity + active model |
| `GET` | `/documents` | List all uploaded documents |
| `POST` | `/documents` | Upload a PDF, start background ingest job |
| `GET` | `/documents/{id}/summary` | Fetch generated Markdown summary |
| `DELETE` | `/documents/{id}` | Delete document, chunks, vectors, file |
| `GET` | `/jobs/{id}` | Poll ingest job status and phase |
| `POST` | `/chat` | Streaming NDJSON chat response |
| `GET` | `/docs` | FastAPI auto-generated OpenAPI docs |

---

## Retrieval Thresholds

pgvector `<=>` returns **cosine distance** in range `[0, 2]`.
- `0` = identical vectors
- `1` = orthogonal (unrelated)
- `2` = opposite

| Threshold | Variable | Used for |
|---|---|---|
| `< 0.7` | `_SEARCH_THRESHOLD` | Document chunk relevance — chunks above this are excluded |
| `< 0.5` | `_MEMORY_THRESHOLD` | Past conversation recall — tighter to avoid hallucinating old context |

Tune `_SEARCH_THRESHOLD` up (e.g. `0.8`) if valid queries are being blocked. Tune it down (e.g. `0.6`) if irrelevant chunks are leaking into answers. The INFO log shows `best cosine=X.XXXX` every query to help calibrate.

---

## Quick Start (no Docker)

### Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension
- Ollama running locally

### 1. Create `env.local`

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/rag_assistant
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=nomic-embed-text
CHAT_MODEL=llama3.2
EMBED_DIM=768
UPLOAD_DIR=uploads
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=5
EMBED_WORKERS=3
LOG_LEVEL=INFO
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 4. Start

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

---

## Quick Start (Docker)

```bash
docker compose up --build
```

Pull models on first run:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
```

Open `http://localhost:8000`.

| Service | Image | Port |
|---|---|---|
| `app` | Built from `Dockerfile` | 8000 |
| `postgres` | `pgvector/pgvector:pg16` | 5432 |
| `ollama` | `ollama/ollama:latest` | 11434 |

`pg_data` and `ollama_models` volumes persist across restarts. `uploads/`, `summaries/`, `logs/` are bind-mounted from the project directory.

See [deployment.md](deployment.md) for the full Railway cloud deployment guide.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | Full PostgreSQL connection string |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model — must match `EMBED_DIM` |
| `CHAT_MODEL` | `llama3.2` | Chat and summarization model |
| `EMBED_DIM` | `768` | Vector dimension — changing this requires rebuilding `vec_chunks` |
| `UPLOAD_DIR` | `uploads` | Directory for uploaded PDFs |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters carried over between chunks for context continuity |
| `TOP_K` | `5` | Max chunks retrieved per query |
| `EMBED_WORKERS` | `3` | Parallel embed threads — use `1` on CPU-only Ollama |
| `LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` |

---

## Startup Checks

On every startup, `main.py` runs two safety checks before accepting requests:

1. **Embedding dimension guard** — reads the stored vector size from `vec_chunks`. If it doesn't match `EMBED_DIM`, the server refuses to start with a clear error. Prevents silent wrong results when the embedding model is swapped.

2. **Orphaned document cleanup** — deletes any `documents` row with no corresponding `chunks` (server crashed mid-ingest). These would show in the sidebar but return no results.

---

## Logging

Controlled by `LOG_LEVEL` in `env.local`.

| Level | What you see |
|---|---|
| `DEBUG` | Per-chunk embed progress, every DB call |
| `INFO` | Startup, uploads, ingest complete, chat queries, cosine distances |
| `WARNING` | Off-topic queries, save_turn failures, degraded Ollama |
| `ERROR` | Hard failures only |

Every chat query logs:
```
search: 3 relevant chunk(s) — best cosine=0.3812 (hybrid BM25+vector)
search: no chunks within threshold 0.7 (best cosine=0.8941)
memory recall: 2 hit(s) — best distance=0.3105
```

---

## Security Notes

- `env.local` is in `.gitignore` — never commit it
- No passwords or keys are hardcoded in any Python module
- All secrets are read through `config.py` from `env.local` or environment variables

---

## Current Feature Status

| Feature | Status |
|---|---|
| PDF upload and text extraction | Working |
| Paragraph-first recursive chunking | Working |
| Parallel embedding (configurable workers) | Working |
| pgvector cosine similarity search | Working |
| Hybrid BM25 + cosine search with RRF fusion | Working |
| Lost-in-the-middle reranking | Working |
| Off-topic query filtering (cosine threshold) | Working |
| Streamed chat responses (NDJSON) | Working |
| Conversation memory across page refreshes | Working |
| Per-document Markdown summary (auto-generated) | Working |
| Summary modal with Download button | Working |
| Sources panel grouped by file with excerpt count | Working |
| Chat input locked when no documents uploaded | Working |
| Background ingest job with per-phase progress | Working |
| Document delete (cascade chunks + vectors) | Working |
| Startup dimension mismatch guard | Working |
| Orphaned document cleanup on startup | Working |
| Docker + docker-compose (app + postgres + ollama) | Working |
| LOG_LEVEL-controlled logging | Working |

---

## What's Next

| Priority | Feature | Notes |
|---|---|---|
| 6 | Table extraction from PDFs | `fitz.find_tables()` → pandas → Markdown tables in chunks |
| 7 | JWT authentication | `python-jose` + `passlib`; protect all routes |
| 8 | Per-user document isolation | `user_id` on documents + conversations; row-level security |
| 9 | Background job queue | Replace `BackgroundTasks` with RQ + Redis for reliability |

### Longer-term roadmap

| Phase | What changes | Files |
|---|---|---|
| **Phase 1** | LlamaIndex for ingest + retrieval | `ingest.py`, `rag.py` |
| **Phase 2** | LangGraph state graph (parallel ingest, conditional retrieval) | new `graph.py`, `main.py` |
| **Phase 3** | Auth + per-user isolation | new `auth.py`, `db.py`, `main.py` |
| **Phase 4** | Redis cache + RQ job queue | new `cache.py`, `queue.py` |
| **Phase 5** | Multi-format loaders (DOCX, HTML, images+OCR) | `ingest.py` |
| **Phase 6** | Structured logging, Prometheus metrics, OpenTelemetry | `logger.py`, new `metrics.py` |

Architecture diagram (including planned phases): [system_architecture.mmd](system_architecture.mmd)
