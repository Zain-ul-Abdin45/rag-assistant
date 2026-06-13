from contextlib import asynccontextmanager
import shutil
import uuid
from pathlib import Path

import ollama
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# logging must be set up before any other local import
from logger import get_logger, setup_logging
setup_logging()

import config
from db import get_db, init_db
from ingest import ingest_pdf
from rag import stream_chat
from summarize import generate_and_save

logger = get_logger(__name__)

UPLOAD_DIR = Path(config.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

# in-memory job tracking: job_id -> {"status": "processing|done|error", ...}
_jobs: dict[str, dict] = {}


# ── startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("starting RAG Assistant — initialising database…")
    init_db()
    logger.info("database ready")

    # Guard: changing EMBED_MODEL after documents are ingested produces silent
    # wrong results. Fail hard on startup so the mismatch is immediately visible.
    conn = get_db()
    stored = conn.execute(
        "SELECT vector_dims(embedding) AS dim FROM vec_chunks LIMIT 1"
    ).fetchone()
    if stored and stored["dim"] and stored["dim"] != config.EMBED_DIM:
        conn.close()
        raise RuntimeError(
            f"Embedding dimension mismatch: config EMBED_DIM={config.EMBED_DIM} "
            f"but stored vectors are {stored['dim']}d. "
            "Changing EMBED_MODEL requires dropping and rebuilding vec_chunks."
        )

    # Clean up documents whose ingestion was interrupted (server restarted mid-job).
    orphaned = conn.execute(
        """
        DELETE FROM documents
        WHERE id NOT IN (SELECT DISTINCT doc_id FROM chunks)
        RETURNING filename
        """
    ).fetchall()
    conn.commit()
    conn.close()
    for row in orphaned:
        logger.warning("removed orphaned document (no chunks ingested): %s", row["filename"])
    if orphaned:
        logger.info("orphan cleanup done — %d document(s) removed", len(orphaned))

    yield
    logger.info("RAG Assistant shutting down")


app = FastAPI(title="RAG Assistant", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        client.list()
        return {"status": "ok", "ollama": True, "chat_model": config.CHAT_MODEL}
    except Exception as e:
        logger.warning("health degraded — ollama unreachable: %s", e)
        return {"status": "degraded", "ollama": False, "error": str(e)}


# ── documents ─────────────────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, pages, created_at FROM documents ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _run_ingest(job_id: str, filepath: str, filename: str) -> None:
    short = job_id[:8]
    logger.info("[%s] ══ job start: %s", short, filename)

    def on_progress(done: int, total: int) -> None:
        _jobs[job_id].update({
            "chunks_done": done,
            "chunks_total": total,
            "phase": "embedding",
        })

    # Phase 1 – text extraction + embedding
    _jobs[job_id]["phase"] = "extracting"
    try:
        result = ingest_pdf(filepath, filename, on_progress=on_progress)
    except Exception as e:
        logger.error("[%s] ingestion failed: %s", short, e, exc_info=True)
        Path(filepath).unlink(missing_ok=True)
        _jobs[job_id] = {"status": "error", "error": str(e)}
        return

    _jobs[job_id] = {"status": "done", "phase": "done", **result}
    logger.info("[%s] ══ embedding done: doc_id=%s chunks=%d", short, result["doc_id"], result["chunks"])

    # Phase 2 – summary (non-blocking: failure does NOT change job status)
    logger.info("[%s] generating summary for %s…", short, filename)
    _jobs[job_id]["phase"] = "summarising"
    try:
        summary_path = generate_and_save(
            filepath, filename,
            pages=result["pages"],
            chunks=result["chunks"],
        )
        _jobs[job_id]["summary_file"] = str(summary_path)
        logger.info("[%s] summary saved → %s", short, summary_path)
    except Exception as se:
        logger.warning("[%s] summary failed (non-fatal): %s", short, se)
    finally:
        _jobs[job_id]["phase"] = "done"


@app.post("/documents")
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    logger.info("POST /documents — received file: %s (%.1f KB)",
                file.filename, (file.size or 0) / 1024)

    if not file.filename.lower().endswith(".pdf"):
        logger.warning("rejected non-PDF upload: %s", file.filename)
        raise HTTPException(400, "Only PDF files are supported.")

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info("file saved → %s", dest)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(_run_ingest, job_id, str(dest), file.filename)

    logger.info("job queued: %s for %s", job_id, file.filename)
    return {"job_id": job_id, "status": "processing"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/documents/{doc_id}/summary")
def get_summary(doc_id: int):
    conn = get_db()
    doc = conn.execute("SELECT filename FROM documents WHERE id = %s", (doc_id,)).fetchone()
    conn.close()
    if not doc:
        raise HTTPException(404, "Document not found.")
    stem = Path(doc["filename"]).stem
    matches = sorted(Path("summaries").glob(f"summary_{stem}_*.md"))
    if not matches:
        raise HTTPException(404, "Summary not yet generated for this document.")
    content = matches[-1].read_text(encoding="utf-8")
    return {"content": content, "filename": doc["filename"]}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    logger.info("DELETE /documents/%d", doc_id)
    conn = get_db()
    doc = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        logger.warning("delete failed — doc_id=%d not found", doc_id)
        raise HTTPException(404, "Document not found.")

    chunk_ids = [
        r["id"]
        for r in conn.execute("SELECT id FROM chunks WHERE doc_id = %s", (doc_id,)).fetchall()
    ]
    for cid in chunk_ids:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = %s", (cid,))
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()

    Path(doc["filepath"]).unlink(missing_ok=True)
    conn.close()
    logger.info("document deleted: doc_id=%d filename=%s", doc_id, doc["filename"])
    return {"status": "deleted"}


# ── chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    session_id: str = ""


@app.post("/chat")
def chat(req: ChatRequest):
    logger.info("POST /chat — query: %.80s…", req.message)
    return StreamingResponse(
        stream_chat(req.message, req.history, req.session_id),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
