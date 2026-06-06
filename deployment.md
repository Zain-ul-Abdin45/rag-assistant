# Deployment Guide

This guide covers local Docker testing, GitHub setup, and production deployment to Railway.

---

## 1. Files added by this guide

| File | Purpose |
|---|---|
| `Dockerfile` | Container image for the FastAPI app |
| `.dockerignore` | Excludes secrets, caches, and generated data from the image |
| `docker-compose.yml` | Local full-stack (app + postgres + ollama) |
| `.gitignore` | Keeps secrets and generated output out of git |

---

## 2. Local Docker testing

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Step 1 — Build and start all services

```bash
docker compose up --build
```

This starts three containers:

| Service | What it runs | Port |
|---|---|---|
| `app` | FastAPI + uvicorn | 8000 |
| `postgres` | pgvector/pgvector:pg16 | 5432 |
| `ollama` | ollama/ollama:latest | 11434 |

### Step 2 — Pull Ollama models

On the first run the Ollama container has no models. Pull them once:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
```

### Step 3 — Open the app

```
http://localhost:8000
```

### Useful commands

```bash
# View app logs
docker compose logs -f app

# Stop everything (keep volumes)
docker compose down

# Stop and delete all data volumes
docker compose down -v

# Rebuild after code changes
docker compose up --build app
```

---

## 3. GitHub setup

### Step 1 — Verify nothing secret is tracked

```bash
git status
```

Confirm `env.local` is **not** listed. If it appears, run:

```bash
git rm --cached env.local
```

### Step 2 — Initialise and push

```bash
git init
git add .
git commit -m "initial commit"
```

Create a new **private** repository on GitHub (do not initialise with a README), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/rag-assistant.git
git branch -M main
git push -u origin main
```

---

## 4. Railway deployment

Railway runs the app container and provides managed PostgreSQL. Ollama runs as a separate Railway service.

### Prerequisites

- [Railway CLI](https://docs.railway.app/develop/cli) installed: `npm i -g @railway/cli`
- Railway account linked: `railway login`

---

### Step 1 — Create the project

```bash
railway init
```

Choose **Empty project** and give it a name (e.g. `rag-assistant`).

---

### Step 2 — Add PostgreSQL

In the Railway dashboard:

1. Click **+ New** → **Database** → **PostgreSQL**
2. Railway provisions a Postgres instance and automatically sets `DATABASE_URL` in your project environment.

> **pgvector check:** Railway's default Postgres image includes pgvector. The app runs `CREATE EXTENSION IF NOT EXISTS vector` on startup — no manual step needed. If it fails, open a Railway shell on the Postgres service and run `CREATE EXTENSION vector;` manually.

---

### Step 3 — Deploy Ollama as a separate Railway service

Ollama is not a managed Railway offering — deploy it as a Docker service:

1. In the Railway dashboard click **+ New** → **Docker Image**
2. Image: `ollama/ollama:latest`
3. Add a volume mount: `/root/.ollama` (so models survive redeploys)
4. Note the **internal hostname** Railway assigns (e.g. `ollama.railway.internal`)

After it starts, open a Railway shell on the Ollama service and pull the models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

> **GPU note:** Railway offers GPU instances but they are billed separately. For CPU-only Ollama the embedding step is slower (~30 s per chunk) but functional. Consider upgrading to a GPU instance for production traffic.

---

### Step 4 — Deploy the app

Link your GitHub repository to a new Railway service:

1. Click **+ New** → **GitHub Repo**
2. Select your `rag-assistant` repository
3. Railway detects the `Dockerfile` automatically

---

### Step 5 — Set environment variables

In the Railway dashboard, open the app service → **Variables** and add:

| Variable | Value | Notes |
|---|---|---|
| `DATABASE_URL` | *(auto-set by Railway Postgres plugin)* | Do not override |
| `OLLAMA_HOST` | `http://ollama.railway.internal:11434` | Internal hostname from Step 3 |
| `EMBED_MODEL` | `nomic-embed-text` | |
| `CHAT_MODEL` | `llama3.2` | |
| `EMBED_DIM` | `768` | |
| `UPLOAD_DIR` | `uploads` | |
| `CHUNK_SIZE` | `800` | |
| `CHUNK_OVERLAP` | `150` | |
| `TOP_K` | `5` | |
| `EMBED_WORKERS` | `1` | Use 1 on CPU-only Ollama to avoid timeout races |
| `LOG_LEVEL` | `WARNING` | Keeps logs minimal in production |

> To set variables via CLI:
> ```bash
> railway variables set LOG_LEVEL=WARNING EMBED_WORKERS=1 ...
> ```

---

### Step 6 — Add a persistent volume for uploads

Uploaded PDFs live in `uploads/`. Without a volume they are lost on every redeploy.

1. Open the app service → **Volumes**
2. Mount path: `/app/uploads`
3. Do the same for `/app/summaries` if you want generated summaries to persist

---

### Step 7 — Deploy

```bash
railway up
```

Or push to `main` — Railway redeploys automatically on every push.

---

### Step 8 — Verify

```bash
railway logs          # tail live logs
railway open          # open the public URL in your browser
```

Hit `GET /health` to confirm Ollama is reachable:

```bash
curl https://YOUR_APP.up.railway.app/health
# {"status":"ok","ollama":true,"chat_model":"llama3.2"}
```

---

## 5. Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | Full PostgreSQL connection string |
| `OLLAMA_HOST` | `http://localhost:11434` | URL of the Ollama server |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for embeddings |
| `CHAT_MODEL` | `llama3.2` | Ollama model used for chat and summaries |
| `EMBED_DIM` | `768` | Embedding vector dimension (must match model) |
| `UPLOAD_DIR` | `uploads` | Directory for uploaded files |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `EMBED_WORKERS` | `3` | Parallel embedding threads |
| `LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` |

---

## 6. Log level guide

| `LOG_LEVEL` | What gets logged | Recommended for |
|---|---|---|
| `DEBUG` | Everything including per-chunk embed progress | Local debugging only |
| `INFO` | Startup · uploads · ingest start/done · chat queries · warnings | Local dev default |
| `WARNING` | Only warnings and errors | Production |
| `ERROR` | Only errors | High-traffic production |

Set in `env.local` for local overrides:

```env
LOG_LEVEL=DEBUG
```

Set as a Railway environment variable for production:

```
LOG_LEVEL=WARNING
```

---

## 7. Post-deployment checklist

- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] Upload a PDF — job completes without error in logs
- [ ] Ask a question — answer streams back correctly
- [ ] `/app/uploads` volume is mounted (files survive redeploy)
- [ ] `env.local` is **not** in the git repository (`git log --all -- env.local` returns nothing)
- [ ] `LOG_LEVEL=WARNING` is set in Railway variables
