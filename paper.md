# Production RAG — Implementation Plan
> Based on **"Searching for Best Practices in Retrieval-Augmented Generation"** (Wang et al., 2024 — arXiv:2407.01219)
> Stack: Python 3.11 · FastAPI · ChromaDB · Groq (free) · No Docker

---

## ✅ Runs 100% Locally — No Docker · Groq Free Tier

```bash
cd rag-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste your GROQ_API_KEY
uvicorn app.main:app --reload  # starts on localhost:8000
```

**Free API key:** [console.groq.com](https://console.groq.com) → Sign up → Create Key → done.

---

## Supported Document Types (Simple)

| Format | How it's read |
|---|---|
| `.txt` | `open(file).read()` — plain string |
| `.pdf` | `pypdf.PdfReader` — extracts text page by page |

That's it. No HTML, no DOCX, no web scraping in v1.

---

## Architecture Pipeline

```
User Query
    │
    ▼
┌───────────────────────┐
│  Query Classifier     │  Groq call, max_tokens=1 → "sufficient" / "insufficient"
└──────────┬────────────┘
           │ "insufficient" → continue
           ▼
┌───────────────────────┐
│  Hybrid Retriever     │  BM25 (rank_bm25) + Dense (bge-small local)
│  α = 0.3              │  score = 0.3·BM25 + 0.7·cosine  ← paper optimal
└──────────┬────────────┘
           │ top-10 chunks
           ▼
┌───────────────────────┐
│  Cross-Encoder Rerank │  ms-marco-MiniLM — local, single batch call
└──────────┬────────────┘
           │ top-5, REVERSED (best chunk closest to question)
           ▼
┌───────────────────────┐
│  Generator            │  Groq — llama-3.3-70b-versatile, temp=0
└───────────────────────┘
           │
           ▼
  {"answer": "...", "sources": [...], "latency_ms": 900}
```

---

## Tech Stack

| Component | Tool |
|---|---|
| API server | FastAPI + Uvicorn |
| Vector store | ChromaDB (local SQLite file, no server) |
| BM25 | `rank-bm25` (in-memory) |
| Embedding | `BAAI/bge-small-en-v1.5` (local HuggingFace, ~130 MB) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, ~90 MB) |
| LLM | Groq — `llama-3.3-70b-versatile` (free tier) |
| PDF parsing | `pypdf` |
| Config | `python-dotenv` |

---

## Project Structure (Complete)

```
rag-app/
│
├── .env.example                 # template — copy to .env
├── .env                         # your secrets — git-ignored
├── requirements.txt             # all pinned deps
├── Makefile                     # shortcuts: make run, make test, make ingest
│
├── data/                        # drop your .txt or .pdf files here
│   └── sample.txt               # test file for development
│
├── chroma_db/                   # auto-created by ChromaDB on first ingest
│   └── ...                      # SQLite + index files (git-ignored)
│
├── app/
│   ├── main.py                  # FastAPI app, startup events
│   ├── config.py                # Settings — reads .env via pydantic-settings
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── ingest.py            # POST /ingest  — upload .txt or .pdf
│   │   └── ask.py               # POST /ask     — full RAG pipeline
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── chunker.py           # sentence-level splitter (txt + pdf)
│       ├── embedder.py          # loads bge-small once, exposes embed()
│       ├── store.py             # ChromaDB wrapper — add / query chunks
│       ├── bm25_index.py        # BM25 index — built from ChromaDB texts
│       ├── retriever.py         # hybrid_search() — BM25 + dense, α=0.3
│       ├── reranker.py          # cross-encoder rerank + reverse repack
│       ├── classifier.py        # Groq zero-shot query router
│       └── generator.py         # Groq llama-3.3 call, returns answer str
│
└── tests/
    ├── conftest.py              # shared fixtures (TestClient, temp chroma dir)
    ├── test_health.py           # GET /health → 200
    ├── test_ingest.py           # POST /ingest with sample .txt and .pdf
    ├── test_classifier.py       # 5 routing tests
    ├── test_retrieval.py        # hybrid search returns expected chunk
    └── test_e2e.py              # ingest → ask → assert answer substring
```

---

## File Responsibilities (What Goes In Each File)

### `app/config.py`
```python
# illustrative
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"
    top_k: int = 10          # candidates from hybrid search
    top_n: int = 5           # kept after reranking
    alpha: float = 0.3       # BM25 weight (paper §3.4.3)
    chunk_size: int = 512    # tokens per chunk (paper §3.2.1)
    chunk_overlap: int = 20  # overlap tokens  (paper §3.2.1)
    chroma_persist_dir: str = "./chroma_db"
    class Config:
        env_file = ".env"
```

### `app/pipeline/chunker.py`
Responsibility: Accept raw text (already extracted from .txt or .pdf), split into sentence-aware chunks of ~512 tokens with 20-token overlap. Return `list[dict]` with keys `text`, `chunk_id`.

```python
# illustrative signature only
def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from .txt or .pdf bytes."""
    ...

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Split on sentence boundaries. Returns [{text, chunk_id}, ...]."""
    ...
```

### `app/pipeline/embedder.py`
Responsibility: Load `BAAI/bge-small-en-v1.5` **once** at app startup. Expose `embed(text: str) -> list[float]` and `embed_batch(texts: list[str]) -> list[list[float]]`.

### `app/pipeline/store.py`
Responsibility: Thin wrapper around ChromaDB. Exposes:
- `add_chunks(chunks, embeddings, source)` — upsert with metadata
- `query(embedding, n_results) -> list[dict]` — returns texts + distances + metadata
- `get_all_texts() -> list[str]` — needed to rebuild BM25 index

### `app/pipeline/bm25_index.py`
Responsibility: Build and hold a `BM25Okapi` index from all stored chunk texts. Exposes `search(query: str) -> list[float]` returning a score per corpus position. **Rebuilt on startup and after every new ingest.**

### `app/pipeline/retriever.py`
Responsibility: `hybrid_search(query, top_k) -> list[ScoredChunk]`.
- Call BM25 for sparse scores, min-max normalize
- Call ChromaDB dense query, invert distances, min-max normalize
- Combine: `0.3 × sparse + 0.7 × dense`
- Return top_k sorted descending

### `app/pipeline/reranker.py`
Responsibility: Load `cross-encoder/ms-marco-MiniLM-L-6-v2` once at startup. `rerank(query, chunks, top_n) -> list[ScoredChunk]` — single batched `.predict()` call. Returns chunks in **ascending** score order (lowest first = "Reverse" repacking — best chunk closest to the question in the prompt).

### `app/pipeline/classifier.py`
Responsibility: `classify(query: str) -> Literal["sufficient", "insufficient"]`. Calls Groq with `max_tokens=1`. Caches result with `functools.lru_cache`. Falls back to `"insufficient"` on API error.

### `app/pipeline/generator.py`
Responsibility: `generate(query: str, chunks: list[ScoredChunk]) -> dict`. Assembles context from chunks (already in reverse order), calls Groq `llama-3.3-70b-versatile` with `temperature=0`, returns `{"answer": str, "sources": list}`.

```python
# illustrative prompt assembly
context = "\n\n---\n\n".join(chunk.text for chunk in chunks)
# chunks are already reversed — highest score is last, closest to the question

messages = [
    {"role": "system", "content":
        "Answer ONLY using the context provided. "
        "If the answer is not in the context, say: I don't know."},
    {"role": "user", "content":
        f"Context:\n{context}\n\nQuestion: {query}"}
]
```

### `app/api/ingest.py`
Responsibility: `POST /ingest` accepts `UploadFile`. Validates extension (`.txt` or `.pdf` only, reject otherwise). Calls `extract_text → chunk_text → embed_batch → store.add_chunks`. Triggers BM25 index rebuild. Returns `{"chunks_stored": N, "source": filename}`.

### `app/api/ask.py`
Responsibility: `POST /ask {"query": "..."}`. Runs full pipeline using a timer around each stage. Returns:
```json
{
  "answer": "...",
  "sources": [{"text": "...", "source": "file.pdf", "chunk_id": "..."}],
  "retrieved": false,
  "latency_ms": {"classify": 280, "retrieve": 120, "rerank": 310, "generate": 650, "total": 1360}
}
```
If classifier returns `"sufficient"` → skip retrieval, call LLM directly, set `"retrieved": false`.

---

## `.env.example`

```bash
# Required
GROQ_API_KEY=your_groq_key_here

# Optional tuning (paper-recommended defaults)
LLM_MODEL=llama-3.3-70b-versatile
TOP_K=10
TOP_N=5
ALPHA=0.3
CHUNK_SIZE=512
CHUNK_OVERLAP=20
CHROMA_PERSIST_DIR=./chroma_db
```

---

## `requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-dotenv==1.0.1
pydantic-settings==2.2.1

# Vector store (no server, local SQLite)
chromadb==0.5.0

# Embeddings + Reranker (local models)
sentence-transformers==2.7.0
torch==2.3.0

# BM25
rank-bm25==0.2.2

# PDF parsing
pypdf==4.2.0

# LLM — Groq free tier
groq==0.9.0

# Testing
pytest==8.1.1
httpx==0.27.0
```

---

## `Makefile`

```makefile
run:
	uvicorn app.main:app --reload

test:
	pytest tests/ -v

ingest:
	curl -X POST localhost:8000/ingest -F "file=@$(FILE)"

ask:
	curl -X POST localhost:8000/ask \
	  -H "Content-Type: application/json" \
	  -d '{"query": "$(Q)"}'
```

Usage:
```bash
make ingest FILE=data/my_doc.pdf
make ask Q="What is this document about?"
```

---

## Milestone Plan

### Milestone 1 — Scaffold & `/health`
- [ ] Create `rag-app/` folder, init venv, install `requirements.txt`
- [ ] `app/config.py` — `Settings` reads `.env`
- [ ] `app/main.py` — FastAPI app with `GET /health → {"status": "ok"}`
- [ ] `pytest tests/test_health.py` → 1 passed

**Acceptance:**
```bash
uvicorn app.main:app --reload
curl localhost:8000/health  # {"status": "ok"}
```

---

### Milestone 2 — Ingest: `.txt` and `.pdf` → ChromaDB
- [ ] `pipeline/chunker.py` — `extract_text()` + `chunk_text()`
- [ ] `pipeline/embedder.py` — load `bge-small-en-v1.5` at startup
- [ ] `pipeline/store.py` — ChromaDB `get_or_create_collection`, `add_chunks`, `query`, `get_all_texts`
- [ ] `api/ingest.py` — `POST /ingest`, validate `.txt`/`.pdf` only
- [ ] `pytest tests/test_ingest.py` — upload `data/sample.txt`, assert `chunks_stored > 0`

**Acceptance:**
```bash
make ingest FILE=data/sample.txt
# → {"chunks_stored": 12, "source": "sample.txt"}
ls chroma_db/   # files appear
```

---

### Milestone 3 — Query Classifier
- [ ] `pipeline/classifier.py` — Groq call, `max_tokens=1`, LRU cache, safe fallback
- [ ] 5 unit tests covering both routing outcomes
- [ ] Overhead ≤ 400 ms

---

### Milestone 4 — Hybrid Retrieval (α = 0.3)
- [ ] `pipeline/bm25_index.py` — build from `store.get_all_texts()` on startup
- [ ] `pipeline/retriever.py` — normalize + combine BM25 and dense, return top_k
- [ ] `GET /search?q=...` endpoint for debugging
- [ ] `pytest tests/test_retrieval.py` — known phrase in top-3

---

### Milestone 5 — Reranking + Reverse Repacking
- [ ] `pipeline/reranker.py` — load cross-encoder at startup, single batch predict
- [ ] Return chunks in ascending score order (reverse = best last in prompt)
- [ ] Latency ≤ 500 ms for 10 chunks on Mac CPU

---

### Milestone 6 — Generator + Full `/ask` Pipeline
- [ ] `pipeline/generator.py` — Groq call, `temperature=0`, per-stage timer
- [ ] `api/ask.py` — wire full pipeline, return answer + sources + latency breakdown
- [ ] `pytest tests/test_e2e.py` — ingest sample → ask → assert expected substring

**Acceptance:**
```bash
make ask Q="What is this document about?"
# → {"answer": "...", "sources": [...], "latency_ms": {...}}
```

---

### Milestone 7 — Polish
- [ ] Structured logging (`structlog`) with `request_id` on every log line
- [ ] Graceful fallback: retrieval failure → LLM-only answer + `"retrieved": false`
- [ ] `POST /admin/rebuild-index` — manual BM25 rebuild trigger
- [ ] README with free API key setup + quickstart

---

## Verification Plan

| Milestone | Command | Pass Condition |
|---|---|---|
| 1 | `pytest tests/test_health.py` | HTTP 200 |
| 2 | `pytest tests/test_ingest.py` | `chunks_stored > 0`, `chroma_db/` created |
| 3 | `pytest tests/test_classifier.py` | 5/5 correct |
| 4 | `pytest tests/test_retrieval.py` | Known chunk in top-3 |
| 5 | Manual smoke test | Best chunk in last position of prompt |
| 6 | `pytest tests/test_e2e.py` | Answer contains expected fact |
| 7 | `make test` | All green |

---

## Key Paper Findings — Guardrails

| Finding | Decision |
|---|---|
| Query classification saves ~5 s AND lifts score | Always include — Milestone 3 |
| α = 0.3 is the BM25/dense hybrid sweet spot | Default in `.env`, configurable |
| Removing reranking = biggest single quality drop | Mandatory — Milestone 5 |
| "Reverse" repacking beats "forward" by ~2% | Best chunk LAST in prompt = closest to question |
| Summarization helps only if hitting token limits | Skip in v1 — Groq supports 128k context |
