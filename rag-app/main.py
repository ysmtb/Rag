"""
main.py — FastAPI Endpoints.

Run natively using FastAPI CLI (no uvicorn needed in code):
  fastapi dev main.py
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import config, build_bm25_index, ingest_document, ask_pipeline, get_store


# ── Load Environment ──────────────────────────────────────────────────────────

load_dotenv()
config.groq_api_key = os.getenv("GROQ_API_KEY", "")
if not config.groq_api_key:
    print("Warning: GROQ_API_KEY not found in .env")


# ── Startup Logic ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Rebuild BM25 index from ChromaDB on startup."""
    print("[startup] Preparing BM25 Index...")
    build_bm25_index()
    yield


app = FastAPI(title="Simplified RAG App", lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    store = get_store()
    return {"status": "ok", "chunks_in_store": store.count()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a .txt or .pdf file."""
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(400, "Only .txt and .pdf allowed.")
    
    file_bytes = await file.read()
    try:
        count = ingest_document(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
        
    return {
        "chunks_stored": count,
        "source": file.filename,
        "total_in_store": get_store().count()
    }


class AskRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(request: AskRequest):
    """Ask a question."""
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    
    # Everything is handled inside the pipeline
    return ask_pipeline(request.query)
