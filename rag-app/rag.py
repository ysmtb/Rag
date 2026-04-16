"""
rag.py — All RAG logic (Vector DB, Embedder, Retriever, Generator) in one simple file.
"""

import re
import io
import time
import functools
import numpy as np
import chromadb
import pypdf
from typing import Optional

import os
import chromadb
from groq import Groq


# ── Configuration (Loaded in main.py) ─────────────────────────────────────────

# We hold settings globally here so main.py can inject them.
# Minimal default configuration.
class RAGConfig:
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    top_k: int = 10
    top_n: int = 5
    alpha: float = 0.3
    chunk_size: int = 512
    chunk_overlap: int = 20
    
    # ── Chroma Cloud Auth ──
    chroma_host: str = os.environ.get("CHROMA_HOST", "api.trychroma.com")
    chroma_tenant: str = os.environ.get("CHROMA_TENANT", "")
    chroma_database: str = os.environ.get("CHROMA_DATABASE", "")
    chroma_api_key: str = os.environ.get("CHROMA_API_KEY", "")

config = RAGConfig()


# ── 1. Document Parsing & Chunking ────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from .txt or .pdf files."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    if ext == "pdf":
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    raise ValueError(f"Unsupported file type: .{ext}. Only .txt and .pdf allowed.")

def chunk_text(text: str) -> list[dict]:
    """Split text into sentence-aware chunks using chunk_size and chunk_overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        count = len(sentence.split())
        if current_token_count + count > config.chunk_size and current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks.append({"text": chunk_str, "chunk_id": f"chunk_{len(chunks)}"})
            overlap_str = " ".join(chunk_str.split()[-config.chunk_overlap:])
            current_sentences = [overlap_str]
            current_token_count = len(overlap_str.split())
        current_sentences.append(sentence)
        current_token_count += count

    if current_sentences:
        chunks.append({"text": " ".join(current_sentences), "chunk_id": f"chunk_{len(chunks)}"})
    return chunks


# ── 2. Models (Loaded lazily) ─────────────────────────────────────────────────

# Caching instances so they don't reload on every request
_vector_store = None

def get_store() -> chromadb.Collection:
    global _vector_store
    if _vector_store is None:
        client = chromadb.CloudClient(
            tenant=config.chroma_tenant, 
            database=config.chroma_database, 
            api_key=config.chroma_api_key
        )
        
        # Cloud Schema handles dense/sparse natively if configured in dashboard
        _vector_store = client.get_or_create_collection(
            name="rag_documents_cloud"
        )
        print(f"[rag] Chroma Cloud ready. Chunks stored: {_vector_store.count()}")
    return _vector_store


# ── 3. Database & Indexing Operations ─────────────────────────────────────────

def ingest_document(file_bytes: bytes, filename: str) -> int:
    """Extract, chunk, embed, and store document in Chroma Cloud."""
    text = extract_text(file_bytes, filename)
    if not text.strip(): raise ValueError("File is empty.")
    
    chunks = chunk_text(text)
    
    # Enforce 16 KiB limit
    valid_chunks = []
    for c in chunks:
        if len(c["text"].encode('utf-8')) <= 16000:
            valid_chunks.append(c)
        else:
            print(f"[rag] Warning: Dropping chunk from {filename} exceeding 16KiB limit.")
    
    if not valid_chunks: return 0
    
    texts = [c["text"] for c in valid_chunks]
    store = get_store()
    ids = [f"{filename}__{c['chunk_id']}" for c in valid_chunks]
    
    # Use source_document_id for GroupBy deduplication
    metadatas = [{"source": filename, "source_document_id": filename, "chunk_id": c["chunk_id"]} for c in valid_chunks]
    
    # Cloud Schema handles dense/sparse natively; upsert raw text!
    store.upsert(ids=ids, documents=texts, metadatas=metadatas)
    return len(valid_chunks)


# ── 4. Query Pipeline ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=512)
def classify_query(query: str) -> str:
    """Decide if we need RAG ('insufficient') or if LLM can answer directly ('sufficient')."""
    try:
        client = Groq(api_key=config.groq_api_key)
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": "Reply with one word: 'sufficient' if query needs no outside info (like translation/grammar) else 'insufficient'."},
                {"role": "user", "content": query},
            ],
            max_tokens=1, temperature=0,
        )
        return "sufficient" if response.choices[0].message.content.lower().startswith("suf") else "insufficient"
    except Exception as e:
        print(f"[rag] Classifier API Error: {e} - defaulting to 'insufficient'")
        return "insufficient"

def hybrid_search(query: str) -> list[dict]:
    """Native Cloud Hybrid Search with Deduplication."""
    store = get_store()
    try:
        # Offload RRF math and grouping to the cloud!
        results = store.search(query=query) \
            .group_by("source_document_id") \
            .limit(config.top_k) \
            .get()
            
        candidates = []
        if results and hasattr(results, 'documents') and results.documents:
            docs = results.documents[0] if isinstance(results.documents[0], list) else results.documents
            metas = results.metadatas[0] if isinstance(results.metadatas[0], list) else results.metadatas
            for d_text, meta in zip(docs, metas):
                candidates.append({
                    "text": d_text,
                    "source": meta.get("source", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "score": 1.0  # RRF score is handled by Chroma
                })
        return candidates
    except Exception as e:
        print(f"[rag] Cloud Search Error: {e}")
        return []

def rerank_and_repack(query: str, chunks: list[dict]) -> list[dict]:
    """Cloud handles reranking natively via RRF, simply repack here."""
    if not chunks: return []
    top = chunks[:config.top_n]
    return list(reversed(top)) # Best chunk placed last

def generate_answer(query: str, reversed_chunks: list[dict]) -> dict:
    """Generate final answer using Groq."""
    context = "\n\n---\n\n".join([f"[{i+1}] (source: {c.get('source', '')})\n{c['text']}" for i, c in enumerate(reversed_chunks)])
    
    client = Groq(api_key=config.groq_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "Answer ONLY using the provided context. If unknown, say: I don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0, max_tokens=512,
    )
    
    sources = [{"source": c.get("source"), "text": c["text"][:150] + "..."} for c in reversed_chunks]
    return {"answer": response.choices[0].message.content.strip(), "sources": sources}


# ── Full Orchestration ────────────────────────────────────────────────────────

def ask_pipeline(query: str) -> dict:
    """Run the complete pipeline end-to-end, measuring latency."""
    t_start = time.perf_counter()
    latency = {}
    
    # 1. Classify
    t0 = time.perf_counter()
    needs_rag = classify_query(query) == "insufficient"
    latency["classify_ms"] = round((time.perf_counter() - t0) * 1000)

    if not needs_rag:
        # LLM answers natively
        t0 = time.perf_counter()
        client = Groq(api_key=config.groq_api_key)
        ans = client.chat.completions.create(model=config.llm_model, messages=[{"role": "user", "content": query}], temperature=0).choices[0].message.content
        latency["total_ms"] = round((time.perf_counter() - t_start) * 1000)
        return {"answer": ans, "sources": [], "retrieved": False, "latency": latency}

    # 2. Retrieve
    t0 = time.perf_counter()
    candidates = hybrid_search(query)
    latency["retrieve_ms"] = round((time.perf_counter() - t0) * 1000)
    
    if not candidates:
        latency["total_ms"] = round((time.perf_counter() - t_start) * 1000)
        return {"answer": "No documents uploaded yet.", "sources": [], "retrieved": False, "latency": latency}

    # 3. Rerank
    t0 = time.perf_counter()
    top_chunks = rerank_and_repack(query, candidates)
    latency["rerank_ms"] = round((time.perf_counter() - t0) * 1000)

    # 4. Generate
    t0 = time.perf_counter()
    result = generate_answer(query, top_chunks)
    result["retrieved"] = True
    latency["generate_ms"] = round((time.perf_counter() - t0) * 1000)
    
    result["latency"] = latency
    result["latency"]["total_ms"] = round((time.perf_counter() - t_start) * 1000)
    return result
