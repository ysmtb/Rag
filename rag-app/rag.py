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

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
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
    chroma_persist_dir: str = "./chroma_db"

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
_embedder_model = None
_reranker_model = None
_vector_store = None
_bm25_index = None
_bm25_texts = []

def get_embedder() -> SentenceTransformer:
    global _embedder_model
    if _embedder_model is None:
        print("[rag] Loading embedding model (BAAI/bge-small-en-v1.5) ...")
        _embedder_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder_model

def get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        print("[rag] Loading reranker model (ms-marco-MiniLM-L-6-v2) ...")
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model

def get_store() -> chromadb.Collection:
    global _vector_store
    if _vector_store is None:
        client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        _vector_store = client.get_or_create_collection(
            name="rag_documents", 
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[rag] ChromaDB ready. Chunks stored: {_vector_store.count()}")
    return _vector_store


# ── 3. Database & Indexing Operations ─────────────────────────────────────────

def ingest_document(file_bytes: bytes, filename: str) -> int:
    """Extract, chunk, embed, and store document in ChromaDB."""
    text = extract_text(file_bytes, filename)
    if not text.strip(): raise ValueError("File is empty.")
    
    chunks = chunk_text(text)
    texts = [c["text"] for c in chunks]
    
    embedder = get_embedder()
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    
    store = get_store()
    ids = [f"{filename}__{c['chunk_id']}" for c in chunks]
    metadatas = [{"source": filename, "chunk_id": c["chunk_id"]} for c in chunks]
    
    store.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    
    # Rebuild BM25 search index
    build_bm25_index()
    return len(chunks)

def build_bm25_index():
    """Extract all text from ChromaDB and rebuild the fast lexical search index."""
    global _bm25_index, _bm25_texts
    store = get_store()
    total = store.count()
    if total == 0: return

    result = store.get(include=["documents"])
    _bm25_texts = result["documents"]
    
    tokenized = [text.lower().split() for text in _bm25_texts]
    _bm25_index = BM25Okapi(tokenized)
    print(f"[rag] BM25 Index rebuilt. {total} chunks ready.")


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
    """Combine BM25 (Keyword) and Sparse Embeddings (Semantic) retrieval."""
    store = get_store()
    if store.count() == 0: return []
    top_k = config.top_k
    
    # 1. Dense (Semantic Search)
    q_embed = get_embedder().encode(query, normalize_embeddings=True).tolist()
    dense_results = store.query(query_embeddings=[q_embed], n_results=min(top_k * 3, store.count()), include=["documents", "metadatas", "distances"])
    
    dense_map = {}
    if dense_results["documents"]:
        for d_text, meta, dist in zip(dense_results["documents"][0], dense_results["metadatas"][0], dense_results["distances"][0]):
            dense_map[d_text] = {"distance": dist, "meta": meta}

    # 2. Sparse (BM25 Keyword Search)
    bm25_map = {}
    if _bm25_index:
        scores = _bm25_index.get_scores(query.lower().split())
        for i, text in enumerate(_bm25_texts):
            if scores[i] > 0: bm25_map[text] = scores[i]

    # Combine and Normalize
    all_texts = set(dense_map.keys()) | set(bm25_map.keys())
    raw_dense = []
    raw_bm25 = []
    candidates = []

    for text in all_texts:
        hit = dense_map.get(text, {"distance": 1.0, "meta": {"source": "bm25", "chunk_id": "unknown"}})
        raw_dense.append(1.0 - hit["distance"])
        raw_bm25.append(bm25_map.get(text, 0.0))
        candidates.append({"text": text, "source": hit["meta"].get("source", ""), "chunk_id": hit["meta"].get("chunk_id", "")})

    def norm(arr):
        a = np.array(arr, dtype=float)
        return [0.0]*len(a) if a.max() == a.min() else ((a - a.min()) / (a.max() - a.min())).tolist()
    
    n_dense, n_bm25 = norm(raw_dense), norm(raw_bm25)
    for i, c in enumerate(candidates):
        c["score"] = (config.alpha * n_bm25[i]) + ((1 - config.alpha) * n_dense[i])
        
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

def rerank_and_repack(query: str, chunks: list[dict]) -> list[dict]:
    """Precise reranking and 'reverse repacking' (best chunk is closest to question)."""
    if not chunks: return []
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    
    for c, score in zip(chunks, scores): c["rerank_score"] = float(score)
    top = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:config.top_n]
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
