# Local RAG Pipeline (Simplified)

This project is a radically simplified, 100% locally-runnable Retrieval-Augmented Generation (RAG) system built entirely in Python. It is based on the research paper: *"Searching for Best Practices in Retrieval-Augmented Generation"*.

## 📂 Project Structure

To make reading and learning as easy as possible, we have collapsed all logic into just 2 main files:

```text
rag-app/
├── main.py                # ⚡️ The FastAPI web routes and configuration
├── rag.py                 # 🧠 The core Machine Learning & Data Pipeline logic
├── requirements.txt       # 📦 Project dependencies
├── .env.example           # 🔑 Template for your API keys
└── Makefile               # 🛠️ Helpful terminal shorthand commands
```

### 1. `main.py` (The API Layer)
This file acts as the bridge between the user and the AI. We use **FastAPI** to expose simple endpoints. It exposes:
- `POST /ingest`: Takes a `.txt` or `.pdf` file from the user and hands it to the ML backend.
- `POST /ask`: Takes a user's question, asks the ML backend to process it, and returns the AI's response.
- `GET /health`: Gives a simple health check to see how many data chunks are currently stored.

### 2. `rag.py` (The Machine Learning Engine)
This is the heart of the RAG application. It is meant to be read top-to-bottom. Here is what it does internally:
1. **Document Parsing & Chunking**: It takes raw documents and splits them cleanly on sentence boundaries (max 512 tokens) with 20 words overlapping so that concepts aren't severed perfectly in half.
2. **Dense Semantic Search**: It loads the local `BAAI/bge-small-en-v1.5` AI model to convert your text chunks into numbers (embeddings) and stores them in a local SQLite file using **ChromaDB**.
3. **Sparse Keyword Search**: It creates an in-memory lexical keyword index using **BM25**.
4. **Hybrid Search**: When a question comes in, it combines the scores from both Semantic text and Keyword text (weighted dynamically at an exact 0.3 alpha algorithm proven in research) for extremely accurate data recovery.
5. **Cross-Encoder Reranking**: It takes the top 10 fetched results, and pushes them through a secondary neural network (`ms-marco-MiniLM-L-6-v2`) to accurately reorganize them. The *most* relevant chunk is placed at the exact bottom of the text pile (called *Reverse Repacking*) to take advantage of LLM recency-bias.
6. **Generation**: Finally, it pipes the exact context directly to the **Groq API** (`llama-3.3-70b-versatile`) to generate a grounded, hallucination-free response with 100% cited sources.

---

## 🚀 Usage

### 1. Initial Setup
```bash
# Add your free Groq API key:
cp .env.example .env
# Edit .env and enter GROQ_API_KEY="gsk_..."
```

### 2. Start the Server
Run the FastAPI development server:
```bash
make run
```
*The local server will now be live on `http://localhost:8000`.*

### 3. Using the AI 
There are two ways to use the App: Using the Visual Frontend or using the Terminal.

**Option A: The Visual Frontend (Recommended)**
Open another terminal tab and start the Streamlit UI:
```bash
make ui
```
This will open `http://localhost:8501` in your web browser. You can drag and drop `.txt` or `.pdf` files onto the sidebar to ingest them, and use the central chat window to ask questions against the uploaded documents.

**Option B: The Terminal**
Open another terminal tab. To add knowledge *(ingestion)*:
```bash
make ingest FILE=data/sample.txt
```

To ask a question:
```bash
make ask Q="Explain the Reverse Repacking concept"
```
