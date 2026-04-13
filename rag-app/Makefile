# ── Dev shortcuts ─────────────────────────────────────
run:
	.venv/bin/fastapi dev main.py
	
ui:
	.venv/bin/streamlit run ui.py


# ── Data helpers ──────────────────────────────────────
# Usage: make ingest FILE=data/my_doc.pdf
ingest:
	curl -s -X POST http://localhost:8000/ingest \
	  -F "file=@$(FILE)" | python3 -m json.tool

# Usage: make ask Q="What is this document about?"
ask:
	curl -s -X POST http://localhost:8000/ask \
	  -H "Content-Type: application/json" \
	  -d "{\"query\": \"$(Q)\"}" | python3 -m json.tool

# ── Setup ─────────────────────────────────────────────
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✅ Done. Edit .env and add your GROQ_API_KEY, then: make run"
