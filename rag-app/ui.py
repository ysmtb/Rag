import streamlit as st
import time
from rag import ask_pipeline, ingest_document

# No API required! We are completely serverless.

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="centered")

st.title("📚 RAG Knowledge Assistant")
st.markdown("Ask questions grounded in the documents you ingest!")

# ── Sidebar: Document Upload & Status ─────────────────────────────────────────
with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    
    if st.button("📤 Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Processing (Extracting, Chunking, Embedding)..."):
                try:
                    # Pass the file directly to our RAG ingestor logic
                    chunks = ingest_document(uploaded_file.getvalue(), uploaded_file.name)
                    st.success(f"✅ Document ingested!\n\n**{chunks}** valid chunks sent to Chroma Cloud.")
                except Exception as e:
                    st.error(f"🚨 Ingestion Error: {str(e)}")
        else:
            st.warning("Please upload a file first.")


# ── Chat Interface ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the backend logic directly!
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            try:
                # Call the pure python function
                data = ask_pipeline(prompt)
                answer_text = data.get("answer", "Error: No answer provided.")
                
                st.markdown(answer_text)
                
                # Display Sources if RAG was used
                if data.get("retrieved") and data.get("sources"):
                    with st.expander("🔍 View Sources & Context", expanded=False):
                        for idx, source in enumerate(data["sources"], 1):
                            st.markdown(f"**Source {idx}:** `{source.get('source', 'unknown')}`")
                            st.caption(f"> {source.get('text', '')}")
                
                # Show latency footer
                total_latency = int((time.time() - start_time) * 1000)
                if not data.get("retrieved"):
                    st.caption(f"⚡ *Answered directly by LLM in {total_latency}ms (Skipped DB Search)*")
                else:
                    st.caption(f"⏱ *RAG Pipeline took {total_latency}ms*")

                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
            except Exception as e:
                st.error(f"🚨 Logic Error: {str(e)}")
