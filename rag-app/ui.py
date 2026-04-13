import streamlit as st
import requests
import time

# Configurations
API_URL = "http://localhost:8000"

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
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Document ingested!\n\n**{data['chunks_stored']}** chunks created.")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Cannot connect to backend. Did you run `make run`?")
        else:
            st.warning("Please upload a file first.")

    st.divider()
    st.header("System Status")
    try:
        health_resp = requests.get(f"{API_URL}/health")
        if health_resp.status_code == 200:
            chunks = health_resp.json().get("chunks_in_store", 0)
            st.metric("Total Fragments Stored", chunks)
            st.success("API is Online ✅")
        else:
            st.error("API is Offline ❌")
    except:
        st.error("API is Offline ❌\nPlease run `make run` in your terminal.")


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

    # Call the backend API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                start_time = time.time()
                response = requests.post(f"{API_URL}/ask", json={"query": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer_text = data["answer"]
                    
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
                
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Cannot connect to API. Please make sure `make run` is running in another terminal window.")
