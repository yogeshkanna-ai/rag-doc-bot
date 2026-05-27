import requests
import streamlit as st
import time

st.set_page_config(
    page_title="RAG Document Q&A Bot",
    page_icon="📄",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000"

st.title("📄 RAG Document Q&A Bot")
st.caption("Upload a PDF and ask questions intelligently.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

with st.sidebar:
    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    upload_clicked = st.button("Upload PDF", use_container_width=True)

    st.markdown("---")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if upload_clicked:
    if uploaded_file is None:
        st.warning("Please choose a PDF first.")
    else:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        try:
            with st.spinner("📄 Uploading and processing PDF..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()

            st.session_state.uploaded = True
            st.session_state.messages = []
            st.success(f"✅ Uploaded: {data['filename']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Upload failed: {e}")

st.divider()

question = st.chat_input("Ask a question about the uploaded PDF")

if question:
    if not st.session_state.uploaded:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        try:
            progress_placeholder = st.empty()

            progress_placeholder.info("🔍 Searching relevant document sections...")
            time.sleep(0.7)

            progress_placeholder.info("🧠 Understanding document context...")
            time.sleep(0.7)

            progress_placeholder.info("🤖 Generating intelligent response...")
            time.sleep(0.7)

            response = requests.get(
                f"{API_URL}/ask",
                params={"question": question},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()

            progress_placeholder.empty()

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", [])
            })

            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Ask failed: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Source Chunks"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(src)
                    st.markdown("---")