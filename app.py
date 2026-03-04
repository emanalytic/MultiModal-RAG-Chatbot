"""
PDF RAG Chatbot — Upload a PDF and chat with it.
Streamlit Version.

Usage:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from pdf_parser import PDFParser
from rag.pipeline import RAGPipeline
from rag.config import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL

# Page config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# Load environment variables
load_dotenv()

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e3440,#2e3440);
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_status" not in st.session_state:
    st.session_state.doc_status = "No document loaded."
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GROQ_API_KEY", "")

def reset_chat():
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    api_key = st.text_input("Groq API Key", value=st.session_state.api_key, type="password")
    if api_key:
        st.session_state.api_key = api_key
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    st.markdown("### Model Configuration")
    llm_model = st.selectbox("LLM Model", [DEFAULT_LLM_MODEL, "llama3-70b-8192", "mixtral-8x7b-32768"], index=0)
    embed_model = st.text_input("Embedding Model", value=DEFAULT_EMBED_MODEL)
    top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=3)
    use_ocr = st.checkbox("Enable OCR", value=True)
    
    if st.button("Load & Index"):
        if uploaded_file is None:
            st.error("Please upload a PDF file first.")
        elif not st.session_state.api_key:
            st.error("Please provide a Groq API Key.")
        else:
            try:
                with st.spinner("Processing PDF..."):
                    # Save uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    output_dir = os.path.join(os.path.dirname(tmp_path), "output")
                    
                    # 1. Parse PDF
                    st.toast("Parsing PDF...")
                    parser = PDFParser(tmp_path, output_dir=output_dir, ocr_images=use_ocr)
                    chunks = parser.parse()
                    json_path = parser.save_json()
                    summary = parser.summary()
                    
                    # 2. Initialize RAG
                    st.toast("Creating embeddings...")
                    rag = RAGPipeline(
                        groq_api_key=st.session_state.api_key,
                        embed_model=embed_model,
                        llm_model=llm_model,
                    )
                    rag.load_chunks(json_path)
                    
                    st.session_state.rag = rag
                    st.session_state.doc_status = (
                        f"**{summary['pdf']}** loaded\n\n"
                        f"{summary['total_pages']} pages\n"
                        f"{len(rag.chunks)} chunks indexed"
                    )
                    st.success("Indexing complete!")
                    reset_chat()
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    if st.button("Clear Chat"):
        reset_chat()
        st.rerun()

    st.markdown("---")
    st.info(st.session_state.doc_status)

# Main Area
st.title("PDF RAG Chatbot")
st.markdown("Upload a document in the sidebar to start chatting with its content.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDF..."):
    if st.session_state.rag is None:
        st.warning("Please upload and index a PDF first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                groq_history = []
                for msg in st.session_state.messages[:-1]:
                    content = msg["content"]
                    if msg["role"] == "assistant":
                        content = content.split("\n\n---\n**Sources:**")[0]
                    groq_history.append({"role": msg["role"], "content": content})

                answer, sources = st.session_state.rag.ask(prompt, groq_history, top_k=top_k)
                
                # Format sources
                source_lines = ["\n\n---\n**Sources:**"]
                for s in sources:
                    line = f"- Page **{s['page']}**"
                    if s.get("section"):
                        sec = s["section"][:60] + ("..." if len(s["section"]) > 60 else "")
                        line += f" · *{sec}*"
                    line += f" · `{s['type']}` · relevance: {s['score']}"
                    source_lines.append(line)
                
                full_answer = answer + "".join(source_lines)
                st.markdown(full_answer)
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
