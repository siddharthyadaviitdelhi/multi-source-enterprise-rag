import os
import tempfile
from pathlib import Path
import streamlit as st
from config import UPLOAD_DIR, SAMPLE_DIR
from ingestion.common import Document
from rag.pipeline import ingest_path, ingest_website, chunk_documents, RAGPipeline

st.set_page_config(page_title="Enterprise Multi-Source RAG", page_icon="🔎", layout="wide")
st.title("Multi-Source Enterprise Knowledge Base")
st.caption("PDF · PowerPoint (PPTX) · CSV · SQLite · public webpages | FAISS · hybrid search · sentence windows · graph heuristic · reranking")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []

with st.sidebar:
    st.header("1. Add sources")
    uploaded = st.file_uploader("Upload PDF, PowerPoint, CSV, or SQLite file(s)", type=["pdf", "pptx", "csv", "db", "sqlite", "sqlite3"], accept_multiple_files=True)
    website_urls = st.text_area("Public webpage URLs (one per line)", placeholder="https://example.com/about")
    use_sample = st.checkbox("Include bundled sample knowledge", value=True)
    if st.button("Build / rebuild knowledge base", type="primary", use_container_width=True):
        raw_docs = []
        errors = []
        if use_sample:
            for p in SAMPLE_DIR.iterdir():
                if p.suffix.lower() in {".csv", ".db", ".sqlite", ".sqlite3", ".pdf"}:
                    try: raw_docs.extend(ingest_path(p))
                    except Exception as e: errors.append(f"{p.name}: {e}")
        if uploaded:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            for file in uploaded:
                target = UPLOAD_DIR / Path(file.name).name
                target.write_bytes(file.getvalue())
                try: raw_docs.extend(ingest_path(target))
                except Exception as e: errors.append(f"{file.name}: {e}")
        for url in [u.strip() for u in website_urls.splitlines() if u.strip()]:
            try: raw_docs.extend(ingest_website(url))
            except Exception as e: errors.append(f"{url}: {e}")
        chunks = chunk_documents(raw_docs)
        if chunks:
            with st.spinner(f"Embedding and indexing {len(chunks)} chunks..."):
                st.session_state.pipeline = RAGPipeline(chunks)
                st.session_state.indexed_names = sorted({d.source for d in chunks})
            st.success(f"Indexed {len(raw_docs)} extracted records into {len(chunks)} chunks.")
        else:
            st.error("No text was indexed. Add sources or enable sample data.")
        for error in errors:
            st.warning(error)

    st.divider()
    st.header("2. Retrieval settings")
    mode = st.selectbox("Retrieval strategy", ["hybrid", "semantic", "sentence_window", "graph"])
    top_k = st.slider("Number of retrieved passages", min_value=1, max_value=10, value=5)
    use_reranker = st.checkbox("Use cross-encoder reranker", value=True)
    st.caption("First use may download embedding/reranker models. Graph mode uses a transparent co-occurrence heuristic.")

if st.session_state.pipeline is None:
    st.info("Build the knowledge base in the sidebar, then ask a question below.")
    st.markdown("**Try:** `What is the company's data retention policy?`")
else:
    st.write("**Indexed sources:** " + ", ".join(st.session_state.indexed_names))
    question = st.text_input("Ask a question about your sources", placeholder="What is the data retention policy?")
    if st.button("Search and answer", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                result = st.session_state.pipeline.answer(question, mode, top_k, use_reranker)
            st.subheader("Answer")
            st.markdown(result["answer"])
            st.subheader("Retrieved source passages")
            for source in result["sources"]:
                with st.expander(f"[{source['id']}] {source['source']} — {source['location'] or source['source_type']}"):
                    st.caption(f"Method: {source['method']} | Score: {source['score']}")
                    st.write(source["text"])
        except Exception as e:
            st.error(f"Request failed: {type(e).__name__}: {e}")

st.divider()
st.caption("Demo project: do not upload confidential or regulated data to third-party APIs without approval.")
