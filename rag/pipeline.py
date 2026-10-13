"""Orchestrates source ingestion, retrieval, reranking and grounded generation."""
from pathlib import Path
import os
import pandas as pd
from groq import Groq
from config import CHUNK_SIZE, CHUNK_OVERLAP, GROQ_MODEL
from ingestion.common import Document, chunk_document
from ingestion.pdf_loader import load_pdf
from ingestion.pptx_loader import load_pptx
from ingestion.csv_loader import load_csv
from ingestion.database_loader import load_sqlite
from ingestion.web_loader import load_website
from retrieval.vector_store import FaissStore
from retrieval.hybrid_search import HybridRetriever
from retrieval.sentence_window import make_sentence_windows, restore_window
from retrieval.graph_retrieval import CooccurrenceGraph
from retrieval.reranker import rerank
from rag.prompts import SYSTEM_PROMPT

def ingest_path(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".pptx":
        return load_pptx(path)
    if suffix == ".csv":
        return load_csv(path)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return load_sqlite(path)
    raise ValueError(f"Unsupported file type: {suffix}. Supported: PDF, PPTX, CSV, SQLite.")

def ingest_website(url):
    return load_website(url)

def chunk_documents(raw_documents):
    output = []
    for doc in raw_documents:
        output.extend(chunk_document(doc, CHUNK_SIZE, CHUNK_OVERLAP))
    return output

class RAGPipeline:
    def __init__(self, documents):
        if not documents:
            raise ValueError("No documents were supplied.")
        self.documents = documents
        self.store = FaissStore().build(documents)
        self.hybrid = HybridRetriever(documents)
        self.graph = CooccurrenceGraph(documents)
        # Sentence-window index is built separately; this is more useful for sentence-level evidence.
        sentence_docs = make_sentence_windows(documents, window=2)
        self.sentence_store = FaissStore().build(sentence_docs) if sentence_docs else None

    def retrieve(self, query, mode="hybrid", top_k=5, use_reranker=True):
        if mode == "semantic":
            results = self.store.search(query, k=top_k)
        elif mode == "hybrid":
            results = self.hybrid.search(query, self.store, k=top_k)
        elif mode == "sentence_window":
            raw = self.sentence_store.search(query, k=top_k) if self.sentence_store else []
            results = [restore_window(x) for x in raw]
        elif mode == "graph":
            results = self.graph.search(query, k=top_k)
            # If the heuristic graph finds no entity overlap, fall back to semantic retrieval.
            if not results:
                results = self.store.search(query, k=top_k)
        else:
            raise ValueError("mode must be semantic, hybrid, sentence_window, or graph")
        if use_reranker and results:
            try:
                results = rerank(query, results, top_k=top_k)
            except Exception as exc:
                # Retrieval remains usable if model download/runtime is unavailable.
                for item in results:
                    item["reranker_note"] = f"Reranker unavailable: {type(exc).__name__}"
        return results[:top_k]

    def answer(self, query, mode="hybrid", top_k=5, use_reranker=True):
        results = self.retrieve(query, mode, top_k, use_reranker)
        if not results:
            return {"answer": "I could not find relevant evidence in the indexed sources.", "sources": [], "retrieval": []}
        source_lines, source_rows = [], []
        for i, item in enumerate(results, start=1):
            doc = item["document"]
            source_id = f"S{i}"
            metadata = doc.metadata or {}
            location = []
            if metadata.get("page"): location.append(f"page {metadata['page']}")
            if metadata.get("row"): location.append(f"row {metadata['row']}")
            if metadata.get("url"): location.append(metadata["url"])
            if metadata.get("table"): location.append(f"table {metadata['table']}")
            loc = ", ".join(location)
            source_lines.append(f"[{source_id}] source={doc.source}; type={doc.source_type}; location={loc}\n{doc.text}")
            source_rows.append({
                "id": source_id, "source": doc.source, "source_type": doc.source_type,
                "location": loc, "text": doc.text, "score": item.get("rerank_score", item.get("score")),
                "method": item.get("method", "unknown")
            })
        context = "\n\n".join(source_lines)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {
                "answer": "GROQ_API_KEY is not set. Retrieval succeeded, but answer generation is disabled. Set the key in your environment and retry.",
                "sources": source_rows, "retrieval": results
            }
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", GROQ_MODEL),
            temperature=0.1,
            max_tokens=900,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nRetrieved context:\n{context}\n\nAnswer with source citations."}
            ]
        )
        return {"answer": completion.choices[0].message.content or "", "sources": source_rows, "retrieval": results}
