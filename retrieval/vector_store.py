"""FAISS inner-product index over normalized sentence-transformer embeddings."""
from pathlib import Path
import json
import faiss
import numpy as np
from ingestion.common import Document
from .embeddings import embed_texts

class FaissStore:
    def __init__(self):
        self.index = None
        self.documents: list[Document] = []

    def build(self, documents: list[Document]):
        if not documents:
            raise ValueError("No documents to index.")
        vectors = embed_texts([d.text for d in documents]).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = documents
        return self

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Build or load the FAISS index first.")
        q = embed_texts([query]).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append({"document": self.documents[int(idx)], "score": float(score), "method": "faiss"})
        return results

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        payload = [d.to_dict() for d in self.documents]
        (directory / "documents.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        store = cls()
        store.index = faiss.read_index(str(directory / "index.faiss"))
        raw = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        store.documents = [Document(**d) for d in raw]
        return store
