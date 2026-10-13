"""Reciprocal Rank Fusion of FAISS semantic results and BM25 keyword results."""
import re
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())

class HybridRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(d.text) for d in documents])

    def search(self, query, vector_store, k=5, candidate_k=12, rrf_k=60):
        dense = vector_store.search(query, k=min(candidate_k, len(self.documents)))
        bm25_scores = self.bm25.get_scores(tokenize(query))
        lexical_ids = sorted(range(len(self.documents)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_k]
        fused = {}
        docs_by_id = {}
        for rank, result in enumerate(dense, start=1):
            doc = result["document"]
            idx = next((i for i, d in enumerate(self.documents) if d is doc), None)
            if idx is not None:
                fused[idx] = fused.get(idx, 0.0) + 1 / (rrf_k + rank)
                docs_by_id[idx] = doc
        for rank, idx in enumerate(lexical_ids, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1 / (rrf_k + rank)
            docs_by_id[idx] = self.documents[idx]
        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{"document": docs_by_id[i], "score": float(fused[i]), "method": "hybrid_rrf"} for i in ranked]
