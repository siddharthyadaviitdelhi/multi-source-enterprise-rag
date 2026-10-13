"""Cross-encoder reranking of retrieved query/document pairs."""
from functools import lru_cache
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)

def rerank(query: str, results: list[dict], top_k: int = 5):
    if not results:
        return []
    model = get_reranker()
    pairs = [(query, item["document"].text) for item in results]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = []
    for item, score in zip(results, scores):
        ranked.append({**item, "rerank_score": float(score)})
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]
