from functools import lru_cache
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts: list[str]):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
