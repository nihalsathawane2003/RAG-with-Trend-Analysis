from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import List

@lru_cache(maxsize=1)
def get_model():
    # Small, fast model
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: List[str]):
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
