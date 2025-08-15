from typing import List, Dict
import numpy as np


class VectorStore:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.embeddings = np.empty((0, dim), dtype="float32")
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.metadatas: List[Dict] = []

    def add(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        arr = np.array(embeddings, dtype="float32")
        self.embeddings = np.vstack([self.embeddings, arr])
        self.ids.extend(ids)
        self.docs.extend(docs)
        self.metadatas.extend(metadatas)

    def upsert(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        # Naive upsert — clears all and re-adds
        self.embeddings = np.empty((0, self.dim), dtype="float32")
        self.ids, self.docs, self.metadatas = [], [], []
        self.add(ids, docs, metadatas, embeddings)

    def query(self, text: str, n: int = 5, embedding_fn=None):
        if embedding_fn is None:
            raise ValueError("Provide embedding_fn(texts) -> embeddings")

        if len(self.ids) == 0:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        query_emb = np.array(embedding_fn([text]), dtype="float32")[0]
        sims = self._cosine_similarity(query_emb, self.embeddings)
        top_idx = np.argsort(sims)[::-1][:n]

        return {
            "ids": [self.ids[i] for i in top_idx],
            "documents": [self.docs[i] for i in top_idx],
            "metadatas": [self.metadatas[i] for i in top_idx],
            "distances": [float(s) for s in sims[top_idx]],
        }

    @staticmethod
    def _cosine_similarity(vec, mat):
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
        return np.dot(mat_norm, vec_norm)
