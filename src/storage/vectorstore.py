from typing import List, Dict
import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim: int = 768):
        """
        FAISS in-memory vector store.
        dim = dimension of embeddings (default 768 for many models)
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []
        self.metadatas = []
        self.ids = []

    def add(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        """
        Add vectors + metadata to FAISS index.
        """
        emb_array = np.array(embeddings).astype("float32")
        self.index.add(emb_array)
        self.ids.extend(ids)
        self.docs.extend(docs)
        self.metadatas.extend(metadatas)

    def upsert(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        """
        Simple upsert: clears and re-adds all data.
        """
        # This is a naive implementation (replace with proper per-ID removal if needed)
        self.index = faiss.IndexFlatL2(self.dim)
        self.ids, self.docs, self.metadatas = [], [], []
        self.add(ids, docs, metadatas, embeddings)

    def query(self, text: str, n: int = 5, embedding_fn=None):
        """
        Query FAISS for similar items.
        """
        if embedding_fn is None:
            raise ValueError("Provide embedding_fn(texts) -> embeddings")

        emb = np.array(embedding_fn([text])).astype("float32")
        distances, idxs = self.index.search(emb, n)

        results = {
            "ids": [self.ids[i] for i in idxs[0] if i < len(self.ids)],
            "documents": [self.docs[i] for i in idxs[0] if i < len(self.docs)],
            "metadatas": [self.metadatas[i] for i in idxs[0] if i < len(self.metadatas)],
            "distances": distances[0].tolist(),
        }
        return results
