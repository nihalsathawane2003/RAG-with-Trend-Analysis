import chromadb
from chromadb.config import Settings
from typing import List, Dict

class VectorStore:
    def __init__(self, path: str = ".chroma"):
        self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="posts", metadata={"hnsw:space":"cosine"})

    def add(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    def upsert(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        # Chroma doesn't have upsert in older versions; emulate with delete+add
        try:
            self.collection.delete(ids=ids)
        except Exception:
            pass
        self.add(ids, docs, metadatas, embeddings)

    def query(self, text: str, n: int = 5, embedding_fn=None):
        if embedding_fn is None:
            raise ValueError("Provide embedding_fn(texts)->embeddings")
        emb = embedding_fn([text])[0]
        res = self.collection.query(query_embeddings=[emb], n_results=n, include=["documents","metadatas","distances"])
        return res
