import os

# Force in-memory Chroma for Streamlit Cloud / limited environments
os.environ["CHROMADB_DEFAULT_DATABASE"] = "duckdb_in_memory"

import chromadb


class VectorStore:
    def __init__(self, persist: bool = False, path: str = ".chroma"):
        if persist:
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(
            name="posts",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    def upsert(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        try:
            self.collection.delete(ids=ids)
        except Exception:
            pass
        self.add(ids, docs, metadatas, embeddings)

    def query(self, text: str, n: int = 5, embedding_fn=None):
        if embedding_fn is None:
            raise ValueError("Provide embedding_fn(texts) -> embeddings")
        emb = embedding_fn([text])[0]
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        return res
