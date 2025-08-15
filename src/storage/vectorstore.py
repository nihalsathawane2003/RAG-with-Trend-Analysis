import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict

class VectorStore:
    def __init__(self, path: str = ".chroma"):
        # Detect if running in a deployment environment (read-only filesystem)
        in_cloud = os.environ.get("DEPLOYMENT", "false").lower() == "true" or not os.access(".", os.W_OK)

        if in_cloud:
            # In-memory Chroma (no persistence)
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )
        else:
            # Persistent Chroma for local dev
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False)
            )

        self.collection = self.client.get_or_create_collection(
            name="posts",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def upsert(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        # Emulate upsert with delete+add
        try:
            self.collection.delete(ids=ids)
        except Exception:
            pass
        self.add(ids, docs, metadatas, embeddings)

    def query(self, text: str, n: int = 5, embedding_fn=None):
        if embedding_fn is None:
            raise ValueError("Provide embedding_fn(texts) -> embeddings")
        emb = embedding_fn([text])[0]
        return self.collection.query(
            query_embeddings=[emb],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
