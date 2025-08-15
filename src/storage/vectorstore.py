import os
from typing import List, Dict

# Force in-memory Chroma for Streamlit Cloud or restricted environments
# This avoids using persistent storage that isn't allowed there
os.environ["CHROMADB_DEFAULT_DATABASE"] = "duckdb_in_memory"

import chromadb


class VectorStore:
    def __init__(self):
        """
        Initializes an in-memory Chroma vector store.
        Works in Streamlit Cloud and other restricted environments.
        """
        # In-memory client (safe for cloud deployment)
        self.client = chromadb.Client()

        # Create or get a cosine-similarity collection
        self.collection = self.client.get_or_create_collection(
            name="posts",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        """
        Adds documents to the vector store.
        """
        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def upsert(self, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings):
        """
        Emulates an upsert by deleting then adding documents.
        """
        try:
            self.collection.delete(ids=ids)
        except Exception:
            pass
        self.add(ids, docs, metadatas, embeddings)

    def query(self, text: str, n: int = 5, embedding_fn=None):
        """
        Queries the vector store for similar documents.
        """
        if embedding_fn is None:
            raise ValueError("Provide embedding_fn(texts) -> embeddings")

        emb = embedding_fn([text])[0]
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        return res
