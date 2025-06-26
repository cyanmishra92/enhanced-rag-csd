
import faiss
import numpy as np
from typing import List, Dict, Any

from .vectordb import VectorDB

class HNSWVectorDB(VectorDB):
    """A vector database that uses Faiss for high-performance similarity search."""

    def __init__(self, dimension: int, m: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.chunks = []
        self.metadata = []

    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add new documents to the index."""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(documents)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        query_embedding = query_embedding.copy()
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1), min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append({
                    "content": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score),
                })
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index.__class__.__name__,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
