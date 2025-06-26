
import faiss
import numpy as np
from typing import List, Dict, Any

from .vectordb import VectorDB

class IVFPQVectorDB(VectorDB):
    """A vector database that uses Faiss for high-performance similarity search."""

    def __init__(self, dimension: int, nlist: int = 100, m: int = 8, nbits: int = 8):
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFPQ(self.quantizer, self.dimension, self.nlist, self.m, self.nbits)
        self.chunks = []
        self.metadata = []

    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add new documents to the index."""
        if not self.index.is_trained:
            self.index.train(embeddings)
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
                    "chunk": self.chunks[idx],
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
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "is_trained": self.index.is_trained,
        }
