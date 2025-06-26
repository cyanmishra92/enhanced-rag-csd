
from .vectordb import VectorDB
from .faiss_vectordb import FaissVectorDB
from .incremental_index import IncrementalVectorStore
from .ivf_flat_vectordb import IVFflatVectorDB
from .ivf_pq_vectordb import IVFPQVectorDB
from .hnsw_vectordb import HNSWVectorDB

class VectorDBFactory:
    """Factory for creating vector databases."""

    @staticmethod
    def create_vectordb(db_type: str, **kwargs) -> VectorDB:
        """Create a vector database based on the given type."""
        if db_type == "faiss":
            return FaissVectorDB(**kwargs)
        elif db_type == "incremental":
            return IncrementalVectorStore(**kwargs)
        elif db_type == "ivf_flat":
            return IVFflatVectorDB(**kwargs)
        elif db_type == "ivf_pq":
            return IVFPQVectorDB(**kwargs)
        elif db_type == "hnsw":
            return HNSWVectorDB(**kwargs)
        else:
            raise ValueError(f"Unknown vector database type: {db_type}")
