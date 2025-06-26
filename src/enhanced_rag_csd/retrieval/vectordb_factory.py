
from typing import List
from .vectordb import VectorDB
from .faiss_vectordb import FaissVectorDB
from .incremental_index import IncrementalVectorStore
from .ivf_flat_vectordb import IVFflatVectorDB
from .ivf_pq_vectordb import IVFPQVectorDB
from .hnsw_vectordb import HNSWVectorDB
from .lsh_vectordb import LSHVectorDB
from .scann_vectordb import ScaNNVectorDB
from .ngt_vectordb import NGTVectorDB

class VectorDBFactory:
    """Factory for creating vector databases."""

    @staticmethod
    def create_vectordb(db_type: str, **kwargs) -> VectorDB:
        """Create a vector database based on the given type."""
        if db_type == "faiss":
            return FaissVectorDB(**kwargs)
        elif db_type == "incremental":
            # Provide default storage_path if not given
            if 'storage_path' not in kwargs:
                kwargs['storage_path'] = './incremental_storage'
            return IncrementalVectorStore(**kwargs)
        elif db_type == "ivf_flat":
            # Use smaller nlist for small datasets
            if 'nlist' not in kwargs:
                kwargs['nlist'] = 10
            return IVFflatVectorDB(**kwargs)
        elif db_type == "ivf_pq":
            # Use smaller nlist for small datasets
            if 'nlist' not in kwargs:
                kwargs['nlist'] = 10
            return IVFPQVectorDB(**kwargs)
        elif db_type == "hnsw":
            return HNSWVectorDB(**kwargs)
        elif db_type == "lsh":
            return LSHVectorDB(**kwargs)
        elif db_type == "scann":
            # Use smaller num_clusters for small datasets
            if 'num_clusters' not in kwargs:
                kwargs['num_clusters'] = 10
            # Remove invalid parameters
            kwargs.pop('nlist', None)
            return ScaNNVectorDB(**kwargs)
        elif db_type == "ngt":
            return NGTVectorDB(**kwargs)
        else:
            raise ValueError(f"Unknown vector database type: {db_type}")
    
    @staticmethod
    def get_available_types() -> List[str]:
        """Get list of available vector database types."""
        return [
            "faiss",           # FAISS flat index
            "incremental",     # Incremental indexing
            "ivf_flat",        # Inverted File with Flat quantizer
            "ivf_pq",          # Inverted File with Product Quantization
            "hnsw",            # Hierarchical Navigable Small World
            "lsh",             # Locality Sensitive Hashing
            "scann",           # ScaNN (Scalable Nearest Neighbors)
            "ngt"              # Neighborhood Graph and Tree
        ]
