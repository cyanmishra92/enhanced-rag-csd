
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class VectorDB(ABC):
    """Abstract base class for vector databases."""

    @abstractmethod
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add new documents to the index."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass
