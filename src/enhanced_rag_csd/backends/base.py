"""
Base interface for CSD backend implementations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class CSDBackendType(Enum):
    """Available CSD backend implementations."""
    ENHANCED_SIMULATOR = "enhanced_simulator"
    SPDK_EMULATOR = "spdk_emulator"
    FIRESIM_FPGA = "firesim_fpga"
    CUSTOM_HARDWARE = "custom_hardware"


class CSDBackendInterface(ABC):
    """Abstract interface for computational storage device backends."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the CSD backend with configuration."""
        self.config = config
        self.metrics = {
            "read_ops": 0,
            "write_ops": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_bytes_read": 0,
            "total_bytes_written": 0,
            "avg_latency": 0.0,
            "backend_type": self.__class__.__name__
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the backend.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def store_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        Store embeddings with metadata.
        
        Args:
            embeddings: Array of embedding vectors
            metadata: List of metadata dictionaries for each embedding
        """
        pass
    
    @abstractmethod 
    def retrieve_embeddings(self, indices: List[int], use_cache: bool = True) -> np.ndarray:
        """
        Retrieve embeddings by indices.
        
        Args:
            indices: List of embedding indices to retrieve
            use_cache: Whether to use cache hierarchy
            
        Returns:
            Array of retrieved embeddings
        """
        pass
    
    @abstractmethod
    def compute_similarities(self, query_embedding: np.ndarray, 
                           candidate_indices: List[int]) -> np.ndarray:
        """
        Compute similarities on the CSD.
        
        Args:
            query_embedding: Query vector
            candidate_indices: Indices of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        pass
    
    @abstractmethod
    def process_era_pipeline(self, query_data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """
        Process complete Encode-Retrieve-Augment pipeline on CSD.
        
        Args:
            query_data: Input query data
            metadata: Query metadata
            
        Returns:
            Augmented data ready for generation
        """
        pass
    
    @abstractmethod
    def p2p_transfer_to_gpu(self, data: np.ndarray) -> str:
        """
        Transfer data from CSD to GPU via P2P.
        
        Args:
            data: Data to transfer
            
        Returns:
            GPU allocation identifier
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the backend.
        
        Returns:
            Dictionary containing performance metrics
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend and clean up resources."""
        pass
    
    def get_backend_type(self) -> CSDBackendType:
        """Get the backend type."""
        return getattr(self, '_backend_type', CSDBackendType.ENHANCED_SIMULATOR)
    
    def is_available(self) -> bool:
        """Check if backend dependencies are available."""
        return True  # Override in subclasses for dependency checking
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the backend."""
        return {
            "backend_type": self.get_backend_type().value,
            "is_available": self.is_available(),
            "description": self.__class__.__doc__ or "No description available",
            "config": self.config
        }