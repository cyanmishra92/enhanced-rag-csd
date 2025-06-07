"""
Enhanced simulator backend - wrapper around existing CSD simulator.
"""

from typing import Dict, List, Any
import numpy as np

from .base import CSDBackendInterface, CSDBackendType
from ..core.csd_emulator import EnhancedCSDSimulator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedSimulatorBackend(CSDBackendInterface):
    """Backend using the enhanced in-house CSD simulator."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._backend_type = CSDBackendType.ENHANCED_SIMULATOR
        self.simulator = None
    
    def initialize(self) -> bool:
        """Initialize the enhanced simulator backend."""
        try:
            self.simulator = EnhancedCSDSimulator(self.config)
            logger.info("Enhanced simulator backend initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize enhanced simulator backend: {e}")
            return False
    
    def store_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Store embeddings using the enhanced simulator."""
        if self.simulator is None:
            raise RuntimeError("Backend not initialized")
        
        self.simulator.store_embeddings(embeddings, metadata)
        self.metrics["write_ops"] += len(embeddings)
        self.metrics["total_bytes_written"] += embeddings.nbytes
    
    def retrieve_embeddings(self, indices: List[int], use_cache: bool = True) -> np.ndarray:
        """Retrieve embeddings using the enhanced simulator."""
        if self.simulator is None:
            raise RuntimeError("Backend not initialized")
        
        result = self.simulator.retrieve_embeddings(indices, use_cache)
        self.metrics["read_ops"] += len(indices) 
        self.metrics["total_bytes_read"] += result.nbytes
        return result
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           candidate_indices: List[int]) -> np.ndarray:
        """Compute similarities using the enhanced simulator."""
        if self.simulator is None:
            raise RuntimeError("Backend not initialized")
        
        return self.simulator.compute_similarities(query_embedding, candidate_indices)
    
    def process_era_pipeline(self, query_data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Process ERA pipeline using the enhanced simulator."""
        if self.simulator is None:
            raise RuntimeError("Backend not initialized")
        
        return self.simulator.process_era_pipeline(query_data, metadata)
    
    def p2p_transfer_to_gpu(self, data: np.ndarray) -> str:
        """Transfer data to GPU using the enhanced simulator."""
        if self.simulator is None:
            raise RuntimeError("Backend not initialized")
        
        return self.simulator.p2p_transfer_to_gpu(data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from the enhanced simulator.""" 
        if self.simulator is None:
            return self.metrics
        
        simulator_metrics = self.simulator.get_metrics()
        
        # Combine backend metrics with simulator metrics
        combined_metrics = {**self.metrics, **simulator_metrics}
        combined_metrics["backend_type"] = self._backend_type.value
        
        return combined_metrics
    
    def shutdown(self) -> None:
        """Shutdown the enhanced simulator."""
        if self.simulator is not None:
            self.simulator.shutdown()
            self.simulator = None
        
        logger.info("Enhanced simulator backend shutdown complete")
    
    def is_available(self) -> bool:
        """Enhanced simulator is always available."""
        return True