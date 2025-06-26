"""
Enhanced simulator backend - wrapper around existing CSD simulator.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import time

from .base import CSDBackendInterface, CSDBackendType
from ..core.csd_emulator import EnhancedCSDSimulator
from ..core.hardware_models import CSDHardwareModel, CSDHardwareType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedSimulatorBackend(CSDBackendInterface):
    """Backend using the enhanced in-house CSD simulator with ML computational offloading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._backend_type = CSDBackendType.ENHANCED_SIMULATOR
        self.simulator = None
        
        # Computational offloading capabilities
        self.compute_units = config.get("csd", {}).get("compute_units", 4)
        self.ml_accelerator_freq = config.get("csd", {}).get("ml_frequency_mhz", 800)  # MHz
        self.embedding_dim = config.get("embedding", {}).get("dimensions", 384)
        
        # Hardware modeling for realistic delays
        hardware_type = config.get("csd", {}).get("hardware_type", "arm_cortex_a78")
        try:
            self.hardware_model = CSDHardwareModel(CSDHardwareType(hardware_type))
            logger.info(f"Using realistic hardware model: {hardware_type}")
        except ValueError:
            # Fallback to ARM cores if invalid type
            self.hardware_model = CSDHardwareModel(CSDHardwareType.ARM_CORTEX_A78)
            logger.warning(f"Invalid hardware type '{hardware_type}', falling back to ARM Cortex-A78")
        
        # ML computation metrics
        self.ml_metrics = {
            "ml_encode_ops": 0,
            "ml_retrieve_ops": 0, 
            "ml_augment_ops": 0,
            "total_ml_compute_time": 0.0,
            "offloaded_computations": 0
        }
    
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
    
    def encode_on_csd(self, queries: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode queries using on-CSD ML computational offloading."""
        start_time = time.time()
        
        # Use realistic hardware model for encoding computation
        data_shape = (len(queries), self.embedding_dim)
        encoding_latency = self.hardware_model.calculate_ml_operation_time(
            "embedding_lookup", data_shape, "fp32"
        )
        time.sleep(encoding_latency)
        
        # Update hardware utilization
        self.hardware_model.update_utilization(0.7)
        
        # Generate normalized embeddings
        embeddings = []
        for query in queries:
            # Simulate encoder computation
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        elapsed = time.time() - start_time
        self.ml_metrics["ml_encode_ops"] += len(queries)
        self.ml_metrics["total_ml_compute_time"] += elapsed
        self.ml_metrics["offloaded_computations"] += 1
        
        stats = {
            "encode_time_ms": elapsed * 1000,
            "queries_encoded": len(queries),
            "compute_units_used": min(self.compute_units, len(queries)),
            "hardware_model": self.hardware_model.hardware_type.value,
            "theoretical_ops": len(queries) * self.embedding_dim,
            "hardware_utilization": self.hardware_model.utilization
        }
        
        logger.debug(f"CSD Encode: {len(queries)} queries in {elapsed*1000:.2f}ms")
        return np.array(embeddings), stats
    
    def retrieve_on_csd(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Retrieve embeddings using on-CSD ML computational offloading."""
        start_time = time.time()
        
        # Use realistic hardware model for similarity computation
        candidates_to_search = min(1000, top_k * 20)  # Search expansion
        data_shape = (candidates_to_search, self.embedding_dim)
        retrieval_latency = self.hardware_model.calculate_ml_operation_time(
            "similarity_compute", data_shape, "fp32"
        )
        time.sleep(retrieval_latency)
        
        # Update hardware utilization
        self.hardware_model.update_utilization(0.8)
        
        # Get candidate embeddings from simulator
        if self.simulator is None:
            # Generate dummy candidates if simulator not available
            candidates = [np.random.randn(self.embedding_dim).astype(np.float32) for _ in range(top_k)]
            for i in range(len(candidates)):
                candidates[i] = candidates[i] / np.linalg.norm(candidates[i])
        else:
            # Use simulator to get actual candidates  
            candidate_indices = list(range(min(50, top_k * 10)))  # Search expansion
            if candidate_indices:
                similarities = self.compute_similarities(query_embedding, candidate_indices)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                candidates = self.retrieve_embeddings(top_indices.tolist())
                candidates = [candidates[i] for i in range(len(candidates))]
            else:
                candidates = [np.random.randn(self.embedding_dim).astype(np.float32) for _ in range(top_k)]
        
        elapsed = time.time() - start_time
        self.ml_metrics["ml_retrieve_ops"] += 1
        self.ml_metrics["total_ml_compute_time"] += elapsed
        self.ml_metrics["offloaded_computations"] += 1
        
        stats = {
            "retrieve_time_ms": elapsed * 1000,
            "candidates_retrieved": len(candidates),
            "compute_units_used": min(self.compute_units, len(candidates))
        }
        
        logger.debug(f"CSD Retrieve: {len(candidates)} candidates in {elapsed*1000:.2f}ms")
        return candidates, stats
    
    def augment_on_csd(self, query: str, retrieved_contexts: List[np.ndarray], metadata: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Augment query with retrieved contexts using on-CSD ML computational offloading."""
        start_time = time.time()
        
        # Simulate augmentation computation on CSD
        augmentation_latency = len(retrieved_contexts) * (0.3 / self.ml_accelerator_freq)  # MHz to ms
        time.sleep(augmentation_latency / 1000)
        
        # Perform context augmentation
        if retrieved_contexts and len(retrieved_contexts) > 0:
            # Create similarity-weighted context
            context_summary = f"Based on {len(retrieved_contexts)} relevant contexts: "
            for i, context in enumerate(retrieved_contexts[:3]):  # Use top 3 contexts
                context_summary += f"[Context {i+1}: relevant information] "
            
            augmented_query = f"{query}\n\nContext: {context_summary}"
        else:
            augmented_query = query
        
        elapsed = time.time() - start_time
        self.ml_metrics["ml_augment_ops"] += 1  
        self.ml_metrics["total_ml_compute_time"] += elapsed
        self.ml_metrics["offloaded_computations"] += 1
        
        stats = {
            "augment_time_ms": elapsed * 1000,
            "contexts_processed": len(retrieved_contexts),
            "compute_units_used": min(self.compute_units, len(retrieved_contexts))
        }
        
        logger.debug(f"CSD Augment: {len(retrieved_contexts)} contexts in {elapsed*1000:.2f}ms")
        return augmented_query, stats

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from the enhanced simulator.""" 
        if self.simulator is None:
            combined_metrics = {**self.metrics, **self.ml_metrics}
        else:
            simulator_metrics = self.simulator.get_metrics()
            # Combine backend metrics with simulator and ML metrics
            combined_metrics = {**self.metrics, **simulator_metrics, **self.ml_metrics}
        
        combined_metrics["backend_type"] = self._backend_type.value
        combined_metrics["computational_offloading"] = True
        combined_metrics["ml_compute_units"] = self.compute_units
        combined_metrics["ml_accelerator_freq_mhz"] = self.ml_accelerator_freq
        
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