"""Configuration module for Enhanced RAG-CSD."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PipelineConfig:
    """Configuration for enhanced pipeline."""
    # Storage paths
    vector_db_path: str
    storage_path: str = "./storage/enhanced"
    
    # Vector database settings
    vector_db: str = "incremental"  # Default vector database type
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Indexing settings
    delta_threshold: int = 10000
    max_delta_indices: int = 5
    
    # CSD emulation settings
    enable_csd_emulation: bool = True
    max_parallel_ops: int = 8
    ssd_bandwidth_mbps: int = 2000
    nand_bandwidth_mbps: int = 500
    compute_latency_ms: float = 0.1
    
    # Pipeline settings
    enable_pipeline_parallel: bool = True
    enable_system_data_flow: bool = False
    flexible_retrieval_interval: int = 3
    
    # Cache settings
    enable_caching: bool = True
    l1_cache_size_mb: int = 64
    l2_cache_size_mb: int = 512
    l3_cache_size_mb: int = 2048
    
    # Retrieval settings
    similarity_metric: str = "cosine"
    retrieval_only: bool = False
    
    # Performance settings
    max_batch_size: int = 32
    embedding_batch_size: int = 64
    prefetch_candidates: bool = False
    async_processing: bool = False
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        # Extract only valid fields
        valid_fields = cls.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_db_path": self.vector_db_path,
            "storage_path": self.storage_path,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "delta_threshold": self.delta_threshold,
            "max_delta_indices": self.max_delta_indices,
            "enable_csd_emulation": self.enable_csd_emulation,
            "max_parallel_ops": self.max_parallel_ops,
            "ssd_bandwidth_mbps": self.ssd_bandwidth_mbps,
            "nand_bandwidth_mbps": self.nand_bandwidth_mbps,
            "enable_pipeline_parallel": self.enable_pipeline_parallel,
            "enable_caching": self.enable_caching,
            "l1_cache_size_mb": self.l1_cache_size_mb,
            "l2_cache_size_mb": self.l2_cache_size_mb,
            "l3_cache_size_mb": self.l3_cache_size_mb,
        }
