#!/usr/bin/env python
"""Script to migrate enhanced implementations to new repository structure."""

import os
import shutil
from pathlib import Path

# Define source and destination mappings
file_mappings = {
    # Core modules
    "../rag-csd/rag_csd/enhanced_pipeline.py": "src/enhanced_rag_csd/core/pipeline.py",
    "../rag-csd/rag_csd/csd/enhanced_simulator.py": "src/enhanced_rag_csd/core/csd_emulator.py",
    
    # Retrieval modules
    "../rag-csd/rag_csd/retrieval/incremental_index.py": "src/enhanced_rag_csd/retrieval/incremental_index.py",
    
    # Evaluation modules
    "../rag-csd/rag_csd/evaluation/accuracy_validator.py": "src/enhanced_rag_csd/evaluation/accuracy_validator.py",
    
    # Benchmark modules
    "../rag-csd/rag_csd/benchmarks/enhanced_benchmark.py": "src/enhanced_rag_csd/benchmarks/runner.py",
    "../rag-csd/rag_csd/benchmarks/visualizer.py": "src/enhanced_rag_csd/benchmarks/visualizer.py",
    
    # Utils - we'll need to extract relevant parts
    "../rag-csd/rag_csd/utils/logger.py": "src/enhanced_rag_csd/utils/logger.py",
    "../rag-csd/rag_csd/utils/metrics.py": "src/enhanced_rag_csd/utils/metrics.py",
    "../rag-csd/rag_csd/utils/error_handling.py": "src/enhanced_rag_csd/utils/error_handling.py",
    
    # Examples
    "../rag-csd/examples/enhanced_demo.py": "examples/demo.py",
    "../rag-csd/examples/run_full_benchmark.py": "examples/benchmark.py",
}

# Additional files we need from the original implementation
additional_files = {
    "../rag-csd/rag_csd/embedding/encoder.py": "src/enhanced_rag_csd/core/encoder.py",
    "../rag-csd/rag_csd/augmentation/augmentor.py": "src/enhanced_rag_csd/core/augmentor.py",
    "../rag-csd/rag_csd/utils/embedding_cache.py": "src/enhanced_rag_csd/utils/embedding_cache.py",
    "../rag-csd/rag_csd/utils/model_cache.py": "src/enhanced_rag_csd/utils/model_cache.py",
    "../rag-csd/rag_csd/utils/text_processor.py": "src/enhanced_rag_csd/utils/text_processor.py",
}

def migrate_files():
    """Migrate files to new structure."""
    print("Starting code migration...")
    
    # Combine all mappings
    all_mappings = {**file_mappings, **additional_files}
    
    for src, dst in all_mappings.items():
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            # Create destination directory
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            print(f"✓ Copied {src} -> {dst}")
        else:
            print(f"✗ Source file not found: {src}")
    
    print("\nMigration complete!")

def create_config_module():
    """Create the config module."""
    config_content = '''"""Configuration module for Enhanced RAG-CSD."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PipelineConfig:
    """Configuration for enhanced pipeline."""
    # Storage paths
    vector_db_path: str
    storage_path: str = "./enhanced_storage"
    
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
'''
    
    with open("src/enhanced_rag_csd/core/config.py", "w") as f:
        f.write(config_content)
    print("✓ Created config.py")

def create_init_files():
    """Create __init__.py files for all packages."""
    packages = [
        "src/enhanced_rag_csd/core",
        "src/enhanced_rag_csd/retrieval", 
        "src/enhanced_rag_csd/benchmarks",
        "src/enhanced_rag_csd/evaluation",
        "src/enhanced_rag_csd/utils",
    ]
    
    for package in packages:
        init_file = Path(package) / "__init__.py"
        init_file.write_text('"""Package initialization."""\n')
        print(f"✓ Created {init_file}")

if __name__ == "__main__":
    migrate_files()
    create_config_module()
    create_init_files()