"""Test that all modules can be imported correctly."""

import pytest


def test_core_imports():
    """Test core module imports."""
    from enhanced_rag_csd import EnhancedRAGPipeline, PipelineConfig
    from enhanced_rag_csd.core import pipeline, config, csd_emulator
    
    assert EnhancedRAGPipeline is not None
    assert PipelineConfig is not None


def test_retrieval_imports():
    """Test retrieval module imports."""
    from enhanced_rag_csd import IncrementalVectorStore
    from enhanced_rag_csd.retrieval import incremental_index
    
    assert IncrementalVectorStore is not None


def test_evaluation_imports():
    """Test evaluation module imports."""
    from enhanced_rag_csd import AccuracyValidator, ValidationDataset
    from enhanced_rag_csd.evaluation import accuracy_validator
    
    assert AccuracyValidator is not None
    assert ValidationDataset is not None


def test_benchmark_imports():
    """Test benchmark module imports."""
    from enhanced_rag_csd.benchmarks import runner, visualizer, baseline_systems
    
    assert runner is not None
    assert visualizer is not None
    assert baseline_systems is not None


def test_utils_imports():
    """Test utils module imports."""
    from enhanced_rag_csd.utils import (
        logger, metrics, error_handling,
        embedding_cache, model_cache, text_processor
    )
    
    assert logger is not None
    assert metrics is not None
    assert error_handling is not None


def test_config_creation():
    """Test configuration object creation."""
    from enhanced_rag_csd import PipelineConfig
    
    config = PipelineConfig(
        vector_db_path="./test_vectors",
        enable_csd_emulation=True
    )
    
    assert config.vector_db_path == "./test_vectors"
    assert config.enable_csd_emulation is True
    assert config.embedding_dim == 384  # default value


def test_matplotlib_backend():
    """Test that matplotlib is configured for headless operation."""
    import matplotlib
    import os
    
    # In headless environment, backend should be Agg
    if os.environ.get('DISPLAY', '') == '':
        assert matplotlib.get_backend() == 'Agg'