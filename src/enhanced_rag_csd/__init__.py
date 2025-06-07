"""
Enhanced RAG-CSD: High-performance RAG with CSD emulation.

This package provides a software-only RAG implementation that emulates
Computational Storage Device benefits through advanced optimizations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.pipeline import EnhancedRAGPipeline
from .core.config import PipelineConfig
from .retrieval.incremental_index import IncrementalVectorStore
from .core.csd_emulator import EnhancedCSDSimulator

# Evaluation imports
from .evaluation.accuracy_validator import (
    AccuracyValidator,
    ValidationDataset,
    DatasetLoader
)

# Benchmark imports (temporarily disabled for import fix)
# from .benchmarks.runner import BenchmarkRunner
# from .benchmarks.visualizer import BenchmarkVisualizer

__all__ = [
    "EnhancedRAGPipeline",
    "PipelineConfig",
    "IncrementalVectorStore",
    "EnhancedCSDSimulator",
    "AccuracyValidator",
    "ValidationDataset",
    "DatasetLoader",
    # "BenchmarkRunner",  # Temporarily disabled
    # "BenchmarkVisualizer",  # Temporarily disabled
]

# Configure matplotlib for headless environments
import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')