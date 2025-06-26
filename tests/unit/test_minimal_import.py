
import pytest

def test_minimal_pipeline_import():
    from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline
    assert EnhancedRAGPipeline is not None
