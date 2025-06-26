
import numpy as np
import pytest

from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline, PipelineConfig


@pytest.mark.parametrize("db_type", ["incremental", "faiss", "ivf_flat", "ivf_pq", "hnsw"])
def test_incremental_add(db_type):
    config = PipelineConfig(
        vector_db_path="./test_db",
        vector_db=db_type,
    )
    pipeline = EnhancedRAGPipeline(config)

    # Add initial documents
    embeddings1 = np.random.rand(10, 384).astype(np.float32)
    documents1 = [f"doc{i}" for i in range(10)]
    metadata1 = [{ "id": i } for i in range(10)]
    pipeline.add_documents(embeddings1, documents1, metadata1)

    # Check initial state
    stats1 = pipeline.vector_store.get_statistics()
    assert stats1["total_vectors"] == 10

    # Add more documents
    embeddings2 = np.random.rand(20, 384).astype(np.float32)
    documents2 = [f"doc{i}" for i in range(10, 30)]
    metadata2 = [{ "id": i } for i in range(10, 30)]
    pipeline.add_documents(embeddings2, documents2, metadata2)

    # Check final state
    stats2 = pipeline.vector_store.get_statistics()
    assert stats2["total_vectors"] == 30

    # Test search
    query = np.random.rand(384).astype(np.float32)
    results = pipeline.vector_store.search(query, top_k=5)
    assert len(results) == 5
