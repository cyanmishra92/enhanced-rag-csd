
import numpy as np

from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline, PipelineConfig

def main():
    for db_type in ["incremental", "faiss", "ivf_flat", "ivf_pq", "hnsw"]:
        print(f"Testing {db_type}...")
        config = PipelineConfig(
            vector_db_path="./test_db",
            vector_db=db_type,
        )
        pipeline = EnhancedRAGPipeline(config)

        # Add initial documents
        embeddings1 = np.random.rand(10, 384).astype(np.float32)
        documents1 = [f"doc{i}" for i in range(10)]
        metadata1 = [{ "id": i } for i in range(10)]
        pipeline.add_documents(documents1, metadata1)

        # Check initial state
        stats1 = pipeline.vector_store.get_statistics()
        # In the pipeline, documents are chunked, so we can't directly compare counts.
        # We'll just check that the index is not empty.
        assert stats1["total_vectors"] > 0

        # Add more documents
        documents2 = [f"doc{i}" for i in range(10, 30)]
        metadata2 = [{ "id": i } for i in range(10, 30)]
        pipeline.add_documents(documents2, metadata2)

        # Check final state
        stats2 = pipeline.vector_store.get_statistics()
        assert stats2["total_vectors"] > stats1["total_vectors"]

        # Test search
        query = np.random.rand(384).astype(np.float32)
        results = pipeline.vector_store.search(query, top_k=5)
        assert len(results) == 5

        print(f"{db_type} test passed!")

if __name__ == "__main__":
    main()
