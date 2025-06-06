#!/usr/bin/env python
"""
Comprehensive demo of the enhanced RAG-CSD system with incremental indexing,
CSD emulation, and performance comparisons.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enhanced_rag_csd.enhanced_pipeline import EnhancedRAGPipeline, PipelineConfig
from enhanced_rag_csd.benchmarks.enhanced_benchmark import EnhancedBenchmarkRunner
from enhanced_rag_csd.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def generate_demo_documents() -> List[Dict[str, Any]]:
    """Generate demo documents for testing."""
    documents = [
        {
            "content": "Computational Storage Devices (CSDs) represent a paradigm shift in data processing "
                      "by bringing computation closer to where data resides. This approach significantly "
                      "reduces data movement overhead and improves overall system efficiency.",
            "metadata": {"source": "csd_intro.txt", "category": "technology"}
        },
        {
            "content": "Retrieval-Augmented Generation (RAG) combines the benefits of large language models "
                      "with dynamic information retrieval. By augmenting queries with relevant context, "
                      "RAG systems can provide more accurate and up-to-date responses.",
            "metadata": {"source": "rag_overview.txt", "category": "ai"}
        },
        {
            "content": "Vector similarity search is a fundamental operation in modern AI systems. "
                      "Efficient algorithms like FAISS enable fast nearest neighbor search in "
                      "high-dimensional spaces, making real-time retrieval possible.",
            "metadata": {"source": "vector_search.txt", "category": "algorithms"}
        },
        {
            "content": "The integration of computational storage with RAG systems opens new possibilities "
                      "for edge computing. By processing data where it's stored, we can build more "
                      "efficient and scalable AI applications.",
            "metadata": {"source": "csd_rag_integration.txt", "category": "innovation"}
        },
        {
            "content": "Incremental indexing allows dynamic updates to vector databases without full rebuilds. "
                      "This approach uses delta indices and drift detection to maintain search quality "
                      "while minimizing computational overhead.",
            "metadata": {"source": "incremental_indexing.txt", "category": "optimization"}
        }
    ]
    
    return documents


def generate_test_queries() -> List[str]:
    """Generate test queries for demonstration."""
    return [
        "What are the benefits of computational storage devices?",
        "How does RAG improve language model responses?",
        "Explain vector similarity search algorithms",
        "What is the advantage of processing data near storage?",
        "How does incremental indexing work?",
        "Compare CSD with traditional storage architectures",
        "What are the key components of a RAG system?",
        "How can we optimize vector search performance?",
        "Describe edge computing applications for AI",
        "What is drift detection in vector databases?"
    ]


def demonstrate_basic_usage(pipeline: EnhancedRAGPipeline) -> None:
    """Demonstrate basic pipeline usage."""
    print("\n" + "="*80)
    print("🔹 BASIC USAGE DEMONSTRATION")
    print("="*80)
    
    # Single query
    query = "What are the benefits of computational storage?"
    print(f"\n📝 Query: {query}")
    
    result = pipeline.query(query, top_k=3)
    
    print(f"\n✅ Results:")
    print(f"   Processing time: {result['processing_time']:.3f}s")
    print(f"   Retrieved {len(result['retrieved_docs'])} relevant documents")
    print(f"\n🔍 Augmented query preview:")
    print(f"   {result['augmented_query'][:200]}...")


def demonstrate_incremental_indexing(pipeline: EnhancedRAGPipeline) -> None:
    """Demonstrate incremental indexing capabilities."""
    print("\n" + "="*80)
    print("🔹 INCREMENTAL INDEXING DEMONSTRATION")
    print("="*80)
    
    # Get initial statistics
    stats = pipeline.get_statistics()
    print(f"\n📊 Initial index statistics:")
    print(f"   Total vectors: {stats['vector_store']['total_vectors']}")
    print(f"   Main index: {stats['vector_store']['main_index_vectors']} vectors")
    print(f"   Delta indices: {stats['vector_store']['delta_indices_count']}")
    
    # Add new documents
    new_docs = [
        {
            "content": "Pipeline parallelism in RAG systems allows concurrent execution of "
                      "retrieval and generation phases, significantly reducing latency.",
            "metadata": {"source": "pipeline_parallel.txt", "category": "optimization"}
        },
        {
            "content": "Memory-mapped files enable efficient access to large vector databases "
                      "by treating disk storage as virtual memory, reducing I/O overhead.",
            "metadata": {"source": "mmap_vectors.txt", "category": "technology"}
        }
    ]
    
    print("\n➕ Adding 2 new documents...")
    add_result = pipeline.add_documents(
        [doc["content"] for doc in new_docs],
        [doc["metadata"] for doc in new_docs]
    )
    
    print(f"   Added in {add_result['processing_time']:.3f}s")
    
    # Check updated statistics
    stats = pipeline.get_statistics()
    print(f"\n📊 Updated index statistics:")
    print(f"   Total vectors: {stats['vector_store']['total_vectors']}")
    print(f"   Delta indices: {stats['vector_store']['delta_indices_count']}")
    
    # Query with new content
    query = "How does pipeline parallelism improve RAG performance?"
    result = pipeline.query(query, top_k=3)
    
    print(f"\n📝 Query with new content: {query}")
    print(f"   Found relevant document: {'pipeline_parallel' in str(result['retrieved_docs'])}")


def demonstrate_batch_processing(pipeline: EnhancedRAGPipeline) -> None:
    """Demonstrate batch processing capabilities."""
    print("\n" + "="*80)
    print("🔹 BATCH PROCESSING DEMONSTRATION")
    print("="*80)
    
    queries = generate_test_queries()[:5]
    
    print(f"\n⚡ Processing {len(queries)} queries in batch mode...")
    
    start_time = time.time()
    batch_results = pipeline.query_batch(queries, top_k=3)
    batch_time = time.time() - start_time
    
    print(f"\n✅ Batch processing completed:")
    print(f"   Total time: {batch_time:.3f}s")
    print(f"   Average per query: {batch_time/len(queries):.3f}s")
    print(f"   Throughput: {len(queries)/batch_time:.2f} queries/second")
    
    # Compare with sequential processing
    print("\n🔄 Comparing with sequential processing...")
    
    start_time = time.time()
    for query in queries:
        pipeline.query(query, top_k=3)
    seq_time = time.time() - start_time
    
    print(f"   Sequential time: {seq_time:.3f}s")
    print(f"   Speedup: {seq_time/batch_time:.2f}x")


def demonstrate_csd_emulation(pipeline: EnhancedRAGPipeline) -> None:
    """Demonstrate CSD emulation features."""
    print("\n" + "="*80)
    print("🔹 CSD EMULATION DEMONSTRATION")
    print("="*80)
    
    if not pipeline.config.enable_csd_emulation:
        print("   ⚠️  CSD emulation is disabled in current configuration")
        return
    
    # Get CSD metrics
    stats = pipeline.get_statistics()
    if 'csd_metrics' in stats:
        metrics = stats['csd_metrics']
        print(f"\n📊 CSD Emulator Metrics:")
        print(f"   Cache hit rate: {metrics['cache_hit_rate']*100:.1f}%")
        print(f"   Average latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"   Storage usage: {metrics['storage_usage_mb']:.1f}MB")
        print(f"   Read operations: {metrics['read_ops']}")
        print(f"   Write operations: {metrics['write_ops']}")
    
    # Demonstrate cache warming
    print("\n🔥 Demonstrating cache warming effect...")
    
    query = "What are the benefits of computational storage?"
    
    # Cold query
    start = time.time()
    result1 = pipeline.query(query, top_k=3)
    cold_time = time.time() - start
    
    # Warm query (should be faster)
    start = time.time()
    result2 = pipeline.query(query, top_k=3)
    warm_time = time.time() - start
    
    print(f"   Cold query time: {cold_time:.3f}s")
    print(f"   Warm query time: {warm_time:.3f}s")
    print(f"   Cache speedup: {cold_time/warm_time:.2f}x")


def run_performance_comparison(vector_db_path: str, output_dir: str) -> None:
    """Run performance comparison between systems."""
    print("\n" + "="*80)
    print("🔹 PERFORMANCE COMPARISON")
    print("="*80)
    
    config = {"embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
    
    # Initialize benchmark runner
    print("\n🔧 Initializing benchmark systems...")
    benchmark = EnhancedBenchmarkRunner(vector_db_path, config)
    
    # Generate test queries
    queries = generate_test_queries()
    
    # Run latency benchmark
    print("\n⏱️  Running latency benchmarks...")
    latency_results = benchmark.run_latency_benchmark(
        queries[:5],  # Use subset for demo
        top_k=5,
        runs_per_query=3,
        warmup_runs=1
    )
    
    # Run throughput benchmark
    print("\n📈 Running throughput benchmarks...")
    throughput_results = benchmark.run_throughput_benchmark(
        queries,
        top_k=5,
        batch_sizes=[1, 4, 8]
    )
    
    # Generate report
    print("\n📊 Generating benchmark report...")
    benchmark.generate_report(latency_results, throughput_results, output_dir)
    
    print(f"\n✅ Benchmark complete! Results saved to: {output_dir}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Enhanced RAG-CSD System Demonstration")
    parser.add_argument("--vector-db", type=str, default="./demo_vectors",
                       help="Path to vector database")
    parser.add_argument("--output-dir", type=str, default="./demo_results",
                       help="Output directory for results")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip performance benchmarking")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level=args.log_level)
    
    print("🚀 Enhanced RAG-CSD System Demonstration")
    print("="*80)
    
    # Initialize pipeline with all features enabled
    config = PipelineConfig(
        vector_db_path=args.vector_db,
        storage_path="./enhanced_storage",
        enable_csd_emulation=True,
        enable_pipeline_parallel=True,
        enable_caching=True,
        delta_threshold=5,  # Small threshold for demo
        max_delta_indices=3
    )
    
    print("\n🔧 Initializing Enhanced RAG-CSD Pipeline...")
    pipeline = EnhancedRAGPipeline(config)
    
    # Add initial documents
    print("\n📚 Adding initial documents...")
    demo_docs = generate_demo_documents()
    result = pipeline.add_documents(
        [doc["content"] for doc in demo_docs],
        [doc["metadata"] for doc in demo_docs]
    )
    print(f"   Added {result['documents_processed']} documents "
          f"({result['chunks_created']} chunks) in {result['processing_time']:.3f}s")
    
    # Run demonstrations
    demonstrate_basic_usage(pipeline)
    demonstrate_incremental_indexing(pipeline)
    demonstrate_batch_processing(pipeline)
    demonstrate_csd_emulation(pipeline)
    
    # Performance comparison
    if not args.skip_benchmark:
        # Create a proper vector database for benchmarking
        os.makedirs(args.vector_db, exist_ok=True)
        
        # Save minimal vector data for benchmarking
        import numpy as np
        embeddings = np.random.randn(100, 384).astype(np.float32)
        np.save(os.path.join(args.vector_db, "embeddings.npy"), embeddings)
        
        chunks = [f"Document chunk {i}" for i in range(100)]
        metadata = [{"doc_id": i, "chunk_id": 0} for i in range(100)]
        
        with open(os.path.join(args.vector_db, "chunks.json"), "w") as f:
            json.dump(chunks, f)
        
        with open(os.path.join(args.vector_db, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        run_performance_comparison(args.vector_db, args.output_dir)
    
    # Final statistics
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)
    
    stats = pipeline.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Cleanup
    pipeline.shutdown()
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()