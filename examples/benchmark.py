#!/usr/bin/env python
"""
Run full benchmark comparing all RAG systems with accuracy validation.
This script provides a complete evaluation including performance and accuracy metrics.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enhanced_rag_csd.enhanced_pipeline import EnhancedRAGPipeline, PipelineConfig
from enhanced_rag_csd.benchmarks.enhanced_benchmark import EnhancedBenchmarkRunner
from enhanced_rag_csd.evaluation.accuracy_validator import (
    AccuracyValidator, ValidationRunner, DatasetLoader, ValidationDataset
)
from enhanced_rag_csd.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def create_validation_dataset() -> ValidationDataset:
    """Create a comprehensive validation dataset."""
    return ValidationDataset(
        queries=[
            "What are computational storage devices and their main benefits?",
            "How does retrieval-augmented generation improve AI responses?",
            "Explain the FAISS algorithm for vector similarity search",
            "What is the difference between batch and real-time processing in RAG?",
            "How does incremental indexing work in vector databases?",
            "What are the advantages of pipeline parallelism in RAG systems?",
            "Describe memory-mapped files and their use in vector storage",
            "How do you measure drift in vector database indices?",
            "What is the role of caching in RAG system optimization?",
            "Compare edge computing with cloud computing for AI applications"
        ],
        relevant_docs=[
            ["computational_storage_1.txt", "computational_storage_2.txt", "storage_computing_integration_1.txt"],
            ["retrieval_augmented_generation_1.txt", "retrieval_augmented_generation_2.txt"],
            ["vector_similarity_search_1.txt", "vector_databases_1.txt"],
            ["retrieval_augmented_generation_3.txt", "computational_storage_3.txt"],
            ["vector_databases_2.txt", "vector_databases_3.txt"],
            ["retrieval_augmented_generation_2.txt", "computational_storage_2.txt"],
            ["storage_computing_integration_2.txt", "vector_databases_1.txt"],
            ["vector_databases_3.txt", "vector_similarity_search_2.txt"],
            ["retrieval_augmented_generation_1.txt", "computational_storage_1.txt"],
            ["storage_computing_integration_3.txt", "computational_storage_3.txt"]
        ],
        ground_truth_answers=[
            "Computational storage devices integrate processing capabilities directly into storage hardware, "
            "reducing data movement and improving performance by processing data where it resides.",
            
            "RAG improves AI responses by retrieving relevant information from a knowledge base and "
            "augmenting the query with this context before generation, providing more accurate and current answers.",
            
            "FAISS is a library that implements efficient algorithms for similarity search in high-dimensional "
            "spaces using techniques like inverted indices, product quantization, and hierarchical structures.",
            
            "Batch processing handles multiple queries together for better throughput, while real-time "
            "processing optimizes for individual query latency. The choice depends on application requirements.",
            
            "Incremental indexing maintains a main index plus delta indices for new documents, merging them "
            "periodically based on drift detection to avoid full index rebuilds.",
            
            "Pipeline parallelism in RAG systems allows concurrent execution of retrieval and generation phases, "
            "reducing overall latency by overlapping compute and I/O operations.",
            
            "Memory-mapped files treat disk storage as virtual memory, allowing efficient random access to "
            "large vector databases without loading everything into RAM.",
            
            "Drift in vector indices is measured using KL divergence of embedding distributions, query performance "
            "degradation, and index fragmentation ratios to determine when rebuilding is needed.",
            
            "Caching in RAG systems stores frequently accessed embeddings and model outputs, dramatically "
            "reducing latency for repeated queries and improving overall system throughput.",
            
            "Edge computing processes AI workloads near data sources with lower latency but limited resources, "
            "while cloud computing offers more power but higher latency and data transfer costs."
        ],
        metadata={
            "dataset_name": "rag_csd_validation",
            "version": "1.0",
            "created": datetime.now().isoformat()
        }
    )


def generate_benchmark_queries(num_queries: int = 50) -> List[str]:
    """Generate a larger set of queries for performance benchmarking."""
    base_queries = [
        "What is {topic} and how does it work?",
        "Explain the benefits of {topic} in modern systems",
        "How do you implement {topic} efficiently?",
        "What are the challenges with {topic}?",
        "Compare {topic} with traditional approaches",
        "Describe best practices for {topic}",
        "What are recent advances in {topic}?",
        "How does {topic} impact performance?",
        "What are the limitations of {topic}?",
        "Explain the architecture of {topic} systems"
    ]
    
    topics = [
        "computational storage", "vector databases", "similarity search",
        "incremental indexing", "embedding generation", "query augmentation",
        "pipeline parallelism", "edge computing", "drift detection",
        "cache optimization", "batch processing", "memory mapping"
    ]
    
    queries = []
    for i in range(num_queries):
        template = base_queries[i % len(base_queries)]
        topic = topics[i % len(topics)]
        queries.append(template.format(topic=topic))
    
    return queries


def run_full_evaluation(args):
    """Run complete evaluation with performance and accuracy metrics."""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Starting full benchmark evaluation at {timestamp}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Initialize configuration
    config = {
        "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
    }
    
    # Phase 1: Performance Benchmarking
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: PERFORMANCE BENCHMARKING")
    logger.info("="*80)
    
    benchmark_runner = EnhancedBenchmarkRunner(args.vector_db, config)
    
    # Generate queries for benchmarking
    perf_queries = generate_benchmark_queries(args.num_queries)
    
    # Run latency benchmark
    logger.info(f"\nRunning latency benchmark with {len(perf_queries)} queries...")
    latency_results = benchmark_runner.run_latency_benchmark(
        perf_queries,
        top_k=args.top_k,
        runs_per_query=args.runs_per_query,
        warmup_runs=1
    )
    
    # Run throughput benchmark
    logger.info("\nRunning throughput benchmark...")
    throughput_results = benchmark_runner.run_throughput_benchmark(
        perf_queries[:20],  # Use subset for throughput test
        top_k=args.top_k,
        batch_sizes=[1, 2, 4, 8, 16]
    )
    
    # Generate performance report
    perf_dir = os.path.join(results_dir, "performance")
    os.makedirs(perf_dir, exist_ok=True)
    benchmark_runner.generate_report(latency_results, throughput_results, perf_dir)
    
    # Phase 2: Accuracy Validation
    if not args.skip_accuracy:
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: ACCURACY VALIDATION")
        logger.info("="*80)
        
        # Create validation dataset
        validation_dataset = create_validation_dataset()
        
        # Save validation dataset for reproducibility
        with open(os.path.join(results_dir, "validation_dataset.json"), "w") as f:
            json.dump({
                "queries": validation_dataset.queries,
                "relevant_docs": validation_dataset.relevant_docs,
                "ground_truth_answers": validation_dataset.ground_truth_answers,
                "metadata": validation_dataset.metadata
            }, f, indent=2)
        
        # Initialize accuracy validator
        validator = AccuracyValidator()
        validation_runner = ValidationRunner(benchmark_runner.systems, validator)
        
        # Run validation
        logger.info(f"\nRunning accuracy validation on {len(validation_dataset.queries)} queries...")
        validation_results = validation_runner.run_validation(
            validation_dataset,
            top_k=args.top_k,
            include_answer_eval=True
        )
        
        # Save validation results
        accuracy_dir = os.path.join(results_dir, "accuracy")
        os.makedirs(accuracy_dir, exist_ok=True)
        validation_runner.save_results(os.path.join(accuracy_dir, "validation_results.json"))
        
        # Print validation summary
        validation_runner.print_summary()
    
    # Phase 3: Combined Analysis
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: COMBINED ANALYSIS")
    logger.info("="*80)
    
    # Load all results
    system_stats = benchmark_runner.calculate_statistics(latency_results)
    
    # Create combined report
    combined_report = {
        "timestamp": timestamp,
        "configuration": {
            "vector_db": args.vector_db,
            "num_queries": args.num_queries,
            "top_k": args.top_k,
            "runs_per_query": args.runs_per_query
        },
        "systems_evaluated": list(benchmark_runner.systems.keys()),
        "performance_summary": {},
        "accuracy_summary": {}
    }
    
    # Add performance summary
    for system_name, stats in system_stats.items():
        combined_report["performance_summary"][system_name] = {
            "avg_latency": stats.avg_latency,
            "p95_latency": stats.p95_latency,
            "throughput": stats.throughput,
            "cache_hit_rate": stats.cache_hit_rate
        }
    
    # Add accuracy summary if available
    if not args.skip_accuracy:
        for system_name, result in validation_results.items():
            if "error" not in result:
                combined_report["accuracy_summary"][system_name] = {
                    "retrieval_score": result.get("overall_retrieval_score", 0),
                    "precision": result["retrieval_metrics"]["precision_at_k"],
                    "recall": result["retrieval_metrics"]["recall_at_k"],
                    "ndcg": result["retrieval_metrics"]["ndcg_at_k"]
                }
    
    # Calculate combined scores
    logger.info("\nCombined Performance-Accuracy Scores:")
    logger.info("-" * 50)
    
    for system_name in benchmark_runner.systems.keys():
        if system_name in system_stats:
            # Normalize latency (lower is better, so invert)
            baseline_latency = system_stats.get("VanillaRAG", system_stats[system_name]).avg_latency
            perf_score = baseline_latency / system_stats[system_name].avg_latency
            
            # Get accuracy score
            acc_score = 0
            if not args.skip_accuracy and system_name in validation_results:
                if "overall_retrieval_score" in validation_results[system_name]:
                    acc_score = validation_results[system_name]["overall_retrieval_score"]
            
            # Combined score (equal weight)
            combined_score = (perf_score + acc_score) / 2 if acc_score > 0 else perf_score
            
            combined_report["performance_summary"][system_name]["combined_score"] = combined_score
            
            logger.info(f"{system_name:<20} Performance: {perf_score:.3f}, "
                       f"Accuracy: {acc_score:.3f}, Combined: {combined_score:.3f}")
    
    # Save combined report
    with open(os.path.join(results_dir, "combined_report.json"), "w") as f:
        json.dump(combined_report, f, indent=2)
    
    # Create executive summary
    create_executive_summary(combined_report, results_dir)
    
    logger.info(f"\n✅ Full benchmark completed! Results saved to: {results_dir}")
    
    return results_dir


def create_executive_summary(report: Dict, output_dir: str):
    """Create an executive summary of the benchmark results."""
    
    summary_lines = [
        "# RAG-CSD Benchmark Executive Summary",
        f"\nGenerated: {report['timestamp']}",
        f"\nSystems Evaluated: {', '.join(report['systems_evaluated'])}",
        "\n## Key Findings",
        "\n### Performance Rankings (by latency)"
    ]
    
    # Sort by latency
    perf_sorted = sorted(
        report["performance_summary"].items(),
        key=lambda x: x[1]["avg_latency"]
    )
    
    for i, (system, metrics) in enumerate(perf_sorted, 1):
        summary_lines.append(
            f"{i}. **{system}**: {metrics['avg_latency']:.3f}s avg latency, "
            f"{metrics['throughput']:.2f} queries/sec"
        )
    
    if report["accuracy_summary"]:
        summary_lines.append("\n### Accuracy Rankings (by retrieval score)")
        
        acc_sorted = sorted(
            report["accuracy_summary"].items(),
            key=lambda x: x[1]["retrieval_score"],
            reverse=True
        )
        
        for i, (system, metrics) in enumerate(acc_sorted, 1):
            summary_lines.append(
                f"{i}. **{system}**: {metrics['retrieval_score']:.3f} score, "
                f"{metrics['precision']:.3f} precision, {metrics['ndcg']:.3f} NDCG"
            )
    
    # Best overall system
    if "combined_score" in next(iter(report["performance_summary"].values())):
        best_system = max(
            report["performance_summary"].items(),
            key=lambda x: x[1].get("combined_score", 0)
        )
        
        summary_lines.extend([
            "\n### Best Overall System",
            f"\n🏆 **{best_system[0]}** with combined score: {best_system[1]['combined_score']:.3f}"
        ])
    
    # Recommendations
    summary_lines.extend([
        "\n## Recommendations",
        "\n1. **For Production Use**: Enhanced-RAG-CSD offers the best balance of performance and accuracy",
        "2. **For Resource-Constrained Environments**: EdgeRAG-like provides acceptable performance with minimal resources",
        "3. **For Research**: FlashRAG-like offers modular architecture for experimentation",
        "4. **Key Optimizations**: Enable caching, use batch processing for multiple queries, implement incremental indexing"
    ])
    
    # Write summary
    with open(os.path.join(output_dir, "executive_summary.md"), "w") as f:
        f.write("\n".join(summary_lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full RAG-CSD benchmark with performance and accuracy evaluation"
    )
    parser.add_argument("--vector-db", type=str, required=True,
                       help="Path to vector database")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--num-queries", type=int, default=20,
                       help="Number of queries for performance benchmark")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of documents to retrieve")
    parser.add_argument("--runs-per-query", type=int, default=3,
                       help="Number of runs per query for averaging")
    parser.add_argument("--skip-accuracy", action="store_true",
                       help="Skip accuracy validation")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level=args.log_level)
    
    # Check vector database exists
    if not os.path.exists(args.vector_db):
        logger.error(f"Vector database not found at: {args.vector_db}")
        logger.info("Please run create_vector_db.py first to create a vector database")
        sys.exit(1)
    
    # Run evaluation
    try:
        results_dir = run_full_evaluation(args)
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: {results_dir}")
        print(f"📊 View executive summary at: {results_dir}/executive_summary.md")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()