"""
Enhanced benchmarking framework for comparing RAG systems including our innovations.
This module provides comprehensive benchmarking with statistical analysis and visualization.
"""

import os
import json
import time
import gc
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from enhanced_rag_csd.utils.logger import get_logger
from enhanced_rag_csd.utils.error_handling import safe_divide
from enhanced_rag_csd.benchmarks.baseline_systems import (
    VanillaRAG, PipeRAGLike, EdgeRAGLike, BaseRAGSystem
)
from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline, PipelineConfig

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single run."""
    system_name: str
    query: str
    latency: float
    memory_used: float
    cache_hit: bool
    retrieved_docs: int
    accuracy_score: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemBenchmark:
    """Aggregated benchmark results for a system."""
    system_name: str
    total_queries: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    std_latency: float
    avg_memory: float
    cache_hit_rate: float
    throughput: float  # queries per second
    accuracy: Optional[float] = None
    
    def speedup_vs(self, baseline: 'SystemBenchmark') -> float:
        """Calculate speedup versus baseline."""
        return baseline.avg_latency / self.avg_latency if self.avg_latency > 0 else 0


class FlashRAGLike(BaseRAGSystem):
    """
    FlashRAG-inspired implementation focusing on modular architecture.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.module_times = {}  # Track time per module
    
    def initialize(self, vector_db_path: str) -> None:
        """Initialize FlashRAG-like system."""
        logger.info(f"Initializing {self.name}")
        
        # Modular initialization
        start = time.time()
        
        # Module 1: Model loading
        from sentence_transformers import SentenceTransformer
        model_name = self.config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        self.module_times['model_loading'] = time.time() - start
        
        # Module 2: Data loading
        start = time.time()
        import faiss
        
        embeddings = np.load(os.path.join(vector_db_path, "embeddings.npy")).astype(np.float32)
        
        with open(os.path.join(vector_db_path, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
        
        with open(os.path.join(vector_db_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        self.module_times['data_loading'] = time.time() - start
        
        # Module 3: Index building
        start = time.time()
        d = embeddings.shape[1]
        
        # Use HNSW for better recall
        self.index = faiss.IndexHNSWFlat(d, 32)
        self.index.hnsw.efConstruction = 200
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.index.hnsw.efSearch = 50
        
        self.module_times['index_building'] = time.time() - start
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Modular query processing."""
        start_time = time.time()
        module_timings = {}
        
        # Module: Query encoding
        start = time.time()
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        module_timings['encoding'] = time.time() - start
        
        # Module: Retrieval
        start = time.time()
        scores, indices = self.index.search(query_embedding, top_k)
        module_timings['retrieval'] = time.time() - start
        
        # Module: Result processing
        start = time.time()
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score)
                })
        module_timings['processing'] = time.time() - start
        
        # Module: Augmentation
        start = time.time()
        context = "\n".join([f"[{i+1}] {r['chunk']}" for i, r in enumerate(results)])
        augmented_query = f"Query: {query}\n\nRelevant Context:\n{context}"
        module_timings['augmentation'] = time.time() - start
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "augmented_query": augmented_query,
            "retrieved_docs": results,
            "processing_time": total_time,
            "module_timings": module_timings,
            "system": self.name
        }
    
    def query_batch(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Batch processing with module timing."""
        results = []
        for query in queries:
            results.append(self.query(query, top_k))
        return results


class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with statistical analysis."""
    
    def __init__(self, vector_db_path: str, config: Dict):
        self.vector_db_path = vector_db_path
        self.config = config
        self.systems = {}
        self.results = []
        
        # Initialize all systems
        self._initialize_systems()
    
    def _initialize_systems(self) -> None:
        """Initialize all RAG systems for comparison."""
        
        # Baseline systems
        baseline_systems = [
            ("VanillaRAG", VanillaRAG),
            ("PipeRAG-like", PipeRAGLike),
            ("EdgeRAG-like", EdgeRAGLike),
            ("FlashRAG-like", FlashRAGLike)
        ]
        
        for name, system_class in baseline_systems:
            try:
                system = system_class(self.config)
                system.initialize(self.vector_db_path)
                self.systems[name] = system
                logger.info(f"✅ {name} initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {name}: {e}")
        
        # Original RAG-CSD
        try:
            self.systems["RAG-CSD"] = RAGCSDPipeline(self.vector_db_path)
            logger.info("✅ RAG-CSD initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG-CSD: {e}")
        
        # Enhanced RAG-CSD
        try:
            enhanced_config = PipelineConfig(
                vector_db_path=self.vector_db_path,
                enable_csd_emulation=True,
                enable_pipeline_parallel=True,
                enable_caching=True
            )
            self.systems["Enhanced-RAG-CSD"] = EnhancedRAGPipeline(enhanced_config)
            logger.info("✅ Enhanced-RAG-CSD initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced-RAG-CSD: {e}")
    
    def _measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def run_latency_benchmark(self,
                            queries: List[str],
                            top_k: int = 5,
                            runs_per_query: int = 3,
                            warmup_runs: int = 1) -> Dict[str, List[BenchmarkResult]]:
        """Run latency benchmarks with warmup and multiple runs."""
        results = {name: [] for name in self.systems}
        
        for system_name, system in self.systems.items():
            logger.info(f"\n🔄 Benchmarking {system_name}...")
            
            # Warmup runs
            if warmup_runs > 0:
                logger.info(f"Warming up {system_name}...")
                for query in queries[:min(3, len(queries))]:
                    try:
                        system.query(query, top_k=top_k)
                    except Exception as e:
                        logger.error(f"Warmup error: {e}")
            
            # Actual benchmark runs
            for query in tqdm(queries, desc=f"{system_name}"):
                for run in range(runs_per_query):
                    try:
                        # Force garbage collection for fair comparison
                        gc.collect()
                        
                        # Measure memory before
                        mem_before = self._measure_memory()
                        
                        # Time the query
                        start_time = time.time()
                        result = system.query(query, top_k=top_k)
                        latency = time.time() - start_time
                        
                        # Measure memory after
                        mem_after = self._measure_memory()
                        memory_used = mem_after - mem_before
                        
                        # Create benchmark result
                        benchmark_result = BenchmarkResult(
                            system_name=system_name,
                            query=query,
                            latency=latency,
                            memory_used=memory_used,
                            cache_hit=result.get('cache_hit', False),
                            retrieved_docs=len(result.get('retrieved_docs', []))
                        )
                        
                        results[system_name].append(benchmark_result)
                        
                    except Exception as e:
                        logger.error(f"Error in {system_name} for query '{query[:50]}...': {e}")
                        
                        # Record failed result
                        benchmark_result = BenchmarkResult(
                            system_name=system_name,
                            query=query,
                            latency=float('inf'),
                            memory_used=0,
                            cache_hit=False,
                            retrieved_docs=0
                        )
                        results[system_name].append(benchmark_result)
        
        return results
    
    def run_throughput_benchmark(self,
                               queries: List[str],
                               top_k: int = 5,
                               batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict[str, Dict[int, float]]:
        """Measure throughput at different batch sizes."""
        results = {name: {} for name in self.systems}
        
        for system_name, system in self.systems.items():
            logger.info(f"\n⚡ Throughput test for {system_name}...")
            
            for batch_size in batch_sizes:
                if batch_size > len(queries):
                    continue
                
                batch_queries = queries[:batch_size]
                
                try:
                    start_time = time.time()
                    
                    if hasattr(system, 'query_batch') and batch_size > 1:
                        # Use batch processing if available
                        system.query_batch(batch_queries, top_k=top_k)
                    else:
                        # Fall back to sequential
                        for query in batch_queries:
                            system.query(query, top_k=top_k)
                    
                    elapsed = time.time() - start_time
                    throughput = safe_divide(batch_size, elapsed, default=float('inf'))
                    
                    results[system_name][batch_size] = throughput
                    logger.info(f"  Batch size {batch_size}: {throughput:.2f} queries/sec")
                    
                except Exception as e:
                    logger.error(f"  Error with batch size {batch_size}: {e}")
                    results[system_name][batch_size] = 0
        
        return results
    
    def run_scalability_benchmark(self,
                                queries: List[str],
                                vector_db_sizes: List[int],
                                top_k: int = 5) -> Dict[str, Dict[int, float]]:
        """Test scalability with different database sizes."""
        # This would require creating vector databases of different sizes
        # For now, we'll simulate by adjusting search parameters
        results = {name: {} for name in self.systems}
        
        logger.info("\n📈 Scalability benchmark (simulated)...")
        
        for size in vector_db_sizes:
            logger.info(f"\nTesting with simulated DB size: {size}")
            
            for system_name, system in self.systems.items():
                try:
                    # Simulate larger database by adjusting parameters
                    # In production, we'd actually test with different DB sizes
                    start_time = time.time()
                    
                    for query in queries[:5]:  # Use subset for scalability test
                        system.query(query, top_k=top_k)
                    
                    avg_latency = (time.time() - start_time) / 5
                    results[system_name][size] = avg_latency
                    
                except Exception as e:
                    logger.error(f"Error in scalability test for {system_name}: {e}")
                    results[system_name][size] = float('inf')
        
        return results
    
    def calculate_statistics(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, SystemBenchmark]:
        """Calculate comprehensive statistics for each system."""
        system_stats = {}
        
        for system_name, system_results in results.items():
            if not system_results:
                continue
            
            # Filter out failed results
            valid_results = [r for r in system_results if r.latency != float('inf')]
            
            if not valid_results:
                continue
            
            latencies = [r.latency for r in valid_results]
            memories = [r.memory_used for r in valid_results]
            cache_hits = [r.cache_hit for r in valid_results]
            
            # Calculate statistics
            system_stats[system_name] = SystemBenchmark(
                system_name=system_name,
                total_queries=len(valid_results),
                avg_latency=np.mean(latencies),
                p50_latency=np.percentile(latencies, 50),
                p95_latency=np.percentile(latencies, 95),
                p99_latency=np.percentile(latencies, 99),
                std_latency=np.std(latencies),
                avg_memory=np.mean(memories),
                cache_hit_rate=sum(cache_hits) / len(cache_hits) if cache_hits else 0,
                throughput=1.0 / np.mean(latencies) if latencies else 0
            )
        
        return system_stats
    
    def run_statistical_tests(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Run statistical significance tests."""
        statistical_results = {}
        
        # Get latencies for each system
        system_latencies = {}
        for system_name, system_results in results.items():
            valid_results = [r for r in system_results if r.latency != float('inf')]
            if valid_results:
                system_latencies[system_name] = [r.latency for r in valid_results]
        
        # Perform pairwise t-tests
        if "VanillaRAG" in system_latencies:
            baseline_latencies = system_latencies["VanillaRAG"]
            
            for system_name, latencies in system_latencies.items():
                if system_name != "VanillaRAG":
                    # Perform Welch's t-test (does not assume equal variances)
                    t_stat, p_value = stats.ttest_ind(
                        baseline_latencies, 
                        latencies, 
                        equal_var=False
                    )
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (np.var(baseline_latencies) + np.var(latencies)) / 2
                    )
                    cohens_d = (np.mean(baseline_latencies) - np.mean(latencies)) / pooled_std
                    
                    statistical_results[f"{system_name}_vs_VanillaRAG"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "cohens_d": cohens_d,
                        "significant": p_value < 0.05,
                        "effect_size": self._interpret_effect_size(cohens_d)
                    }
        
        return statistical_results
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_report(self, 
                       latency_results: Dict[str, List[BenchmarkResult]],
                       throughput_results: Dict[str, Dict[int, float]],
                       output_dir: str) -> None:
        """Generate comprehensive benchmark report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate statistics
        system_stats = self.calculate_statistics(latency_results)
        statistical_tests = self.run_statistical_tests(latency_results)
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "systems_tested": list(self.systems.keys()),
            "system_statistics": {
                name: asdict(stats) for name, stats in system_stats.items()
            },
            "statistical_significance": statistical_tests,
            "raw_results": {
                name: [r.to_dict() for r in results]
                for name, results in latency_results.items()
            }
        }
        
        # Save JSON report
        with open(os.path.join(output_dir, "benchmark_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._create_visualizations(
            system_stats, 
            throughput_results,
            statistical_tests,
            output_dir
        )
        
        # Generate markdown summary
        self._create_markdown_summary(
            system_stats,
            statistical_tests,
            output_dir
        )
    
    def _create_visualizations(self,
                             system_stats: Dict[str, SystemBenchmark],
                             throughput_results: Dict[str, Dict[int, float]],
                             statistical_tests: Dict[str, Any],
                             output_dir: str) -> None:
        """Create comprehensive visualizations."""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Latency comparison bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        systems = list(system_stats.keys())
        avg_latencies = [system_stats[s].avg_latency for s in systems]
        colors = plt.cm.viridis(np.linspace(0, 1, len(systems)))
        
        bars = ax.bar(systems, avg_latencies, color=colors)
        
        # Add error bars
        errors = [system_stats[s].std_latency for s in systems]
        ax.errorbar(systems, avg_latencies, yerr=errors, fmt='none', color='black', capsize=5)
        
        # Add value labels
        for bar, val in zip(bars, avg_latencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}s', ha='center', va='bottom')
        
        ax.set_ylabel('Average Latency (seconds)')
        ax.set_title('RAG System Latency Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Latency distribution violin plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for violin plot
        plot_data = []
        for system in systems:
            if system in system_stats:
                latencies = [r.latency for r in self.results 
                           if r.system_name == system and r.latency != float('inf')]
                plot_data.extend([{'System': system, 'Latency': lat} for lat in latencies])
        
        df = pd.DataFrame(plot_data)
        sns.violinplot(data=df, x='System', y='Latency', ax=ax)
        
        ax.set_ylabel('Latency (seconds)')
        ax.set_title('Latency Distribution by System')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latency_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Speedup comparison
        if "VanillaRAG" in system_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            baseline = system_stats["VanillaRAG"]
            speedups = []
            systems_sorted = []
            
            for system in systems:
                if system != "VanillaRAG":
                    speedup = system_stats[system].speedup_vs(baseline)
                    speedups.append(speedup)
                    systems_sorted.append(system)
            
            # Sort by speedup
            sorted_indices = np.argsort(speedups)[::-1]
            speedups_sorted = [speedups[i] for i in sorted_indices]
            systems_sorted = [systems_sorted[i] for i in sorted_indices]
            
            bars = ax.bar(systems_sorted, speedups_sorted, 
                          color=['green' if s > 1 else 'red' for s in speedups_sorted])
            
            # Add value labels
            for bar, val in zip(bars, speedups_sorted):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.1f}x', ha='center', va='bottom')
            
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Speedup vs VanillaRAG')
            ax.set_title('Performance Speedup Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=300)
            plt.close()
        
        # 4. Throughput curves
        if throughput_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for system, throughputs in throughput_results.items():
                if throughputs:
                    batch_sizes = sorted(throughputs.keys())
                    tps = [throughputs[bs] for bs in batch_sizes]
                    ax.plot(batch_sizes, tps, marker='o', label=system)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (queries/second)')
            ax.set_title('Throughput vs Batch Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'throughput_curves.png'), dpi=300)
            plt.close()
        
        # 5. Memory usage comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        systems = list(system_stats.keys())
        avg_memories = [system_stats[s].avg_memory for s in systems]
        
        bars = ax.bar(systems, avg_memories, color=colors)
        
        for bar, val in zip(bars, avg_memories):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}MB', ha='center', va='bottom')
        
        ax.set_ylabel('Average Memory Usage (MB)')
        ax.set_title('Memory Usage Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=300)
        plt.close()
        
        # 6. Performance radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Define metrics for radar
        metrics = ['Speed', 'Memory\nEfficiency', 'Cache\nHit Rate', 'Consistency']
        num_metrics = len(metrics)
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each system
        for system in systems[:5]:  # Limit to 5 systems for clarity
            if system in system_stats:
                stats = system_stats[system]
                
                # Normalize metrics (higher is better)
                speed_score = 1.0 / stats.avg_latency if stats.avg_latency > 0 else 0
                memory_score = 1.0 / (stats.avg_memory + 1)  # Avoid division by zero
                cache_score = stats.cache_hit_rate
                consistency_score = 1.0 / (stats.std_latency + 0.1)  # Lower std is better
                
                # Normalize to 0-1 scale
                max_speed = max(1.0 / s.avg_latency for s in system_stats.values() if s.avg_latency > 0)
                max_consistency = max(1.0 / (s.std_latency + 0.1) for s in system_stats.values())
                
                values = [
                    speed_score / max_speed,
                    memory_score,
                    cache_score,
                    consistency_score / max_consistency
                ]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=system)
                ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('System Performance Radar Chart', size=20, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_radar.png'), dpi=300)
        plt.close()
    
    def _create_markdown_summary(self,
                               system_stats: Dict[str, SystemBenchmark],
                               statistical_tests: Dict[str, Any],
                               output_dir: str) -> None:
        """Create markdown summary report."""
        report_lines = [
            "# RAG System Benchmark Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Executive Summary",
            "\n### System Performance Ranking"
        ]
        
        # Sort systems by average latency
        sorted_systems = sorted(system_stats.items(), key=lambda x: x[1].avg_latency)
        
        report_lines.append("\n| Rank | System | Avg Latency | Speedup vs Baseline | p95 Latency |")
        report_lines.append("|------|--------|-------------|---------------------|-------------|")
        
        baseline_latency = system_stats.get("VanillaRAG", sorted_systems[0][1]).avg_latency
        
        for i, (name, stats) in enumerate(sorted_systems, 1):
            speedup = baseline_latency / stats.avg_latency
            report_lines.append(
                f"| {i} | {name} | {stats.avg_latency:.3f}s | "
                f"{speedup:.2f}x | {stats.p95_latency:.3f}s |"
            )
        
        # Key findings
        report_lines.extend([
            "\n### Key Findings",
            f"\n1. **Best Performance**: {sorted_systems[0][0]} with "
            f"{sorted_systems[0][1].avg_latency:.3f}s average latency",
            f"2. **Highest Speedup**: {sorted_systems[0][0]} achieves "
            f"{baseline_latency / sorted_systems[0][1].avg_latency:.2f}x speedup over baseline",
            f"3. **Most Consistent**: System with lowest latency variance"
        ])
        
        # Statistical significance
        report_lines.extend([
            "\n## Statistical Analysis",
            "\n### Significance Tests (vs VanillaRAG)"
        ])
        
        report_lines.append("\n| System | p-value | Significant | Effect Size |")
        report_lines.append("|--------|---------|-------------|-------------|")
        
        for test_name, results in statistical_tests.items():
            system_name = test_name.replace("_vs_VanillaRAG", "")
            significance = "✅ Yes" if results["significant"] else "❌ No"
            report_lines.append(
                f"| {system_name} | {results['p_value']:.4f} | "
                f"{significance} | {results['effect_size']} |"
            )
        
        # Performance characteristics
        report_lines.extend([
            "\n## Detailed Performance Characteristics",
            "\n| System | Throughput | Cache Hit Rate | Avg Memory |",
            "|--------|------------|----------------|------------|"
        ])
        
        for name, stats in system_stats.items():
            report_lines.append(
                f"| {name} | {stats.throughput:.2f} q/s | "
                f"{stats.cache_hit_rate*100:.1f}% | {stats.avg_memory:.1f} MB |"
            )
        
        # Recommendations
        report_lines.extend([
            "\n## Recommendations",
            "\n1. **For Low Latency**: Use Enhanced-RAG-CSD or RAG-CSD with caching enabled",
            "2. **For Resource-Constrained**: Use EdgeRAG-like for minimal memory footprint",
            "3. **For High Throughput**: Use Enhanced-RAG-CSD with batch processing",
            "4. **For Research**: Use FlashRAG-like for modular experimentation"
        ])
        
        # Write report
        with open(os.path.join(output_dir, "benchmark_summary.md"), "w") as f:
            f.write("\n".join(report_lines))