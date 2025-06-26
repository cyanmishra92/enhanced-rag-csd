#!/usr/bin/env python3
"""
Comprehensive benchmark of all emulator backends with performance analysis and plotting.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced_rag_csd.backends import CSDBackendManager, CSDBackendType
from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)

class EmulatorBenchmark:
    """Comprehensive benchmark for all emulator backends."""
    
    def __init__(self, output_dir: str = "results/emulator_benchmark"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        self.manager = CSDBackendManager()
        self.results = {}
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark on all available backends."""
        
        print("🚀 Enhanced RAG-CSD Emulator Benchmark")
        print("=" * 60)
        
        # Test configuration
        config = {
            "vector_db_path": f"{self.output_dir}/test_data",
            "embedding": {"dimensions": 384},
            "opencsd": {
                "simulation_mode": True,
                "ebpf_program_dir": f"{self.output_dir}/ebpf"
            },
            "spdk_vfio": {
                "simulation_mode": True,
                "shared_memory_mb": 1024
            }
        }
        
        # Backends to test
        backends_to_test = [
            (CSDBackendType.ENHANCED_SIMULATOR, "Enhanced Simulator"),
            (CSDBackendType.MOCK_SPDK, "Mock SPDK"),
            (CSDBackendType.OPENCSD_EMULATOR, "OpenCSD Emulator"),
            (CSDBackendType.SPDK_VFIO_USER, "SPDK vfio-user"),
        ]
        
        for backend_type, name in backends_to_test:
            print(f"\n📊 Benchmarking {name}...")
            
            try:
                result = self._benchmark_backend(backend_type, config, name)
                if result:
                    self.results[backend_type.value] = result
                    print(f"  ✅ {name} benchmark completed")
                else:
                    print(f"  ❌ {name} benchmark failed")
                    
            except Exception as e:
                print(f"  ❌ {name} benchmark error: {e}")
        
        # Generate analysis and plots
        self._generate_analysis()
        self._generate_plots()
        self._save_results()
        
        return self.results
    
    def _benchmark_backend(self, backend_type: CSDBackendType, config: Dict[str, Any], name: str) -> Dict[str, Any]:
        """Benchmark a specific backend."""
        
        # Create backend (force simulation mode for next-gen backends)
        if backend_type in [CSDBackendType.OPENCSD_EMULATOR, CSDBackendType.SPDK_VFIO_USER]:
            # Create directly with simulation mode
            if backend_type == CSDBackendType.OPENCSD_EMULATOR:
                from src.enhanced_rag_csd.backends.opencsd_backend import OpenCSDBackend
                backend = OpenCSDBackend(config)
            else:
                from src.enhanced_rag_csd.backends.spdk_vfio_user_backend import SPDKVfioUserBackend
                backend = SPDKVfioUserBackend(config)
            
            if not backend.initialize():
                return None
        else:
            backend = self.manager.create_backend(backend_type, config)
            if not backend:
                return None
        
        try:
            # Test data
            num_embeddings = 100
            query_iterations = 50
            embeddings = np.random.randn(num_embeddings, 384).astype(np.float32)
            metadata = [{"id": i, "text": f"document_{i}"} for i in range(num_embeddings)]
            
            # 1. Store benchmark
            store_start = time.time()
            backend.store_embeddings(embeddings, metadata)
            store_time = time.time() - store_start
            
            # 2. Retrieve benchmark
            retrieve_times = []
            for _ in range(20):
                indices = np.random.choice(num_embeddings, 10, replace=False).tolist()
                start = time.time()
                retrieved = backend.retrieve_embeddings(indices)
                retrieve_times.append(time.time() - start)
            
            avg_retrieve_time = np.mean(retrieve_times)
            
            # 3. Similarity benchmark
            similarity_times = []
            total_start_time = time.time()
            for i in range(query_iterations):
                query = embeddings[i % num_embeddings]
                candidates = np.random.choice(num_embeddings, 20, replace=False).tolist()
                
                start = time.time()
                similarities = backend.compute_similarities(query, candidates)
                elapsed = time.time() - start
                if elapsed > 0:  # Prevent negative or zero timing
                    similarity_times.append(elapsed)
            
            total_elapsed_time = time.time() - total_start_time
            avg_similarity_time = np.mean(similarity_times) if similarity_times else 0.001
            # Fix throughput calculation: use actual wall-clock time
            throughput = query_iterations / total_elapsed_time if total_elapsed_time > 0 else 0
            
            # 4. ERA Pipeline benchmark
            era_times = []
            for i in range(10):
                start = time.time()
                era_result = backend.process_era_pipeline(embeddings[i], {"top_k": 5})
                era_times.append(time.time() - start)
            
            avg_era_time = np.mean(era_times)
            
            # 5. Computational offloading benchmark (if supported)
            offloading_results = {}
            if backend.supports_feature("arbitrary_computation"):
                # Test ML primitives
                test_data = np.random.randn(64, 384).astype(np.float32)
                
                for operation in ["softmax", "matrix_multiply", "attention"]:
                    try:
                        start = time.time()
                        if operation == "softmax":
                            result = backend.offload_computation("softmax", test_data[0], {"temperature": 1.0})
                        elif operation == "matrix_multiply":
                            result = backend.offload_computation("matrix_multiply", test_data, {"matrix_b": test_data.T})
                        elif operation == "attention":
                            result = backend.offload_computation("attention", test_data, {"seq_len": 64, "d_model": 384, "scale": 0.16})
                        
                        exec_time = time.time() - start
                        offloading_results[operation] = {
                            "execution_time": exec_time,
                            "result_shape": result.shape,
                            "successful": True
                        }
                    except Exception as e:
                        offloading_results[operation] = {
                            "execution_time": 0,
                            "error": str(e),
                            "successful": False
                        }
            
            # 6. Get comprehensive metrics
            metrics = backend.get_metrics()
            accelerator_info = backend.get_accelerator_info()
            
            # 7. Feature analysis
            features = {}
            test_features = [
                "basic_storage", "basic_retrieval", "basic_similarity", "era_pipeline",
                "p2p_transfer", "cache_hierarchy", "ebpf_offloading", "shared_memory",
                "parallel_processing", "computational_storage", "arbitrary_computation"
            ]
            
            for feature in test_features:
                features[feature] = backend.supports_feature(feature)
            
            result = {
                "name": name,
                "backend_type": backend_type.value,
                "performance": {
                    "store_time": store_time,
                    "avg_retrieve_time": avg_retrieve_time,
                    "avg_similarity_time": avg_similarity_time,
                    "avg_era_time": avg_era_time,
                    "throughput_qps": throughput,
                    "latency_ms": avg_similarity_time * 1000
                },
                "computational_offloading": offloading_results,
                "metrics": metrics,
                "accelerator_info": accelerator_info,
                "features": features,
                "test_config": {
                    "num_embeddings": num_embeddings,
                    "query_iterations": query_iterations,
                    "embedding_dim": 384
                }
            }
            
            return result
            
        finally:
            backend.shutdown()
    
    def _generate_analysis(self):
        """Generate comprehensive performance analysis."""
        
        analysis_file = f"{self.output_dir}/BENCHMARK_ANALYSIS.md"
        
        with open(analysis_file, 'w') as f:
            f.write("# Enhanced RAG-CSD Emulator Benchmark Analysis\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance comparison
            f.write("## Performance Comparison\n\n")
            f.write("| Backend | Latency (ms) | Throughput (q/s) | Store Time (s) | ERA Time (ms) |\n")
            f.write("|---------|--------------|------------------|----------------|---------------|\n")
            
            for backend_name, result in self.results.items():
                perf = result["performance"]
                f.write(f"| {result['name']} | {perf['latency_ms']:.2f} | {perf['throughput_qps']:.1f} | {perf['store_time']:.3f} | {perf['avg_era_time']*1000:.2f} |\n")
            
            # Feature matrix
            f.write("\n## Feature Support Matrix\n\n")
            if self.results:
                all_features = list(next(iter(self.results.values()))["features"].keys())
                f.write("| Feature | " + " | ".join(result["name"] for result in self.results.values()) + " |\n")
                f.write("|---------|" + "---|" * len(self.results) + "\n")
                
                for feature in all_features:
                    row = f"| {feature} |"
                    for result in self.results.values():
                        status = "✅" if result["features"][feature] else "❌"
                        row += f" {status} |"
                    f.write(row + "\n")
            
            # Computational offloading analysis
            f.write("\n## Computational Offloading Results\n\n")
            for backend_name, result in self.results.items():
                offloading = result["computational_offloading"]
                if offloading:
                    f.write(f"### {result['name']}\n\n")
                    for operation, details in offloading.items():
                        if details["successful"]:
                            f.write(f"- **{operation}**: {details['execution_time']*1000:.2f}ms, result shape: {details['result_shape']}\n")
                        else:
                            f.write(f"- **{operation}**: Failed - {details.get('error', 'Unknown error')}\n")
                    f.write("\n")
            
            # Performance analysis
            f.write("\n## Performance Analysis\n\n")
            
            if len(self.results) >= 2:
                # Find fastest backend
                fastest_backend = min(self.results.values(), key=lambda x: x["performance"]["latency_ms"])
                highest_throughput = max(self.results.values(), key=lambda x: x["performance"]["throughput_qps"])
                
                f.write(f"**Fastest Latency**: {fastest_backend['name']} ({fastest_backend['performance']['latency_ms']:.2f}ms)\n\n")
                f.write(f"**Highest Throughput**: {highest_throughput['name']} ({highest_throughput['performance']['throughput_qps']:.1f} q/s)\n\n")
                
                # Speedup analysis
                baseline = None
                for result in self.results.values():
                    if "enhanced_simulator" in result["backend_type"]:
                        baseline = result
                        break
                
                if baseline:
                    f.write("### Speedup Analysis (vs Enhanced Simulator)\n\n")
                    for backend_name, result in self.results.items():
                        if result != baseline:
                            speedup = baseline["performance"]["latency_ms"] / result["performance"]["latency_ms"]
                            throughput_ratio = result["performance"]["throughput_qps"] / baseline["performance"]["throughput_qps"]
                            f.write(f"- **{result['name']}**: {speedup:.2f}x speedup, {throughput_ratio:.2f}x throughput\n")
            
            f.write(f"\n## Accelerator Information\n\n")
            for backend_name, result in self.results.items():
                accel_info = result["accelerator_info"]
                f.write(f"### {result['name']}\n\n")
                for key, value in accel_info.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
        
        print(f"  📋 Analysis saved to {analysis_file}")
    
    def _generate_plots(self):
        """Generate comprehensive performance plots."""
        
        if not self.results:
            print("  ⚠️  No results to plot")
            return
        
        # Prepare data
        backends = [result["name"] for result in self.results.values()]
        latencies = [result["performance"]["latency_ms"] for result in self.results.values()]
        throughputs = [result["performance"]["throughput_qps"] for result in self.results.values()]
        store_times = [result["performance"]["store_time"] for result in self.results.values()]
        era_times = [result["performance"]["avg_era_time"] * 1000 for result in self.results.values()]
        
        # Plot 1: Latency Comparison
        plt.figure(figsize=(12, 8))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = plt.bar(backends, latencies, color=colors[:len(backends)])
        plt.title('Emulator Backend Latency Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Backend Type', fontsize=12)
        plt.ylabel('Average Latency (ms)', fontsize=12)
        plt.yscale('log')
        
        # Add value labels on bars
        for bar, latency in zip(bars, latencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{latency:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/latency_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Throughput vs Latency
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(latencies, throughputs, s=200, c=colors[:len(backends)], alpha=0.7)
        
        for i, backend in enumerate(backends):
            plt.annotate(backend, (latencies[i], throughputs[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.title('Throughput vs Latency Trade-off', fontsize=16, fontweight='bold')
        plt.xlabel('Latency (ms)', fontsize=12)
        plt.ylabel('Throughput (queries/sec)', fontsize=12)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/throughput_vs_latency.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Comprehensive Performance Radar
        if len(backends) >= 2:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Normalize metrics for radar chart
            metrics = ['Latency', 'Throughput', 'Store Speed', 'ERA Speed']
            
            # Invert latency and times (lower is better)
            max_latency = max(latencies)
            max_store = max(store_times)
            max_era = max(era_times)
            
            normalized_data = []
            for i in range(len(backends)):
                normalized = [
                    (max_latency - latencies[i]) / max_latency,  # Inverted
                    throughputs[i] / max(throughputs),
                    (max_store - store_times[i]) / max_store,  # Inverted
                    (max_era - era_times[i]) / max_era  # Inverted
                ]
                normalized_data.append(normalized)
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, (backend, data) in enumerate(zip(backends, normalized_data)):
                values = data + data[:1]  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=2, label=backend, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Comprehensive Performance Comparison\n(Higher = Better)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/performance_radar.pdf", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 4: Feature Support Heatmap
        if self.results:
            features_data = []
            feature_names = []
            backend_names = []
            
            for result in self.results.values():
                backend_names.append(result["name"])
                features = result["features"]
                if not feature_names:
                    feature_names = list(features.keys())
                features_data.append([int(features[f]) for f in feature_names])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(features_data, cmap='RdYlGn', aspect='auto')
            
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_yticks(np.arange(len(backend_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_yticklabels(backend_names)
            
            # Add text annotations
            for i in range(len(backend_names)):
                for j in range(len(feature_names)):
                    text = "✅" if features_data[i][j] else "❌"
                    ax.text(j, i, text, ha="center", va="center", fontsize=12)
            
            ax.set_title('Feature Support Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/feature_matrix.pdf", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  📊 Plots saved to {self.output_dir}/plots/")
    
    def _save_results(self):
        """Save detailed results to JSON."""
        results_file = f"{self.output_dir}/benchmark_results.json"
        
        # Convert numpy types to JSON serializable
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return obj  # Keep tuples as tuples
            return obj
        
        # Deep conversion of results
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = self._convert_for_json(value)
        
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "benchmark_config": {
                    "embedding_dim": 384,
                    "num_embeddings": 100,
                    "query_iterations": 50
                },
                "results": json_results
            }, f, indent=2, default=convert_numpy)
        
        print(f"  💾 Results saved to {results_file}")
    
    def _convert_for_json(self, obj):
        """Recursively convert objects for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists for JSON
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def main():
    """Run the comprehensive emulator benchmark."""
    
    benchmark = EmulatorBenchmark()
    results = benchmark.run_benchmark()
    
    print(f"\n🎉 Emulator Benchmark Completed!")
    print(f"📁 Results: {benchmark.output_dir}")
    print(f"📊 Plots: {benchmark.output_dir}/plots/")
    print(f"📋 Analysis: {benchmark.output_dir}/BENCHMARK_ANALYSIS.md")
    
    # Summary
    if results:
        print(f"\n📈 Quick Summary:")
        for backend_name, result in results.items():
            perf = result["performance"]
            print(f"  {result['name']}: {perf['latency_ms']:.2f}ms, {perf['throughput_qps']:.1f} q/s")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)