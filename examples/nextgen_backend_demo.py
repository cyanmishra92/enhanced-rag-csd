#!/usr/bin/env python3
"""
Next-Generation CSD Backend Demonstration.

This script demonstrates the new backend architecture supporting:
- OpenCSD with eBPF computational offloading
- SPDK vfio-user with shared memory P2P
- Hardware abstraction layer for accelerator-agnostic design
- Enhanced Mock SPDK with 3-level cache hierarchy
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_rag_csd.backends import (
    CSDBackendManager, CSDBackendType, CSDHardwareAbstractionLayer, AcceleratorType
)
from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class NextGenBackendDemo:
    """Demonstration of next-generation CSD backend capabilities."""
    
    def __init__(self):
        self.manager = CSDBackendManager()
        self.hal = CSDHardwareAbstractionLayer()
        
        # Demo configuration
        self.demo_config = {
            "vector_db_path": "/tmp/nextgen_demo",
            "storage_path": "/tmp/nextgen_storage",
            "embedding": {"dimensions": 384},
            "csd": {
                "ssd_bandwidth_mbps": 3000,
                "nand_bandwidth_mbps": 800,
                "compute_latency_ms": 0.05,
                "max_parallel_ops": 16
            },
            "cache": {
                "l1_cache_mb": 128,
                "l2_cache_mb": 1024,
                "l3_cache_mb": 4096
            },
            "opencsd": {
                "zns_device": "/dev/nvme0n1",
                "mount_point": "/tmp/flufflefs_demo",
                "ebpf_program_dir": "./ebpf_kernels"
            },
            "spdk_vfio": {
                "socket_path": "/tmp/demo_vfio.sock",
                "shared_memory_mb": 2048,
                "nvme_size_gb": 32,
                "queue_depth": 512,
                "compute_cores": 8
            }
        }
        
        # Test data
        self.num_embeddings = 100
        self.embedding_dim = 384
        self.test_embeddings = np.random.randn(self.num_embeddings, self.embedding_dim).astype(np.float32)
        self.test_embeddings = self.test_embeddings / np.linalg.norm(self.test_embeddings, axis=1, keepdims=True)
        self.test_metadata = [{"id": i, "content": f"Demo document {i}"} for i in range(self.num_embeddings)]
    
    def run_demo(self) -> None:
        """Run comprehensive next-generation backend demonstration."""
        print("🚀 Enhanced RAG-CSD Next-Generation Backend Demo")
        print("=" * 60)
        
        # Step 1: Hardware detection and analysis
        self._demo_hardware_detection()
        
        # Step 2: Backend availability analysis
        self._demo_backend_analysis()
        
        # Step 3: Test available backends
        self._demo_backend_testing()
        
        # Step 4: Feature comparison
        self._demo_feature_comparison()
        
        # Step 5: Performance benchmarking
        self._demo_performance_benchmarking()
        
        print("\n🎉 Next-generation backend demonstration completed!")
    
    def _demo_hardware_detection(self) -> None:
        """Demonstrate hardware detection capabilities."""
        print("\n🔍 Hardware Detection & Analysis")
        print("-" * 40)
        
        # Detect available hardware
        hardware = self.hal.detect_available_hardware()
        print(f"Detected hardware accelerators:")
        for accel_type, available in hardware.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"  {accel_type.value.upper()}: {status}")
        
        # Get hardware report
        report = self.hal.get_hardware_report()
        print(f"\nPlatform: {report['platform']}")
        print(f"Architecture: {report['architecture']}")
        
        if report['available_accelerators']:
            print(f"\nAccelerator Performance:")
            for accel in report['available_accelerators']:
                perf = accel['performance']
                print(f"  {accel['type'].upper()}:")
                print(f"    Compute Units: {perf['compute_units']}")
                print(f"    Peak Performance: {perf.get('peak_performance_gflops', 'N/A')} GFLOPS")
                print(f"    Memory Bandwidth: {perf.get('memory_bandwidth_gbps', 'N/A')} GB/s")
        
        # Backend recommendations
        print(f"\nRecommended Backends:")
        for rec in report['recommended_backends']:
            config_desc = f"{rec['config']['preferred_accelerator']}"
            if rec['config']['require_real_hardware']:
                config_desc += " (real hardware)"
            print(f"  {config_desc}: {rec['recommended_backend']}")
    
    def _demo_backend_analysis(self) -> None:
        """Demonstrate backend availability and capabilities."""
        print("\n📊 Backend Availability Analysis")
        print("-" * 40)
        
        available_backends = self.manager.get_available_backends()
        print(f"Available backends: {len(available_backends)}")
        
        for backend_type in available_backends:
            print(f"\n✅ {backend_type.value.upper()}")
            
            # Create backend to get detailed info
            backend = self.manager.create_backend(backend_type, self.demo_config, enable_fallback=False)
            if backend:
                try:
                    info = backend.get_backend_info()
                    print(f"    Description: {info.get('description', 'N/A')}")
                    
                    if hasattr(backend, 'get_accelerator_info'):
                        accel_info = backend.get_accelerator_info()
                        print(f"    Accelerator: {accel_info.get('accelerator_type', 'N/A')}")
                        print(f"    Compute Units: {accel_info.get('compute_units', 'N/A')}")
                        print(f"    Supports Offloading: {accel_info.get('supports_offloading', False)}")
                    
                    # Check feature support
                    features = [
                        "ebpf_offloading", "shared_memory", "cache_hierarchy", 
                        "parallel_processing", "real_time_processing"
                    ]
                    supported = [f for f in features if backend.supports_feature(f)]
                    if supported:
                        print(f"    Advanced Features: {', '.join(supported)}")
                    
                finally:
                    backend.shutdown()
        
        # Show unavailable backends
        all_backends = [bt for bt in CSDBackendType]
        unavailable = [bt for bt in all_backends if bt not in available_backends]
        
        if unavailable:
            print(f"\n❌ Unavailable backends ({len(unavailable)}):")
            for backend_type in unavailable:
                status = self.manager.get_backend_status(backend_type)
                print(f"    {backend_type.value}: {status.value}")
    
    def _demo_backend_testing(self) -> None:
        """Test functionality of available backends."""
        print("\n🧪 Backend Functionality Testing")
        print("-" * 40)
        
        available_backends = self.manager.get_available_backends()
        
        for backend_type in available_backends:
            print(f"\nTesting {backend_type.value.upper()}...")
            
            backend = self.manager.create_backend(backend_type, self.demo_config)
            if not backend:
                print(f"  ❌ Failed to create backend")
                continue
            
            try:
                # Test basic operations
                test_embeddings = self.test_embeddings[:20]  # Use subset for demo
                test_metadata = self.test_metadata[:20]
                
                # Store embeddings
                start_time = time.time()
                backend.store_embeddings(test_embeddings, test_metadata)
                store_time = time.time() - start_time
                print(f"  ✅ Store: {len(test_embeddings)} embeddings in {store_time:.3f}s")
                
                # Retrieve embeddings
                start_time = time.time()
                retrieved = backend.retrieve_embeddings([0, 1, 2, 3, 4])
                retrieve_time = time.time() - start_time
                print(f"  ✅ Retrieve: 5 embeddings in {retrieve_time:.3f}s")
                
                # Compute similarities
                query_embedding = test_embeddings[0]
                candidate_indices = [1, 2, 3, 4, 5]
                start_time = time.time()
                similarities = backend.compute_similarities(query_embedding, candidate_indices)
                similarity_time = time.time() - start_time
                print(f"  ✅ Similarities: {len(similarities)} computed in {similarity_time:.3f}s")
                
                # ERA pipeline
                start_time = time.time()
                augmented = backend.process_era_pipeline(
                    query_embedding, 
                    {"top_k": 3, "mode": "similarity"}
                )
                era_time = time.time() - start_time
                print(f"  ✅ ERA Pipeline: Completed in {era_time:.3f}s")
                
                # Get metrics
                metrics = backend.get_metrics()
                cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
                read_ops = metrics.get("read_ops", 0)
                write_ops = metrics.get("write_ops", 0)
                print(f"  📊 Metrics: R:{read_ops}, W:{write_ops}, Cache:{cache_hit_rate:.1%}")
                
                # Backend-specific metrics
                if "cache_hierarchy" in metrics:
                    cache_info = metrics["cache_hierarchy"]
                    print(f"  🏗️  Cache: L1:{cache_info.get('l1_hits', 0)}, "
                          f"L2:{cache_info.get('l2_hits', 0)}, L3:{cache_info.get('l3_hits', 0)}")
                
                if "opencsd_info" in metrics:
                    opencsd_info = metrics["opencsd_info"]
                    print(f"  🔧 OpenCSD: eBPF programs:{opencsd_info.get('ebpf_programs_loaded', 0)}, "
                          f"ZNS ops:{metrics.get('zns_operations', 0)}")
                
                if "spdk_vfio_info" in metrics:
                    vfio_info = metrics["spdk_vfio_info"]
                    print(f"  ⚡ SPDK vfio-user: Shared memory:{vfio_info.get('shared_memory_size_mb', 0)}MB, "
                          f"P2P transfers:{metrics.get('p2p_transfers', 0)}")
                
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
            
            finally:
                backend.shutdown()
    
    def _demo_feature_comparison(self) -> None:
        """Compare features across different backends."""
        print("\n🔍 Feature Comparison Matrix")
        print("-" * 40)
        
        features = [
            "basic_storage", "basic_retrieval", "basic_similarity",
            "era_pipeline", "p2p_transfer", "cache_hierarchy",
            "ebpf_offloading", "shared_memory", "parallel_processing",
            "real_time_processing", "computational_storage"
        ]
        
        available_backends = self.manager.get_available_backends()
        
        # Create comparison matrix
        print(f"{'Feature':<20} ", end="")
        for backend_type in available_backends:
            print(f"{backend_type.value[:12]:<12} ", end="")
        print()
        
        print("-" * (20 + len(available_backends) * 12))
        
        for feature in features:
            print(f"{feature:<20} ", end="")
            
            for backend_type in available_backends:
                backend = self.manager.create_backend(backend_type, self.demo_config, enable_fallback=False)
                if backend:
                    supported = "✅" if backend.supports_feature(feature) else "❌"
                    backend.shutdown()
                else:
                    supported = "❓"
                print(f"{supported:<12} ", end="")
            print()
    
    def _demo_performance_benchmarking(self) -> None:
        """Benchmark performance across backends."""
        print("\n⚡ Performance Benchmarking")
        print("-" * 40)
        
        available_backends = self.manager.get_available_backends()
        benchmark_results = {}
        
        for backend_type in available_backends:
            print(f"\nBenchmarking {backend_type.value.upper()}...")
            
            backend = self.manager.create_backend(backend_type, self.demo_config)
            if not backend:
                continue
            
            try:
                # Benchmark parameters
                num_embeddings = 50
                num_queries = 20
                
                test_embeddings = self.test_embeddings[:num_embeddings]
                test_metadata = self.test_metadata[:num_embeddings]
                
                # Store benchmark
                start_time = time.time()
                backend.store_embeddings(test_embeddings, test_metadata)
                store_time = time.time() - start_time
                
                # Query benchmark
                query_times = []
                for i in range(num_queries):
                    query_embedding = test_embeddings[i % num_embeddings]
                    candidate_indices = [(i + j + 1) % num_embeddings for j in range(10)]
                    
                    query_start = time.time()
                    similarities = backend.compute_similarities(query_embedding, candidate_indices)
                    query_time = time.time() - query_start
                    query_times.append(query_time)
                
                avg_query_time = np.mean(query_times)
                throughput = num_queries / sum(query_times)
                
                # ERA pipeline benchmark
                era_times = []
                for i in range(5):  # Test 5 ERA operations
                    query_embedding = test_embeddings[i]
                    era_start = time.time()
                    augmented = backend.process_era_pipeline(
                        query_embedding, 
                        {"top_k": 5, "mode": "similarity"}
                    )
                    era_time = time.time() - era_start
                    era_times.append(era_time)
                
                avg_era_time = np.mean(era_times)
                
                # Store results
                benchmark_results[backend_type.value] = {
                    "store_time": store_time,
                    "avg_query_time": avg_query_time,
                    "throughput": throughput,
                    "avg_era_time": avg_era_time
                }
                
                print(f"  📊 Store: {store_time:.3f}s for {num_embeddings} embeddings")
                print(f"  📊 Query: {avg_query_time:.3f}s average ({throughput:.1f} q/s)")
                print(f"  📊 ERA: {avg_era_time:.3f}s average")
                
            except Exception as e:
                print(f"  ❌ Benchmark failed: {e}")
            
            finally:
                backend.shutdown()
        
        # Summary comparison
        if len(benchmark_results) > 1:
            print(f"\n📈 Performance Summary:")
            fastest_query = min(benchmark_results.items(), key=lambda x: x[1]["avg_query_time"])
            fastest_era = min(benchmark_results.items(), key=lambda x: x[1]["avg_era_time"])
            highest_throughput = max(benchmark_results.items(), key=lambda x: x[1]["throughput"])
            
            print(f"  🏆 Fastest Query: {fastest_query[0]} ({fastest_query[1]['avg_query_time']:.3f}s)")
            print(f"  🏆 Fastest ERA: {fastest_era[0]} ({fastest_era[1]['avg_era_time']:.3f}s)")
            print(f"  🏆 Highest Throughput: {highest_throughput[0]} ({highest_throughput[1]['throughput']:.1f} q/s)")


def main():
    """Run the next-generation backend demonstration."""
    try:
        demo = NextGenBackendDemo()
        demo.run_demo()
        return True
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)