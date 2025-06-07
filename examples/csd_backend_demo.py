#!/usr/bin/env python3
"""
Demonstration of Enhanced RAG-CSD with pluggable CSD backends.

This script shows how to use different computational storage emulator backends
while preserving the existing performance characteristics.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_rag_csd.backends import CSDBackendManager, CSDBackendType
from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


def demo_backend_selection():
    """Demonstrate backend selection and availability checking."""
    print("🔍 Enhanced RAG-CSD Backend Selection Demo")
    print("=" * 50)
    
    # Initialize backend manager
    manager = CSDBackendManager()
    
    # Show available backends
    available = manager.get_available_backends()
    print(f"Available backends: {[bt.value for bt in available]}")
    
    # Show backend status
    backend_info = manager.get_backend_info()
    print("\nBackend Status:")
    for backend, status in backend_info["backend_status"].items():
        status_icon = "✅" if status == "available" else "❌"
        print(f"  {status_icon} {backend}: {status}")
    
    return manager


def demo_enhanced_simulator(manager: CSDBackendManager):
    """Demonstrate enhanced simulator backend."""
    print("\n🚀 Testing Enhanced Simulator Backend")
    print("-" * 40)
    
    config = {
        "vector_db_path": "./demo_vectors",
        "storage_path": "./demo_storage",
        "embedding": {"dimensions": 384},
        "csd": {
            "ssd_bandwidth_mbps": 2000,
            "nand_bandwidth_mbps": 500,
            "compute_latency_ms": 0.1,
            "max_parallel_ops": 8
        }
    }
    
    # Create backend
    backend = manager.create_backend(CSDBackendType.ENHANCED_SIMULATOR, config)
    if backend is None:
        print("❌ Failed to create enhanced simulator backend")
        return
    
    print("✅ Enhanced simulator backend created successfully")
    
    # Test basic operations
    test_backend_operations(backend, "Enhanced Simulator")
    
    # Cleanup
    backend.shutdown()


def demo_spdk_emulator(manager: CSDBackendManager):
    """Demonstrate SPDK emulator backend."""
    print("\n🔧 Testing SPDK Emulator Backend")
    print("-" * 40)
    
    # Check if SPDK backend is available
    spdk_status = manager.get_backend_status(CSDBackendType.SPDK_EMULATOR)
    if spdk_status.value != "available":
        print(f"❌ SPDK emulator not available: {spdk_status.value}")
        print("   Install SPDK, QEMU, and libvfio-user for full emulation")
        return
    
    config = {
        "vector_db_path": "./demo_vectors_spdk",
        "storage_path": "./demo_storage_spdk", 
        "embedding": {"dimensions": 384},
        "spdk": {
            "nvme_size_gb": 10,
            "rpc_socket": "/tmp/spdk_demo.sock",
            "virtual_queues": 8
        }
    }
    
    # Create backend
    backend = manager.create_backend(CSDBackendType.SPDK_EMULATOR, config)
    if backend is None:
        print("❌ Failed to create SPDK emulator backend")
        return
    
    print("✅ SPDK emulator backend created successfully")
    
    # Test basic operations
    test_backend_operations(backend, "SPDK Emulator")
    
    # Cleanup
    backend.shutdown()


def test_backend_operations(backend, backend_name: str):
    """Test basic operations on a CSD backend."""
    print(f"\n📊 Testing {backend_name} Operations:")
    
    # Generate test embeddings
    num_embeddings = 100
    embedding_dim = 384
    embeddings = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create metadata
    metadata = [{"id": i, "content": f"Document {i}"} for i in range(num_embeddings)]
    
    # Test 1: Store embeddings
    print("  📝 Storing embeddings...", end=" ")
    start_time = time.time()
    backend.store_embeddings(embeddings, metadata)
    store_time = time.time() - start_time
    print(f"✅ {store_time:.3f}s")
    
    # Test 2: Retrieve embeddings
    print("  📖 Retrieving embeddings...", end=" ")
    test_indices = list(range(0, min(50, num_embeddings), 5))
    start_time = time.time()
    retrieved = backend.retrieve_embeddings(test_indices)
    retrieve_time = time.time() - start_time
    print(f"✅ {retrieve_time:.3f}s ({len(test_indices)} embeddings)")
    
    # Test 3: Compute similarities
    print("  🔍 Computing similarities...", end=" ")
    query_embedding = embeddings[0]
    candidate_indices = list(range(10, min(60, num_embeddings)))
    start_time = time.time()
    similarities = backend.compute_similarities(query_embedding, candidate_indices)
    similarity_time = time.time() - start_time
    print(f"✅ {similarity_time:.3f}s ({len(candidate_indices)} candidates)")
    
    # Test 4: ERA pipeline
    print("  ⚡ ERA pipeline...", end=" ")
    start_time = time.time()
    augmented = backend.process_era_pipeline(
        query_embedding, 
        {"top_k": 5, "mode": "similarity"}
    )
    era_time = time.time() - start_time
    print(f"✅ {era_time:.3f}s")
    
    # Test 5: P2P transfer
    print("  🔄 P2P transfer...", end=" ")
    test_data = np.random.randn(1000).astype(np.float32)
    start_time = time.time()
    gpu_alloc = backend.p2p_transfer_to_gpu(test_data)
    p2p_time = time.time() - start_time
    print(f"✅ {p2p_time:.3f}s (allocation: {gpu_alloc[:20]}...)")
    
    # Show metrics
    metrics = backend.get_metrics()
    print(f"\n  📈 Performance Metrics:")
    print(f"    Backend Type: {metrics.get('backend_type', 'unknown')}")
    print(f"    Read Ops: {metrics.get('read_ops', 0)}")
    print(f"    Write Ops: {metrics.get('write_ops', 0)}")
    print(f"    Cache Hit Rate: {metrics.get('cache_hit_rate', 0)*100:.1f}%")
    print(f"    Total Latency: {store_time + retrieve_time + similarity_time + era_time:.3f}s")


def demo_fallback_mechanism(manager: CSDBackendManager):
    """Demonstrate fallback mechanism."""
    print("\n🔄 Testing Fallback Mechanism")
    print("-" * 40)
    
    config = {
        "vector_db_path": "./demo_vectors_fallback",
        "csd_backend": "spdk_emulator",  # Request SPDK
        "enable_fallback": True
    }
    
    # This should fallback to enhanced simulator if SPDK unavailable
    backend = manager.create_backend(
        CSDBackendType.SPDK_EMULATOR,
        config,
        enable_fallback=True
    )
    
    if backend:
        backend_type = backend.get_backend_type()
        print(f"✅ Successfully created backend: {backend_type.value}")
        
        # Quick test
        test_embeddings = np.random.randn(10, 384).astype(np.float32)
        test_metadata = [{"id": i} for i in range(10)]
        
        backend.store_embeddings(test_embeddings, test_metadata)
        retrieved = backend.retrieve_embeddings([0, 1, 2])
        print(f"✅ Fallback backend operational ({len(retrieved)} embeddings retrieved)")
        
        backend.shutdown()
    else:
        print("❌ Fallback mechanism failed")


def demo_performance_comparison():
    """Compare performance between different backends."""
    print("\n⚡ Performance Comparison")
    print("-" * 40)
    
    manager = CSDBackendManager()
    available_backends = manager.get_available_backends()
    
    if len(available_backends) < 2:
        print("⚠️  Need at least 2 backends for comparison")
        print(f"   Available: {[bt.value for bt in available_backends]}")
        return
    
    # Test configuration
    config = {
        "vector_db_path": "./perf_test_vectors",
        "embedding": {"dimensions": 384}
    }
    
    results = {}
    
    for backend_type in available_backends[:2]:  # Test first 2 available
        print(f"\n  Testing {backend_type.value}...")
        
        backend = manager.create_backend(backend_type, config)
        if backend is None:
            continue
        
        # Performance test
        num_embeddings = 50
        embeddings = np.random.randn(num_embeddings, 384).astype(np.float32)
        metadata = [{"id": i} for i in range(num_embeddings)]
        
        # Measure operations
        start_time = time.time()
        backend.store_embeddings(embeddings, metadata)
        store_time = time.time() - start_time
        
        start_time = time.time()
        retrieved = backend.retrieve_embeddings(list(range(10)))
        retrieve_time = time.time() - start_time
        
        start_time = time.time()
        similarities = backend.compute_similarities(embeddings[0], list(range(20)))
        compute_time = time.time() - start_time
        
        total_time = store_time + retrieve_time + compute_time
        
        results[backend_type.value] = {
            "store_time": store_time,
            "retrieve_time": retrieve_time,
            "compute_time": compute_time,
            "total_time": total_time
        }
        
        backend.shutdown()
    
    # Show comparison
    print(f"\n  📊 Performance Results:")
    print(f"  {'Backend':<20} {'Store':<10} {'Retrieve':<10} {'Compute':<10} {'Total':<10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for backend_name, times in results.items():
        print(f"  {backend_name:<20} "
              f"{times['store_time']*1000:>8.1f}ms "
              f"{times['retrieve_time']*1000:>8.1f}ms "
              f"{times['compute_time']*1000:>8.1f}ms "
              f"{times['total_time']*1000:>8.1f}ms")


def main():
    """Main demonstration function."""
    print("🎯 Enhanced RAG-CSD Backend Integration Demo")
    print("============================================\n")
    
    try:
        # Demo 1: Backend selection
        manager = demo_backend_selection()
        
        # Demo 2: Enhanced simulator
        demo_enhanced_simulator(manager)
        
        # Demo 3: SPDK emulator (if available)
        demo_spdk_emulator(manager)
        
        # Demo 4: Fallback mechanism
        demo_fallback_mechanism(manager)
        
        # Demo 5: Performance comparison
        demo_performance_comparison()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. Install SPDK, QEMU, and libvfio-user for full CSD emulation")
        print("   2. Extend the framework with custom backends")
        print("   3. Run comprehensive benchmarks with different backends")
        print("   4. Contribute new emulator integrations")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    main()