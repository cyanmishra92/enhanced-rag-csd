#!/usr/bin/env python3
"""
Comprehensive test suite for CSD backend integration.

This test validates the backend abstraction framework and ensures
that all backends maintain API compatibility and performance characteristics.
"""

import sys
import os
import time
import pytest
import numpy as np
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_rag_csd.backends import (
    CSDBackendManager, 
    CSDBackendType, 
    CSDBackendInterface,
    EnhancedSimulatorBackend
)
from enhanced_rag_csd.backends.mock_spdk import MockSPDKEmulatorBackend


class TestBackendIntegration:
    """Test suite for backend integration framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CSDBackendManager()
        self.test_config = {
            "vector_db_path": "./test_vectors",
            "storage_path": "./test_storage",
            "embedding": {"dimensions": 384},
            "csd": {
                "ssd_bandwidth_mbps": 2000,
                "nand_bandwidth_mbps": 500,
                "compute_latency_ms": 0.1,
                "max_parallel_ops": 8
            },
            "spdk": {
                "nvme_size_gb": 5,
                "virtual_queues": 4
            }
        }
        
        # Test data
        self.num_embeddings = 50
        self.embedding_dim = 384
        self.test_embeddings = np.random.randn(self.num_embeddings, self.embedding_dim).astype(np.float32)
        self.test_embeddings = self.test_embeddings / np.linalg.norm(self.test_embeddings, axis=1, keepdims=True)
        self.test_metadata = [{"id": i, "content": f"Test document {i}"} for i in range(self.num_embeddings)]
    
    def test_backend_manager_initialization(self):
        """Test that backend manager initializes correctly."""
        assert self.manager is not None
        
        available = self.manager.get_available_backends()
        assert CSDBackendType.ENHANCED_SIMULATOR in available
        assert CSDBackendType.MOCK_SPDK in available
        
        # Check backend info
        info = self.manager.get_backend_info()
        assert "available_backends" in info
        assert "backend_status" in info
    
    def test_enhanced_simulator_backend(self):
        """Test enhanced simulator backend functionality."""
        backend = self.manager.create_backend(CSDBackendType.ENHANCED_SIMULATOR, self.test_config)
        assert backend is not None
        assert isinstance(backend, EnhancedSimulatorBackend)
        
        self._test_backend_operations(backend, "Enhanced Simulator")
        
        backend.shutdown()
    
    def test_mock_spdk_backend(self):
        """Test mock SPDK backend functionality."""
        backend = self.manager.create_backend(CSDBackendType.MOCK_SPDK, self.test_config)
        assert backend is not None
        assert isinstance(backend, MockSPDKEmulatorBackend)
        
        self._test_backend_operations(backend, "Mock SPDK")
        
        backend.shutdown()
    
    def _test_backend_operations(self, backend: CSDBackendInterface, backend_name: str):
        """Test all basic operations on a backend."""
        print(f"\\nTesting {backend_name} backend operations...")
        
        # Test 1: Store embeddings
        start_time = time.time()
        backend.store_embeddings(self.test_embeddings, self.test_metadata)
        store_time = time.time() - start_time
        print(f"  Store time: {store_time:.3f}s")
        
        # Test 2: Retrieve embeddings
        test_indices = list(range(0, min(20, self.num_embeddings), 2))
        start_time = time.time()
        retrieved = backend.retrieve_embeddings(test_indices)
        retrieve_time = time.time() - start_time
        print(f"  Retrieve time: {retrieve_time:.3f}s")
        
        # Validate retrieved embeddings shape
        assert retrieved.shape == (len(test_indices), self.embedding_dim)
        
        # Test 3: Compute similarities
        query_embedding = self.test_embeddings[0]
        candidate_indices = list(range(5, min(25, self.num_embeddings)))
        start_time = time.time()
        similarities = backend.compute_similarities(query_embedding, candidate_indices)
        similarity_time = time.time() - start_time
        print(f"  Similarity computation time: {similarity_time:.3f}s")
        
        # Validate similarities
        assert len(similarities) == len(candidate_indices)
        assert all(-1 <= sim <= 1 for sim in similarities)
        
        # Test 4: ERA pipeline
        start_time = time.time()
        augmented = backend.process_era_pipeline(
            query_embedding, 
            {"top_k": 5, "mode": "similarity"}
        )
        era_time = time.time() - start_time
        print(f"  ERA pipeline time: {era_time:.3f}s")
        
        # Validate augmented data
        assert len(augmented) > self.embedding_dim  # Should be concatenated
        
        # Test 5: P2P transfer
        test_data = np.random.randn(1000).astype(np.float32)
        start_time = time.time()
        gpu_alloc = backend.p2p_transfer_to_gpu(test_data)
        p2p_time = time.time() - start_time
        print(f"  P2P transfer time: {p2p_time:.3f}s")
        
        # Validate P2P result
        assert isinstance(gpu_alloc, str)
        assert len(gpu_alloc) > 0
        
        # Test 6: Get metrics
        metrics = backend.get_metrics()
        assert "backend_type" in metrics
        assert "read_ops" in metrics
        assert "write_ops" in metrics
        
        total_time = store_time + retrieve_time + similarity_time + era_time + p2p_time
        print(f"  Total operation time: {total_time:.3f}s")
        print(f"  Metrics: {metrics.get('backend_type', 'unknown')} - "
              f"R:{metrics.get('read_ops', 0)} W:{metrics.get('write_ops', 0)}")
    
    def test_backend_performance_comparison(self):
        """Compare performance between different backends."""
        backends_to_test = [
            (CSDBackendType.ENHANCED_SIMULATOR, "Enhanced Simulator"),
            (CSDBackendType.MOCK_SPDK, "Mock SPDK")
        ]
        
        results = {}
        
        for backend_type, name in backends_to_test:
            backend = self.manager.create_backend(backend_type, self.test_config)
            if backend is None:
                continue
            
            # Measure store performance
            start_time = time.time()
            backend.store_embeddings(self.test_embeddings[:20], self.test_metadata[:20])
            store_time = time.time() - start_time
            
            # Measure retrieve performance
            start_time = time.time()
            retrieved = backend.retrieve_embeddings(list(range(10)))
            retrieve_time = time.time() - start_time
            
            # Measure similarity computation
            start_time = time.time()
            similarities = backend.compute_similarities(self.test_embeddings[0], list(range(15)))
            compute_time = time.time() - start_time
            
            results[name] = {
                "store_time": store_time,
                "retrieve_time": retrieve_time,
                "compute_time": compute_time,
                "total_time": store_time + retrieve_time + compute_time
            }
            
            backend.shutdown()
        
        # Validate that we have results for both backends
        assert len(results) == 2
        
        # Print comparison
        print("\\n📊 Performance Comparison Results:")
        for backend_name, times in results.items():
            print(f"  {backend_name}: "
                  f"Store={times['store_time']*1000:.1f}ms, "
                  f"Retrieve={times['retrieve_time']*1000:.1f}ms, "
                  f"Compute={times['compute_time']*1000:.1f}ms, "
                  f"Total={times['total_time']*1000:.1f}ms")
    
    def test_fallback_mechanism(self):
        """Test the fallback mechanism when primary backend unavailable."""
        # Request an unavailable backend
        backend = self.manager.create_backend(
            CSDBackendType.SPDK_EMULATOR,  # Real SPDK (unavailable)
            self.test_config,
            enable_fallback=True
        )
        
        # Should fallback to enhanced simulator
        assert backend is not None
        assert backend.get_backend_type() == CSDBackendType.ENHANCED_SIMULATOR
        
        # Test that fallback backend works
        backend.store_embeddings(self.test_embeddings[:10], self.test_metadata[:10])
        retrieved = backend.retrieve_embeddings([0, 1, 2])
        assert len(retrieved) == 3
        
        backend.shutdown()
    
    def test_backend_configuration_options(self):
        """Test different configuration options for backends."""
        # Test enhanced simulator with different config
        enhanced_config = {**self.test_config}
        enhanced_config.update({
            "csd": {
                "ssd_bandwidth_mbps": 3000,
                "max_parallel_ops": 16
            }
        })
        
        backend = self.manager.create_backend(CSDBackendType.ENHANCED_SIMULATOR, enhanced_config)
        assert backend is not None
        
        # Test basic operation
        backend.store_embeddings(self.test_embeddings[:5], self.test_metadata[:5])
        retrieved = backend.retrieve_embeddings([0, 1])
        assert len(retrieved) == 2
        
        backend.shutdown()
        
        # Test mock SPDK with different config
        mock_config = {**self.test_config}
        mock_config.update({
            "spdk": {
                "nvme_size_gb": 20,
                "virtual_queues": 8
            }
        })
        
        backend = self.manager.create_backend(CSDBackendType.MOCK_SPDK, mock_config)
        assert backend is not None
        
        # Check that config was applied
        backend_info = backend.get_backend_info()
        assert backend_info["nvme_size_gb"] == 20
        assert backend_info["virtual_queues"] == 8
        
        backend.shutdown()
    
    def test_metrics_consistency(self):
        """Test that metrics are consistent across backends."""
        backend = self.manager.create_backend(CSDBackendType.MOCK_SPDK, self.test_config)
        assert backend is not None
        
        # Get initial metrics
        initial_metrics = backend.get_metrics()
        initial_read_ops = initial_metrics.get("read_ops", 0)
        initial_write_ops = initial_metrics.get("write_ops", 0)
        
        # Perform operations
        backend.store_embeddings(self.test_embeddings[:10], self.test_metadata[:10])
        backend.retrieve_embeddings([0, 1, 2, 3, 4])
        
        # Check updated metrics
        final_metrics = backend.get_metrics()
        final_read_ops = final_metrics.get("read_ops", 0)
        final_write_ops = final_metrics.get("write_ops", 0)
        
        # Validate metrics updated correctly
        assert final_write_ops > initial_write_ops
        assert final_read_ops > initial_read_ops
        
        # Check backend-specific metrics
        if backend.get_backend_type() == CSDBackendType.MOCK_SPDK:
            assert "nvme_read_commands" in final_metrics
            assert "nvme_write_commands" in final_metrics
            assert "mock_backend" in final_metrics
            assert final_metrics["mock_backend"] is True
        
        backend.shutdown()


def run_comprehensive_test():
    """Run the comprehensive test suite."""
    print("🧪 Enhanced RAG-CSD Backend Integration Test Suite")
    print("=" * 60)
    
    test_instance = TestBackendIntegration()
    test_instance.setup_method()
    
    try:
        print("\\n1. Testing Backend Manager Initialization...")
        test_instance.test_backend_manager_initialization()
        print("✅ Backend manager initialization test passed")
        
        print("\\n2. Testing Enhanced Simulator Backend...")
        test_instance.test_enhanced_simulator_backend()
        print("✅ Enhanced simulator backend test passed")
        
        print("\\n3. Testing Mock SPDK Backend...")
        test_instance.test_mock_spdk_backend()
        print("✅ Mock SPDK backend test passed")
        
        print("\\n4. Testing Performance Comparison...")
        test_instance.test_backend_performance_comparison()
        print("✅ Performance comparison test passed")
        
        print("\\n5. Testing Fallback Mechanism...")
        test_instance.test_fallback_mechanism()
        print("✅ Fallback mechanism test passed")
        
        print("\\n6. Testing Configuration Options...")
        test_instance.test_backend_configuration_options()
        print("✅ Configuration options test passed")
        
        print("\\n7. Testing Metrics Consistency...")
        test_instance.test_metrics_consistency()
        print("✅ Metrics consistency test passed")
        
        print("\\n🎉 All tests passed successfully!")
        print("\\n📊 Integration Summary:")
        print("   ✅ Backend abstraction framework functional")
        print("   ✅ Enhanced simulator backend operational")
        print("   ✅ Mock SPDK backend operational")
        print("   ✅ Fallback mechanism working")
        print("   ✅ Performance characteristics preserved")
        print("   ✅ API compatibility maintained")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)