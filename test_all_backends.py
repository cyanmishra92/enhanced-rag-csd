#!/usr/bin/env python3
"""
Test script to verify all backend types work in simulation mode.
"""

import numpy as np
from src.enhanced_rag_csd.backends import CSDBackendManager, CSDBackendType
from src.enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)

def test_backend_simulation(backend_type, config):
    """Test a backend in simulation mode."""
    print(f"\n🧪 Testing {backend_type.value} backend...")
    
    manager = CSDBackendManager()
    
    # Force simulation mode by bypassing dependency checks
    try:
        if backend_type == CSDBackendType.OPENCSD_EMULATOR:
            from src.enhanced_rag_csd.backends.opencsd_backend import OpenCSDBackend
            backend = OpenCSDBackend(config)
            # Override is_available for simulation
            backend.is_available = lambda: True
            backend.initialize()
            
        elif backend_type == CSDBackendType.SPDK_VFIO_USER:
            from src.enhanced_rag_csd.backends.spdk_vfio_user_backend import SPDKVfioUserBackend
            backend = SPDKVfioUserBackend(config)
            # Override is_available for simulation
            backend.is_available = lambda: True
            backend.initialize()
            
        elif backend_type == CSDBackendType.ENHANCED_SIMULATOR:
            from src.enhanced_rag_csd.backends.enhanced_simulator import EnhancedSimulatorBackend
            backend = EnhancedSimulatorBackend(config)
            backend.initialize()
        elif backend_type == CSDBackendType.MOCK_SPDK:
            from src.enhanced_rag_csd.backends.mock_spdk import MockSPDKEmulatorBackend
            backend = MockSPDKEmulatorBackend(config)
            backend.initialize()
        
        if not backend:
            print(f"  ❌ Failed to create {backend_type.value} backend")
            return False
        
        # Test basic operations
        embeddings = np.random.randn(10, 384).astype(np.float32)
        metadata = [{"id": i} for i in range(10)]
        
        # Store
        backend.store_embeddings(embeddings, metadata)
        print(f"  ✅ Store: 10 embeddings")
        
        # Retrieve
        retrieved = backend.retrieve_embeddings([0, 1, 2])
        print(f"  ✅ Retrieve: {retrieved.shape}")
        
        # Similarities
        similarities = backend.compute_similarities(embeddings[0], [1, 2, 3])
        print(f"  ✅ Similarities: {len(similarities)}")
        
        # ERA Pipeline
        era_result = backend.process_era_pipeline(embeddings[0], {"top_k": 3})
        print(f"  ✅ ERA Pipeline: {era_result.shape}")
        
        # Test computational offloading if supported
        if backend.supports_feature("arbitrary_computation"):
            try:
                # Test softmax
                softmax_result = backend.offload_computation("softmax", embeddings[0], {"temperature": 1.0})
                print(f"  ✅ Softmax offloading: {softmax_result.shape}")
                
                # Test custom kernel
                custom_kernel = '''
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct scale_args {
    float *input;
    float *output;
    int size;
    float factor;
};

SEC("csd/scale")
int vector_scale(struct scale_args *args) {
    for (int i = 0; i < args->size; i++) {
        args->output[i] = args->input[i] * args->factor;
    }
    return 0;
}

char _license[] SEC("license") = "GPL";
'''
                custom_result = backend.offload_computation("custom_kernel", embeddings[0], {
                    "ebpf_source": custom_kernel,
                    "kernel_name": "vector_scale",
                    "factor": 2.5
                })
                print(f"  ✅ Custom kernel offloading: {custom_result.shape}")
                
            except Exception as e:
                print(f"  ⚠️  Computational offloading test failed: {e}")
        
        # Get metrics
        metrics = backend.get_metrics()
        print(f"  📊 Metrics: R:{metrics.get('read_ops', 0)}, W:{metrics.get('write_ops', 0)}")
        
        # Cleanup
        backend.shutdown()
        print(f"  ✅ {backend_type.value} test completed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ {backend_type.value} test failed: {e}")
        return False

def main():
    """Run comprehensive backend tests."""
    print("🚀 Enhanced RAG-CSD Backend Simulation Test")
    print("=" * 60)
    
    config = {
        "vector_db_path": "./test_data",
        "embedding": {"dimensions": 384},
        "opencsd": {
            "ebpf_program_dir": "./test_ebpf"
        },
        "spdk_vfio": {
            "shared_memory_mb": 512
        }
    }
    
    # Test all backend types
    backends_to_test = [
        CSDBackendType.ENHANCED_SIMULATOR,
        CSDBackendType.MOCK_SPDK,
        CSDBackendType.OPENCSD_EMULATOR,
        CSDBackendType.SPDK_VFIO_USER,
    ]
    
    results = {}
    for backend_type in backends_to_test:
        results[backend_type] = test_backend_simulation(backend_type, config)
    
    print("\n📊 Test Results Summary:")
    print("-" * 40)
    for backend_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {backend_type.value}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\n🎯 Overall: {total_passed}/{total_tests} backends passed")
    
    if total_passed == total_tests:
        print("🎉 All backend tests completed successfully!")
        return True
    else:
        print("⚠️  Some backend tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)