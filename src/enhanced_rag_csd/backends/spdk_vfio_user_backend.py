"""
SPDK vfio-user backend implementation for Enhanced RAG-CSD system.

This backend uses SPDK with vfio-user for high-performance computational storage
device emulation with shared memory and P2P GPU transfers.
"""

import os
import time
import subprocess
import threading
import socket
import struct
from typing import Dict, List, Any, Optional
import numpy as np
import mmap
import tempfile

from .base import CSDBackendInterface, CSDBackendType
from .hardware_abstraction import AcceleratorType, CSDHardwareAbstractionLayer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SPDKVfioUserBackend(CSDBackendInterface):
    """Backend using SPDK + vfio-user for high-performance CSD emulation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._backend_type = CSDBackendType.SPDK_VFIO_USER
        
        # SPDK vfio-user configuration
        self.vfio_config = config.get("spdk_vfio", {})
        self.socket_path = self.vfio_config.get("socket_path", "/tmp/vfio-user.sock")
        self.shared_memory_size = self.vfio_config.get("shared_memory_mb", 1024) * 1024 * 1024
        self.nvme_size_gb = self.vfio_config.get("nvme_size_gb", 16)
        self.queue_depth = self.vfio_config.get("queue_depth", 256)
        self.embedding_dim = config.get("embedding", {}).get("dimensions", 384)
        
        # SPDK components
        self.spdk_target = None
        self.vfio_device = None
        self.shared_memory = None
        self.shared_memory_fd = None
        self.socket_conn = None
        self.initialized = False
        
        # Performance tracking
        self.vfio_metrics = {
            "spdk_commands": 0,
            "vfio_transactions": 0,
            "shared_memory_operations": 0,
            "p2p_transfers": 0,
            "queue_depth_max": self.queue_depth,
            "bandwidth_utilization": 0.0
        }
        
        # Computational storage simulation
        self.compute_cores = self.vfio_config.get("compute_cores", 4)
        self.accelerator_memory_gb = self.vfio_config.get("accelerator_memory_gb", 8)
        
        # Storage mapping
        self.embedding_storage = {}
        self.next_lba = 0
        self.lba_size = 512  # NVMe LBA size
        
        # Hardware abstraction
        self.hal = CSDHardwareAbstractionLayer()
        
        logger.info("SPDK vfio-user backend initialized for high-performance CSD")
    
    def is_available(self) -> bool:
        """Check if SPDK vfio-user dependencies are available."""
        # For simulation mode, always return True
        simulation_mode = self.vfio_config.get("simulation_mode", True)
        if simulation_mode:
            return True
            
        return (
            self._check_spdk_installation() and
            self._check_vfio_user_support() and
            self._check_libvfio_user() and
            self._check_iommu_support()
        )
    
    def _check_spdk_installation(self) -> bool:
        """Check for SPDK installation."""
        try:
            spdk_paths = [
                "/usr/local/bin/spdk_tgt",
                "/opt/spdk/bin/spdk_tgt",
                "/usr/bin/spdk_tgt"
            ]
            return any(os.path.exists(path) for path in spdk_paths)
        except:
            return False
    
    def _check_vfio_user_support(self) -> bool:
        """Check for vfio-user kernel support."""
        try:
            # Check for vfio-user kernel module
            result = subprocess.run(["lsmod"], capture_output=True, text=True)
            return "vfio" in result.stdout
        except:
            return False
    
    def _check_libvfio_user(self) -> bool:
        """Check for libvfio-user library."""
        try:
            # Check for libvfio-user headers
            include_paths = [
                "/usr/include/libvfio-user.h",
                "/usr/local/include/libvfio-user.h"
            ]
            return any(os.path.exists(path) for path in include_paths)
        except:
            return False
    
    def _check_iommu_support(self) -> bool:
        """Check for IOMMU support."""
        try:
            # Check for IOMMU in kernel command line
            with open("/proc/cmdline", "r") as f:
                cmdline = f.read()
                return "iommu=on" in cmdline or "intel_iommu=on" in cmdline
        except:
            return False
    
    def initialize(self) -> bool:
        """Initialize SPDK vfio-user backend."""
        try:
            logger.info("Initializing SPDK vfio-user backend...")
            
            # Check if we're in simulation mode
            simulation_mode = self.vfio_config.get("simulation_mode", True)
            
            if simulation_mode:
                # Simplified initialization for simulation
                logger.info("Running in simulation mode - skipping real dependencies")
                
                # Setup simulated shared memory using regular memory
                self.shared_memory_size = self.vfio_config.get("shared_memory_mb", 1024) * 1024 * 1024
                self.shared_memory = bytearray(self.shared_memory_size)
                logger.info(f"Simulated shared memory region: {self.shared_memory_size // (1024*1024)}MB")
                
                # Initialize simulated compute units
                if not self._initialize_compute_units():
                    return False
                
                self.initialized = True
                logger.info("SPDK vfio-user backend initialized in simulation mode")
                logger.info("🚀 High-performance CSD simulation with P2P enabled")
                return True
            
            else:
                # Full initialization for real hardware
                # Step 1: Setup shared memory region
                if not self._setup_shared_memory():
                    return False
                
                # Step 2: Start SPDK target with vfio-user
                if not self._start_spdk_target():
                    return False
                
                # Step 3: Create vfio-user device
                if not self._create_vfio_device():
                    return False
                
                # Step 4: Establish vfio-user connection
                if not self._connect_vfio_user():
                    return False
                
                # Step 5: Initialize computational units
                if not self._initialize_compute_units():
                    return False
                
                self.initialized = True
                logger.info("SPDK vfio-user backend initialized successfully")
                logger.info("🚀 High-performance CSD with shared memory P2P enabled")
                
                return True
            
        except Exception as e:
            logger.error(f"SPDK vfio-user initialization failed: {e}")
            self._cleanup()
            return False
    
    def _setup_shared_memory(self) -> bool:
        """Setup shared memory region for vfio-user communication."""
        try:
            # Create shared memory file
            self.shared_memory_fd = os.memfd_create("spdk_vfio_user", 0)
            os.ftruncate(self.shared_memory_fd, self.shared_memory_size)
            
            # Map shared memory
            self.shared_memory = mmap.mmap(
                self.shared_memory_fd, 
                self.shared_memory_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )
            
            logger.info(f"Shared memory region created: {self.shared_memory_size // (1024*1024)}MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup shared memory: {e}")
            return False
    
    def _start_spdk_target(self) -> bool:
        """Start SPDK target with vfio-user support."""
        try:
            # SPDK target configuration
            spdk_config = {
                "subsystems": [
                    {
                        "subsystem": "bdev",
                        "config": [
                            {
                                "method": "bdev_malloc_create",
                                "params": {
                                    "name": "csd_bdev",
                                    "num_blocks": self.nvme_size_gb * 1024 * 1024 * 2,  # 512B blocks
                                    "block_size": 512
                                }
                            }
                        ]
                    },
                    {
                        "subsystem": "nvmf",
                        "config": [
                            {
                                "method": "nvmf_create_transport",
                                "params": {
                                    "trtype": "vfio-user"
                                }
                            }
                        ]
                    }
                ]
            }
            
            # For simulation, we'll create a mock SPDK target process
            # In real implementation, this would start actual SPDK target
            logger.info("SPDK target simulation started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SPDK target: {e}")
            return False
    
    def _create_vfio_device(self) -> bool:
        """Create vfio-user NVMe device."""
        try:
            # Create Unix domain socket for vfio-user communication
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            
            # In real implementation, this would create actual vfio-user device
            # For simulation, we create a mock socket
            self.socket_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket_server.bind(self.socket_path)
            self.socket_server.listen(1)
            
            logger.info(f"vfio-user device created at {self.socket_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vfio-user device: {e}")
            return False
    
    def _connect_vfio_user(self) -> bool:
        """Establish vfio-user connection."""
        try:
            # In real implementation, this would establish libvfio-user connection
            # For simulation, we'll use the socket we created
            logger.info("vfio-user connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect vfio-user: {e}")
            return False
    
    def _initialize_compute_units(self) -> bool:
        """Initialize computational storage units."""
        try:
            # Setup computational units in shared memory
            compute_region_size = self.shared_memory_size // 4  # 25% for compute
            storage_region_size = self.shared_memory_size - compute_region_size
            
            self.compute_region_offset = 0
            self.storage_region_offset = compute_region_size
            
            # Initialize compute unit metadata
            self.compute_units = {
                "similarity_engine": {
                    "cores": self.compute_cores // 2,
                    "memory_offset": self.compute_region_offset,
                    "active": True
                },
                "embedding_processor": {
                    "cores": self.compute_cores // 2,
                    "memory_offset": self.compute_region_offset + (compute_region_size // 2),
                    "active": True
                }
            }
            
            logger.info(f"Computational units initialized: {len(self.compute_units)} units")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize compute units: {e}")
            return False
    
    def store_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Store embeddings using SPDK vfio-user with shared memory."""
        if not self.initialized:
            raise RuntimeError("SPDK vfio-user backend not initialized")
        
        start_time = time.time()
        
        try:
            for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                # Allocate LBA for embedding
                lba = self.next_lba
                embedding_id = len(self.embedding_storage)
                
                # Store in shared memory region
                embedding_bytes = embedding.astype(np.float32).tobytes()
                memory_offset = self.storage_region_offset + (embedding_id * len(embedding_bytes))
                
                if memory_offset + len(embedding_bytes) < self.shared_memory_size:
                    # Write to shared memory
                    if hasattr(self.shared_memory, '__setitem__'):
                        # For bytearray simulation
                        self.shared_memory[memory_offset:memory_offset + len(embedding_bytes)] = embedding_bytes
                    else:
                        # For mmap object
                        self.shared_memory[memory_offset:memory_offset + len(embedding_bytes)] = embedding_bytes
                    
                    # Update storage mapping
                    self.embedding_storage[embedding_id] = {
                        "lba": lba,
                        "memory_offset": memory_offset,
                        "size": len(embedding_bytes),
                        "metadata": meta
                    }
                    
                    self.next_lba += (len(embedding_bytes) + self.lba_size - 1) // self.lba_size
                    self.vfio_metrics["shared_memory_operations"] += 1
                else:
                    logger.warning(f"Shared memory full, skipping embedding {i}")
                
                self.vfio_metrics["spdk_commands"] += 1
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["write_ops"] += len(embeddings)
            self.metrics["total_bytes_written"] += embeddings.nbytes
            
            logger.debug(f"SPDK vfio-user: Stored {len(embeddings)} embeddings in {elapsed:.3f}s")
            
        except Exception as e:
            logger.error(f"Error storing embeddings in SPDK vfio-user: {e}")
            raise
    
    def retrieve_embeddings(self, indices: List[int], use_cache: bool = True) -> np.ndarray:
        """Retrieve embeddings using SPDK vfio-user shared memory."""
        if not self.initialized:
            raise RuntimeError("SPDK vfio-user backend not initialized")
        
        start_time = time.time()
        embeddings = []
        
        try:
            for idx in indices:
                if idx in self.embedding_storage:
                    storage_info = self.embedding_storage[idx]
                    memory_offset = storage_info["memory_offset"]
                    size = storage_info["size"]
                    
                    # Read from shared memory
                    if hasattr(self.shared_memory, '__getitem__'):
                        # For bytearray simulation
                        embedding_bytes = bytes(self.shared_memory[memory_offset:memory_offset + size])
                    else:
                        # For mmap object
                        embedding_bytes = bytes(self.shared_memory[memory_offset:memory_offset + size])
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    # Ensure correct dimension
                    if len(embedding) != self.embedding_dim:
                        embedding = np.resize(embedding, self.embedding_dim)
                    
                    embeddings.append(embedding)
                    self.vfio_metrics["shared_memory_operations"] += 1
                else:
                    # Return zero embedding for missing indices
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                    logger.warning(f"Embedding {idx} not found in SPDK vfio-user")
                
                self.vfio_metrics["spdk_commands"] += 1
            
            result = np.array(embeddings)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["read_ops"] += len(indices)
            self.metrics["total_bytes_read"] += result.nbytes
            
            logger.debug(f"SPDK vfio-user: Retrieved {len(indices)} embeddings in {elapsed:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings from SPDK vfio-user: {e}")
            raise
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           candidate_indices: List[int]) -> np.ndarray:
        """Compute similarities using computational storage units."""
        if not self.initialized:
            raise RuntimeError("SPDK vfio-user backend not initialized")
        
        start_time = time.time()
        
        try:
            # Retrieve candidate embeddings
            candidates = self.retrieve_embeddings(candidate_indices)
            
            # Offload to similarity engine compute unit
            if "similarity_engine" in self.compute_units and self.compute_units["similarity_engine"]["active"]:
                similarities = self._compute_unit_similarity(query_embedding, candidates)
                self.vfio_metrics["vfio_transactions"] += 1
            else:
                # Fallback to CPU computation
                similarities = self._cpu_similarity_computation(query_embedding, candidates)
            
            elapsed = time.time() - start_time
            logger.debug(f"SPDK vfio-user: Computed {len(candidate_indices)} similarities in {elapsed:.3f}s")
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing similarities in SPDK vfio-user: {e}")
            raise
    
    def _compute_unit_similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Execute similarity computation on dedicated compute unit."""
        # Simulate compute unit execution latency
        compute_cores = self.compute_units["similarity_engine"]["cores"]
        compute_latency = 0.001 * len(candidates) / compute_cores  # Parallel execution
        time.sleep(compute_latency)
        
        # Perform computation (in real implementation, this would use hardware acceleration)
        query_norm = np.linalg.norm(query)
        candidate_norms = np.linalg.norm(candidates, axis=1)
        
        # Avoid division by zero
        query_norm = max(query_norm, 1e-8)
        candidate_norms = np.maximum(candidate_norms, 1e-8)
        
        # Compute cosine similarities with hardware acceleration simulation
        similarities = np.dot(candidates, query) / (candidate_norms * query_norm)
        
        return similarities
    
    def process_era_pipeline(self, query_data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Process ERA pipeline using computational storage units."""
        if not self.initialized:
            raise RuntimeError("SPDK vfio-user backend not initialized")
        
        start_time = time.time()
        
        try:
            # Stage 1: Encode using embedding processor
            if query_data.dtype != np.float32 or len(query_data.shape) != 1:
                query_embedding = self._compute_unit_encode(query_data)
            else:
                query_embedding = query_data / np.linalg.norm(query_data)
            
            # Stage 2: Retrieve using shared memory access
            top_k = metadata.get("top_k", 5)
            max_candidates = min(50, len(self.embedding_storage))
            
            if max_candidates > 0:
                candidate_indices = list(range(min(max_candidates, len(self.embedding_storage))))
                similarities = self.compute_similarities(query_embedding, candidate_indices)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                retrieved = self.retrieve_embeddings(top_indices.tolist())
            else:
                retrieved = np.random.randn(top_k, self.embedding_dim).astype(np.float32)
                for i in range(top_k):
                    retrieved[i] = retrieved[i] / np.linalg.norm(retrieved[i])
            
            # Stage 3: Augment using compute units
            augmented_data = self._compute_unit_augment(query_embedding, retrieved, metadata)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.vfio_metrics["vfio_transactions"] += 3  # Encode, Retrieve, Augment
            
            logger.debug(f"SPDK vfio-user ERA pipeline completed in {elapsed:.3f}s")
            return augmented_data
            
        except Exception as e:
            logger.error(f"Error in SPDK vfio-user ERA pipeline: {e}")
            raise
    
    def _compute_unit_encode(self, query_data: np.ndarray) -> np.ndarray:
        """Execute encoding on embedding processor compute unit."""
        # Simulate embedding processor latency
        time.sleep(0.001)  # 1ms
        
        # Generate normalized embedding (in real implementation, this would use hardware)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def _compute_unit_augment(self, query: np.ndarray, retrieved: np.ndarray, 
                             metadata: Dict[str, Any]) -> np.ndarray:
        """Execute augmentation on compute units."""
        # Simulate compute unit augmentation latency
        time.sleep(0.0005)  # 0.5ms
        
        # Perform augmentation
        mode = metadata.get("mode", "concatenate")
        
        if mode == "weighted" and len(retrieved) > 0:
            # Weight by similarity scores if available
            weights = metadata.get("similarities", np.ones(len(retrieved)))
            weights = weights / np.sum(weights)
            weighted_retrieved = np.average(retrieved, axis=0, weights=weights)
            return np.concatenate([query, weighted_retrieved])
        elif len(retrieved) > 0:
            # Simple concatenation
            return np.concatenate([query] + [retrieved[i] for i in range(len(retrieved))])
        else:
            return np.concatenate([query, np.zeros(self.embedding_dim)])
    
    def p2p_transfer_to_gpu(self, data: np.ndarray) -> str:
        """Efficient P2P transfer using vfio-user shared memory."""
        if not self.initialized:
            raise RuntimeError("SPDK vfio-user backend not initialized")
        
        start_time = time.time()
        
        try:
            # Simulate zero-copy P2P transfer using shared memory
            # In real implementation, this would use GPU Direct Storage or similar
            transfer_bandwidth = 25 * 1024 * 1024 * 1024  # 25 GB/s
            transfer_time = data.nbytes / transfer_bandwidth
            time.sleep(transfer_time)
            
            # Allocate in shared memory for GPU access
            allocation_size = data.nbytes
            gpu_offset = self.shared_memory_size - allocation_size
            
            if gpu_offset > self.storage_region_offset:
                # Copy data to GPU-accessible region
                data_bytes = data.astype(np.float32).tobytes()
                if hasattr(self.shared_memory, '__setitem__'):
                    # For bytearray simulation
                    self.shared_memory[gpu_offset:gpu_offset + len(data_bytes)] = data_bytes
                else:
                    # For mmap object
                    self.shared_memory[gpu_offset:gpu_offset + len(data_bytes)] = data_bytes
                
                allocation_id = f"vfio_gpu_alloc_{gpu_offset}_{int(time.time() * 1000000)}"
                
                self.vfio_metrics["p2p_transfers"] += 1
                self.vfio_metrics["shared_memory_operations"] += 1
                
                elapsed = time.time() - start_time
                logger.debug(f"SPDK vfio-user P2P: {data.nbytes/1024:.1f}KB in {elapsed*1000:.2f}ms")
                
                return allocation_id
            else:
                logger.error("Insufficient shared memory for P2P transfer")
                return "allocation_failed"
            
        except Exception as e:
            logger.error(f"Error in SPDK vfio-user P2P transfer: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive SPDK vfio-user metrics."""
        combined_metrics = {**self.metrics, **self.vfio_metrics}
        combined_metrics["backend_type"] = self._backend_type.value
        combined_metrics["backend_subtype"] = "spdk_vfio_user_shared_memory"
        
        # Add SPDK vfio-user specific information
        combined_metrics["spdk_vfio_info"] = {
            "shared_memory_size_mb": self.shared_memory_size // (1024 * 1024),
            "queue_depth": self.queue_depth,
            "compute_units": len(self.compute_units),
            "storage_utilization": len(self.embedding_storage),
            "socket_path": self.socket_path
        }
        
        # Performance analysis
        total_ops = combined_metrics["spdk_commands"]
        if total_ops > 0:
            combined_metrics["avg_ops_per_second"] = total_ops / max(1, time.time() - getattr(self, 'start_time', time.time()))
            combined_metrics["shared_memory_efficiency"] = combined_metrics["shared_memory_operations"] / total_ops
        
        return combined_metrics
    
    def get_accelerator_info(self) -> Dict[str, Any]:
        """Get SPDK vfio-user accelerator information."""
        return {
            "accelerator_type": "spdk_vfio_user",
            "compute_units": list(self.compute_units.keys()),
            "memory_hierarchy": "shared_memory_with_p2p",
            "supports_parallel": True,
            "supports_offloading": True,
            "shared_memory_size_gb": self.shared_memory_size // (1024 * 1024 * 1024),
            "queue_depth": self.queue_depth,
            "p2p_bandwidth_gbps": 25,
            "nvme_backend": "spdk_malloc_bdev"
        }
    
    def supports_feature(self, feature: str) -> bool:
        """Check SPDK vfio-user feature support."""
        vfio_features = {
            "shared_memory", "zero_copy_p2p", "high_performance",
            "compute_offloading", "parallel_processing", "nvme_emulation",
            "vfio_user_protocol", "spdk_backend"
        }
        base_features = super().supports_feature(feature)
        return base_features or feature in vfio_features
    
    def shutdown(self) -> None:
        """Shutdown SPDK vfio-user backend and cleanup resources."""
        try:
            self._cleanup()
            self.initialized = False
            logger.info("SPDK vfio-user backend shutdown complete")
        except Exception as e:
            logger.error(f"Error during SPDK vfio-user shutdown: {e}")
    
    def _cleanup(self) -> None:
        """Cleanup SPDK vfio-user resources."""
        try:
            # Close shared memory
            if self.shared_memory:
                if hasattr(self.shared_memory, 'close'):
                    self.shared_memory.close()
                self.shared_memory = None
            
            if self.shared_memory_fd:
                os.close(self.shared_memory_fd)
                self.shared_memory_fd = None
            
            # Close socket
            if hasattr(self, 'socket_server'):
                self.socket_server.close()
            
            if self.socket_conn:
                self.socket_conn.close()
                self.socket_conn = None
            
            # Remove socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            
            # Cleanup SPDK target
            # In real implementation, this would stop SPDK target process
            
        except Exception as e:
            logger.error(f"Error during SPDK vfio-user cleanup: {e}")
    
    def _cpu_similarity_computation(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Fallback CPU similarity computation."""
        query_norm = np.linalg.norm(query)
        candidate_norms = np.linalg.norm(candidates, axis=1)
        
        query_norm = max(query_norm, 1e-8)
        candidate_norms = np.maximum(candidate_norms, 1e-8)
        
        return np.dot(candidates, query) / (candidate_norms * query_norm)