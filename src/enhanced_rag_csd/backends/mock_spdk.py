"""
Mock SPDK emulator backend for testing the integration framework.

This provides a simplified version that simulates SPDK behavior without
requiring actual SPDK installation. Used for testing and development.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .base import CSDBackendInterface, CSDBackendType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SPDKCacheHierarchy:
    """3-level cache hierarchy for SPDK backend with realistic CSD behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        # Cache configuration
        self.l1_size_mb = config.get("l1_cache_mb", 64)     # L1: Fast SRAM-like
        self.l2_size_mb = config.get("l2_cache_mb", 512)    # L2: Fast NVMe
        self.l3_size_mb = config.get("l3_cache_mb", 2048)   # L3: Slower NVMe
        
        self.embedding_dim = config.get("embedding_dim", 384)
        self.embedding_size = self.embedding_dim * 4  # float32
        
        # Calculate cache capacities
        self.l1_capacity = (self.l1_size_mb * 1024 * 1024) // self.embedding_size
        self.l2_capacity = (self.l2_size_mb * 1024 * 1024) // self.embedding_size
        self.l3_capacity = (self.l3_size_mb * 1024 * 1024) // self.embedding_size
        
        # Cache storage with LRU ordering
        self.l1_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.l2_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.l3_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        
        # Access statistics
        self.access_counts = {}
        self.cache_stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0,
            "total_accesses": 0,
            "promotions": 0, "demotions": 0
        }
        
        # Cache latencies (microseconds)
        self.l1_latency_us = 1     # Ultra-fast on-chip cache
        self.l2_latency_us = 10    # Fast NVMe region
        self.l3_latency_us = 50    # Standard NVMe region
        self.nvme_latency_us = 100 # Main NVMe storage
        
        self.lock = threading.Lock()
        
        logger.info(f"SPDK Cache: L1={self.l1_capacity} L2={self.l2_capacity} L3={self.l3_capacity} embeddings")
    
    def get_embedding(self, index: int, nvme_read_func) -> Tuple[np.ndarray, str]:
        """Get embedding with cache hierarchy lookup."""
        with self.lock:
            self.cache_stats["total_accesses"] += 1
            self.access_counts[index] = self.access_counts.get(index, 0) + 1
            
            # Check L1 cache first
            if index in self.l1_cache:
                time.sleep(self.l1_latency_us / 1_000_000)
                self.cache_stats["l1_hits"] += 1
                embedding = self.l1_cache[index]
                # Move to end (most recently used)
                self.l1_cache.move_to_end(index)
                return embedding.copy(), "l1_hit"
            
            # Check L2 cache
            if index in self.l2_cache:
                time.sleep(self.l2_latency_us / 1_000_000)
                self.cache_stats["l2_hits"] += 1
                embedding = self.l2_cache[index]
                # Promote to L1
                self._promote_to_l1(index, embedding)
                return embedding.copy(), "l2_hit"
            
            # Check L3 cache
            if index in self.l3_cache:
                time.sleep(self.l3_latency_us / 1_000_000)
                self.cache_stats["l3_hits"] += 1
                embedding = self.l3_cache[index]
                # Promote to L2 and possibly L1
                self._promote_to_l2(index, embedding)
                return embedding.copy(), "l3_hit"
            
            # Cache miss - read from NVMe
            time.sleep(self.nvme_latency_us / 1_000_000)
            self.cache_stats["l1_misses"] += 1
            self.cache_stats["l2_misses"] += 1
            self.cache_stats["l3_misses"] += 1
            
            # Read from main storage
            embedding = nvme_read_func(index)
            
            # Store in appropriate cache level based on access pattern
            if self.access_counts.get(index, 0) > 3:
                self._promote_to_l1(index, embedding)
            elif self.access_counts.get(index, 0) > 1:
                self._promote_to_l2(index, embedding)
            else:
                self._promote_to_l3(index, embedding)
            
            return embedding.copy(), "cache_miss"
    
    def put_embedding(self, index: int, embedding: np.ndarray) -> None:
        """Store embedding in cache hierarchy."""
        with self.lock:
            # Always store new embeddings in L1 for fast access
            self._promote_to_l1(index, embedding)
    
    def _promote_to_l1(self, index: int, embedding: np.ndarray) -> None:
        """Promote embedding to L1 cache."""
        # Remove from lower levels if present
        self.l2_cache.pop(index, None)
        self.l3_cache.pop(index, None)
        
        # Add to L1
        self.l1_cache[index] = embedding.copy()
        
        # Evict if over capacity
        if len(self.l1_cache) > self.l1_capacity:
            evicted_idx, evicted_embedding = self.l1_cache.popitem(last=False)
            # Demote to L2
            self._promote_to_l2(evicted_idx, evicted_embedding)
            self.cache_stats["demotions"] += 1
        
        self.cache_stats["promotions"] += 1
    
    def _promote_to_l2(self, index: int, embedding: np.ndarray) -> None:
        """Promote embedding to L2 cache."""
        # Remove from L3 if present
        self.l3_cache.pop(index, None)
        
        # Add to L2
        self.l2_cache[index] = embedding.copy()
        
        # Evict if over capacity
        if len(self.l2_cache) > self.l2_capacity:
            evicted_idx, evicted_embedding = self.l2_cache.popitem(last=False)
            # Demote to L3
            self._promote_to_l3(evicted_idx, evicted_embedding)
            self.cache_stats["demotions"] += 1
    
    def _promote_to_l3(self, index: int, embedding: np.ndarray) -> None:
        """Promote embedding to L3 cache."""
        # Add to L3
        self.l3_cache[index] = embedding.copy()
        
        # Evict if over capacity (LRU)
        if len(self.l3_cache) > self.l3_capacity:
            self.l3_cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            total_hits = (self.cache_stats["l1_hits"] + 
                         self.cache_stats["l2_hits"] + 
                         self.cache_stats["l3_hits"])
            total_accesses = self.cache_stats["total_accesses"]
            hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0
            
            return {
                **self.cache_stats,
                "cache_hit_rate": hit_rate,
                "l1_utilization": len(self.l1_cache) / self.l1_capacity,
                "l2_utilization": len(self.l2_cache) / self.l2_capacity,
                "l3_utilization": len(self.l3_cache) / self.l3_capacity,
                "total_cached_embeddings": len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
            }


class MockNVMeController:
    """Mock NVMe controller that simulates real NVMe command processing."""
    
    def __init__(self, size_gb: int = 10):
        self.size_gb = size_gb
        self.size_bytes = size_gb * 1024 * 1024 * 1024
        self.block_size = 512
        self.total_blocks = self.size_bytes // self.block_size
        
        # Simulated storage (in-memory for testing)
        self.storage = {}
        self.command_count = 0
        self.lock = threading.Lock()
        
        # Performance characteristics
        self.read_latency_us = 100  # 100 microseconds
        self.write_latency_us = 200  # 200 microseconds
        self.bandwidth_mbps = 3500  # NVMe SSD bandwidth
        
        logger.info(f"Mock NVMe controller initialized: {size_gb}GB, {self.total_blocks} blocks")
    
    def read_blocks(self, lba: int, num_blocks: int) -> bytes:
        """Simulate NVMe read command."""
        with self.lock:
            self.command_count += 1
            
            # Simulate read latency
            data_size = num_blocks * self.block_size
            latency = self.read_latency_us / 1_000_000  # Convert to seconds
            bandwidth_delay = data_size / (self.bandwidth_mbps * 1024 * 1024)
            total_delay = latency + bandwidth_delay
            time.sleep(total_delay)
            
            # Read data from simulated storage
            data = bytearray(data_size)
            for i in range(num_blocks):
                block_lba = lba + i
                if block_lba in self.storage:
                    start_offset = i * self.block_size
                    end_offset = start_offset + self.block_size
                    data[start_offset:end_offset] = self.storage[block_lba]
            
            logger.debug(f"Mock NVMe read: LBA={lba}, blocks={num_blocks}, "
                        f"latency={total_delay*1000:.2f}ms")
            
            return bytes(data)
    
    def write_blocks(self, lba: int, data: bytes) -> bool:
        """Simulate NVMe write command."""
        with self.lock:
            self.command_count += 1
            
            num_blocks = len(data) // self.block_size
            if len(data) % self.block_size != 0:
                num_blocks += 1
                # Pad data to block boundary
                data = data + b'\x00' * (num_blocks * self.block_size - len(data))
            
            # Simulate write latency
            latency = self.write_latency_us / 1_000_000
            bandwidth_delay = len(data) / (self.bandwidth_mbps * 1024 * 1024)
            total_delay = latency + bandwidth_delay
            time.sleep(total_delay)
            
            # Write data to simulated storage
            for i in range(num_blocks):
                block_lba = lba + i
                start_offset = i * self.block_size
                end_offset = start_offset + self.block_size
                self.storage[block_lba] = data[start_offset:end_offset]
            
            logger.debug(f"Mock NVMe write: LBA={lba}, blocks={num_blocks}, "
                        f"latency={total_delay*1000:.2f}ms")
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "total_commands": self.command_count,
            "storage_blocks_used": len(self.storage),
            "storage_utilization": len(self.storage) / self.total_blocks,
            "read_latency_us": self.read_latency_us,
            "write_latency_us": self.write_latency_us,
            "bandwidth_mbps": self.bandwidth_mbps
        }


class MockSPDKEmulatorBackend(CSDBackendInterface):
    """Mock SPDK emulator backend for testing without SPDK installation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._backend_type = CSDBackendType.MOCK_SPDK
        
        # Mock SPDK configuration
        self.spdk_config = config.get("spdk", {})
        self.nvme_size_gb = self.spdk_config.get("nvme_size_gb", 10)
        self.virtual_queues = self.spdk_config.get("virtual_queues", 8)
        self.embedding_dim = config.get("embedding", {}).get("dimensions", 384)
        
        # Mock components
        self.nvme_controller = None
        self.cache_hierarchy = None
        self.thread_pool = None
        self.initialized = False
        
        # Mock metrics
        self.mock_metrics = {
            "nvme_read_commands": 0,
            "nvme_write_commands": 0,
            "total_nvme_commands": 0,
            "emulation_overhead_ms": 0.0,
            "p2p_transfers": 0,
            "similarity_computations": 0,
            "era_pipeline_executions": 0,
            "parallel_operations": 0,
            "mock_backend": True
        }
        
        # Performance configuration
        self.max_parallel_ops = config.get("csd", {}).get("max_parallel_ops", 8)
        self.compute_latency_ms = config.get("csd", {}).get("compute_latency_ms", 0.1)
        
        # ML computational offloading capabilities
        self.ml_compute_units = config.get("csd", {}).get("ml_compute_units", 6)
        self.ml_accelerator_freq = config.get("csd", {}).get("ml_frequency_mhz", 600)  # MHz
        self.vector_processing_units = config.get("csd", {}).get("vector_units", 2)
        
        # ML computation metrics
        self.ml_metrics = {
            "ml_encode_ops": 0,
            "ml_retrieve_ops": 0,
            "ml_augment_ops": 0,
            "vector_operations": 0,
            "total_ml_compute_time": 0.0,
            "offloaded_computations": 0,
            "cache_accelerated_ops": 0
        }
        
        # Storage mapping
        self.embedding_to_lba = {}  # embedding_index -> LBA
        self.next_lba = 0
        self.embedding_size_blocks = (self.embedding_dim * 4) // 512 + 1  # float32, round up
    
    def initialize(self) -> bool:
        """Initialize the mock SPDK emulator backend."""
        try:
            logger.info("Initializing Mock SPDK Emulator Backend")
            
            # Create mock NVMe controller
            self.nvme_controller = MockNVMeController(self.nvme_size_gb)
            
            # Initialize cache hierarchy
            cache_config = {
                "embedding_dim": self.embedding_dim,
                **self.config.get("cache", {})
            }
            self.cache_hierarchy = SPDKCacheHierarchy(cache_config)
            
            # Initialize thread pool for parallel operations
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel_ops)
            
            self.initialized = True
            logger.info("Mock SPDK emulator backend initialized successfully")
            logger.info(f"Cache hierarchy: L1={self.cache_hierarchy.l1_capacity}, "
                       f"L2={self.cache_hierarchy.l2_capacity}, "
                       f"L3={self.cache_hierarchy.l3_capacity} embeddings")
            logger.info("⚠️  Note: This is a mock backend for testing. "
                       "Install SPDK for real CSD emulation.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize mock SPDK backend: {e}")
            return False
    
    def store_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Store embeddings using mock NVMe write commands."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            # Convert embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            # Allocate LBA for this embedding
            lba = self.next_lba
            embedding_index = len(self.embedding_to_lba)
            self.embedding_to_lba[embedding_index] = lba
            
            # Write to mock NVMe
            success = self.nvme_controller.write_blocks(lba, embedding_bytes)
            if success:
                self.next_lba += self.embedding_size_blocks
                self.mock_metrics["nvme_write_commands"] += 1
                
                # Store in cache hierarchy for future fast access
                self.cache_hierarchy.put_embedding(embedding_index, embedding)
            else:
                logger.error(f"Failed to write embedding {i}")
        
        # Update metrics
        elapsed = time.time() - start_time
        self.metrics["write_ops"] += len(embeddings)
        self.metrics["total_bytes_written"] += embeddings.nbytes
        self.mock_metrics["total_nvme_commands"] = (
            self.mock_metrics["nvme_read_commands"] + 
            self.mock_metrics["nvme_write_commands"]
        )
        self.mock_metrics["emulation_overhead_ms"] += elapsed * 1000
        
        logger.debug(f"Mock SPDK: Stored {len(embeddings)} embeddings in {elapsed:.3f}s")
    
    def retrieve_embeddings(self, indices: List[int], use_cache: bool = True) -> np.ndarray:
        """Retrieve embeddings using cache hierarchy and mock NVMe reads."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        embeddings = []
        cache_hits = 0
        
        def _read_from_nvme(idx: int) -> np.ndarray:
            """Helper function to read embedding from NVMe storage."""
            if idx not in self.embedding_to_lba:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            lba = self.embedding_to_lba[idx]
            data = self.nvme_controller.read_blocks(lba, self.embedding_size_blocks)
            
            # Convert back to embedding
            embedding_size = self.embedding_dim * 4  # float32
            embedding_data = data[:embedding_size]
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
            
            # Handle size mismatch due to block alignment
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            elif len(embedding) < self.embedding_dim:
                # Pad with zeros if needed
                padded = np.zeros(self.embedding_dim, dtype=np.float32)
                padded[:len(embedding)] = embedding
                embedding = padded
            
            self.mock_metrics["nvme_read_commands"] += 1
            return embedding
        
        for idx in indices:
            if use_cache:
                # Try cache hierarchy first
                embedding, cache_result = self.cache_hierarchy.get_embedding(idx, _read_from_nvme)
                if cache_result != "cache_miss":
                    cache_hits += 1
            else:
                # Direct NVMe read
                embedding = _read_from_nvme(idx)
            
            embeddings.append(embedding)
        
        result = np.array(embeddings)
        
        # Update metrics
        elapsed = time.time() - start_time
        self.metrics["read_ops"] += len(indices)
        self.metrics["total_bytes_read"] += result.nbytes
        self.metrics["cache_hits"] += cache_hits
        self.metrics["cache_misses"] += len(indices) - cache_hits
        
        # Calculate cache hit rate
        total_reads = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        self.metrics["cache_hit_rate"] = self.metrics["cache_hits"] / total_reads if total_reads > 0 else 0.0
        
        self.mock_metrics["total_nvme_commands"] = (
            self.mock_metrics["nvme_read_commands"] + 
            self.mock_metrics["nvme_write_commands"]
        )
        self.mock_metrics["emulation_overhead_ms"] += elapsed * 1000
        
        logger.debug(f"Mock SPDK: Retrieved {len(indices)} embeddings in {elapsed:.3f}s "
                    f"(cache hits: {cache_hits}/{len(indices)})")
        return result
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           candidate_indices: List[int]) -> np.ndarray:
        """Compute similarities using parallel mock CSD compute units."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        
        # Retrieve candidate embeddings
        candidates = self.retrieve_embeddings(candidate_indices)
        
        # Parallel similarity computation simulation
        def _compute_batch_similarities(batch_candidates: np.ndarray, batch_start: int) -> np.ndarray:
            """Compute similarities for a batch of candidates."""
            # Simulate CSD compute latency per batch
            time.sleep(self.compute_latency_ms / 1000)
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_normalized = query_embedding / query_norm
            else:
                query_normalized = query_embedding
            
            # Normalize candidate embeddings
            candidate_norms = np.linalg.norm(batch_candidates, axis=1)
            candidate_norms = np.where(candidate_norms == 0, 1, candidate_norms)
            candidates_normalized = batch_candidates / candidate_norms.reshape(-1, 1)
            
            # Compute cosine similarities
            batch_similarities = np.dot(candidates_normalized, query_normalized)
            return batch_similarities
        
        # Split into batches for parallel processing
        batch_size = max(1, len(candidate_indices) // self.max_parallel_ops)
        futures = []
        
        if len(candidate_indices) > batch_size:
            # Use thread pool for large datasets
            for i in range(0, len(candidates), batch_size):
                batch_candidates = candidates[i:i + batch_size]
                future = self.thread_pool.submit(_compute_batch_similarities, batch_candidates, i)
                futures.append(future)
            
            # Collect results
            similarities = []
            for future in futures:
                batch_similarities = future.result()
                similarities.extend(batch_similarities)
            
            similarities = np.array(similarities)
            self.mock_metrics["parallel_operations"] += len(futures)
        else:
            # Direct computation for small datasets
            similarities = _compute_batch_similarities(candidates, 0)
        
        # Update metrics
        elapsed = time.time() - start_time
        self.mock_metrics["similarity_computations"] += len(candidate_indices)
        self.mock_metrics["emulation_overhead_ms"] += elapsed * 1000
        
        logger.debug(f"Mock SPDK: Computed {len(candidate_indices)} similarities in {elapsed:.3f}s")
        
        return similarities
    
    def process_era_pipeline(self, query_data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Process complete ERA pipeline using mock SPDK CSD with realistic simulation."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        
        # **Stage 1: Encode on CSD**
        encode_start = time.time()
        
        if query_data.dtype != np.float32 or len(query_data.shape) != 1:
            # Simulate encoding latency for text input
            time.sleep(1.0 / 1000)  # 1ms encoding latency
            query_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        else:
            # Already encoded embedding
            query_embedding = query_data / np.linalg.norm(query_data) if np.linalg.norm(query_data) > 0 else query_data
        
        encode_time = time.time() - encode_start
        
        # **Stage 2: Retrieve on CSD**
        retrieve_start = time.time()
        
        top_k = metadata.get("top_k", 5)
        search_expansion = metadata.get("search_expansion", 10)  # Search more than top_k
        max_candidates = min(search_expansion * top_k, len(self.embedding_to_lba))
        
        if max_candidates > 0:
            candidate_indices = list(range(min(max_candidates, len(self.embedding_to_lba))))
            similarities = self.compute_similarities(query_embedding, candidate_indices)
            
            # Get top-k most similar
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            retrieved = self.retrieve_embeddings(top_indices.tolist())
            retrieved_similarities = similarities[top_indices]
        else:
            # Generate dummy retrieved documents for empty database
            retrieved = np.random.randn(top_k, self.embedding_dim).astype(np.float32)
            for i in range(top_k):
                retrieved[i] = retrieved[i] / np.linalg.norm(retrieved[i])
            retrieved_similarities = np.random.uniform(0.1, 0.9, top_k)
        
        retrieve_time = time.time() - retrieve_start
        
        # **Stage 3: Augment on CSD**
        augment_start = time.time()
        
        # Simulate augmentation compute latency
        time.sleep(0.5 / 1000)  # 0.5ms augmentation latency
        
        # Create augmented representation
        # Method 1: Weighted concatenation based on similarity scores
        mode = metadata.get("mode", "similarity")
        
        if mode == "similarity":
            # Weight embeddings by similarity scores
            weights = retrieved_similarities / np.sum(retrieved_similarities) if np.sum(retrieved_similarities) > 0 else np.ones(len(retrieved)) / len(retrieved)
            weighted_retrieved = np.average(retrieved, axis=0, weights=weights)
            
            # Concatenate query with weighted retrieved context
            augmented_data = np.concatenate([query_embedding, weighted_retrieved])
        
        elif mode == "concatenate":
            # Simple concatenation of all embeddings
            augmented_data = np.concatenate([query_embedding] + [retrieved[i] for i in range(len(retrieved))])
        
        else:
            # Default: just return query with average of retrieved
            avg_retrieved = np.mean(retrieved, axis=0) if len(retrieved) > 0 else np.zeros(self.embedding_dim)
            augmented_data = np.concatenate([query_embedding, avg_retrieved])
        
        augment_time = time.time() - augment_start
        
        # Update metrics
        total_time = time.time() - start_time
        self.mock_metrics["era_pipeline_executions"] += 1
        self.mock_metrics["emulation_overhead_ms"] += total_time * 1000
        
        logger.debug(f"Mock SPDK ERA Pipeline: Total={total_time*1000:.2f}ms "
                    f"(Encode={encode_time*1000:.2f}ms, Retrieve={retrieve_time*1000:.2f}ms, "
                    f"Augment={augment_time*1000:.2f}ms)")
        
        return augmented_data
    
    def p2p_transfer_to_gpu(self, data: np.ndarray) -> str:
        """Simulate P2P transfer from mock CSD to GPU."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        # Simulate P2P transfer with realistic bandwidth
        p2p_bandwidth_gbps = 12  # 12 GB/s
        transfer_time = data.nbytes / (p2p_bandwidth_gbps * 1024 * 1024 * 1024)
        time.sleep(transfer_time)
        
        self.mock_metrics["p2p_transfers"] += 1
        allocation_id = f"mock_gpu_alloc_{self.mock_metrics['p2p_transfers']}"
        
        logger.debug(f"Mock P2P transfer: {data.nbytes/1024:.1f}KB in {transfer_time*1000:.2f}ms")
        
        return allocation_id
    
    def encode_on_csd(self, queries: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode queries using CSD ML computational offloading with cache acceleration."""
        start_time = time.time()
        
        # Simulate ML encoding computation on CSD with vector processing units
        encoding_latency = len(queries) * (1.0 / self.ml_accelerator_freq)  # MHz to ms conversion
        # Account for vector processing parallelism
        if len(queries) > self.vector_processing_units:
            parallel_batches = len(queries) // self.vector_processing_units
            encoding_latency = parallel_batches * (1.0 / self.ml_accelerator_freq)
        
        time.sleep(encoding_latency / 1000)
        
        # Generate normalized embeddings
        embeddings = []
        for query in queries:
            # Simulate encoder computation with cache optimization
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        elapsed = time.time() - start_time
        self.ml_metrics["ml_encode_ops"] += len(queries)
        self.ml_metrics["vector_operations"] += len(queries)
        self.ml_metrics["total_ml_compute_time"] += elapsed
        self.ml_metrics["offloaded_computations"] += 1
        
        # Use cache hierarchy to accelerate future operations
        for i, embedding in enumerate(embeddings):
            cache_key = f"encoded_query_{hash(queries[i])}"
            self.cache_hierarchy.put_embedding(i, embedding)
            self.ml_metrics["cache_accelerated_ops"] += 1
        
        stats = {
            "encode_time_ms": elapsed * 1000,
            "queries_encoded": len(queries),
            "vector_units_used": min(self.vector_processing_units, len(queries)),
            "cache_utilization": self.cache_hierarchy.get_stats()["cache_hit_rate"]
        }
        
        logger.debug(f"CSD Encode: {len(queries)} queries in {elapsed*1000:.2f}ms using {min(self.vector_processing_units, len(queries))} vector units")
        return np.array(embeddings), stats
    
    def retrieve_on_csd(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Retrieve embeddings using CSD ML computational offloading with 3-level cache."""
        start_time = time.time()
        
        # Simulate retrieval computation on CSD with cache acceleration
        cache_stats = self.cache_hierarchy.get_stats()
        base_latency = 1.5 / self.ml_accelerator_freq  # MHz to ms conversion
        
        # Cache acceleration reduces latency
        cache_speedup = 1.0 + (cache_stats["cache_hit_rate"] * 2.0)  # Up to 3x speedup
        retrieval_latency = base_latency / cache_speedup
        time.sleep(retrieval_latency / 1000)
        
        # Get candidate embeddings using cache hierarchy
        max_candidates = min(100, top_k * 20)  # Search expansion
        candidates = []
        
        if len(self.embedding_to_lba) > 0:
            candidate_indices = list(range(min(max_candidates, len(self.embedding_to_lba))))
            similarities = self.compute_similarities(query_embedding, candidate_indices)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            retrieved_embeddings = self.retrieve_embeddings(top_indices.tolist())
            candidates = [retrieved_embeddings[i] for i in range(len(retrieved_embeddings))]
        else:
            # Generate dummy candidates
            candidates = [np.random.randn(self.embedding_dim).astype(np.float32) for _ in range(top_k)]
            for i in range(len(candidates)):
                candidates[i] = candidates[i] / np.linalg.norm(candidates[i])
        
        elapsed = time.time() - start_time
        self.ml_metrics["ml_retrieve_ops"] += 1
        self.ml_metrics["vector_operations"] += len(candidates)
        self.ml_metrics["total_ml_compute_time"] += elapsed
        self.ml_metrics["offloaded_computations"] += 1
        self.ml_metrics["cache_accelerated_ops"] += 1
        
        stats = {
            "retrieve_time_ms": elapsed * 1000,
            "candidates_retrieved": len(candidates),
            "cache_speedup": cache_speedup,
            "cache_hierarchy_stats": cache_stats
        }
        
        logger.debug(f"CSD Retrieve: {len(candidates)} candidates in {elapsed*1000:.2f}ms with {cache_speedup:.1f}x cache speedup")
        return candidates, stats
    
    def augment_on_csd(self, query: str, retrieved_contexts: List[np.ndarray], metadata: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Augment query with retrieved contexts using CSD ML computational offloading."""
        start_time = time.time()
        
        # Simulate augmentation computation on CSD with parallel processing
        num_contexts = len(retrieved_contexts)
        augmentation_latency = num_contexts * (0.4 / self.ml_accelerator_freq)  # MHz to ms
        
        # Use parallel processing for multiple contexts
        if num_contexts > self.ml_compute_units:
            parallel_batches = num_contexts // self.ml_compute_units
            augmentation_latency = parallel_batches * (0.4 / self.ml_accelerator_freq)
        
        time.sleep(augmentation_latency / 1000)
        
        # Perform context augmentation with vector processing
        if retrieved_contexts and len(retrieved_contexts) > 0:
            # Compute context similarities for weighting
            context_weights = []
            for context in retrieved_contexts:
                # Simulate similarity computation on vector units
                weight = np.random.uniform(0.3, 1.0)  # Simulated relevance score
                context_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(context_weights)
            context_weights = [w / total_weight for w in context_weights] if total_weight > 0 else [1.0 / len(context_weights)] * len(context_weights)
            
            # Create weighted context summary
            context_summary = f"Based on {len(retrieved_contexts)} relevant contexts (weights: {[f'{w:.2f}' for w in context_weights[:3]]}):\n"
            for i, (context, weight) in enumerate(zip(retrieved_contexts[:3], context_weights[:3])):
                context_summary += f"[Context {i+1} (relevance: {weight:.2f}): processed information] "
            
            augmented_query = f"{query}\n\nContext: {context_summary}"
        else:
            augmented_query = query
        
        elapsed = time.time() - start_time
        self.ml_metrics["ml_augment_ops"] += 1
        self.ml_metrics["vector_operations"] += num_contexts
        self.ml_metrics["total_ml_compute_time"] += elapsed
        self.ml_metrics["offloaded_computations"] += 1
        
        stats = {
            "augment_time_ms": elapsed * 1000,
            "contexts_processed": num_contexts,
            "ml_units_used": min(self.ml_compute_units, num_contexts),
            "parallel_efficiency": min(self.ml_compute_units, num_contexts) / self.ml_compute_units
        }
        
        logger.debug(f"CSD Augment: {num_contexts} contexts in {elapsed*1000:.2f}ms using {min(self.ml_compute_units, num_contexts)} ML units")
        return augmented_query, stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including cache and SPDK-specific metrics."""
        # Combine base metrics with mock-specific and ML metrics
        combined = {**self.metrics, **self.mock_metrics, **self.ml_metrics}
        combined["backend_type"] = self._backend_type.value
        combined["backend_subtype"] = "mock_implementation"
        combined["computational_offloading"] = True
        combined["ml_compute_units"] = self.ml_compute_units
        combined["ml_accelerator_freq_mhz"] = self.ml_accelerator_freq
        combined["vector_processing_units"] = self.vector_processing_units
        
        # Add cache hierarchy stats
        if self.cache_hierarchy:
            cache_stats = self.cache_hierarchy.get_stats()
            combined["cache_hierarchy"] = cache_stats
            # Update overall cache hit rate from cache hierarchy
            combined["cache_hit_rate"] = cache_stats.get("cache_hit_rate", 0.0)
        
        # Add NVMe controller stats
        if self.nvme_controller:
            nvme_stats = self.nvme_controller.get_stats()
            combined["nvme_controller"] = nvme_stats
        
        # Add performance analysis
        combined["performance_analysis"] = {
            "avg_similarity_compute_time": (
                self.mock_metrics["emulation_overhead_ms"] / 
                max(1, self.mock_metrics["similarity_computations"])
            ),
            "avg_era_pipeline_time": (
                self.mock_metrics["emulation_overhead_ms"] / 
                max(1, self.mock_metrics["era_pipeline_executions"])
            ),
            "parallel_efficiency": (
                self.mock_metrics["parallel_operations"] / 
                max(1, self.mock_metrics["similarity_computations"])
            )
        }
        
        return combined
    
    def shutdown(self) -> None:
        """Shutdown the mock SPDK emulator backend."""
        try:
            if self.thread_pool:
                logger.info("Shutting down thread pool")
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None
            
            if self.cache_hierarchy:
                logger.info("Clearing cache hierarchy")
                self.cache_hierarchy = None
            
            if self.nvme_controller:
                logger.info("Shutting down mock NVMe controller")
                self.nvme_controller = None
            
            self.initialized = False
            logger.info("Mock SPDK emulator backend shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during mock SPDK shutdown: {e}")
    
    def is_available(self) -> bool:
        """Mock SPDK backend is always available for testing."""
        return True
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the mock backend."""
        info = super().get_backend_info()
        info.update({
            "is_mock": True,
            "nvme_size_gb": self.nvme_size_gb,
            "virtual_queues": self.virtual_queues,
            "embedding_dim": self.embedding_dim,
            "embedding_size_blocks": self.embedding_size_blocks,
            "description": "Mock SPDK backend for testing and development"
        })
        return info