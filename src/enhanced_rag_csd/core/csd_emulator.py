"""
Enhanced CSD Simulator with advanced storage emulation capabilities.
This module provides a more realistic simulation of Computational Storage Devices
with features like memory-mapped files, parallel I/O, and cache-aware processing.
"""

import os
import time
import mmap
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock
from typing import Dict, List, Optional, Tuple, Any
import psutil
from dataclasses import dataclass
from collections import deque
import hashlib

from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StorageMetrics:
    """Metrics for storage operations."""
    read_ops: int = 0
    write_ops: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    avg_latency: float = 0.0


class CacheHierarchy:
    """Multi-level cache hierarchy for vector storage."""
    
    def __init__(self, l1_size_mb: int = 64, l2_size_mb: int = 512, l3_size_mb: int = 2048):
        self.l1_cache = {}  # Hot embeddings (in-memory)
        self.l2_cache = {}  # Warm embeddings (simulated SSD)
        self.l3_cache = {}  # Cold embeddings (simulated disk)
        
        self.l1_size = l1_size_mb * 1024 * 1024
        self.l2_size = l2_size_mb * 1024 * 1024
        self.l3_size = l3_size_mb * 1024 * 1024
        
        self.l1_lru = deque()
        self.l2_lru = deque()
        self.l3_lru = deque()
        
        self.access_counts = {}
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache hierarchy."""
        with self.lock:
            # Check L1
            if key in self.l1_cache:
                self._update_lru(self.l1_lru, key)
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return self.l1_cache[key]
            
            # Check L2
            if key in self.l2_cache:
                embedding = self.l2_cache[key]
                self._promote_to_l1(key, embedding)
                return embedding
            
            # Check L3
            if key in self.l3_cache:
                embedding = self.l3_cache[key]
                self._promote_to_l2(key, embedding)
                return embedding
            
            return None
    
    def put(self, key: str, embedding: np.ndarray) -> None:
        """Put embedding into cache hierarchy."""
        with self.lock:
            embedding_size = embedding.nbytes
            
            # Add to L1 if it fits
            if self._get_cache_size(self.l1_cache) + embedding_size <= self.l1_size:
                self.l1_cache[key] = embedding
                self.l1_lru.append(key)
            else:
                # Evict from L1 if needed
                self._evict_from_l1()
                self.l1_cache[key] = embedding
                self.l1_lru.append(key)
    
    def _update_lru(self, lru_queue: deque, key: str) -> None:
        """Update LRU position for accessed key."""
        if key in lru_queue:
            lru_queue.remove(key)
        lru_queue.append(key)
    
    def _get_cache_size(self, cache: Dict) -> int:
        """Calculate total size of cache in bytes."""
        return sum(v.nbytes for v in cache.values())
    
    def _promote_to_l1(self, key: str, embedding: np.ndarray) -> None:
        """Promote embedding from L2/L3 to L1."""
        self._evict_from_l1()
        self.l1_cache[key] = embedding
        self.l1_lru.append(key)
        
        # Remove from lower levels
        if key in self.l2_cache:
            del self.l2_cache[key]
            self.l2_lru.remove(key)
        if key in self.l3_cache:
            del self.l3_cache[key]
            self.l3_lru.remove(key)
    
    def _promote_to_l2(self, key: str, embedding: np.ndarray) -> None:
        """Promote embedding from L3 to L2."""
        self._evict_from_l2()
        self.l2_cache[key] = embedding
        self.l2_lru.append(key)
        
        if key in self.l3_cache:
            del self.l3_cache[key]
            self.l3_lru.remove(key)
    
    def _evict_from_l1(self) -> None:
        """Evict least recently used item from L1."""
        while self._get_cache_size(self.l1_cache) >= self.l1_size and self.l1_lru:
            evict_key = self.l1_lru.popleft()
            if evict_key in self.l1_cache:
                embedding = self.l1_cache.pop(evict_key)
                # Demote to L2
                self._evict_from_l2()
                self.l2_cache[evict_key] = embedding
                self.l2_lru.append(evict_key)
    
    def _evict_from_l2(self) -> None:
        """Evict least recently used item from L2."""
        while self._get_cache_size(self.l2_cache) >= self.l2_size and self.l2_lru:
            evict_key = self.l2_lru.popleft()
            if evict_key in self.l2_cache:
                embedding = self.l2_cache.pop(evict_key)
                # Demote to L3
                self._evict_from_l3()
                self.l3_cache[evict_key] = embedding
                self.l3_lru.append(evict_key)
    
    def _evict_from_l3(self) -> None:
        """Evict least recently used item from L3."""
        while self._get_cache_size(self.l3_cache) >= self.l3_size and self.l3_lru:
            evict_key = self.l3_lru.popleft()
            if evict_key in self.l3_cache:
                del self.l3_cache[evict_key]


class MemoryMappedStorage:
    """Memory-mapped file storage for efficient vector access."""
    
    def __init__(self, storage_path: str, embedding_dim: int = 384):
        self.storage_path = storage_path
        self.embedding_dim = embedding_dim
        self.embedding_size = embedding_dim * 4  # float32
        
        os.makedirs(storage_path, exist_ok=True)
        self.vectors_file = os.path.join(storage_path, "vectors.bin")
        self.metadata_file = os.path.join(storage_path, "metadata.json")
        
        self.mmap_file = None
        self.num_vectors = 0
        self._init_storage()
    
    def _init_storage(self) -> None:
        """Initialize memory-mapped storage."""
        if os.path.exists(self.vectors_file):
            # Open existing file
            self.file_handle = open(self.vectors_file, "r+b")
            file_size = os.path.getsize(self.vectors_file)
            self.num_vectors = file_size // self.embedding_size
            
            if file_size > 0:
                self.mmap_file = mmap.mmap(
                    self.file_handle.fileno(), 
                    file_size,
                    access=mmap.ACCESS_WRITE
                )
        else:
            # Create new file
            self.file_handle = open(self.vectors_file, "w+b")
            # Pre-allocate space for initial vectors
            initial_size = 1000 * self.embedding_size
            self.file_handle.write(b'\0' * initial_size)
            self.file_handle.flush()
            
            self.mmap_file = mmap.mmap(
                self.file_handle.fileno(),
                initial_size,
                access=mmap.ACCESS_WRITE
            )
    
    def read_vector(self, index: int) -> np.ndarray:
        """Read a vector from memory-mapped storage."""
        if index >= self.num_vectors:
            raise IndexError(f"Index {index} out of bounds")
        
        offset = index * self.embedding_size
        self.mmap_file.seek(offset)
        data = self.mmap_file.read(self.embedding_size)
        return np.frombuffer(data, dtype=np.float32)
    
    def write_vector(self, index: int, vector: np.ndarray) -> None:
        """Write a vector to memory-mapped storage."""
        if vector.shape[0] != self.embedding_dim:
            raise ValueError(f"Vector dimension mismatch: {vector.shape[0]} != {self.embedding_dim}")
        
        offset = index * self.embedding_size
        
        # Extend file if necessary
        required_size = (index + 1) * self.embedding_size
        current_size = len(self.mmap_file)
        
        if required_size > current_size:
            self._extend_storage(required_size)
        
        self.mmap_file.seek(offset)
        self.mmap_file.write(vector.astype(np.float32).tobytes())
        self.mmap_file.flush()
        
        if index >= self.num_vectors:
            self.num_vectors = index + 1
    
    def _extend_storage(self, new_size: int) -> None:
        """Extend the memory-mapped file."""
        self.mmap_file.close()
        self.file_handle.seek(0, 2)  # Seek to end
        
        # Extend in chunks
        extension_size = max(new_size - self.file_handle.tell(), 1000 * self.embedding_size)
        self.file_handle.write(b'\0' * extension_size)
        self.file_handle.flush()
        
        # Re-map the file
        self.mmap_file = mmap.mmap(
            self.file_handle.fileno(),
            self.file_handle.tell(),
            access=mmap.ACCESS_WRITE
        )
    
    def batch_read(self, indices: List[int]) -> np.ndarray:
        """Read multiple vectors efficiently."""
        vectors = np.zeros((len(indices), self.embedding_dim), dtype=np.float32)
        
        # Sort indices for sequential access
        sorted_indices = sorted(enumerate(indices), key=lambda x: x[1])
        
        for i, idx in sorted_indices:
            vectors[i] = self.read_vector(idx)
        
        return vectors
    
    def close(self) -> None:
        """Close memory-mapped storage."""
        if self.mmap_file:
            self.mmap_file.close()
        if self.file_handle:
            self.file_handle.close()


class EnhancedCSDSimulator:
    """Enhanced CSD simulator with realistic storage characteristics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Storage configuration
        self.storage_path = config.get("storage_path", "./csd_storage")
        self.embedding_dim = config.get("embedding", {}).get("dimensions", 384)
        
        # Performance parameters
        self.ssd_bandwidth_mbps = config.get("csd", {}).get("ssd_bandwidth_mbps", 2000)
        self.nand_bandwidth_mbps = config.get("csd", {}).get("nand_bandwidth_mbps", 500)
        self.compute_latency_ms = config.get("csd", {}).get("compute_latency_ms", 0.1)
        self.max_parallel_ops = config.get("csd", {}).get("max_parallel_ops", 8)
        
        # Initialize components
        self.storage = MemoryMappedStorage(self.storage_path, self.embedding_dim)
        self.cache_hierarchy = CacheHierarchy()
        self.metrics = StorageMetrics()
        
        # Parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_ops)
        self.semaphore = Semaphore(self.max_parallel_ops)
        
        logger.info(f"Enhanced CSD Simulator initialized with {self.max_parallel_ops} parallel ops")
    
    def store_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Store embeddings with metadata."""
        start_time = time.time()
        
        # Parallel write operations
        futures = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            future = self.executor.submit(self._store_single_embedding, i, embedding, meta)
            futures.append(future)
        
        # Wait for all writes to complete
        for future in as_completed(futures):
            future.result()
        
        elapsed = time.time() - start_time
        self.metrics.write_ops += len(embeddings)
        self.metrics.total_bytes_written += embeddings.nbytes
        
        logger.info(f"Stored {len(embeddings)} embeddings in {elapsed:.2f}s")
    
    def _store_single_embedding(self, index: int, embedding: np.ndarray, metadata: Dict) -> None:
        """Store a single embedding with caching."""
        # Generate cache key
        cache_key = hashlib.md5(str(metadata).encode()).hexdigest()
        
        # Store in memory-mapped file
        self.storage.write_vector(index, embedding)
        
        # Add to cache
        self.cache_hierarchy.put(cache_key, embedding)
        
        # Simulate storage latency
        data_size_mb = embedding.nbytes / (1024 * 1024)
        latency = data_size_mb / self.ssd_bandwidth_mbps
        time.sleep(latency)
    
    def retrieve_embeddings(self, indices: List[int], use_cache: bool = True) -> np.ndarray:
        """Retrieve embeddings with parallel I/O."""
        start_time = time.time()
        
        if use_cache:
            # Try cache first
            cached_embeddings = []
            cache_misses = []
            
            for idx in indices:
                cache_key = f"idx_{idx}"
                cached = self.cache_hierarchy.get(cache_key)
                if cached is not None:
                    cached_embeddings.append((idx, cached))
                    self.metrics.cache_hits += 1
                else:
                    cache_misses.append(idx)
                    self.metrics.cache_misses += 1
            
            # Parallel retrieval for cache misses
            if cache_misses:
                miss_embeddings = self._parallel_retrieve(cache_misses)
                
                # Update cache
                for idx, embedding in zip(cache_misses, miss_embeddings):
                    cache_key = f"idx_{idx}"
                    self.cache_hierarchy.put(cache_key, embedding)
                
                # Combine results
                all_embeddings = dict(cached_embeddings)
                for idx, embedding in zip(cache_misses, miss_embeddings):
                    all_embeddings[idx] = embedding
                
                # Return in original order
                result = np.array([all_embeddings[idx] for idx in indices])
            else:
                result = np.array([emb for _, emb in cached_embeddings])
        else:
            # Direct retrieval without cache
            result = self._parallel_retrieve(indices)
        
        elapsed = time.time() - start_time
        self.metrics.read_ops += len(indices)
        self.metrics.total_bytes_read += result.nbytes
        self.metrics.avg_latency = elapsed / len(indices)
        
        return result
    
    def _parallel_retrieve(self, indices: List[int]) -> np.ndarray:
        """Retrieve embeddings in parallel."""
        # Sort indices for better sequential access
        sorted_indices = sorted(indices)
        
        # Determine optimal batch size based on cache line size
        cpu_cache_line = 64  # bytes
        vectors_per_cache_line = cpu_cache_line // (self.embedding_dim * 4)
        batch_size = max(vectors_per_cache_line, len(indices) // self.max_parallel_ops)
        
        futures = []
        for i in range(0, len(sorted_indices), batch_size):
            batch = sorted_indices[i:i + batch_size]
            future = self.executor.submit(self._retrieve_batch, batch)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        
        # Restore original order
        index_map = {idx: i for i, idx in enumerate(indices)}
        ordered_results = [None] * len(indices)
        for idx, embedding in results:
            ordered_results[index_map[idx]] = embedding
        
        return np.array(ordered_results)
    
    def _retrieve_batch(self, indices: List[int]) -> List[Tuple[int, np.ndarray]]:
        """Retrieve a batch of embeddings."""
        with self.semaphore:
            # Simulate storage access
            embeddings = self.storage.batch_read(indices)
            
            # Simulate bandwidth constraints
            data_size_mb = embeddings.nbytes / (1024 * 1024)
            latency = data_size_mb / self.nand_bandwidth_mbps
            time.sleep(latency)
            
            return list(zip(indices, embeddings))
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           candidate_indices: List[int]) -> np.ndarray:
        """Compute similarities with CSD acceleration."""
        start_time = time.time()
        
        # Retrieve candidate embeddings
        candidates = self.retrieve_embeddings(candidate_indices)
        
        # Parallel similarity computation
        num_candidates = len(candidates)
        batch_size = max(1, num_candidates // self.max_parallel_ops)
        
        futures = []
        for i in range(0, num_candidates, batch_size):
            batch = candidates[i:i + batch_size]
            future = self.executor.submit(
                self._compute_batch_similarities,
                query_embedding,
                batch
            )
            futures.append(future)
        
        # Collect results
        similarities = []
        for future in as_completed(futures):
            similarities.extend(future.result())
        
        elapsed = time.time() - start_time
        logger.debug(f"Computed {num_candidates} similarities in {elapsed:.3f}s")
        
        return np.array(similarities)
    
    def _compute_batch_similarities(self, query: np.ndarray, 
                                  candidates: np.ndarray) -> List[float]:
        """Compute similarities for a batch."""
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(candidates_norm, query_norm)
        
        # Simulate compute latency
        time.sleep(self.compute_latency_ms / 1000 * len(candidates))
        
        return similarities.tolist()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        cache_hit_rate = (self.metrics.cache_hits / 
                         (self.metrics.cache_hits + self.metrics.cache_misses + 1e-10))
        
        return {
            "read_ops": self.metrics.read_ops,
            "write_ops": self.metrics.write_ops,
            "cache_hit_rate": cache_hit_rate,
            "total_bytes_read": self.metrics.total_bytes_read,
            "total_bytes_written": self.metrics.total_bytes_written,
            "avg_latency_ms": self.metrics.avg_latency * 1000,
            "storage_usage_mb": self.storage.num_vectors * self.storage.embedding_size / (1024 * 1024)
        }
    
    def shutdown(self) -> None:
        """Shutdown the simulator."""
        self.executor.shutdown(wait=True)
        self.storage.close()
        logger.info("Enhanced CSD Simulator shutdown complete")