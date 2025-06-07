"""
Enhanced RAG-CSD Pipeline with CSD emulation, incremental indexing, and advanced optimizations.
This module provides the main pipeline interface with all performance improvements.
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import numpy as np

from enhanced_rag_csd.core.encoder import Encoder
from enhanced_rag_csd.retrieval.incremental_index import IncrementalVectorStore
from enhanced_rag_csd.core.augmentor import Augmentor
from enhanced_rag_csd.core.csd_emulator import EnhancedCSDSimulator
from enhanced_rag_csd.core.system_data_flow import SystemDataFlow, SystemDataFlowConfig
from enhanced_rag_csd.core.system_memory import MemoryConfig
from enhanced_rag_csd.utils.logger import get_logger
from enhanced_rag_csd.utils.metrics import MetricsCollector
from enhanced_rag_csd.utils.embedding_cache import get_embedding_cache
from enhanced_rag_csd.utils.model_cache import get_model_cache
from enhanced_rag_csd.utils.text_processor import get_text_processor

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for enhanced pipeline."""
    # Storage paths
    vector_db_path: str
    storage_path: str = "./enhanced_storage"
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Indexing settings
    delta_threshold: int = 10000
    max_delta_indices: int = 5
    
    # CSD emulation settings
    enable_csd_emulation: bool = True
    max_parallel_ops: int = 8
    ssd_bandwidth_mbps: int = 2000
    nand_bandwidth_mbps: int = 500
    
    # Pipeline settings
    enable_pipeline_parallel: bool = True
    flexible_retrieval_interval: int = 3
    enable_system_data_flow: bool = False
    
    # Cache settings
    enable_caching: bool = True
    l1_cache_size_mb: int = 64
    l2_cache_size_mb: int = 512
    l3_cache_size_mb: int = 2048
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


class WorkloadClassifier:
    """Classifies workload characteristics for optimization."""
    
    def __init__(self):
        self.query_history = []
        self.latency_history = []
    
    def classify(self, queries: Union[str, List[str]]) -> str:
        """Classify workload type."""
        if isinstance(queries, str):
            return "single"
        
        num_queries = len(queries)
        
        if num_queries == 1:
            return "single"
        elif num_queries < 10:
            return "small_batch"
        elif num_queries < 100:
            return "medium_batch"
        else:
            return "large_batch"
    
    def recommend_strategy(self, workload_type: str) -> Dict[str, Any]:
        """Recommend optimization strategy based on workload."""
        strategies = {
            "single": {
                "mode": "sync",
                "prefetch": False,
                "batch_size": 1,
                "cache_priority": "latency"
            },
            "small_batch": {
                "mode": "async",
                "prefetch": True,
                "batch_size": 4,
                "cache_priority": "balanced"
            },
            "medium_batch": {
                "mode": "batch",
                "prefetch": True,
                "batch_size": 16,
                "cache_priority": "throughput"
            },
            "large_batch": {
                "mode": "batch",
                "prefetch": True,
                "batch_size": 32,
                "cache_priority": "throughput"
            }
        }
        
        return strategies.get(workload_type, strategies["single"])


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with all optimizations."""
    
    def __init__(self, config: Union[Dict, PipelineConfig]):
        if isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config
        
        logger.info("Initializing Enhanced RAG-CSD Pipeline")
        
        # Initialize components
        self._init_components()
        
        # Metrics and monitoring
        self.metrics = MetricsCollector()
        self.workload_classifier = WorkloadClassifier()
        
        # Pipeline state
        self.is_initialized = False
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_ops)
        
        logger.info("Enhanced RAG-CSD Pipeline initialized successfully")
    
    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Model and caches
        if self.config.enable_caching:
            self.model_cache = get_model_cache()
            self.embedding_cache = get_embedding_cache()
        
        # Embedding encoder
        self.encoder = Encoder({"model": self.config.embedding_model})
        
        # Incremental vector store
        self.vector_store = IncrementalVectorStore(
            storage_path=os.path.join(self.config.storage_path, "vectors"),
            dimension=self.config.embedding_dim,
            delta_threshold=self.config.delta_threshold,
            max_delta_indices=self.config.max_delta_indices
        )
        
        # CSD emulator
        if self.config.enable_csd_emulation:
            csd_config = {
                "storage_path": os.path.join(self.config.storage_path, "csd"),
                "embedding": {"dimensions": self.config.embedding_dim},
                "csd": {
                    "max_parallel_ops": self.config.max_parallel_ops,
                    "ssd_bandwidth_mbps": self.config.ssd_bandwidth_mbps,
                    "nand_bandwidth_mbps": self.config.nand_bandwidth_mbps
                },
                "system": {
                    "enable_integration": self.config.enable_system_data_flow,
                    "enable_p2p": True,
                    "csd_memory_mb": 4096,
                    "pcie_bandwidth_mbps": 15750,
                    "p2p_bandwidth_mbps": 12000
                }
            }
            self.csd_simulator = EnhancedCSDSimulator(csd_config)
        else:
            self.csd_simulator = None
        
        # System data flow
        if self.config.enable_system_data_flow:
            memory_config = MemoryConfig(
                dram_capacity_mb=16384,
                gpu_memory_mb=8192,
                csd_memory_mb=4096,
                enable_p2p=True
            )
            system_config = SystemDataFlowConfig(
                memory_config=memory_config,
                enable_async_processing=True,
                enable_prefetching=True
            )
            self.system_data_flow = SystemDataFlow(system_config)
        else:
            self.system_data_flow = None
        
        # Query augmentor
        self.augmentor = Augmentor()
        
        # Text processor
        self.text_processor = get_text_processor()
    
    def add_documents(self, 
                     documents: List[str],
                     metadata: Optional[List[Dict]] = None,
                     chunk_size: int = 512,
                     chunk_overlap: int = 50) -> Dict[str, Any]:
        """Add new documents to the index."""
        start_time = time.time()
        
        if metadata is None:
            metadata = [{"doc_id": i} for i in range(len(documents))]
        
        # Process documents into chunks
        all_chunks = []
        all_metadata = []
        
        for doc, meta in zip(documents, metadata):
            chunks = self.text_processor.chunk_text_optimized(
                doc, 
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_meta = meta.copy()
                chunk_meta["chunk_id"] = i
                all_metadata.append(chunk_meta)
        
        logger.info(f"Processing {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Encode chunks
        if self.config.enable_caching:
            # Check cache for existing embeddings
            embeddings = []
            uncached_chunks = []
            uncached_indices = []
            
            for i, chunk in enumerate(all_chunks):
                cached = self.embedding_cache.get(chunk)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    uncached_chunks.append(chunk)
                    uncached_indices.append(i)
                    embeddings.append(None)
            
            # Encode uncached chunks
            if uncached_chunks:
                new_embeddings = self.encoder.encode(uncached_chunks)
                
                # Update cache and results
                for i, idx in enumerate(uncached_indices):
                    embeddings[idx] = new_embeddings[i]
                    self.embedding_cache.put(uncached_chunks[i], new_embeddings[i])
            
            embeddings = np.array(embeddings)
        else:
            embeddings = self.encoder.encode(all_chunks)
        
        # Add to vector store
        self.vector_store.add_documents(embeddings, all_chunks, all_metadata)
        
        # Store in CSD simulator if enabled
        if self.csd_simulator:
            self.csd_simulator.store_embeddings(embeddings, all_metadata)
        
        elapsed = time.time() - start_time
        
        result = {
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "processing_time": elapsed,
            "chunks_per_second": len(all_chunks) / elapsed
        }
        
        self.metrics.record_timing("add_documents", elapsed)
        self.metrics.record_count("documents_processed", len(documents))
        
        return result
    
    def query(self, 
             query: str,
             top_k: int = 5,
             include_metadata: bool = True) -> Dict[str, Any]:
        """Process a single query with all optimizations."""
        start_time = time.time()
        
        # Classify workload and get strategy
        workload_type = self.workload_classifier.classify(query)
        strategy = self.workload_classifier.recommend_strategy(workload_type)
        
        # Use system data flow if enabled
        if self.config.enable_system_data_flow and self.system_data_flow is not None:
            result = self._query_system_data_flow(query, top_k, include_metadata)
        elif self.config.enable_pipeline_parallel:
            result = self._query_pipeline_parallel(query, top_k, include_metadata)
        else:
            result = self._query_sequential(query, top_k, include_metadata)
        
        # Record metrics
        elapsed = time.time() - start_time
        result["processing_time"] = elapsed
        result["strategy"] = strategy
        
        self.metrics.record_timing("query", elapsed)
        self.metrics.record_count("queries_processed")
        
        return result
    
    def _query_system_data_flow(self,
                               query: str,
                               top_k: int,
                               include_metadata: bool) -> Dict[str, Any]:
        """Query processing using complete system data flow."""
        # Convert query to numpy array for system processing
        query_embedding = self._encode_query(query)
        query_data = query_embedding.astype(np.float32)
        
        query_metadata = {
            "query_id": f"query_{int(time.time() * 1000)}",
            "top_k": top_k,
            "original_query": query
        }
        
        # Process through system data flow
        result = self.system_data_flow.process_query(query_data, query_metadata, async_processing=False)
        
        # Format result for compatibility
        formatted_result = {
            "query": query,
            "augmented_query": f"System processed: {query} with {result.get('gpu_data_size_bytes', 0)} bytes",
            "retrieved_docs": top_k if include_metadata else f"{top_k} documents retrieved",
            "top_k": top_k,
            "system_data_flow": True,
            "generation_time_ms": result.get("generation_time_ms", 0),
            "data_flow_path": result.get("data_flow", "DRAM→CSD→GPU")
        }
        
        return formatted_result
    
    def _query_sequential(self, 
                         query: str,
                         top_k: int,
                         include_metadata: bool) -> Dict[str, Any]:
        """Sequential query processing."""
        # Encode query
        query_embedding = self._encode_query(query)
        
        # Retrieve documents
        retrieved_docs = self._retrieve_documents(query_embedding, top_k)
        
        # Augment query
        augmented_query = self._augment_query(query, retrieved_docs)
        
        return {
            "query": query,
            "augmented_query": augmented_query,
            "retrieved_docs": retrieved_docs if include_metadata else len(retrieved_docs),
            "top_k": top_k
        }
    
    def _query_pipeline_parallel(self,
                               query: str,
                               top_k: int,
                               include_metadata: bool) -> Dict[str, Any]:
        """Pipeline parallel query processing."""
        # Start encoding
        encode_future = self._executor.submit(self._encode_query, query)
        
        # Prepare for retrieval (pre-warm caches, etc.)
        if self.csd_simulator:
            prefetch_future = self._executor.submit(self._prefetch_candidates, top_k * 2)
        
        # Get encoding result
        query_embedding = encode_future.result()
        
        # Start retrieval
        retrieve_future = self._executor.submit(
            self._retrieve_documents, 
            query_embedding, 
            top_k
        )
        
        # Prepare augmentation while retrieval is running
        augment_prep_future = self._executor.submit(self._prepare_augmentation)
        
        # Get retrieval results
        retrieved_docs = retrieve_future.result()
        
        # Complete augmentation
        augmented_query = self._augment_query(query, retrieved_docs)
        
        return {
            "query": query,
            "augmented_query": augmented_query,
            "retrieved_docs": retrieved_docs if include_metadata else len(retrieved_docs),
            "top_k": top_k,
            "pipeline_parallel": True
        }
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query with caching."""
        if self.config.enable_caching:
            cached = self.embedding_cache.get(query)
            if cached is not None:
                self.metrics.record_count("cache_hits")
                return cached
            
            self.metrics.record_count("cache_misses")
        
        embedding = self.encoder.encode([query])[0]
        
        if self.config.enable_caching:
            self.embedding_cache.put(query, embedding)
        
        return embedding
    
    def _retrieve_documents(self, 
                          query_embedding: np.ndarray,
                          top_k: int) -> List[Dict[str, Any]]:
        """Retrieve documents using vector store or CSD."""
        if self.csd_simulator and self.config.enable_csd_emulation:
            # Use CSD for retrieval
            indices = list(range(self.vector_store.get_statistics()["total_vectors"]))
            similarities = self.csd_simulator.compute_similarities(
                query_embedding,
                indices[:1000]  # Limit for demo
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Fetch documents from vector store
            # This is simplified - in production we'd have better integration
            results = self.vector_store.search(query_embedding, top_k)
        else:
            # Use vector store directly
            results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    def _augment_query(self, 
                      query: str,
                      retrieved_docs: List[Dict[str, Any]]) -> str:
        """Augment query with retrieved documents."""
        return self.augmentor.augment(query, retrieved_docs)
    
    def _prefetch_candidates(self, num_candidates: int) -> None:
        """Prefetch candidate vectors for CSD."""
        # This would intelligently prefetch likely candidates
        pass
    
    def _prepare_augmentation(self) -> None:
        """Prepare augmentation resources."""
        # Pre-compile templates, warm caches, etc.
        pass
    
    async def query_batch_async(self,
                              queries: List[str],
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries asynchronously."""
        tasks = []
        
        for query in queries:
            task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.query,
                    query,
                    top_k,
                    False  # Don't include full metadata for batch
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return results
    
    def query_batch(self,
                   queries: List[str],
                   top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries with batching optimization."""
        start_time = time.time()
        
        # Classify workload
        workload_type = self.workload_classifier.classify(queries)
        strategy = self.workload_classifier.recommend_strategy(workload_type)
        
        # Batch encode queries
        query_embeddings = np.array([self._encode_query(q) for q in queries])
        
        # Batch retrieve
        all_results = []
        
        if self.csd_simulator and self.config.enable_csd_emulation:
            # CSD batch processing
            for i in range(0, len(queries), strategy["batch_size"]):
                batch_queries = queries[i:i + strategy["batch_size"]]
                batch_embeddings = query_embeddings[i:i + strategy["batch_size"]]
                
                batch_results = []
                for q, emb in zip(batch_queries, batch_embeddings):
                    docs = self._retrieve_documents(emb, top_k)
                    aug = self._augment_query(q, docs)
                    batch_results.append({
                        "query": q,
                        "augmented_query": aug,
                        "retrieved_docs": len(docs),
                        "top_k": top_k
                    })
                
                all_results.extend(batch_results)
        else:
            # Regular batch processing
            for q, emb in zip(queries, query_embeddings):
                docs = self._retrieve_documents(emb, top_k)
                aug = self._augment_query(q, docs)
                all_results.append({
                    "query": q,
                    "augmented_query": aug,
                    "retrieved_docs": len(docs),
                    "top_k": top_k
                })
        
        elapsed = time.time() - start_time
        
        self.metrics.record_timing("batch_query", elapsed)
        self.metrics.record_count("batch_queries_processed", len(queries))
        
        return all_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            "vector_store": self.vector_store.get_statistics(),
            "metrics": self.metrics.get_all_metrics_summary(),
            "config": {
                "csd_emulation": self.config.enable_csd_emulation,
                "pipeline_parallel": self.config.enable_pipeline_parallel,
                "caching": self.config.enable_caching,
                "system_data_flow": self.config.enable_system_data_flow
            }
        }
        
        if self.csd_simulator:
            stats["csd_metrics"] = self.csd_simulator.get_metrics()
        
        if self.system_data_flow:
            stats["system_data_flow_metrics"] = self.system_data_flow.get_system_metrics()
        
        return stats
    
    def shutdown(self) -> None:
        """Cleanup resources."""
        logger.info("Shutting down Enhanced RAG-CSD Pipeline")
        
        self._executor.shutdown(wait=True)
        
        if self.csd_simulator:
            self.csd_simulator.shutdown()
        
        if self.system_data_flow:
            self.system_data_flow.cleanup()
        
        logger.info("Shutdown complete")