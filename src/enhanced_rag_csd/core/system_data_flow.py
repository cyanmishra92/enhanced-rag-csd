"""
Enhanced System Data Flow for RAG-CSD.
This module implements the complete system data flow including:
DRAM → CSD (encode, retrieve, augment) → GPU (generate)
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from concurrent.futures import Future
import numpy as np

from enhanced_rag_csd.core.system_memory import (
    SystemMemoryManager, MemoryType, MemoryConfig
)
from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemDataFlowConfig:
    """Configuration for system data flow."""
    # Memory configuration
    memory_config: MemoryConfig
    
    # Processing settings
    enable_async_processing: bool = True
    enable_prefetching: bool = True
    prefetch_lookahead: int = 2
    
    # Pipeline settings
    pipeline_depth: int = 3
    enable_overlapping: bool = True
    
    # Optimization settings
    enable_data_locality: bool = True
    enable_compression: bool = False
    compression_ratio: float = 0.7


@dataclass
class ProcessingStage:
    """Represents a processing stage in the pipeline."""
    stage_name: str
    input_memory_type: MemoryType
    output_memory_type: MemoryType
    processing_time_ms: float
    memory_overhead_factor: float = 1.0


@dataclass
class DataFlowMetrics:
    """Metrics for system data flow."""
    total_queries_processed: int = 0
    total_processing_time_s: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Stage-specific metrics
    dram_to_csd_time_s: float = 0.0
    csd_processing_time_s: float = 0.0
    csd_to_gpu_time_s: float = 0.0
    
    # Transfer metrics
    total_data_transferred_mb: float = 0.0
    p2p_transfers: int = 0
    pcie_transfers: int = 0
    
    # Memory metrics
    peak_dram_usage_mb: float = 0.0
    peak_gpu_usage_mb: float = 0.0
    peak_csd_usage_mb: float = 0.0


class SystemDataFlow:
    """
    Manages the complete system data flow for RAG-CSD processing.
    
    Data Flow:
    1. Query arrives in DRAM
    2. DRAM → CSD: Transfer query data for processing
    3. CSD Processing: Encode → Retrieve → Augment
    4. CSD → GPU: P2P transfer of augmented data
    5. GPU Processing: Generate response
    """
    
    def __init__(self, config: SystemDataFlowConfig):
        self.config = config
        self.memory_manager = SystemMemoryManager(config.memory_config)
        self.metrics = DataFlowMetrics()
        
        # Define processing stages
        self.stages = {
            "encode": ProcessingStage(
                stage_name="encode",
                input_memory_type=MemoryType.CSD,
                output_memory_type=MemoryType.CSD,
                processing_time_ms=1.0,
                memory_overhead_factor=1.2
            ),
            "retrieve": ProcessingStage(
                stage_name="retrieve",
                input_memory_type=MemoryType.CSD,
                output_memory_type=MemoryType.CSD,
                processing_time_ms=5.0,
                memory_overhead_factor=2.0
            ),
            "augment": ProcessingStage(
                stage_name="augment",
                input_memory_type=MemoryType.CSD,
                output_memory_type=MemoryType.CSD,
                processing_time_ms=0.5,
                memory_overhead_factor=1.5
            ),
            "generate": ProcessingStage(
                stage_name="generate",
                input_memory_type=MemoryType.GPU,
                output_memory_type=MemoryType.GPU,
                processing_time_ms=100.0,  # Fixed generation time
                memory_overhead_factor=3.0
            )
        }
        
        # Pipeline state
        self._pipeline_queue = asyncio.Queue(maxsize=self.config.pipeline_depth)
        self._active_transfers = {}
        
        logger.info("System Data Flow initialized with P2P transfers and pipeline parallelism")
    
    def process_query(self, 
                     query_data: np.ndarray, 
                     query_metadata: Dict[str, Any],
                     async_processing: bool = None) -> Union[Dict[str, Any], Future]:
        """
        Process a single query through the complete system data flow.
        
        Args:
            query_data: Query data (typically text embedding or raw text)
            query_metadata: Metadata about the query
            async_processing: Whether to process asynchronously
            
        Returns:
            Processing result or Future for async processing
        """
        if async_processing is None:
            async_processing = self.config.enable_async_processing
        
        if async_processing:
            return asyncio.create_task(self._process_query_async(query_data, query_metadata))
        else:
            return self._process_query_sync(query_data, query_metadata)
    
    def _process_query_sync(self, 
                           query_data: np.ndarray, 
                           query_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous query processing."""
        start_time = time.time()
        query_id = query_metadata.get("query_id", f"query_{int(time.time() * 1000)}")
        
        logger.debug(f"Starting synchronous processing for {query_id}")
        
        try:
            # Stage 1: Store query in DRAM
            dram_allocation_id = self._stage_dram_input(query_data, query_metadata)
            
            # Stage 2: Transfer DRAM → CSD
            csd_allocation_id = self._stage_dram_to_csd(dram_allocation_id, query_metadata)
            
            # Stage 3: CSD Processing (Encode → Retrieve → Augment)
            augmented_allocation_id = self._stage_csd_processing(csd_allocation_id, query_metadata)
            
            # Stage 4: Transfer CSD → GPU (P2P)
            gpu_allocation_id = self._stage_csd_to_gpu_p2p(augmented_allocation_id, query_metadata)
            
            # Stage 5: GPU Generation (simulated)
            result = self._stage_gpu_generation(gpu_allocation_id, query_metadata)
            
            # Cleanup intermediate allocations
            self._cleanup_allocations([dram_allocation_id, csd_allocation_id, 
                                     augmented_allocation_id, gpu_allocation_id])
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, query_data.nbytes)
            
            result.update({
                "processing_time_ms": processing_time * 1000,
                "query_id": query_id,
                "data_flow": "DRAM→CSD→GPU"
            })
            
            logger.debug(f"Completed synchronous processing for {query_id} in {processing_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in synchronous processing for {query_id}: {e}")
            raise
    
    async def _process_query_async(self, 
                                  query_data: np.ndarray, 
                                  query_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronous query processing with pipeline parallelism."""
        start_time = time.time()
        query_id = query_metadata.get("query_id", f"query_{int(time.time() * 1000)}")
        
        logger.debug(f"Starting asynchronous processing for {query_id}")
        
        try:
            # Stage 1: Store query in DRAM
            dram_allocation_id = self._stage_dram_input(query_data, query_metadata)
            
            # Create async tasks for pipeline stages
            tasks = []
            
            # Stage 2: Transfer DRAM → CSD (async)
            dram_to_csd_task = asyncio.create_task(
                self._async_dram_to_csd(dram_allocation_id, query_metadata)
            )
            tasks.append(("dram_to_csd", dram_to_csd_task))
            
            # Wait for DRAM → CSD to complete
            csd_allocation_id = await dram_to_csd_task
            
            # Stage 3: CSD Processing (async)
            csd_processing_task = asyncio.create_task(
                self._async_csd_processing(csd_allocation_id, query_metadata)
            )
            tasks.append(("csd_processing", csd_processing_task))
            
            # Wait for CSD processing to complete
            augmented_allocation_id = await csd_processing_task
            
            # Stage 4: CSD → GPU P2P (async)
            csd_to_gpu_task = asyncio.create_task(
                self._async_csd_to_gpu_p2p(augmented_allocation_id, query_metadata)
            )
            tasks.append(("csd_to_gpu", csd_to_gpu_task))
            
            # Wait for P2P transfer to complete
            gpu_allocation_id = await csd_to_gpu_task
            
            # Stage 5: GPU Generation
            result = self._stage_gpu_generation(gpu_allocation_id, query_metadata)
            
            # Cleanup
            self._cleanup_allocations([dram_allocation_id, csd_allocation_id, 
                                     augmented_allocation_id, gpu_allocation_id])
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, query_data.nbytes)
            
            result.update({
                "processing_time_ms": processing_time * 1000,
                "query_id": query_id,
                "data_flow": "DRAM→CSD→GPU (async)",
                "pipeline_stages": len(tasks)
            })
            
            logger.debug(f"Completed asynchronous processing for {query_id} in {processing_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in asynchronous processing for {query_id}: {e}")
            raise
    
    def _stage_dram_input(self, 
                         query_data: np.ndarray, 
                         query_metadata: Dict[str, Any]) -> str:
        """Stage 1: Store query data in DRAM."""
        stage_start = time.time()
        
        # Allocate memory in DRAM
        allocation_id = self.memory_manager.allocate_memory(
            MemoryType.DRAM, 
            query_data.nbytes, 
            query_data
        )
        
        stage_time = time.time() - stage_start
        logger.debug(f"DRAM input stage: {stage_time*1000:.2f}ms, {query_data.nbytes/1024:.1f}KB")
        
        return allocation_id
    
    def _stage_dram_to_csd(self, 
                          dram_allocation_id: str, 
                          query_metadata: Dict[str, Any]) -> str:
        """Stage 2: Transfer data from DRAM to CSD."""
        stage_start = time.time()
        
        # Transfer data DRAM → CSD
        csd_allocation_id = self.memory_manager.transfer_data(
            MemoryType.DRAM, 
            dram_allocation_id, 
            MemoryType.CSD, 
            async_transfer=False
        )
        
        stage_time = time.time() - stage_start
        self.metrics.dram_to_csd_time_s += stage_time
        self.metrics.pcie_transfers += 1
        
        logger.debug(f"DRAM→CSD transfer: {stage_time*1000:.2f}ms")
        
        return csd_allocation_id
    
    async def _async_dram_to_csd(self, 
                                dram_allocation_id: str, 
                                query_metadata: Dict[str, Any]) -> str:
        """Async version of DRAM to CSD transfer."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._stage_dram_to_csd, dram_allocation_id, query_metadata
        )
    
    def _stage_csd_processing(self, 
                             csd_allocation_id: str, 
                             query_metadata: Dict[str, Any]) -> str:
        """Stage 3: CSD Processing (Encode → Retrieve → Augment)."""
        stage_start = time.time()
        
        # Get input data
        input_data = self.memory_manager.get_data(MemoryType.CSD, csd_allocation_id)
        if input_data is None:
            raise ValueError("Invalid CSD allocation for processing")
        
        # Simulate the three CSD processing stages
        current_data = input_data
        current_allocation_id = csd_allocation_id
        
        for stage_name in ["encode", "retrieve", "augment"]:
            stage = self.stages[stage_name]
            
            # Simulate processing time
            time.sleep(stage.processing_time_ms / 1000)
            
            # Simulate memory growth (e.g., retrieved documents)
            if stage_name == "retrieve":
                # Simulate adding retrieved documents
                augmented_size = int(current_data.nbytes * stage.memory_overhead_factor)
                augmented_data = np.zeros(augmented_size, dtype=np.uint8)
                augmented_data[:current_data.nbytes] = current_data.view(np.uint8)
                
                # Allocate new memory for augmented data
                new_allocation_id = self.memory_manager.allocate_memory(
                    MemoryType.CSD, 
                    augmented_data.nbytes, 
                    augmented_data
                )
                
                # Free old allocation if different
                if current_allocation_id != csd_allocation_id:
                    self.memory_manager.deallocate_memory(MemoryType.CSD, current_allocation_id)
                
                current_data = augmented_data
                current_allocation_id = new_allocation_id
            
            logger.debug(f"CSD {stage_name} stage: {stage.processing_time_ms:.2f}ms")
        
        total_stage_time = time.time() - stage_start
        self.metrics.csd_processing_time_s += total_stage_time
        
        logger.debug(f"CSD processing complete: {total_stage_time*1000:.2f}ms")
        
        return current_allocation_id
    
    async def _async_csd_processing(self, 
                                   csd_allocation_id: str, 
                                   query_metadata: Dict[str, Any]) -> str:
        """Async version of CSD processing."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._stage_csd_processing, csd_allocation_id, query_metadata
        )
    
    def _stage_csd_to_gpu_p2p(self, 
                             augmented_allocation_id: str, 
                             query_metadata: Dict[str, Any]) -> str:
        """Stage 4: P2P transfer from CSD to GPU."""
        stage_start = time.time()
        
        # P2P transfer CSD → GPU
        gpu_allocation_id = self.memory_manager.transfer_data(
            MemoryType.CSD, 
            augmented_allocation_id, 
            MemoryType.GPU, 
            async_transfer=False
        )
        
        stage_time = time.time() - stage_start
        self.metrics.csd_to_gpu_time_s += stage_time
        self.metrics.p2p_transfers += 1
        
        logger.debug(f"CSD→GPU P2P transfer: {stage_time*1000:.2f}ms")
        
        return gpu_allocation_id
    
    async def _async_csd_to_gpu_p2p(self, 
                                   augmented_allocation_id: str, 
                                   query_metadata: Dict[str, Any]) -> str:
        """Async version of CSD to GPU P2P transfer."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._stage_csd_to_gpu_p2p, augmented_allocation_id, query_metadata
        )
    
    def _stage_gpu_generation(self, 
                             gpu_allocation_id: str, 
                             query_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: GPU Generation (simulated)."""
        stage_start = time.time()
        
        # Get GPU data
        gpu_data = self.memory_manager.get_data(MemoryType.GPU, gpu_allocation_id)
        if gpu_data is None:
            raise ValueError("Invalid GPU allocation for generation")
        
        # Simulate generation time (fixed regardless of system since GPU compute is constant)
        generation_stage = self.stages["generate"]
        time.sleep(generation_stage.processing_time_ms / 1000)
        
        stage_time = time.time() - stage_start
        
        logger.debug(f"GPU generation: {stage_time*1000:.2f}ms")
        
        return {
            "generated_response": f"Generated response from {gpu_data.nbytes} bytes of augmented data",
            "generation_time_ms": stage_time * 1000,
            "gpu_data_size_bytes": gpu_data.nbytes
        }
    
    def _cleanup_allocations(self, allocation_ids: List[str]) -> None:
        """Clean up memory allocations."""
        memory_types = [MemoryType.DRAM, MemoryType.CSD, MemoryType.GPU]
        
        for allocation_id in allocation_ids:
            for memory_type in memory_types:
                try:
                    self.memory_manager.deallocate_memory(memory_type, allocation_id)
                except:
                    # Allocation might not exist in this memory type
                    pass
    
    def _update_metrics(self, processing_time: float, data_size_bytes: int) -> None:
        """Update processing metrics."""
        self.metrics.total_queries_processed += 1
        self.metrics.total_processing_time_s += processing_time
        self.metrics.avg_latency_ms = (
            self.metrics.total_processing_time_s / max(self.metrics.total_queries_processed, 1)
        ) * 1000
        self.metrics.total_data_transferred_mb += data_size_bytes / (1024 * 1024)
        
        # Update peak memory usage
        memory_stats = self.memory_manager.get_comprehensive_stats()
        self.metrics.peak_dram_usage_mb = max(
            self.metrics.peak_dram_usage_mb,
            memory_stats["memory_subsystems"]["dram"]["allocated_mb"]
        )
        self.metrics.peak_gpu_usage_mb = max(
            self.metrics.peak_gpu_usage_mb,
            memory_stats["memory_subsystems"]["gpu"]["allocated_mb"]
        )
        self.metrics.peak_csd_usage_mb = max(
            self.metrics.peak_csd_usage_mb,
            memory_stats["memory_subsystems"]["csd"]["allocated_mb"]
        )
    
    def process_batch(self, 
                     queries_data: List[np.ndarray], 
                     queries_metadata: List[Dict[str, Any]],
                     batch_size: int = 4) -> List[Dict[str, Any]]:
        """Process multiple queries with batching optimization."""
        start_time = time.time()
        
        logger.info(f"Processing batch of {len(queries_data)} queries with batch_size={batch_size}")
        
        results = []
        
        # Process in batches
        for i in range(0, len(queries_data), batch_size):
            batch_queries = queries_data[i:i + batch_size]
            batch_metadata = queries_metadata[i:i + batch_size]
            
            # Process batch concurrently if async is enabled
            if self.config.enable_async_processing:
                batch_results = asyncio.run(self._process_batch_async(batch_queries, batch_metadata))
            else:
                batch_results = [
                    self._process_query_sync(query, metadata)
                    for query, metadata in zip(batch_queries, batch_metadata)
                ]
            
            results.extend(batch_results)
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing complete: {len(queries_data)} queries in {total_time:.2f}s "
                   f"({total_time/len(queries_data)*1000:.1f}ms per query)")
        
        return results
    
    async def _process_batch_async(self, 
                                  batch_queries: List[np.ndarray], 
                                  batch_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of queries asynchronously."""
        tasks = [
            self._process_query_async(query, metadata)
            for query, metadata in zip(batch_queries, batch_metadata)
        ]
        
        return await asyncio.gather(*tasks)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        memory_stats = self.memory_manager.get_comprehensive_stats()
        
        return {
            "data_flow_metrics": {
                "total_queries_processed": self.metrics.total_queries_processed,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "total_processing_time_s": self.metrics.total_processing_time_s,
                "dram_to_csd_time_s": self.metrics.dram_to_csd_time_s,
                "csd_processing_time_s": self.metrics.csd_processing_time_s,
                "csd_to_gpu_time_s": self.metrics.csd_to_gpu_time_s,
                "total_data_transferred_mb": self.metrics.total_data_transferred_mb,
                "p2p_transfers": self.metrics.p2p_transfers,
                "pcie_transfers": self.metrics.pcie_transfers
            },
            "memory_metrics": memory_stats,
            "system_config": {
                "async_processing": self.config.enable_async_processing,
                "p2p_enabled": self.config.memory_config.enable_p2p,
                "pipeline_depth": self.config.pipeline_depth
            }
        }
    
    def cleanup(self) -> None:
        """Cleanup system resources."""
        self.memory_manager.cleanup()
        logger.info("System Data Flow cleanup complete")