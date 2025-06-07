"""
System Memory Management for Enhanced RAG-CSD.
This module simulates the complete memory hierarchy including DRAM, GPU memory, 
and computational storage device memory with realistic transfer characteristics.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future

from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryType(Enum):
    """Types of memory in the system."""
    DRAM = "dram"
    GPU = "gpu"
    CSD = "csd"


@dataclass
class MemoryConfig:
    """Configuration for memory subsystem."""
    # Memory capacities (in MB)
    dram_capacity_mb: int = 16384  # 16GB
    gpu_memory_mb: int = 8192      # 8GB
    csd_memory_mb: int = 4096      # 4GB
    
    # Transfer bandwidths (MB/s)
    dram_bandwidth_mbps: int = 51200     # DDR4-3200 dual channel
    gpu_memory_bandwidth_mbps: int = 900000  # GPU HBM
    pcie_bandwidth_mbps: int = 15750     # PCIe 4.0 x16
    p2p_bandwidth_mbps: int = 12000      # P2P GPU-Storage
    
    # Transfer latencies (microseconds)
    dram_latency_us: float = 0.1
    gpu_memory_latency_us: float = 0.01
    pcie_latency_us: float = 2.0
    p2p_latency_us: float = 1.0
    
    # System settings
    enable_p2p: bool = True
    memory_alignment: int = 64  # bytes


@dataclass
class MemoryAllocation:
    """Represents a memory allocation."""
    allocation_id: str
    memory_type: MemoryType
    size_bytes: int
    data: Optional[np.ndarray] = None
    timestamp: float = 0.0
    access_count: int = 0


@dataclass
class TransferMetrics:
    """Metrics for memory transfers."""
    total_transfers: int = 0
    total_bytes_transferred: int = 0
    total_transfer_time: float = 0.0
    avg_bandwidth_mbps: float = 0.0
    p2p_transfers: int = 0
    pcie_transfers: int = 0


class SystemBus:
    """Simulates the system bus for inter-component communication."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.transfer_metrics = TransferMetrics()
        self._bus_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def transfer_data(self, 
                     source: MemoryType, 
                     dest: MemoryType, 
                     data: np.ndarray,
                     async_transfer: bool = False) -> Union[np.ndarray, Future]:
        """
        Transfer data between memory subsystems.
        
        Args:
            source: Source memory type
            dest: Destination memory type
            data: Data to transfer
            async_transfer: Whether to perform transfer asynchronously
            
        Returns:
            Transferred data or Future for async transfers
        """
        if async_transfer:
            future = self._executor.submit(self._perform_transfer, source, dest, data)
            return future
        else:
            return self._perform_transfer(source, dest, data)
    
    def _perform_transfer(self, 
                         source: MemoryType, 
                         dest: MemoryType, 
                         data: np.ndarray) -> np.ndarray:
        """Perform the actual data transfer with realistic timing."""
        start_time = time.time()
        data_size_bytes = data.nbytes
        
        # Determine transfer path and characteristics
        bandwidth_mbps, latency_us, is_p2p = self._get_transfer_characteristics(source, dest)
        
        with self._bus_lock:
            # Simulate transfer latency
            time.sleep(latency_us / 1_000_000)  # Convert to seconds
            
            # Simulate bandwidth constraints
            transfer_time = data_size_bytes / (bandwidth_mbps * 1024 * 1024)
            time.sleep(transfer_time)
            
            # Update metrics
            self.transfer_metrics.total_transfers += 1
            self.transfer_metrics.total_bytes_transferred += data_size_bytes
            self.transfer_metrics.total_transfer_time += time.time() - start_time
            
            if is_p2p:
                self.transfer_metrics.p2p_transfers += 1
            else:
                self.transfer_metrics.pcie_transfers += 1
            
            # Calculate average bandwidth
            if self.transfer_metrics.total_transfer_time > 0:
                self.transfer_metrics.avg_bandwidth_mbps = (
                    self.transfer_metrics.total_bytes_transferred / 
                    (self.transfer_metrics.total_transfer_time * 1024 * 1024)
                )
        
        logger.debug(f"Transfer {source.value}→{dest.value}: {data_size_bytes/1024:.1f}KB "
                    f"in {(time.time() - start_time)*1000:.2f}ms")
        
        return data.copy()  # Simulate data copy
    
    def _get_transfer_characteristics(self, 
                                    source: MemoryType, 
                                    dest: MemoryType) -> Tuple[int, float, bool]:
        """Get transfer bandwidth, latency, and P2P flag for a transfer path."""
        # P2P transfers (GPU ↔ CSD)
        if self.config.enable_p2p and {source, dest} == {MemoryType.GPU, MemoryType.CSD}:
            return self.config.p2p_bandwidth_mbps, self.config.p2p_latency_us, True
        
        # PCIe transfers (DRAM ↔ GPU, DRAM ↔ CSD)
        if source == MemoryType.DRAM or dest == MemoryType.DRAM:
            return self.config.pcie_bandwidth_mbps, self.config.pcie_latency_us, False
        
        # Default to PCIe for any other transfers
        return self.config.pcie_bandwidth_mbps, self.config.pcie_latency_us, False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get transfer metrics."""
        return {
            "total_transfers": self.transfer_metrics.total_transfers,
            "total_bytes_mb": self.transfer_metrics.total_bytes_transferred / (1024 * 1024),
            "total_transfer_time_s": self.transfer_metrics.total_transfer_time,
            "avg_bandwidth_mbps": self.transfer_metrics.avg_bandwidth_mbps,
            "p2p_transfers": self.transfer_metrics.p2p_transfers,
            "pcie_transfers": self.transfer_metrics.pcie_transfers
        }


class MemorySubsystem:
    """Represents a memory subsystem (DRAM, GPU, or CSD)."""
    
    def __init__(self, memory_type: MemoryType, capacity_mb: int, bandwidth_mbps: int):
        self.memory_type = memory_type
        self.capacity_bytes = capacity_mb * 1024 * 1024
        self.bandwidth_mbps = bandwidth_mbps
        self.allocated_bytes = 0
        self.allocations: Dict[str, MemoryAllocation] = {}
        self._lock = threading.Lock()
        self._allocation_counter = 0
    
    def allocate(self, size_bytes: int, data: Optional[np.ndarray] = None) -> str:
        """Allocate memory and return allocation ID."""
        with self._lock:
            if self.allocated_bytes + size_bytes > self.capacity_bytes:
                # Try to free some memory using LRU
                self._free_lru_allocations(size_bytes)
                
                if self.allocated_bytes + size_bytes > self.capacity_bytes:
                    raise MemoryError(f"Not enough {self.memory_type.value} memory: "
                                    f"requested {size_bytes/1024/1024:.1f}MB, "
                                    f"available {(self.capacity_bytes - self.allocated_bytes)/1024/1024:.1f}MB")
            
            allocation_id = f"{self.memory_type.value}_{self._allocation_counter}"
            self._allocation_counter += 1
            
            allocation = MemoryAllocation(
                allocation_id=allocation_id,
                memory_type=self.memory_type,
                size_bytes=size_bytes,
                data=data,
                timestamp=time.time()
            )
            
            self.allocations[allocation_id] = allocation
            self.allocated_bytes += size_bytes
            
            logger.debug(f"Allocated {size_bytes/1024:.1f}KB in {self.memory_type.value} "
                        f"({self.allocated_bytes/1024/1024:.1f}MB/{self.capacity_bytes/1024/1024:.1f}MB used)")
            
            return allocation_id
    
    def deallocate(self, allocation_id: str) -> None:
        """Deallocate memory."""
        with self._lock:
            if allocation_id in self.allocations:
                allocation = self.allocations.pop(allocation_id)
                self.allocated_bytes -= allocation.size_bytes
                logger.debug(f"Deallocated {allocation.size_bytes/1024:.1f}KB from {self.memory_type.value}")
    
    def get_allocation(self, allocation_id: str) -> Optional[MemoryAllocation]:
        """Get allocation by ID."""
        with self._lock:
            allocation = self.allocations.get(allocation_id)
            if allocation:
                allocation.access_count += 1
                allocation.timestamp = time.time()
            return allocation
    
    def _free_lru_allocations(self, needed_bytes: int) -> None:
        """Free least recently used allocations to make space."""
        # Sort by timestamp (oldest first)
        sorted_allocations = sorted(
            self.allocations.items(), 
            key=lambda x: x[1].timestamp
        )
        
        freed_bytes = 0
        for allocation_id, allocation in sorted_allocations:
            if freed_bytes >= needed_bytes:
                break
            
            self.allocations.pop(allocation_id)
            self.allocated_bytes -= allocation.size_bytes
            freed_bytes += allocation.size_bytes
            
            logger.debug(f"LRU freed {allocation.size_bytes/1024:.1f}KB from {self.memory_type.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory subsystem statistics."""
        with self._lock:
            return {
                "type": self.memory_type.value,
                "capacity_mb": self.capacity_bytes / (1024 * 1024),
                "allocated_mb": self.allocated_bytes / (1024 * 1024),
                "utilization_percent": (self.allocated_bytes / self.capacity_bytes) * 100,
                "num_allocations": len(self.allocations),
                "bandwidth_mbps": self.bandwidth_mbps
            }


class SystemMemoryManager:
    """Comprehensive memory management for the entire system."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # Initialize memory subsystems
        self.dram = MemorySubsystem(
            MemoryType.DRAM, 
            self.config.dram_capacity_mb, 
            self.config.dram_bandwidth_mbps
        )
        self.gpu_memory = MemorySubsystem(
            MemoryType.GPU, 
            self.config.gpu_memory_mb, 
            self.config.gpu_memory_bandwidth_mbps
        )
        self.csd_memory = MemorySubsystem(
            MemoryType.CSD, 
            self.config.csd_memory_mb, 
            self.config.pcie_bandwidth_mbps
        )
        
        # Initialize system bus
        self.system_bus = SystemBus(self.config)
        
        self._subsystems = {
            MemoryType.DRAM: self.dram,
            MemoryType.GPU: self.gpu_memory,
            MemoryType.CSD: self.csd_memory
        }
        
        logger.info(f"System Memory Manager initialized: "
                   f"DRAM={self.config.dram_capacity_mb}MB, "
                   f"GPU={self.config.gpu_memory_mb}MB, "
                   f"CSD={self.config.csd_memory_mb}MB")
    
    def allocate_memory(self, 
                       memory_type: MemoryType, 
                       size_bytes: int, 
                       data: Optional[np.ndarray] = None) -> str:
        """Allocate memory in specified subsystem."""
        subsystem = self._subsystems[memory_type]
        return subsystem.allocate(size_bytes, data)
    
    def deallocate_memory(self, memory_type: MemoryType, allocation_id: str) -> None:
        """Deallocate memory from specified subsystem."""
        subsystem = self._subsystems[memory_type]
        subsystem.deallocate(allocation_id)
    
    def transfer_data(self, 
                     source_type: MemoryType, 
                     source_allocation_id: str,
                     dest_type: MemoryType, 
                     async_transfer: bool = False) -> Union[str, Future]:
        """
        Transfer data between memory subsystems.
        
        Args:
            source_type: Source memory type
            source_allocation_id: Source allocation ID
            dest_type: Destination memory type
            async_transfer: Whether to perform transfer asynchronously
            
        Returns:
            Destination allocation ID or Future for async transfers
        """
        # Get source data
        source_subsystem = self._subsystems[source_type]
        source_allocation = source_subsystem.get_allocation(source_allocation_id)
        
        if not source_allocation or source_allocation.data is None:
            raise ValueError(f"Invalid source allocation: {source_allocation_id}")
        
        if async_transfer:
            future = self.system_bus.transfer_data(
                source_type, dest_type, source_allocation.data, async_transfer=True
            )
            
            def _complete_async_transfer(transfer_future):
                transferred_data = transfer_future.result()
                dest_subsystem = self._subsystems[dest_type]
                return dest_subsystem.allocate(transferred_data.nbytes, transferred_data)
            
            # Chain the allocation after transfer completes
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=1)
            return executor.submit(_complete_async_transfer, future)
        else:
            # Perform synchronous transfer
            transferred_data = self.system_bus.transfer_data(
                source_type, dest_type, source_allocation.data, async_transfer=False
            )
            
            # Allocate in destination
            dest_subsystem = self._subsystems[dest_type]
            return dest_subsystem.allocate(transferred_data.nbytes, transferred_data)
    
    def get_data(self, memory_type: MemoryType, allocation_id: str) -> Optional[np.ndarray]:
        """Get data from memory allocation."""
        subsystem = self._subsystems[memory_type]
        allocation = subsystem.get_allocation(allocation_id)
        return allocation.data if allocation else None
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system memory statistics."""
        stats = {
            "memory_subsystems": {
                mem_type.value: subsystem.get_stats() 
                for mem_type, subsystem in self._subsystems.items()
            },
            "system_bus": self.system_bus.get_metrics(),
            "config": {
                "p2p_enabled": self.config.enable_p2p,
                "dram_capacity_mb": self.config.dram_capacity_mb,
                "gpu_memory_mb": self.config.gpu_memory_mb,
                "csd_memory_mb": self.config.csd_memory_mb
            }
        }
        
        # Calculate total system utilization
        total_capacity = (self.config.dram_capacity_mb + 
                         self.config.gpu_memory_mb + 
                         self.config.csd_memory_mb)
        total_allocated = sum(
            subsystem.allocated_bytes / (1024 * 1024) 
            for subsystem in self._subsystems.values()
        )
        
        stats["system_summary"] = {
            "total_capacity_mb": total_capacity,
            "total_allocated_mb": total_allocated,
            "system_utilization_percent": (total_allocated / total_capacity) * 100
        }
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("System Memory Manager cleanup complete")