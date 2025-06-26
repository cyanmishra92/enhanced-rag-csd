"""
Realistic Hardware Models for CSD Computational and Communication Delays.

This module provides accurate timing models based on real hardware specifications 
for various CSD implementations including FPGA, ARM cores, and commercial CSDs.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CSDHardwareType(Enum):
    """Types of CSD hardware implementations."""
    AMD_VERSAL_FPGA = "amd_versal_fpga"           # AMD Versal ACAP FPGA
    ARM_CORTEX_A78 = "arm_cortex_a78"             # ARM Cortex-A78 cores
    SK_HYNIX_CSD = "sk_hynix_csd"                 # SK-Hynix computational SSD
    SAMSUNG_SMARTSSD = "samsung_smartssd"         # Samsung SmartSSD
    XILINX_ALVEO = "xilinx_alveo"                 # Xilinx Alveo U50/U55
    CUSTOM_ASIC = "custom_asic"                   # Custom ASIC implementation


@dataclass
class ComputeSpecs:
    """Computational specifications for hardware."""
    peak_ops_per_sec: float        # Peak operations per second
    memory_bandwidth_gbps: float   # Memory bandwidth in GB/s
    cache_size_mb: int             # On-chip cache size
    power_watts: float             # Power consumption
    frequency_mhz: int             # Operating frequency
    parallel_units: int           # Number of parallel compute units
    vector_width: int             # SIMD vector width
    precision: str                # "fp32", "fp16", "int8", "bfloat16"


@dataclass
class CommunicationSpecs:
    """Communication specifications for hardware."""
    pcie_gen: int                  # PCIe generation (3, 4, 5)
    pcie_lanes: int               # Number of PCIe lanes
    max_bandwidth_gbps: float     # Maximum bandwidth
    latency_ns: int               # Base communication latency
    dma_engines: int              # Number of DMA engines
    queue_depth: int              # Maximum queue depth


class CSDHardwareModel:
    """Model for a specific CSD hardware implementation."""
    
    def __init__(self, hardware_type: CSDHardwareType):
        self.hardware_type = hardware_type
        self.compute_specs = self._get_compute_specs()
        self.comm_specs = self._get_communication_specs()
        
        # Runtime state
        self.utilization = 0.0  # Current utilization (0.0-1.0)
        self.thermal_state = 0.5  # Thermal state (0.0-1.0, affects performance)
        
        logger.info(f"Initialized {hardware_type.value} CSD model")
    
    def _get_compute_specs(self) -> ComputeSpecs:
        """Get computational specifications based on hardware type."""
        specs = {
            CSDHardwareType.AMD_VERSAL_FPGA: ComputeSpecs(
                peak_ops_per_sec=1.8e12,      # 1.8 TOP/s for ML inference
                memory_bandwidth_gbps=58.6,    # HBM2 bandwidth
                cache_size_mb=64,              # On-chip BRAM
                power_watts=75,                # Typical power consumption
                frequency_mhz=700,             # DSP operating frequency
                parallel_units=1968,           # DSP48 blocks
                vector_width=16,               # Vector processing width
                precision="fp16"               # Optimized for fp16 ML
            ),
            
            CSDHardwareType.ARM_CORTEX_A78: ComputeSpecs(
                peak_ops_per_sec=4.2e11,      # ~420 GOPS with NEON
                memory_bandwidth_gbps=25.6,    # LPDDR5 bandwidth
                cache_size_mb=4,               # L3 cache size
                power_watts=5,                 # Mobile SoC power
                frequency_mhz=3000,            # Max boost frequency
                parallel_units=8,              # Typical core count
                vector_width=8,                # NEON vector width
                precision="fp32"               # General purpose
            ),
            
            CSDHardwareType.SK_HYNIX_CSD: ComputeSpecs(
                peak_ops_per_sec=2.1e11,      # ~210 GOPS estimated
                memory_bandwidth_gbps=12.8,    # DDR4-3200 interface
                cache_size_mb=8,               # On-SSD cache
                power_watts=15,                # SSD + compute power
                frequency_mhz=1200,            # Embedded processor
                parallel_units=4,              # Parallel compute engines
                vector_width=4,                # Limited vectorization
                precision="fp32"               # Mixed precision
            ),
            
            CSDHardwareType.SAMSUNG_SMARTSSD: ComputeSpecs(
                peak_ops_per_sec=1.6e11,      # ~160 GOPS Xilinx FPGA
                memory_bandwidth_gbps=14.0,    # NVMe + DDR4
                cache_size_mb=16,              # Large cache buffer
                power_watts=25,                # Higher power for compute
                frequency_mhz=250,             # Conservative FPGA freq
                parallel_units=16,             # FPGA parallel units
                vector_width=8,                # FPGA vector processing
                precision="fp16"               # Optimized precision
            ),
            
            CSDHardwareType.XILINX_ALVEO: ComputeSpecs(
                peak_ops_per_sec=3.7e12,      # 3.7 TOP/s peak
                memory_bandwidth_gbps=77.0,    # HBM2 high bandwidth
                cache_size_mb=128,             # Large on-chip memory
                power_watts=150,               # High-performance power
                frequency_mhz=1000,            # High FPGA frequency
                parallel_units=2592,           # DSP units
                vector_width=32,               # Wide vector processing
                precision="fp16"               # ML optimized
            ),
            
            CSDHardwareType.CUSTOM_ASIC: ComputeSpecs(
                peak_ops_per_sec=5.0e12,      # 5 TOP/s custom design
                memory_bandwidth_gbps=100.0,   # Custom memory interface
                cache_size_mb=256,             # Large custom cache
                power_watts=200,               # High-end power
                frequency_mhz=1500,            # Custom frequency
                parallel_units=1024,           # Optimized units
                vector_width=64,               # Very wide vectors
                precision="bfloat16"           # Custom precision
            )
        }
        
        return specs[self.hardware_type]
    
    def _get_communication_specs(self) -> CommunicationSpecs:
        """Get communication specifications based on hardware type."""
        specs = {
            CSDHardwareType.AMD_VERSAL_FPGA: CommunicationSpecs(
                pcie_gen=4,
                pcie_lanes=16,
                max_bandwidth_gbps=32.0,       # PCIe 4.0 x16
                latency_ns=500,                # Low FPGA latency
                dma_engines=8,                 # Multiple DMA engines
                queue_depth=256
            ),
            
            CSDHardwareType.ARM_CORTEX_A78: CommunicationSpecs(
                pcie_gen=4,
                pcie_lanes=8,
                max_bandwidth_gbps=16.0,       # PCIe 4.0 x8
                latency_ns=800,                # Higher latency
                dma_engines=4,                 # Limited DMA
                queue_depth=128
            ),
            
            CSDHardwareType.SK_HYNIX_CSD: CommunicationSpecs(
                pcie_gen=4,
                pcie_lanes=4,
                max_bandwidth_gbps=8.0,        # Standard NVMe
                latency_ns=1200,               # SSD controller latency
                dma_engines=2,                 # Basic DMA
                queue_depth=64
            ),
            
            CSDHardwareType.SAMSUNG_SMARTSSD: CommunicationSpecs(
                pcie_gen=4,
                pcie_lanes=4,
                max_bandwidth_gbps=8.0,        # Standard NVMe
                latency_ns=1000,               # Optimized latency
                dma_engines=4,                 # Enhanced DMA
                queue_depth=128
            ),
            
            CSDHardwareType.XILINX_ALVEO: CommunicationSpecs(
                pcie_gen=4,
                pcie_lanes=16,
                max_bandwidth_gbps=32.0,       # High-end PCIe
                latency_ns=300,                # Very low latency
                dma_engines=16,                # Many DMA engines
                queue_depth=512
            ),
            
            CSDHardwareType.CUSTOM_ASIC: CommunicationSpecs(
                pcie_gen=5,
                pcie_lanes=16,
                max_bandwidth_gbps=64.0,       # PCIe 5.0 x16
                latency_ns=200,                # Optimal latency
                dma_engines=32,                # Maximum DMA
                queue_depth=1024
            )
        }
        
        return specs[self.hardware_type]
    
    def calculate_ml_operation_time(self, operation: str, data_shape: Tuple[int, ...], 
                                   data_type: str = "fp32") -> float:
        """Calculate time for ML operation based on hardware specs."""
        # Base operation counts
        operation_counts = {
            "matrix_multiply": np.prod(data_shape) * data_shape[-1],  # O(n³) for square matrices
            "vector_add": np.prod(data_shape),                        # O(n)
            "dot_product": data_shape[-1] if len(data_shape) > 0 else 1,  # O(n)
            "softmax": np.prod(data_shape) * 3,                      # exp + sum + div
            "relu": np.prod(data_shape),                             # O(n)
            "attention": np.prod(data_shape) * data_shape[-1] * 2,   # Q·K + softmax·V
            "embedding_lookup": data_shape[-1] if len(data_shape) > 0 else 384,  # Memory access
            "similarity_compute": data_shape[-1] if len(data_shape) > 0 else 384,  # Dot product
            "layer_norm": np.prod(data_shape) * 2,                   # Mean + variance
            "convolution": np.prod(data_shape) * 9,                  # 3x3 kernel estimate
        }
        
        total_ops = operation_counts.get(operation, np.prod(data_shape))
        
        # Data type efficiency multipliers
        type_multipliers = {
            "fp32": 1.0,
            "fp16": 2.0,    # 2x throughput for fp16
            "bfloat16": 2.0, 
            "int8": 4.0,    # 4x throughput for int8
            "int16": 2.0
        }
        
        # Precision-specific performance
        precision_ops_per_sec = self.compute_specs.peak_ops_per_sec
        if data_type in type_multipliers:
            precision_ops_per_sec *= type_multipliers[data_type]
        
        # Calculate theoretical time
        theoretical_time = total_ops / precision_ops_per_sec
        
        # Apply real-world efficiency factors
        utilization_factor = 0.7 + (0.3 * (1.0 - self.utilization))  # Lower utilization = higher efficiency
        thermal_factor = 1.0 + (0.3 * self.thermal_state)            # Thermal throttling
        memory_bound_factor = self._calculate_memory_bound_factor(total_ops, data_shape)
        
        actual_time = theoretical_time * thermal_factor * memory_bound_factor / utilization_factor
        
        # Add hardware-specific overheads
        setup_overhead = self._get_operation_setup_time(operation)
        
        return actual_time + setup_overhead
    
    def calculate_memory_transfer_time(self, data_size_bytes: int, 
                                     transfer_type: str = "host_to_device") -> float:
        """Calculate memory transfer time based on communication specs."""
        # Transfer types and their efficiency factors
        transfer_efficiency = {
            "host_to_device": 0.8,     # PCIe downstream efficiency
            "device_to_host": 0.85,    # PCIe upstream efficiency  
            "device_to_device": 0.95,  # On-device transfer
            "p2p_transfer": 0.9        # Peer-to-peer transfer
        }
        
        efficiency = transfer_efficiency.get(transfer_type, 0.8)
        effective_bandwidth = self.comm_specs.max_bandwidth_gbps * 1e9 * efficiency  # bytes/sec
        
        # Calculate transfer time
        transfer_time = data_size_bytes / effective_bandwidth
        
        # Add latency overhead
        latency_overhead = self.comm_specs.latency_ns * 1e-9
        
        # DMA setup overhead
        dma_setup_overhead = 50e-6  # 50μs typical DMA setup
        
        return transfer_time + latency_overhead + dma_setup_overhead
    
    def _calculate_memory_bound_factor(self, total_ops: int, data_shape: Tuple[int, ...]) -> float:
        """Calculate if operation is memory-bound or compute-bound."""
        # Estimate memory accesses
        data_size_bytes = np.prod(data_shape) * 4  # Assume fp32
        memory_accesses = data_size_bytes / self.compute_specs.cache_size_mb / 1024 / 1024
        
        # Arithmetic intensity (ops per byte)
        arithmetic_intensity = total_ops / data_size_bytes if data_size_bytes > 0 else 1.0
        
        # Hardware-specific arithmetic intensity threshold
        roofline_threshold = self.compute_specs.peak_ops_per_sec / (self.compute_specs.memory_bandwidth_gbps * 1e9)
        
        if arithmetic_intensity < roofline_threshold:
            # Memory-bound operation
            memory_bound_factor = 1.0 + (2.0 * memory_accesses)  # Penalty for cache misses
        else:
            # Compute-bound operation
            memory_bound_factor = 1.0
        
        return min(memory_bound_factor, 5.0)  # Cap at 5x penalty
    
    def _get_operation_setup_time(self, operation: str) -> float:
        """Get setup overhead time for different operations."""
        # Hardware-specific setup times (in seconds)
        base_setup = {
            CSDHardwareType.AMD_VERSAL_FPGA: 10e-6,      # 10μs FPGA kernel launch
            CSDHardwareType.ARM_CORTEX_A78: 1e-6,        # 1μs ARM setup
            CSDHardwareType.SK_HYNIX_CSD: 50e-6,         # 50μs SSD controller
            CSDHardwareType.SAMSUNG_SMARTSSD: 30e-6,     # 30μs optimized
            CSDHardwareType.XILINX_ALVEO: 5e-6,          # 5μs optimized FPGA
            CSDHardwareType.CUSTOM_ASIC: 1e-6            # 1μs custom ASIC
        }
        
        operation_multipliers = {
            "matrix_multiply": 2.0,      # Complex setup
            "attention": 1.5,            # Moderate setup
            "convolution": 3.0,          # Most complex setup
            "softmax": 1.0,              # Simple setup
            "embedding_lookup": 0.5,     # Very simple
        }
        
        base_time = base_setup.get(self.hardware_type, 10e-6)
        multiplier = operation_multipliers.get(operation, 1.0)
        
        return base_time * multiplier
    
    def update_utilization(self, new_utilization: float) -> None:
        """Update current hardware utilization."""
        self.utilization = max(0.0, min(1.0, new_utilization))
        
        # Thermal state increases with sustained high utilization
        if self.utilization > 0.8:
            self.thermal_state = min(1.0, self.thermal_state + 0.01)
        else:
            self.thermal_state = max(0.0, self.thermal_state - 0.005)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "hardware_type": self.hardware_type.value,
            "compute_specs": {
                "peak_ops_per_sec": f"{self.compute_specs.peak_ops_per_sec:.2e}",
                "memory_bandwidth_gbps": self.compute_specs.memory_bandwidth_gbps,
                "frequency_mhz": self.compute_specs.frequency_mhz,
                "parallel_units": self.compute_specs.parallel_units,
                "precision": self.compute_specs.precision
            },
            "communication_specs": {
                "pcie_config": f"PCIe {self.comm_specs.pcie_gen}.0 x{self.comm_specs.pcie_lanes}",
                "max_bandwidth_gbps": self.comm_specs.max_bandwidth_gbps,
                "latency_ns": self.comm_specs.latency_ns,
                "dma_engines": self.comm_specs.dma_engines
            },
            "runtime_state": {
                "utilization": f"{self.utilization:.1%}",
                "thermal_state": f"{self.thermal_state:.1%}",
                "effective_performance": f"{(1.0 - self.thermal_state * 0.3):.1%}"
            }
        }


class CSDHardwareManager:
    """Manager for multiple CSD hardware models."""
    
    def __init__(self):
        self.hardware_models: Dict[str, CSDHardwareModel] = {}
        self.active_model: Optional[CSDHardwareModel] = None
        
    def add_hardware(self, name: str, hardware_type: CSDHardwareType) -> CSDHardwareModel:
        """Add a new hardware model."""
        model = CSDHardwareModel(hardware_type)
        self.hardware_models[name] = model
        
        if self.active_model is None:
            self.active_model = model
        
        logger.info(f"Added hardware model '{name}' of type {hardware_type.value}")
        return model
    
    def set_active_hardware(self, name: str) -> None:
        """Set the active hardware model."""
        if name not in self.hardware_models:
            raise ValueError(f"Hardware model '{name}' not found")
        
        self.active_model = self.hardware_models[name]
        logger.info(f"Set active hardware to '{name}'")
    
    def benchmark_operation(self, operation: str, data_shape: Tuple[int, ...], 
                          hardware_list: Optional[list] = None) -> Dict[str, float]:
        """Benchmark an operation across multiple hardware types."""
        if hardware_list is None:
            hardware_list = list(self.hardware_models.keys())
        
        results = {}
        for hw_name in hardware_list:
            if hw_name in self.hardware_models:
                model = self.hardware_models[hw_name]
                exec_time = model.calculate_ml_operation_time(operation, data_shape)
                results[hw_name] = exec_time
        
        return results
    
    def get_optimal_hardware(self, operation: str, data_shape: Tuple[int, ...]) -> str:
        """Find the optimal hardware for a given operation."""
        benchmarks = self.benchmark_operation(operation, data_shape)
        
        if not benchmarks:
            return ""
        
        optimal_hw = min(benchmarks, key=benchmarks.get)
        logger.info(f"Optimal hardware for {operation} with shape {data_shape}: {optimal_hw}")
        
        return optimal_hw