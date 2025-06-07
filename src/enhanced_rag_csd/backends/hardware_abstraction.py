"""
Hardware Abstraction Layer for Next-Generation Computational Storage Devices.

This module provides an accelerator-agnostic interface for computational storage
that supports CPU, GPU, FPGA, DPU, and other acceleration architectures.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import numpy as np
import platform
import subprocess
import os

from .base import CSDBackendType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AcceleratorType(Enum):
    """Types of hardware accelerators for computational storage."""
    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"
    DPU = "dpu"              # Data Processing Unit (NVIDIA BlueField, etc.)
    ASIC = "asic"            # Application-Specific Integrated Circuit
    SMARTNIC = "smartnic"    # Smart Network Interface Card
    SMARTSSD = "smartssd"    # Smart SSD with embedded processors
    HYBRID = "hybrid"        # Multiple accelerator types


class AcceleratorInterface(ABC):
    """Abstract interface for hardware accelerators."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accelerator_type = AcceleratorType.CPU  # Default
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the accelerator."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if accelerator is available and functional."""
        pass
    
    @abstractmethod
    def execute_computation(self, computation_type: str, data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Execute computation on the accelerator."""
        pass
    
    @abstractmethod
    def get_performance_info(self) -> Dict[str, Any]:
        """Get accelerator performance characteristics."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the accelerator."""
        self.is_initialized = False


class CPUAcceleratorInterface(AcceleratorInterface):
    """CPU-based computational acceleration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.accelerator_type = AcceleratorType.CPU
        self.num_cores = config.get("cpu_cores", os.cpu_count())
        self.use_simd = config.get("use_simd", True)
    
    def initialize(self) -> bool:
        """Initialize CPU accelerator."""
        try:
            # Check for SIMD support
            if self.use_simd:
                self._check_simd_support()
            
            self.is_initialized = True
            logger.info(f"CPU accelerator initialized with {self.num_cores} cores")
            return True
        except Exception as e:
            logger.error(f"CPU accelerator initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """CPU is always available."""
        return True
    
    def execute_computation(self, computation_type: str, data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Execute computation using optimized CPU instructions."""
        if computation_type == "similarity":
            return self._cpu_similarity_computation(data, metadata)
        elif computation_type == "embedding":
            return self._cpu_embedding_computation(data, metadata)
        elif computation_type == "augmentation":
            return self._cpu_augmentation_computation(data, metadata)
        else:
            raise NotImplementedError(f"CPU computation '{computation_type}' not implemented")
    
    def _cpu_similarity_computation(self, query: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Optimized CPU similarity computation using vectorization."""
        candidates = metadata.get("candidates", np.array([]))
        if len(candidates) == 0:
            return np.array([])
        
        # Use NumPy's optimized BLAS for similarity computation
        query_norm = np.linalg.norm(query)
        candidate_norms = np.linalg.norm(candidates, axis=1)
        
        # Avoid division by zero
        query_norm = max(query_norm, 1e-8)
        candidate_norms = np.maximum(candidate_norms, 1e-8)
        
        # Compute cosine similarities using vectorized operations
        similarities = np.dot(candidates, query) / (candidate_norms * query_norm)
        return similarities
    
    def _check_simd_support(self) -> bool:
        """Check for SIMD instruction support."""
        try:
            if platform.machine() in ['x86_64', 'AMD64']:
                # Check for AVX/SSE support
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                return 'avx' in info.get('flags', []) or 'sse' in info.get('flags', [])
        except ImportError:
            logger.warning("cpuinfo not available, assuming SIMD support")
        return True
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get CPU performance characteristics."""
        return {
            "accelerator_type": self.accelerator_type.value,
            "compute_units": self.num_cores,
            "peak_performance_gflops": self.num_cores * 2.0,  # Estimated
            "memory_bandwidth_gbps": 50.0,  # Typical DDR4
            "supports_simd": self.use_simd,
            "architecture": platform.machine()
        }


class GPUAcceleratorInterface(AcceleratorInterface):
    """GPU-based computational acceleration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.accelerator_type = AcceleratorType.GPU
        self.gpu_id = config.get("gpu_id", 0)
        self.use_cuda = config.get("use_cuda", True)
        self.use_tensor_cores = config.get("use_tensor_cores", True)
    
    def initialize(self) -> bool:
        """Initialize GPU accelerator."""
        try:
            if self.use_cuda:
                return self._initialize_cuda()
            else:
                return self._initialize_opencl()
        except Exception as e:
            logger.error(f"GPU accelerator initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPU is available."""
        try:
            if self.use_cuda:
                import torch
                return torch.cuda.is_available() and torch.cuda.device_count() > self.gpu_id
            else:
                # Check for OpenCL
                return self._check_opencl_available()
        except ImportError:
            return False
    
    def _initialize_cuda(self) -> bool:
        """Initialize CUDA GPU."""
        try:
            import torch
            self.device = torch.device(f"cuda:{self.gpu_id}")
            
            # Test GPU functionality
            test_tensor = torch.randn(100, 100, device=self.device)
            result = torch.mm(test_tensor, test_tensor.T)
            
            self.is_initialized = True
            logger.info(f"CUDA GPU {self.gpu_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"CUDA initialization failed: {e}")
            return False
    
    def execute_computation(self, computation_type: str, data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Execute computation on GPU."""
        try:
            import torch
            
            if computation_type == "similarity":
                return self._gpu_similarity_computation(data, metadata)
            elif computation_type == "embedding":
                return self._gpu_embedding_computation(data, metadata)
            else:
                raise NotImplementedError(f"GPU computation '{computation_type}' not implemented")
        except ImportError:
            logger.error("PyTorch not available for GPU computation")
            raise
    
    def _gpu_similarity_computation(self, query: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """GPU-accelerated similarity computation."""
        import torch
        
        candidates = metadata.get("candidates", np.array([]))
        if len(candidates) == 0:
            return np.array([])
        
        # Transfer to GPU
        query_tensor = torch.from_numpy(query).float().to(self.device)
        candidates_tensor = torch.from_numpy(candidates).float().to(self.device)
        
        # GPU-accelerated cosine similarity
        query_norm = torch.norm(query_tensor)
        candidate_norms = torch.norm(candidates_tensor, dim=1)
        
        similarities = torch.mm(candidates_tensor, query_tensor.unsqueeze(1)).squeeze()
        similarities = similarities / (candidate_norms * query_norm)
        
        # Transfer back to CPU
        return similarities.cpu().numpy()
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get GPU performance characteristics."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.gpu_id)
                return {
                    "accelerator_type": self.accelerator_type.value,
                    "device_name": props.name,
                    "compute_units": props.multi_processor_count,
                    "memory_total_gb": props.total_memory / (1024**3),
                    "peak_performance_tflops": props.multi_processor_count * 2.0,  # Estimated
                    "supports_tensor_cores": self.use_tensor_cores and props.major >= 7,
                    "cuda_compute_capability": f"{props.major}.{props.minor}"
                }
        except ImportError:
            pass
        
        return {
            "accelerator_type": self.accelerator_type.value,
            "status": "not_available"
        }


class FPGAAcceleratorInterface(AcceleratorInterface):
    """FPGA-based computational acceleration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.accelerator_type = AcceleratorType.FPGA
        self.fpga_device = config.get("fpga_device", "/dev/xdma0_user")
        self.bitstream_path = config.get("bitstream_path", None)
        self.pcie_id = config.get("pcie_id", None)
    
    def initialize(self) -> bool:
        """Initialize FPGA accelerator."""
        try:
            # Check for FPGA device
            if not os.path.exists(self.fpga_device):
                logger.error(f"FPGA device {self.fpga_device} not found")
                return False
            
            # Load bitstream if provided
            if self.bitstream_path and os.path.exists(self.bitstream_path):
                self._load_bitstream()
            
            self.is_initialized = True
            logger.info("FPGA accelerator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"FPGA accelerator initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if FPGA is available."""
        return (
            os.path.exists(self.fpga_device) and
            os.access(self.fpga_device, os.R_OK | os.W_OK)
        )
    
    def execute_computation(self, computation_type: str, data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Execute computation on FPGA."""
        if computation_type == "similarity":
            return self._fpga_similarity_computation(data, metadata)
        else:
            raise NotImplementedError(f"FPGA computation '{computation_type}' not implemented")
    
    def _fpga_similarity_computation(self, query: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """FPGA-accelerated similarity computation (placeholder)."""
        # This would interface with actual FPGA hardware/drivers
        logger.info("Executing similarity computation on FPGA")
        
        # Placeholder: fall back to CPU computation
        cpu_accelerator = CPUAcceleratorInterface(self.config)
        return cpu_accelerator._cpu_similarity_computation(query, metadata)
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get FPGA performance characteristics."""
        return {
            "accelerator_type": self.accelerator_type.value,
            "fpga_device": self.fpga_device,
            "bitstream_loaded": self.bitstream_path is not None,
            "compute_units": "configurable",
            "peak_performance": "depends_on_design",
            "memory_bandwidth": "configurable"
        }


class CSDHardwareAbstractionLayer:
    """Hardware abstraction layer for computational storage devices."""
    
    def __init__(self):
        self.accelerator_types = {
            AcceleratorType.CPU: CPUAcceleratorInterface,
            AcceleratorType.GPU: GPUAcceleratorInterface,
            AcceleratorType.FPGA: FPGAAcceleratorInterface,
            # Add more accelerator types as implemented
        }
        self._detected_hardware = None
    
    def detect_available_hardware(self) -> Dict[AcceleratorType, bool]:
        """Detect available accelerator hardware."""
        if self._detected_hardware is None:
            self._detected_hardware = {}
            
            for accel_type, accel_class in self.accelerator_types.items():
                try:
                    accel_instance = accel_class({})
                    self._detected_hardware[accel_type] = accel_instance.is_available()
                except Exception as e:
                    logger.debug(f"Error detecting {accel_type.value}: {e}")
                    self._detected_hardware[accel_type] = False
        
        return self._detected_hardware
    
    def get_optimal_backend(self, config: Dict[str, Any]) -> CSDBackendType:
        """Select optimal backend based on available hardware and requirements."""
        hardware = self.detect_available_hardware()
        preferred_accel = config.get("preferred_accelerator", "auto")
        require_real_hardware = config.get("require_real_hardware", False)
        
        # Priority order for backend selection
        if preferred_accel == "fpga" and hardware.get(AcceleratorType.FPGA, False):
            return CSDBackendType.OPENCSD_EMULATOR  # Best for FPGA research
        elif preferred_accel == "gpu" and hardware.get(AcceleratorType.GPU, False):
            return CSDBackendType.GPU_ACCELERATED   # GPU computational storage
        elif hardware.get(AcceleratorType.FPGA, False) and not require_real_hardware:
            return CSDBackendType.OPENCSD_EMULATOR  # FPGA emulation available
        elif hardware.get(AcceleratorType.GPU, False):
            return CSDBackendType.SPDK_VFIO_USER    # High-performance with GPU
        elif not require_real_hardware:
            return CSDBackendType.MOCK_SPDK         # Enhanced simulation
        else:
            return CSDBackendType.ENHANCED_SIMULATOR # Fallback
    
    def create_accelerator(self, accel_type: AcceleratorType, 
                          config: Dict[str, Any]) -> Optional[AcceleratorInterface]:
        """Create and initialize an accelerator instance."""
        if accel_type not in self.accelerator_types:
            logger.error(f"Accelerator type {accel_type.value} not supported")
            return None
        
        try:
            accel_class = self.accelerator_types[accel_type]
            accelerator = accel_class(config)
            
            if accelerator.initialize():
                return accelerator
            else:
                logger.error(f"Failed to initialize {accel_type.value} accelerator")
                return None
        except Exception as e:
            logger.error(f"Error creating {accel_type.value} accelerator: {e}")
            return None
    
    def get_hardware_report(self) -> Dict[str, Any]:
        """Generate comprehensive hardware availability report."""
        hardware = self.detect_available_hardware()
        
        report = {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "available_accelerators": [],
            "recommended_backends": []
        }
        
        for accel_type, available in hardware.items():
            if available:
                try:
                    accel_instance = self.accelerator_types[accel_type]({})
                    if accel_instance.is_available():
                        perf_info = accel_instance.get_performance_info()
                        report["available_accelerators"].append({
                            "type": accel_type.value,
                            "performance": perf_info
                        })
                except Exception as e:
                    logger.debug(f"Error getting info for {accel_type.value}: {e}")
        
        # Add recommended backends based on available hardware
        configs = [
            {"preferred_accelerator": "auto", "require_real_hardware": False},
            {"preferred_accelerator": "fpga", "require_real_hardware": True},
            {"preferred_accelerator": "gpu", "require_real_hardware": True}
        ]
        
        for config in configs:
            backend = self.get_optimal_backend(config)
            report["recommended_backends"].append({
                "config": config,
                "recommended_backend": backend.value
            })
        
        return report