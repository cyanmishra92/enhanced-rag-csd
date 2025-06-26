"""
CSD Backend Manager for runtime backend selection and fallback handling.
"""

from typing import Dict, Any, Optional, Type, List
import importlib
from enum import Enum

from .base import CSDBackendInterface, CSDBackendType
from .enhanced_simulator import EnhancedSimulatorBackend
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BackendStatus(Enum):
    """Backend availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable" 
    ERROR = "error"
    NOT_INSTALLED = "not_installed"


class CSDBackendManager:
    """Manager for CSD backend selection, initialization, and fallback handling."""
    
    def __init__(self):
        self._backends: Dict[CSDBackendType, Type[CSDBackendInterface]] = {}
        self._backend_status: Dict[CSDBackendType, BackendStatus] = {}
        self._current_backend: Optional[CSDBackendInterface] = None
        
        self._register_default_backends()
        self._check_backend_availability()
    
    def _register_default_backends(self) -> None:
        """Register default backend implementations."""
        # Enhanced simulator (always available)
        self._backends[CSDBackendType.ENHANCED_SIMULATOR] = EnhancedSimulatorBackend
        
        # Realistic CSD backend (always available)
        try:
            from .realistic_csd_backend import RealisticCSDBackend
            self._backends[CSDBackendType.REALISTIC_CSD] = RealisticCSDBackend
            logger.info("Realistic CSD backend registered")
        except ImportError as e:
            logger.warning(f"Realistic CSD backend not available: {e}")
        
        # SPDK emulator (conditional availability)
        try:
            from .spdk_emulator import SPDKEmulatorBackend
            self._backends[CSDBackendType.SPDK_EMULATOR] = SPDKEmulatorBackend
            logger.info("SPDK emulator backend registered")
        except ImportError as e:
            logger.warning(f"SPDK emulator backend not available: {e}")
        
        # Mock SPDK for testing (always try to register)
        try:
            from .mock_spdk import MockSPDKEmulatorBackend
            self._backends[CSDBackendType.MOCK_SPDK] = MockSPDKEmulatorBackend
            logger.info("Mock SPDK emulator backend registered for testing")
        except ImportError as e:
            logger.warning(f"Mock SPDK backend not available: {e}")
        
        # Next-generation emulator backends
        try:
            from .opencsd_backend import OpenCSDBackend
            self._backends[CSDBackendType.OPENCSD_EMULATOR] = OpenCSDBackend
            logger.info("OpenCSD emulator backend registered")
        except ImportError as e:
            logger.warning(f"OpenCSD backend not available: {e}")
        
        try:
            from .spdk_vfio_user_backend import SPDKVfioUserBackend
            self._backends[CSDBackendType.SPDK_VFIO_USER] = SPDKVfioUserBackend
            logger.info("SPDK vfio-user backend registered")
        except ImportError as e:
            logger.warning(f"SPDK vfio-user backend not available: {e}")
        
        # Hardware abstraction layer backends
        try:
            from .hardware_abstraction import CSDHardwareAbstractionLayer
            self.hal = CSDHardwareAbstractionLayer()
            logger.info("Hardware abstraction layer initialized")
        except ImportError as e:
            logger.warning(f"Hardware abstraction layer not available: {e}")
        
        # Placeholder for future backends
        # self._backends[CSDBackendType.FEMU_SMARTSSD] = FEMUSmartSSDBackend
        # self._backends[CSDBackendType.FIRESIM_FPGA] = FireSimBackend
        # self._backends[CSDBackendType.GPU_ACCELERATED] = GPUAcceleratedBackend
    
    def _check_backend_availability(self) -> None:
        """Check availability of all registered backends."""
        for backend_type, backend_class in self._backends.items():
            try:
                # Create a dummy instance to check availability
                dummy_config = {"vector_db_path": "/tmp/test"}
                backend_instance = backend_class(dummy_config)
                
                if backend_instance.is_available():
                    self._backend_status[backend_type] = BackendStatus.AVAILABLE
                    logger.info(f"Backend {backend_type.value} is available")
                else:
                    self._backend_status[backend_type] = BackendStatus.UNAVAILABLE
                    logger.warning(f"Backend {backend_type.value} dependencies not available")
                    
            except Exception as e:
                self._backend_status[backend_type] = BackendStatus.ERROR
                logger.error(f"Error checking backend {backend_type.value}: {e}")
    
    def get_available_backends(self) -> List[CSDBackendType]:
        """Get list of available backend types."""
        return [
            backend_type for backend_type, status in self._backend_status.items()
            if status == BackendStatus.AVAILABLE
        ]
    
    def get_backend_status(self, backend_type: CSDBackendType) -> BackendStatus:
        """Get status of a specific backend."""
        return self._backend_status.get(backend_type, BackendStatus.NOT_INSTALLED)
    
    def create_backend(self, 
                      backend_type: CSDBackendType, 
                      config: Dict[str, Any],
                      enable_fallback: bool = True) -> Optional[CSDBackendInterface]:
        """
        Create and initialize a CSD backend.
        
        Args:
            backend_type: Type of backend to create
            config: Configuration for the backend
            enable_fallback: Whether to fallback to enhanced simulator if primary fails
            
        Returns:
            Initialized backend instance or None if failed
        """
        # Check if requested backend is available
        if self.get_backend_status(backend_type) != BackendStatus.AVAILABLE:
            logger.warning(f"Requested backend {backend_type.value} not available")
            
            if enable_fallback and backend_type != CSDBackendType.ENHANCED_SIMULATOR:
                logger.info("Falling back to enhanced simulator")
                return self.create_backend(
                    CSDBackendType.ENHANCED_SIMULATOR, 
                    config, 
                    enable_fallback=False
                )
            else:
                return None
        
        # Create backend instance
        try:
            backend_class = self._backends[backend_type]
            backend_instance = backend_class(config)
            
            # Initialize the backend
            if backend_instance.initialize():
                logger.info(f"Successfully created {backend_type.value} backend")
                return backend_instance
            else:
                logger.error(f"Failed to initialize {backend_type.value} backend")
                
                if enable_fallback and backend_type != CSDBackendType.ENHANCED_SIMULATOR:
                    logger.info("Falling back to enhanced simulator")
                    return self.create_backend(
                        CSDBackendType.ENHANCED_SIMULATOR,
                        config,
                        enable_fallback=False
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error creating {backend_type.value} backend: {e}")
            
            if enable_fallback and backend_type != CSDBackendType.ENHANCED_SIMULATOR:
                logger.info("Falling back to enhanced simulator due to error")
                return self.create_backend(
                    CSDBackendType.ENHANCED_SIMULATOR,
                    config,
                    enable_fallback=False
                )
            
            return None
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about all backends."""
        info = {
            "available_backends": [bt.value for bt in self.get_available_backends()],
            "backend_status": {
                bt.value: status.value for bt, status in self._backend_status.items()
            },
            "registered_backends": list(self._backends.keys())
        }
        
        # Add detailed info for each backend
        for backend_type, backend_class in self._backends.items():
            try:
                dummy_config = {"vector_db_path": "/tmp/test"}
                backend_instance = backend_class(dummy_config)
                info[f"{backend_type.value}_info"] = backend_instance.get_backend_info()
            except Exception as e:
                info[f"{backend_type.value}_info"] = {"error": str(e)}
        
        return info
    
    def register_backend(self, 
                        backend_type: CSDBackendType, 
                        backend_class: Type[CSDBackendInterface]) -> None:
        """
        Register a custom backend implementation.
        
        Args:
            backend_type: Type identifier for the backend
            backend_class: Backend implementation class
        """
        if not issubclass(backend_class, CSDBackendInterface):
            raise ValueError("Backend class must inherit from CSDBackendInterface")
        
        self._backends[backend_type] = backend_class
        
        # Check availability of newly registered backend
        try:
            dummy_config = {"vector_db_path": "/tmp/test"}
            backend_instance = backend_class(dummy_config)
            
            if backend_instance.is_available():
                self._backend_status[backend_type] = BackendStatus.AVAILABLE
            else:
                self._backend_status[backend_type] = BackendStatus.UNAVAILABLE
                
        except Exception as e:
            self._backend_status[backend_type] = BackendStatus.ERROR
            logger.error(f"Error checking custom backend {backend_type.value}: {e}")
        
        logger.info(f"Registered custom backend: {backend_type.value}")
    
    def get_recommended_backend(self, config: Dict[str, Any]) -> CSDBackendType:
        """
        Get recommended backend based on configuration and availability.
        
        Args:
            config: User configuration
            
        Returns:
            Recommended backend type
        """
        # Check user preference
        user_preference = config.get("csd_backend", "enhanced_simulator")
        
        try:
            preferred_type = CSDBackendType(user_preference)
            if self.get_backend_status(preferred_type) == BackendStatus.AVAILABLE:
                return preferred_type
        except ValueError:
            logger.warning(f"Invalid backend type specified: {user_preference}")
        
        # Priority order for automatic selection (next-gen first)
        priority_order = [
            CSDBackendType.OPENCSD_EMULATOR,     # Real eBPF computational offloading
            CSDBackendType.SPDK_VFIO_USER,       # High-performance shared memory
            CSDBackendType.SPDK_EMULATOR,        # Real SPDK emulation
            CSDBackendType.MOCK_SPDK,            # Enhanced testing backend
            CSDBackendType.ENHANCED_SIMULATOR,   # Always available fallback
        ]
        
        # Use hardware abstraction layer if available
        if hasattr(self, 'hal'):
            try:
                hal_recommendation = self.hal.get_optimal_backend(config)
                if self.get_backend_status(hal_recommendation) == BackendStatus.AVAILABLE:
                    return hal_recommendation
            except Exception as e:
                logger.debug(f"HAL recommendation failed: {e}")
        
        for backend_type in priority_order:
            if self.get_backend_status(backend_type) == BackendStatus.AVAILABLE:
                return backend_type
        
        # Should never reach here since enhanced simulator is always available
        return CSDBackendType.ENHANCED_SIMULATOR