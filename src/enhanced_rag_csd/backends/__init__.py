"""
CSD Backend abstraction layer for Enhanced RAG-CSD.

This module provides a pluggable architecture for different computational
storage device emulation backends while maintaining API compatibility.
"""

from .base import CSDBackendInterface, CSDBackendType
from .enhanced_simulator import EnhancedSimulatorBackend  
from .backend_manager import CSDBackendManager

try:
    from .realistic_csd_backend import RealisticCSDBackend
except ImportError:
    RealisticCSDBackend = None

try:
    from .mock_spdk import MockSPDKEmulatorBackend
except ImportError:
    MockSPDKEmulatorBackend = None

try:
    from .opencsd_backend import OpenCSDBackend
except ImportError:
    OpenCSDBackend = None

try:
    from .spdk_vfio_user_backend import SPDKVfioUserBackend
except ImportError:
    SPDKVfioUserBackend = None

try:
    from .hardware_abstraction import CSDHardwareAbstractionLayer, AcceleratorType
except ImportError:
    CSDHardwareAbstractionLayer = None
    AcceleratorType = None

__all__ = [
    'CSDBackendInterface',
    'CSDBackendType', 
    'EnhancedSimulatorBackend',
    'RealisticCSDBackend',
    'CSDBackendManager',
    'MockSPDKEmulatorBackend',
    'OpenCSDBackend',
    'SPDKVfioUserBackend',
    'CSDHardwareAbstractionLayer',
    'AcceleratorType'
]