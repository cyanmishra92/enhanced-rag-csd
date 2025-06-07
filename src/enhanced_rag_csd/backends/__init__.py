"""
CSD Backend abstraction layer for Enhanced RAG-CSD.

This module provides a pluggable architecture for different computational
storage device emulation backends while maintaining API compatibility.
"""

from .base import CSDBackendInterface, CSDBackendType
from .enhanced_simulator import EnhancedSimulatorBackend  
from .backend_manager import CSDBackendManager

__all__ = [
    'CSDBackendInterface',
    'CSDBackendType', 
    'EnhancedSimulatorBackend',
    'CSDBackendManager'
]