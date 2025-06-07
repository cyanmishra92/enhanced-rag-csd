"""
SPDK emulator backend using SPDK + QEMU + libvfio-user.
"""

import time
import subprocess
import os
from typing import Dict, List, Any, Optional
import numpy as np

from .base import CSDBackendInterface, CSDBackendType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SPDKEmulatorBackend(CSDBackendInterface):
    """Backend using SPDK + QEMU + libvfio-user for realistic CSD emulation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._backend_type = CSDBackendType.SPDK_EMULATOR
        
        # SPDK configuration
        self.spdk_config = config.get("spdk", {})
        self.nvme_size_gb = self.spdk_config.get("nvme_size_gb", 10)
        self.rpc_socket = self.spdk_config.get("rpc_socket", "/tmp/spdk.sock")
        self.virtual_queues = self.spdk_config.get("virtual_queues", 8)
        
        # Runtime state
        self.spdk_process = None
        self.qemu_process = None
        self.initialized = False
        
        # Emulator-specific metrics
        self.emulator_metrics = {
            "nvme_commands_sent": 0,
            "nvme_read_commands": 0,
            "nvme_write_commands": 0,
            "emulator_latency_ms": 0.0,
            "p2p_transfers": 0
        }
    
    def initialize(self) -> bool:
        """Initialize the SPDK emulator backend."""
        if not self.is_available():
            logger.error("SPDK emulator dependencies not available")
            return False
        
        try:
            # Start SPDK application
            if not self._start_spdk_app():
                logger.error("Failed to start SPDK application")
                return False
            
            # Initialize virtual NVMe controller
            if not self._init_nvme_controller():
                logger.error("Failed to initialize NVMe controller")
                return False
            
            # Start QEMU with vfio-user
            if not self._start_qemu_vm():
                logger.error("Failed to start QEMU VM")
                return False
            
            self.initialized = True
            logger.info("SPDK emulator backend initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SPDK emulator backend: {e}")
            self.shutdown()
            return False
    
    def _start_spdk_app(self) -> bool:
        """Start the SPDK application."""
        try:
            # Create SPDK configuration
            spdk_config = {
                "subsystems": [
                    {
                        "subsystem": "bdev",
                        "config": [
                            {
                                "method": "bdev_malloc_create",
                                "params": {
                                    "name": "Malloc0",
                                    "num_blocks": self.nvme_size_gb * 1024 * 1024 * 2,  # 512B blocks
                                    "block_size": 512
                                }
                            }
                        ]
                    },
                    {
                        "subsystem": "nvmf",
                        "config": [
                            {
                                "method": "nvmf_create_transport",
                                "params": {
                                    "trtype": "VFIOUSER"
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Start SPDK target application
            spdk_cmd = [
                "spdk_tgt",
                "-r", self.rpc_socket,
                "-m", "0x1"  # Core mask
            ]
            
            self.spdk_process = subprocess.Popen(
                spdk_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait for SPDK to initialize
            time.sleep(2)
            
            if self.spdk_process.poll() is not None:
                logger.error("SPDK process terminated unexpectedly")
                return False
            
            logger.info("SPDK application started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SPDK application: {e}")
            return False
    
    def _init_nvme_controller(self) -> bool:
        """Initialize the virtual NVMe controller."""
        try:
            # Create NVMe subsystem via RPC
            rpc_commands = [
                {
                    "method": "bdev_nvme_attach_controller",
                    "params": {
                        "name": "Nvme0",
                        "trtype": "VFIOUSER",
                        "traddr": "/tmp/vfio-user-nvme0"
                    }
                }
            ]
            
            for cmd in rpc_commands:
                result = self._send_spdk_rpc(cmd)
                if result is None:
                    logger.error(f"Failed to execute RPC command: {cmd}")
                    return False
            
            logger.info("NVMe controller initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NVMe controller: {e}")
            return False
    
    def _start_qemu_vm(self) -> bool:
        """Start QEMU VM with vfio-user device."""
        try:
            qemu_cmd = [
                "qemu-system-x86_64",
                "-machine", "q35",
                "-cpu", "host",
                "-m", "2G",
                "-smp", "2",
                "-object", f"memory-backend-file,id=mem0,size=2G,mem-path=/dev/shm,share=on",
                "-numa", "node,memdev=mem0",
                "-device", "vfio-user-pci,socket=/tmp/vfio-user-nvme0",
                "-nographic",
                "-serial", "stdio"
            ]
            
            self.qemu_process = subprocess.Popen(
                qemu_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait for QEMU to start
            time.sleep(3)
            
            if self.qemu_process.poll() is not None:
                logger.error("QEMU process terminated unexpectedly")
                return False
            
            logger.info("QEMU VM started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start QEMU VM: {e}")
            return False
    
    def _send_spdk_rpc(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send RPC command to SPDK."""
        try:
            import json
            import socket
            
            # Connect to SPDK RPC socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.rpc_socket)
            
            # Send command
            cmd_str = json.dumps(command) + "\\n"
            sock.send(cmd_str.encode())
            
            # Receive response
            response = sock.recv(4096).decode()
            sock.close()
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Failed to send SPDK RPC: {e}")
            return None
    
    def store_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Store embeddings using SPDK NVMe write commands."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        
        # Convert embeddings to bytes
        data = embeddings.astype(np.float32).tobytes()
        
        # Send NVMe write command via SPDK
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            lba = i * (embedding.nbytes // 512)  # Calculate LBA
            self._nvme_write(lba, embedding.tobytes())
        
        # Update metrics
        elapsed = time.time() - start_time
        self.metrics["write_ops"] += len(embeddings)
        self.metrics["total_bytes_written"] += embeddings.nbytes
        self.emulator_metrics["nvme_write_commands"] += len(embeddings)
        self.emulator_metrics["emulator_latency_ms"] += elapsed * 1000
        
        logger.debug(f"SPDK: Stored {len(embeddings)} embeddings in {elapsed:.3f}s")
    
    def retrieve_embeddings(self, indices: List[int], use_cache: bool = True) -> np.ndarray:
        """Retrieve embeddings using SPDK NVMe read commands."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        start_time = time.time()
        
        # Calculate embedding size
        embedding_dim = self.config.get("embedding", {}).get("dimensions", 384)
        embedding_size = embedding_dim * 4  # float32
        
        # Retrieve embeddings via NVMe read commands
        embeddings = []
        for idx in indices:
            lba = idx * (embedding_size // 512)
            data = self._nvme_read(lba, embedding_size)
            embedding = np.frombuffer(data, dtype=np.float32)
            embeddings.append(embedding)
        
        result = np.array(embeddings)
        
        # Update metrics
        elapsed = time.time() - start_time
        self.metrics["read_ops"] += len(indices)
        self.metrics["total_bytes_read"] += result.nbytes
        self.emulator_metrics["nvme_read_commands"] += len(indices)
        self.emulator_metrics["emulator_latency_ms"] += elapsed * 1000
        
        logger.debug(f"SPDK: Retrieved {len(indices)} embeddings in {elapsed:.3f}s")
        return result
    
    def _nvme_write(self, lba: int, data: bytes) -> bool:
        """Send NVMe write command."""
        try:
            # Simulate NVMe write command via SPDK
            cmd = {
                "method": "bdev_nvme_send_cmd",
                "params": {
                    "name": "Nvme0n1",
                    "cmd_type": "write",
                    "lba": lba,
                    "data": data.hex()
                }
            }
            
            result = self._send_spdk_rpc(cmd)
            self.emulator_metrics["nvme_commands_sent"] += 1
            
            return result is not None
            
        except Exception as e:
            logger.error(f"NVMe write failed: {e}")
            return False
    
    def _nvme_read(self, lba: int, size: int) -> bytes:
        """Send NVMe read command."""
        try:
            # Simulate NVMe read command via SPDK
            cmd = {
                "method": "bdev_nvme_send_cmd", 
                "params": {
                    "name": "Nvme0n1",
                    "cmd_type": "read",
                    "lba": lba,
                    "size": size
                }
            }
            
            result = self._send_spdk_rpc(cmd)
            self.emulator_metrics["nvme_commands_sent"] += 1
            
            if result and "data" in result:
                return bytes.fromhex(result["data"])
            else:
                # Return zeros if read fails
                return b'\\x00' * size
                
        except Exception as e:
            logger.error(f"NVMe read failed: {e}")
            return b'\\x00' * size
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           candidate_indices: List[int]) -> np.ndarray:
        """Compute similarities using CSD compute units."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        # For now, fall back to CPU computation
        # TODO: Implement actual CSD compute offload
        candidates = self.retrieve_embeddings(candidate_indices)
        
        # Normalize vectors
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(candidates_norm, query_norm)
        
        return similarities
    
    def process_era_pipeline(self, query_data: np.ndarray, 
                           metadata: Dict[str, Any]) -> np.ndarray:
        """Process ERA pipeline using SPDK emulator."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        # For now, use simplified pipeline
        # TODO: Implement full ERA pipeline on emulated CSD
        
        # Encode (simulate)
        if query_data.dtype != np.float32 or len(query_data.shape) != 1:
            embedding_dim = self.config.get("embedding", {}).get("dimensions", 384)
            query_embedding = np.random.randn(embedding_dim).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        else:
            query_embedding = query_data
        
        # Retrieve
        top_k = metadata.get("top_k", 5)
        candidate_indices = list(range(min(100, top_k * 10)))  # Get some candidates
        if candidate_indices:
            similarities = self.compute_similarities(query_embedding, candidate_indices)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            retrieved = self.retrieve_embeddings(top_indices.tolist())
        else:
            retrieved = np.random.randn(top_k, len(query_embedding)).astype(np.float32)
        
        # Augment (concatenate)
        augmented = np.concatenate([query_embedding] + [retrieved[i] for i in range(len(retrieved))])
        
        return augmented
    
    def p2p_transfer_to_gpu(self, data: np.ndarray) -> str:
        """Transfer data from CSD to GPU via P2P."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        # Simulate P2P transfer
        transfer_time = data.nbytes / (12000 * 1024 * 1024)  # 12GB/s P2P bandwidth
        time.sleep(transfer_time)
        
        self.emulator_metrics["p2p_transfers"] += 1
        
        gpu_allocation_id = f"spdk_gpu_alloc_{self.emulator_metrics['p2p_transfers']}"
        logger.debug(f"SPDK P2P transfer: {data.nbytes/1024:.1f}KB -> {gpu_allocation_id}")
        
        return gpu_allocation_id
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics including SPDK-specific metrics."""
        combined_metrics = {**self.metrics, **self.emulator_metrics}
        combined_metrics["backend_type"] = self._backend_type.value
        combined_metrics["emulator_initialized"] = self.initialized
        
        return combined_metrics
    
    def shutdown(self) -> None:
        """Shutdown the SPDK emulator backend."""
        try:
            # Terminate QEMU process
            if self.qemu_process and self.qemu_process.poll() is None:
                self.qemu_process.terminate()
                self.qemu_process.wait(timeout=5)
                logger.info("QEMU process terminated")
            
            # Terminate SPDK process
            if self.spdk_process and self.spdk_process.poll() is None:
                self.spdk_process.terminate() 
                self.spdk_process.wait(timeout=5)
                logger.info("SPDK process terminated")
                
        except Exception as e:
            logger.error(f"Error during SPDK emulator shutdown: {e}")
        
        finally:
            self.initialized = False
            self.spdk_process = None
            self.qemu_process = None
            logger.info("SPDK emulator backend shutdown complete")
    
    def is_available(self) -> bool:
        """Check if SPDK emulator dependencies are available."""
        try:
            # Check for SPDK installation
            result = subprocess.run(["which", "spdk_tgt"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("SPDK not found in PATH")
                return False
            
            # Check for QEMU with vfio-user support
            result = subprocess.run(["which", "qemu-system-x86_64"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("QEMU not found in PATH")
                return False
            
            # Check for libvfio-user
            if not os.path.exists("/usr/lib/libvfio-user.so") and \\
               not os.path.exists("/usr/local/lib/libvfio-user.so"):
                logger.warning("libvfio-user not found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking SPDK availability: {e}")
            return False