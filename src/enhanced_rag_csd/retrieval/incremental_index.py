"""
Incremental indexing system with drift detection for dynamic document management.
This module provides efficient incremental updates to vector indices with automatic
drift detection and index rebuilding when necessary.
"""

import os
import json
import time
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
import faiss
from scipy.stats import entropy
from sklearn.decomposition import PCA

from enhanced_rag_csd.utils.logger import get_logger
from enhanced_rag_csd.utils.error_handling import handle_exceptions

logger = get_logger(__name__)


@dataclass
class IndexVersion:
    """Version information for an index."""
    version_id: str
    created_at: datetime
    num_vectors: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndexVersion':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class DriftMetrics:
    """Metrics for index drift detection."""
    kl_divergence: float
    performance_degradation: float
    fragmentation_ratio: float
    drift_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class DriftDetector:
    """Detects when index drift requires rebuilding."""
    
    def __init__(self, 
                 kl_threshold: float = 0.1,
                 perf_threshold: float = 0.2,
                 frag_threshold: float = 0.3,
                 window_size: int = 1000):
        self.kl_threshold = kl_threshold
        self.perf_threshold = perf_threshold
        self.frag_threshold = frag_threshold
        self.window_size = window_size
        
        # Baseline distribution (will be set on first index build)
        self.baseline_distribution = None
        self.baseline_pca = None
        
        # Performance tracking
        self.query_latencies = deque(maxlen=window_size)
        self.baseline_latency = None
        
        # Fragmentation tracking
        self.delta_indices_count = 0
        self.main_index_size = 0
    
    def initialize_baseline(self, embeddings: np.ndarray) -> None:
        """Initialize baseline distribution from main index."""
        logger.info("Initializing drift detector baseline")
        
        # Compute baseline distribution using PCA
        n_components = min(50, embeddings.shape[1])
        self.baseline_pca = PCA(n_components=n_components)
        reduced = self.baseline_pca.fit_transform(embeddings)
        
        # Compute histogram of reduced dimensions
        self.baseline_distribution = self._compute_distribution(reduced)
        
        # Set baseline performance
        if self.query_latencies:
            self.baseline_latency = np.median(list(self.query_latencies))
        else:
            self.baseline_latency = 0.1  # Default 100ms
        
        self.main_index_size = len(embeddings)
    
    def _compute_distribution(self, embeddings: np.ndarray, bins: int = 50) -> np.ndarray:
        """Compute distribution histogram from embeddings."""
        # Flatten and compute histogram
        flattened = embeddings.flatten()
        hist, _ = np.histogram(flattened, bins=bins, density=True)
        
        # Normalize to probability distribution
        hist = hist + 1e-10  # Avoid zeros
        hist = hist / hist.sum()
        
        return hist
    
    def compute_kl_divergence(self, new_embeddings: np.ndarray) -> float:
        """Compute KL divergence between baseline and new distributions."""
        if self.baseline_distribution is None or self.baseline_pca is None:
            return 0.0
        
        try:
            # Transform new embeddings using baseline PCA
            reduced = self.baseline_pca.transform(new_embeddings)
            new_distribution = self._compute_distribution(reduced)
            
            # Compute KL divergence
            kl_div = entropy(new_distribution, self.baseline_distribution)
            
            return float(kl_div)
        except Exception as e:
            logger.warning(f"Error computing KL divergence: {e}")
            return 0.0
    
    def compute_performance_degradation(self, current_latency: float) -> float:
        """Compute performance degradation ratio."""
        self.query_latencies.append(current_latency)
        
        if self.baseline_latency is None or self.baseline_latency == 0:
            return 0.0
        
        # Calculate median of recent latencies
        if len(self.query_latencies) < 10:
            return 0.0
        
        recent_median = np.median(list(self.query_latencies))
        degradation = (recent_median - self.baseline_latency) / self.baseline_latency
        
        return max(0.0, degradation)
    
    def compute_fragmentation(self, num_delta_indices: int, total_delta_size: int) -> float:
        """Compute index fragmentation ratio."""
        if self.main_index_size == 0:
            return 0.0
        
        # Fragmentation increases with number of delta indices and their size
        frag_by_count = num_delta_indices / 10.0  # Normalize by expected max
        frag_by_size = total_delta_size / self.main_index_size
        
        fragmentation = 0.5 * frag_by_count + 0.5 * frag_by_size
        
        return min(1.0, fragmentation)
    
    def detect_drift(self, 
                    new_embeddings: Optional[np.ndarray] = None,
                    query_latency: Optional[float] = None,
                    num_delta_indices: int = 0,
                    total_delta_size: int = 0) -> Tuple[bool, DriftMetrics]:
        """Detect if index drift exceeds thresholds."""
        
        # Compute individual metrics
        kl_div = 0.0
        if new_embeddings is not None and len(new_embeddings) > 0:
            kl_div = self.compute_kl_divergence(new_embeddings)
        
        perf_deg = 0.0
        if query_latency is not None:
            perf_deg = self.compute_performance_degradation(query_latency)
        
        frag = self.compute_fragmentation(num_delta_indices, total_delta_size)
        
        # Weighted drift score
        drift_score = (0.4 * (kl_div / self.kl_threshold) +
                      0.4 * (perf_deg / self.perf_threshold) +
                      0.2 * (frag / self.frag_threshold))
        
        # Create metrics object
        metrics = DriftMetrics(
            kl_divergence=kl_div,
            performance_degradation=perf_deg,
            fragmentation_ratio=frag,
            drift_score=drift_score,
            timestamp=datetime.now()
        )
        
        # Drift detected if score > 1.0
        drift_detected = drift_score > 1.0
        
        if drift_detected:
            logger.warning(f"Index drift detected! Score: {drift_score:.3f}")
            logger.info(f"KL divergence: {kl_div:.3f}, Performance degradation: {perf_deg:.3f}, "
                       f"Fragmentation: {frag:.3f}")
        
        return drift_detected, metrics


class DeltaIndex:
    """A delta index for incremental updates."""
    
    def __init__(self, index_id: str, dimension: int):
        self.index_id = index_id
        self.dimension = dimension
        self.created_at = datetime.now()
        
        # FAISS flat index for delta
        self.index = faiss.IndexFlatIP(dimension)
        
        # Metadata storage
        self.chunks = []
        self.metadata = []
        
        # Track modifications
        self.num_additions = 0
        self.last_modified = self.created_at
    
    def add(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict]) -> None:
        """Add new documents to delta index."""
        if len(embeddings) != len(chunks) or len(embeddings) != len(metadata):
            raise ValueError("Mismatched lengths for embeddings, chunks, and metadata")
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        self.num_additions += len(embeddings)
        self.last_modified = datetime.now()
    
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search in delta index."""
        # Normalize query
        query_embedding = query_embedding.copy()
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), min(k, self.index.ntotal))
        
        return scores[0], indices[0]
    
    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal
    
    def save(self, path: str) -> None:
        """Save delta index to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "delta.index"))
        
        # Save metadata
        metadata = {
            "index_id": self.index_id,
            "dimension": self.dimension,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "num_additions": self.num_additions,
            "chunks": self.chunks,
            "metadata": self.metadata
        }
        
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'DeltaIndex':
        """Load delta index from disk."""
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)
        
        # Create instance
        delta = cls(meta["index_id"], meta["dimension"])
        delta.created_at = datetime.fromisoformat(meta["created_at"])
        delta.last_modified = datetime.fromisoformat(meta["last_modified"])
        delta.num_additions = meta["num_additions"]
        delta.chunks = meta["chunks"]
        delta.metadata = meta["metadata"]
        
        # Load FAISS index
        delta.index = faiss.read_index(os.path.join(path, "delta.index"))
        
        return delta


class IncrementalVectorStore:
    """Vector store with incremental indexing and drift detection."""
    
    def __init__(self, 
                 storage_path: str,
                 dimension: int = 384,
                 delta_threshold: int = 10000,
                 max_delta_indices: int = 5):
        self.storage_path = storage_path
        self.dimension = dimension
        self.delta_threshold = delta_threshold
        self.max_delta_indices = max_delta_indices
        
        # Create directories
        os.makedirs(storage_path, exist_ok=True)
        self.main_index_path = os.path.join(storage_path, "main")
        self.delta_path = os.path.join(storage_path, "deltas")
        self.versions_path = os.path.join(storage_path, "versions")
        
        os.makedirs(self.delta_path, exist_ok=True)
        os.makedirs(self.versions_path, exist_ok=True)
        
        # Initialize components
        self.main_index = None
        self.main_chunks = []
        self.main_metadata = []
        
        self.delta_indices = []
        self.current_delta = None
        
        self.drift_detector = DriftDetector()
        self.version_history = []
        
        # Load existing index if available
        self._load_indices()
    
    def _load_indices(self) -> None:
        """Load existing indices from disk."""
        # Load main index
        main_index_file = os.path.join(self.main_index_path, "main.index")
        if os.path.exists(main_index_file):
            logger.info("Loading existing main index")
            self.main_index = faiss.read_index(main_index_file)
            
            # Load metadata
            with open(os.path.join(self.main_index_path, "metadata.json"), "r") as f:
                meta = json.load(f)
                self.main_chunks = meta["chunks"]
                self.main_metadata = meta["metadata"]
            
            # Initialize drift detector baseline
            # Note: In production, we'd reconstruct embeddings or store them
            logger.info(f"Main index loaded with {self.main_index.ntotal} vectors")
        
        # Load delta indices
        if os.path.exists(self.delta_path):
            for delta_dir in sorted(os.listdir(self.delta_path)):
                delta_path = os.path.join(self.delta_path, delta_dir)
                if os.path.isdir(delta_path):
                    delta = DeltaIndex.load(delta_path)
                    self.delta_indices.append(delta)
                    logger.info(f"Loaded delta index {delta.index_id} with {delta.size()} vectors")
        
        # Load version history
        version_file = os.path.join(self.versions_path, "history.json")
        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                version_data = json.load(f)
                self.version_history = [IndexVersion.from_dict(v) for v in version_data]
    
    @handle_exceptions(default_return=None)
    def add_documents(self, 
                     embeddings: np.ndarray,
                     chunks: List[str],
                     metadata: List[Dict]) -> None:
        """Add new documents to the index."""
        if len(embeddings) != len(chunks) or len(embeddings) != len(metadata):
            raise ValueError("Mismatched lengths for embeddings, chunks, and metadata")
        
        # Create new delta if needed
        if self.current_delta is None or self.current_delta.size() >= self.delta_threshold:
            self._create_new_delta()
        
        # Add to current delta
        self.current_delta.add(embeddings, chunks, metadata)
        logger.info(f"Added {len(embeddings)} documents to delta index {self.current_delta.index_id}")
        
        # Check if we need to merge
        if len(self.delta_indices) >= self.max_delta_indices:
            logger.info(f"Maximum delta indices ({self.max_delta_indices}) reached, checking drift")
            
            # Check drift with current batch
            drift_detected, metrics = self.drift_detector.detect_drift(
                new_embeddings=embeddings,
                num_delta_indices=len(self.delta_indices),
                total_delta_size=sum(d.size() for d in self.delta_indices)
            )
            
            if drift_detected:
                self._rebuild_main_index()
            else:
                self._merge_oldest_delta()
    
    def _create_new_delta(self) -> None:
        """Create a new delta index."""
        delta_id = f"delta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_delta = DeltaIndex(delta_id, self.dimension)
        self.delta_indices.append(self.current_delta)
        logger.info(f"Created new delta index: {delta_id}")
    
    def _merge_oldest_delta(self) -> None:
        """Merge oldest delta into main index."""
        if not self.delta_indices:
            return
        
        oldest_delta = self.delta_indices[0]
        logger.info(f"Merging delta index {oldest_delta.index_id} into main index")
        
        # Extract vectors from delta
        # Note: In production, we'd have a method to extract raw embeddings
        # For now, we'll just update metadata
        
        if self.main_index is None:
            # Create new main index from delta
            self.main_index = faiss.IndexFlatIP(self.dimension)
            self.main_chunks = []
            self.main_metadata = []
        
        # Add delta content to main
        # This is simplified - in production we'd properly merge the indices
        self.main_chunks.extend(oldest_delta.chunks)
        self.main_metadata.extend(oldest_delta.metadata)
        
        # Remove delta
        self.delta_indices.pop(0)
        
        # Save updated main index
        self._save_main_index()
    
    def _rebuild_main_index(self) -> None:
        """Rebuild main index from scratch."""
        logger.info("Rebuilding main index due to drift")
        
        # Create version snapshot
        version = IndexVersion(
            version_id=f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(),
            num_vectors=self.main_index.ntotal if self.main_index else 0,
            metadata={"reason": "drift_detected"}
        )
        self.version_history.append(version)
        
        # Combine all indices
        all_chunks = self.main_chunks.copy() if self.main_chunks else []
        all_metadata = self.main_metadata.copy() if self.main_metadata else []
        
        for delta in self.delta_indices:
            all_chunks.extend(delta.chunks)
            all_metadata.extend(delta.metadata)
        
        # Create new main index
        # Note: In production, we'd re-encode all documents
        self.main_index = faiss.IndexFlatIP(self.dimension)
        self.main_chunks = all_chunks
        self.main_metadata = all_metadata
        
        # Clear deltas
        self.delta_indices = []
        self.current_delta = None
        
        # Save everything
        self._save_main_index()
        self._save_version_history()
        
        # Clean up old deltas
        self._cleanup_deltas()
        
        logger.info(f"Main index rebuilt with {len(self.main_chunks)} documents")
    
    def search(self, 
              query_embedding: np.ndarray,
              top_k: int = 10,
              include_deltas: bool = True) -> List[Dict[str, Any]]:
        """Search across main and delta indices."""
        start_time = time.time()
        
        all_results = []
        
        # Search main index
        if self.main_index and self.main_index.ntotal > 0:
            scores, indices = self.main_index.search(
                query_embedding.reshape(1, -1), 
                min(top_k, self.main_index.ntotal)
            )
            
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    all_results.append({
                        "chunk": self.main_chunks[idx],
                        "metadata": self.main_metadata[idx],
                        "score": float(score),
                        "source": "main"
                    })
        
        # Search delta indices
        if include_deltas:
            for delta in self.delta_indices:
                if delta.size() > 0:
                    scores, indices = delta.search(query_embedding, top_k)
                    
                    for idx, score in zip(indices, scores):
                        if idx >= 0:
                            all_results.append({
                                "chunk": delta.chunks[idx],
                                "metadata": delta.metadata[idx],
                                "score": float(score),
                                "source": delta.index_id
                            })
        
        # Sort by score and take top k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        results = all_results[:top_k]
        
        # Track query latency for drift detection
        query_latency = time.time() - start_time
        self.drift_detector.compute_performance_degradation(query_latency)
        
        return results
    
    def _save_main_index(self) -> None:
        """Save main index to disk."""
        os.makedirs(self.main_index_path, exist_ok=True)
        
        if self.main_index:
            faiss.write_index(self.main_index, os.path.join(self.main_index_path, "main.index"))
        
        # Save metadata
        metadata = {
            "chunks": self.main_chunks,
            "metadata": self.main_metadata,
            "dimension": self.dimension,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.main_index_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def _save_version_history(self) -> None:
        """Save version history."""
        version_data = [v.to_dict() for v in self.version_history]
        
        with open(os.path.join(self.versions_path, "history.json"), "w") as f:
            json.dump(version_data, f, indent=2)
    
    def _cleanup_deltas(self) -> None:
        """Clean up old delta indices."""
        import shutil
        
        if os.path.exists(self.delta_path):
            for delta_dir in os.listdir(self.delta_path):
                shutil.rmtree(os.path.join(self.delta_path, delta_dir))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_vectors = 0
        
        if self.main_index:
            total_vectors += self.main_index.ntotal
        
        delta_vectors = sum(d.size() for d in self.delta_indices)
        total_vectors += delta_vectors
        
        return {
            "total_vectors": total_vectors,
            "main_index_vectors": self.main_index.ntotal if self.main_index else 0,
            "delta_indices_count": len(self.delta_indices),
            "delta_vectors": delta_vectors,
            "version_count": len(self.version_history),
            "storage_path": self.storage_path
        }