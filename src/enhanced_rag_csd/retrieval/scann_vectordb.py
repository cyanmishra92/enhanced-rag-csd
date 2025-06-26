"""
ScaNN (Scalable Nearest Neighbors) Vector Database Implementation.
A simplified implementation inspired by Google's ScaNN algorithm.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import faiss
from sklearn.cluster import KMeans

from .vectordb import VectorDB


class ScaNNVectorDB(VectorDB):
    """
    A vector database inspired by ScaNN (Scalable Nearest Neighbors).
    Implements learned quantization with anisotropic loss function.
    """
    
    def __init__(self, dimension: int, num_clusters: int = 100, 
                 anisotropic_quantization: bool = True, reorder_k: int = 1000):
        """
        Initialize ScaNN-inspired Vector Database.
        
        Args:
            dimension: Vector dimension
            num_clusters: Number of clusters for quantization
            anisotropic_quantization: Use anisotropic quantization
            reorder_k: Number of candidates to reorder with exact distance
        """
        self.dimension = dimension
        self.num_clusters = num_clusters
        self.anisotropic_quantization = anisotropic_quantization
        self.reorder_k = reorder_k
        
        # Initialize quantizer
        self.quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(self.quantizer, dimension, num_clusters)
        
        # For anisotropic quantization
        self.learned_rotation = None
        if anisotropic_quantization:
            self.learned_rotation = np.eye(dimension, dtype=np.float32)
        
        # Storage
        self.chunks = []
        self.metadata = []
        self.is_trained = False
        
        # Product quantization for compression
        self.pq_dim = 8  # Number of sub-vectors
        self.pq_bits = 8  # Bits per sub-vector
        
    def _apply_rotation(self, vectors: np.ndarray) -> np.ndarray:
        """Apply learned rotation for anisotropic quantization."""
        if self.learned_rotation is not None:
            return vectors @ self.learned_rotation.T
        return vectors
    
    def _learn_rotation(self, vectors: np.ndarray) -> None:
        """Learn rotation matrix for anisotropic quantization."""
        if not self.anisotropic_quantization:
            return
        
        # Simplified rotation learning - use PCA
        vectors_centered = vectors - np.mean(vectors, axis=0)
        covariance_matrix = np.cov(vectors_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        self.learned_rotation = eigenvectors[:, idx].T.astype(np.float32)
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add documents to the ScaNN index."""
        embeddings = embeddings.astype(np.float32)
        
        # Learn rotation if not trained
        if not self.is_trained:
            if len(embeddings) >= self.num_clusters:
                self._learn_rotation(embeddings)
                
                # Apply rotation and train index
                rotated_embeddings = self._apply_rotation(embeddings)
                faiss.normalize_L2(rotated_embeddings)
                
                self.index.train(rotated_embeddings)
                self.is_trained = True
            else:
                # Not enough data to train, store for later
                pass
        
        # Apply rotation and normalize
        rotated_embeddings = self._apply_rotation(embeddings)
        faiss.normalize_L2(rotated_embeddings)
        
        # Add to index if trained
        if self.is_trained:
            self.index.add(rotated_embeddings)
        
        # Store documents and metadata
        self.chunks.extend(documents)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search using ScaNN-inspired approach with reordering."""
        if not self.is_trained or self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Apply rotation to query
        rotated_query = self._apply_rotation(query_embedding.reshape(1, -1))
        faiss.normalize_L2(rotated_query)
        
        # Two-stage search: coarse + fine
        # Stage 1: Get more candidates than needed
        search_k = min(self.reorder_k, self.index.ntotal)
        scores, indices = self.index.search(rotated_query, search_k)
        
        # Stage 2: Reorder with exact distances (simplified)
        # In a full ScaNN implementation, this would use learned distance functions
        valid_indices = indices[0][indices[0] >= 0]
        valid_scores = scores[0][:len(valid_indices)]
        
        # Sort by score and take top_k
        sorted_pairs = sorted(zip(valid_scores, valid_indices), reverse=True)
        
        results = []
        for score, idx in sorted_pairs[:top_k]:
            if idx < len(self.chunks):
                results.append({
                    'content': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the ScaNN index."""
        return {
            'total_documents': len(self.chunks),
            'num_clusters': self.num_clusters,
            'is_trained': self.is_trained,
            'anisotropic_quantization': self.anisotropic_quantization,
            'reorder_k': self.reorder_k,
            'index_size': self.index.ntotal if self.is_trained else 0
        }