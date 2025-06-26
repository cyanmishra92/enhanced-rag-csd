"""
Locality Sensitive Hashing (LSH) Vector Database Implementation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import hashlib
from collections import defaultdict
import random

from .vectordb import VectorDB


class LSHVectorDB(VectorDB):
    """A vector database using Locality Sensitive Hashing for approximate nearest neighbor search."""
    
    def __init__(self, dimension: int, num_hashes: int = 10, num_bands: int = 5):
        """
        Initialize LSH Vector Database.
        
        Args:
            dimension: Vector dimension
            num_hashes: Number of hash functions per band
            num_bands: Number of bands for LSH
        """
        self.dimension = dimension
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        
        # Initialize random hyperplanes for LSH
        self.hyperplanes = []
        for _ in range(num_bands * num_hashes):
            hyperplane = np.random.randn(dimension)
            hyperplane = hyperplane / np.linalg.norm(hyperplane)
            self.hyperplanes.append(hyperplane)
        
        # Storage
        self.hash_tables = [defaultdict(list) for _ in range(num_bands)]
        self.vectors = []
        self.chunks = []
        self.metadata = []
    
    def _hash_vector(self, vector: np.ndarray) -> List[str]:
        """Generate LSH hashes for a vector."""
        hashes = []
        
        for band in range(self.num_bands):
            band_hash = []
            start_idx = band * self.num_hashes
            
            for i in range(self.num_hashes):
                hyperplane = self.hyperplanes[start_idx + i]
                hash_val = 1 if np.dot(vector, hyperplane) >= 0 else 0
                band_hash.append(str(hash_val))
            
            # Combine hash values for this band
            band_signature = ''.join(band_hash)
            hashes.append(band_signature)
        
        return hashes
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add documents to the LSH index."""
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        for i, (embedding, doc, meta) in enumerate(zip(embeddings, documents, metadata)):
            doc_id = len(self.vectors)
            
            # Store vector and metadata
            self.vectors.append(embedding)
            self.chunks.append(doc)
            self.metadata.append(meta)
            
            # Generate LSH hashes and add to hash tables
            hashes = self._hash_vector(embedding)
            for band_idx, hash_signature in enumerate(hashes):
                self.hash_tables[band_idx][hash_signature].append(doc_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar documents using LSH."""
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Get candidate documents from LSH
        candidates = set()
        query_hashes = self._hash_vector(query_embedding)
        
        for band_idx, hash_signature in enumerate(query_hashes):
            if hash_signature in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][hash_signature])
        
        # If no candidates found, return empty results
        if not candidates:
            return []
        
        # Calculate exact similarities for candidates
        similarities = []
        for doc_id in candidates:
            vector = self.vectors[doc_id]
            similarity = np.dot(query_embedding, vector)
            similarities.append((similarity, doc_id))
        
        # Sort by similarity and return top_k
        similarities.sort(reverse=True)
        
        results = []
        for similarity, doc_id in similarities[:top_k]:
            results.append({
                'content': self.chunks[doc_id],
                'metadata': self.metadata[doc_id],
                'score': float(similarity)
            })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the LSH index."""
        total_docs = len(self.vectors)
        
        # Calculate average bucket size
        bucket_sizes = []
        for hash_table in self.hash_tables:
            bucket_sizes.extend([len(bucket) for bucket in hash_table.values()])
        
        avg_bucket_size = np.mean(bucket_sizes) if bucket_sizes else 0
        max_bucket_size = max(bucket_sizes) if bucket_sizes else 0
        
        return {
            'total_documents': total_docs,
            'num_bands': self.num_bands,
            'num_hashes_per_band': self.num_hashes,
            'total_buckets': sum(len(table) for table in self.hash_tables),
            'avg_bucket_size': float(avg_bucket_size),
            'max_bucket_size': int(max_bucket_size)
        }