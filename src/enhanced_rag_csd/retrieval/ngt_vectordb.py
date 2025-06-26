"""
NGT (Neighborhood Graph and Tree) Vector Database Implementation.
A simplified implementation inspired by Yahoo's NGT algorithm.
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple
import heapq
import random
from collections import defaultdict

from .vectordb import VectorDB


class NGTVectorDB(VectorDB):
    """
    A vector database inspired by NGT (Neighborhood Graph and Tree).
    Combines graph-based search with tree-based indexing.
    """
    
    def __init__(self, dimension: int, max_edges: int = 10, 
                 search_depth: int = 3, tree_fanout: int = 2):
        """
        Initialize NGT-inspired Vector Database.
        
        Args:
            dimension: Vector dimension
            max_edges: Maximum edges per node in the graph
            search_depth: Depth for graph traversal
            tree_fanout: Fanout for the tree structure
        """
        self.dimension = dimension
        self.max_edges = max_edges
        self.search_depth = search_depth
        self.tree_fanout = tree_fanout
        
        # Graph structure: node_id -> [neighbor_ids]
        self.graph = defaultdict(list)
        
        # Tree structure for initial search
        self.tree = {}  # node_id -> {'children': [], 'vector': np.array, 'is_leaf': bool}
        self.root_id = None
        
        # Storage
        self.vectors = []
        self.chunks = []
        self.metadata = []
        
        # For efficient nearest neighbor computation
        self.vector_norms = []
    
    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between vectors."""
        return np.linalg.norm(v1 - v2)
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def _build_graph_edges(self, new_node_id: int) -> None:
        """Build graph edges for a new node using k-NN graph construction."""
        if len(self.vectors) <= 1:
            return
        
        new_vector = self.vectors[new_node_id]
        
        # Find k nearest neighbors for the new node
        distances = []
        for i, vector in enumerate(self.vectors):
            if i != new_node_id:
                dist = self._euclidean_distance(new_vector, vector)
                distances.append((dist, i))
        
        # Sort by distance and take closest neighbors
        distances.sort()
        k_neighbors = min(self.max_edges, len(distances))
        
        # Add edges (bidirectional)
        for _, neighbor_id in distances[:k_neighbors]:
            self.graph[new_node_id].append(neighbor_id)
            
            # Add reverse edge if neighbor doesn't have too many edges
            if len(self.graph[neighbor_id]) < self.max_edges:
                self.graph[neighbor_id].append(new_node_id)
            else:
                # Replace farthest neighbor if new node is closer
                neighbor_vector = self.vectors[neighbor_id]
                neighbor_distances = []
                
                for connected_id in self.graph[neighbor_id]:
                    connected_vector = self.vectors[connected_id]
                    dist = self._euclidean_distance(neighbor_vector, connected_vector)
                    neighbor_distances.append((dist, connected_id))
                
                neighbor_distances.sort(reverse=True)  # Sort by distance (farthest first)
                new_dist = self._euclidean_distance(neighbor_vector, new_vector)
                
                if new_dist < neighbor_distances[0][0]:
                    # Remove farthest neighbor and add new node
                    farthest_id = neighbor_distances[0][1]
                    self.graph[neighbor_id].remove(farthest_id)
                    self.graph[neighbor_id].append(new_node_id)
    
    def _build_tree_structure(self) -> None:
        """Build tree structure for efficient initial search."""
        if not self.vectors:
            return
        
        # Simple tree construction: randomly select root and build levels
        if self.root_id is None:
            self.root_id = 0
            self.tree[0] = {
                'children': [],
                'vector': self.vectors[0],
                'is_leaf': len(self.vectors) == 1
            }
        
        # Add new nodes to tree (simplified insertion)
        for node_id in range(len(self.tree), len(self.vectors)):
            # Find closest existing node as parent
            best_parent = self.root_id
            best_distance = self._euclidean_distance(
                self.vectors[node_id], 
                self.tree[best_parent]['vector']
            )
            
            for existing_id in self.tree:
                if not self.tree[existing_id]['is_leaf']:
                    continue
                
                dist = self._euclidean_distance(
                    self.vectors[node_id], 
                    self.tree[existing_id]['vector']
                )
                
                if dist < best_distance:
                    best_distance = dist
                    best_parent = existing_id
            
            # Add new node
            self.tree[node_id] = {
                'children': [],
                'vector': self.vectors[node_id],
                'is_leaf': True
            }
            
            # Update parent
            if len(self.tree[best_parent]['children']) < self.tree_fanout:
                self.tree[best_parent]['children'].append(node_id)
                self.tree[best_parent]['is_leaf'] = False
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add documents to the NGT index."""
        start_id = len(self.vectors)
        
        # Store vectors and metadata
        for embedding, doc, meta in zip(embeddings, documents, metadata):
            self.vectors.append(embedding.astype(np.float32))
            self.chunks.append(doc)
            self.metadata.append(meta)
            self.vector_norms.append(np.linalg.norm(embedding))
        
        # Build graph edges for new nodes
        for i in range(start_id, len(self.vectors)):
            self._build_graph_edges(i)
        
        # Rebuild tree structure
        self._build_tree_structure()
    
    def _graph_search(self, query: np.ndarray, entry_points: List[int], top_k: int) -> List[Tuple[float, int]]:
        """Perform graph-based search starting from entry points."""
        visited = set()
        candidates = []  # Min-heap: (distance, node_id)
        
        # Initialize with entry points
        for entry_id in entry_points:
            if entry_id < len(self.vectors):
                dist = self._euclidean_distance(query, self.vectors[entry_id])
                heapq.heappush(candidates, (dist, entry_id))
                visited.add(entry_id)
        
        results = []
        
        # Graph traversal
        for _ in range(min(self.search_depth * len(self.vectors), 1000)):
            if not candidates:
                break
            
            current_dist, current_id = heapq.heappop(candidates)
            results.append((current_dist, current_id))
            
            # Explore neighbors
            for neighbor_id in self.graph[current_id]:
                if neighbor_id not in visited and neighbor_id < len(self.vectors):
                    visited.add(neighbor_id)
                    dist = self._euclidean_distance(query, self.vectors[neighbor_id])
                    heapq.heappush(candidates, (dist, neighbor_id))
        
        # Sort results by distance and return top_k
        results.sort()
        return results[:top_k]
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search using NGT-inspired approach."""
        if not self.vectors:
            return []
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Stage 1: Tree-based search to find entry points
        entry_points = []
        if self.root_id is not None:
            # Simple tree traversal to find good entry points
            queue = [self.root_id]
            while queue and len(entry_points) < 3:
                node_id = queue.pop(0)
                entry_points.append(node_id)
                
                if not self.tree[node_id]['is_leaf']:
                    queue.extend(self.tree[node_id]['children'])
        
        if not entry_points:
            entry_points = [0]  # Fallback to first node
        
        # Stage 2: Graph-based search
        search_results = self._graph_search(query_embedding, entry_points, top_k * 2)
        
        # Convert to final results format
        results = []
        for dist, node_id in search_results[:top_k]:
            if node_id < len(self.chunks):
                # Convert distance to similarity score (higher is better)
                similarity = 1.0 / (1.0 + dist)
                
                results.append({
                    'content': self.chunks[node_id],
                    'metadata': self.metadata[node_id],
                    'score': float(similarity)
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the NGT index."""
        total_edges = sum(len(neighbors) for neighbors in self.graph.values())
        avg_edges = total_edges / len(self.graph) if self.graph else 0
        
        return {
            'total_documents': len(self.vectors),
            'max_edges_per_node': self.max_edges,
            'avg_edges_per_node': float(avg_edges),
            'total_edges': total_edges,
            'tree_nodes': len(self.tree),
            'search_depth': self.search_depth
        }