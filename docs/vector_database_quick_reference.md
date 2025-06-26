# Vector Database Quick Reference

## 🚀 Quick Algorithm Selection

| **Need** | **Algorithm** | **Command** |
|----------|---------------|-------------|
| **Fastest** | FAISS | `VectorDBFactory.create_vectordb("faiss", dimension=384)` |
| **Most Accurate** | HNSW | `VectorDBFactory.create_vectordb("hnsw", dimension=384, m=16)` |
| **Memory Efficient** | LSH | `VectorDBFactory.create_vectordb("lsh", dimension=384)` |
| **Large Scale** | IVF-Flat | `VectorDBFactory.create_vectordb("ivf_flat", dimension=384)` |
| **Maximum Compression** | IVF-PQ | `VectorDBFactory.create_vectordb("ivf_pq", dimension=384)` |
| **Real-time Updates** | Incremental | `VectorDBFactory.create_vectordb("incremental", dimension=384)` |
| **Research/Advanced** | ScaNN | `VectorDBFactory.create_vectordb("scann", dimension=384)` |
| **Hybrid Approach** | NGT | `VectorDBFactory.create_vectordb("ngt", dimension=384)` |

## 📊 Performance Summary

| Algorithm | Speed | Memory | Accuracy | Scalability |
|-----------|-------|--------|----------|-------------|
| **FAISS** | ⚡⚡⚡ | 🔶🔶 | ⭐⭐⭐ | ⭐⭐ |
| **HNSW** | ⚡⚡ | 🔶🔶 | ⭐⭐⭐ | ⭐⭐⭐ |
| **IVF-Flat** | ⚡⚡ | 🔶 | ⭐⭐ | ⭐⭐⭐ |
| **IVF-PQ** | ⚡ | 🔶 | ⭐⭐ | ⭐⭐⭐ |
| **LSH** | ⚡⚡ | 🔶 | ⭐ | ⭐⭐⭐ |
| **ScaNN** | ⚡⚡⚡ | 🔶 | ⭐⭐⭐ | ⭐⭐⭐ |
| **NGT** | ⚡ | 🔶🔶 | ⭐⭐ | ⭐⭐ |
| **Incremental** | ⚡⚡ | 🔶🔶 | ⭐⭐ | ⭐⭐ |

## 🎯 Use Case Matrix

| **Dataset Size** | **Accuracy Need** | **Memory Limit** | **Recommended** |
|------------------|-------------------|------------------|-----------------|
| < 10K | High | No | FAISS |
| < 10K | Medium | Yes | LSH |
| 10K-100K | High | No | HNSW |
| 10K-100K | Medium | Yes | IVF-Flat |
| 100K-1M | High | No | HNSW |
| 100K-1M | Medium | Yes | IVF-Flat |
| > 1M | High | No | ScaNN |
| > 1M | Medium | Yes | IVF-PQ |

## ⚡ One-Liner Examples

```python
# Import factory
from enhanced_rag_csd.retrieval.vectordb_factory import VectorDBFactory

# Speed champion
db = VectorDBFactory.create_vectordb("faiss", dimension=384)

# Accuracy champion  
db = VectorDBFactory.create_vectordb("hnsw", dimension=384, m=32, ef_search=200)

# Memory champion
db = VectorDBFactory.create_vectordb("lsh", dimension=384, num_hashes=5)

# Scale champion
db = VectorDBFactory.create_vectordb("ivf_flat", dimension=384, nlist=1000)

# Compression champion
db = VectorDBFactory.create_vectordb("ivf_pq", dimension=384, m=8)

# Research champion
db = VectorDBFactory.create_vectordb("scann", dimension=384, anisotropic_quantization=True)
```