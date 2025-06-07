# Enhanced RAG-CSD Simulator Documentation

## Overview

The Enhanced RAG-CSD (Retrieval-Augmented Generation with Computational Storage Devices) simulator is a comprehensive system that emulates the behavior of computational storage devices for accelerating RAG workloads. This simulator models the complete pipeline from query encoding through document retrieval and query augmentation.

## Architecture

### Core Components

#### 1. Enhanced CSD Simulator (`core/csd_emulator.py`)

The heart of the system that simulates computational storage device characteristics:

**Key Features:**
- **Multi-level Cache Hierarchy**: L1 (64MB), L2 (512MB), L3 (2048MB) cache simulation
- **Memory-Mapped Storage**: Efficient vector storage using memory-mapped files
- **Parallel I/O Operations**: Concurrent read/write operations with configurable parallelism
- **Realistic Storage Modeling**: SSD and NAND bandwidth constraints simulation
- **Performance Metrics**: Comprehensive metrics collection for analysis

**Storage Metrics Tracked:**
- Read/write operations count
- Cache hit/miss ratios
- Data transfer volumes
- Average latency per operation
- Storage utilization

#### 2. Enhanced RAG Pipeline (`core/pipeline.py`)

The main orchestrator that coordinates all components:

**Key Features:**
- **Workload Classification**: Automatically classifies queries (single, small_batch, medium_batch, large_batch)
- **Pipeline Parallelism**: Overlaps encoding, retrieval, and augmentation stages
- **Adaptive Optimization**: Selects strategies based on workload characteristics
- **Batch Processing**: Optimized batch operations for throughput

#### 3. Encoder (`core/encoder.py`)

Handles query and document encoding with CSD integration:

**Key Features:**
- **CSD Offloading**: Simulates embedding computation on storage devices
- **Embedding Cache**: LRU cache for frequently accessed embeddings
- **Model Cache**: Efficient model loading and caching
- **Batch Optimization**: Configurable batch sizes for optimal throughput

#### 4. Augmentor (`core/augmentor.py`)

Combines queries with retrieved documents:

**Augmentation Strategies:**
- **Concatenation**: Simple joining of query and context
- **Template-based**: Structured formatting using templates
- **Weighted**: Relevance-score based prioritization

## Current System Data Flow

### Standard RAG Pipeline

```
Query → Encode → Retrieve → Augment → Generate
```

### Current CSD-Enhanced Pipeline

```
Query → [CSD: Encode] → [CSD: Vector Search] → [CSD: Retrieve] → [Host: Augment] → [Host: Generate]
```

## Performance Characteristics

### Simulated Hardware Parameters

- **SSD Bandwidth**: 2000 MB/s (configurable)
- **NAND Bandwidth**: 500 MB/s (configurable)
- **Compute Latency**: 0.1ms per operation (configurable)
- **Max Parallel Operations**: 8 (configurable)

### Cache Behavior

- **L1 Cache**: Hot embeddings in memory (64MB default)
- **L2 Cache**: Warm embeddings simulating SSD (512MB default)
- **L3 Cache**: Cold embeddings simulating disk (2048MB default)
- **LRU Eviction**: Least Recently Used replacement policy

## Configuration

### Key Configuration Parameters

```python
config = {
    "storage_path": "./enhanced_storage",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "enable_csd_emulation": True,
    "max_parallel_ops": 8,
    "ssd_bandwidth_mbps": 2000,
    "nand_bandwidth_mbps": 500,
    "enable_pipeline_parallel": True,
    "enable_caching": True
}
```

## Current Limitations

### System-Level Gaps

1. **Memory Hierarchy Modeling**:
   - No explicit DRAM modeling
   - Missing host-storage data transfer simulation
   - No P2P GPU memory transfer simulation

2. **System Integration**:
   - Generation stage not integrated with CSD pipeline
   - Missing GPU memory management
   - No explicit system-level bottleneck modeling

3. **Data Movement**:
   - Query input from DRAM not modeled
   - Augmented data transfer to GPU not simulated
   - No P2P storage-to-GPU transfer paths

## Usage Example

```python
from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline

# Initialize pipeline
config = {
    "vector_db_path": "./vector_db",
    "storage_path": "./storage",
    "enable_csd_emulation": True
}

pipeline = EnhancedRAGPipeline(config)

# Add documents
documents = ["Document 1 content", "Document 2 content"]
pipeline.add_documents(documents)

# Query processing
result = pipeline.query("What is the main topic?", top_k=5)
print(result["augmented_query"])

# Batch processing
queries = ["Query 1", "Query 2", "Query 3"]
results = pipeline.query_batch(queries, top_k=5)

# Get performance metrics
stats = pipeline.get_statistics()
print(stats["csd_metrics"])
```

## Metrics and Analysis

### Available Metrics

- **CSD Metrics**: Cache performance, I/O operations, latency
- **Pipeline Metrics**: End-to-end latency, throughput
- **Component Metrics**: Encoding time, retrieval time, augmentation time

### Performance Analysis

The simulator provides detailed performance breakdowns enabling:
- Bottleneck identification
- Cache optimization analysis  
- Parallel processing efficiency evaluation
- System-level performance modeling

## Future Enhancements

### Planned System-Level Improvements

1. **Complete System Data Flow**: Model DRAM→Storage→GPU data paths
2. **GPU Integration**: Simulate generation stage with GPU memory management
3. **P2P Transfers**: Direct storage-to-GPU data movement
4. **System Bottleneck Modeling**: Comprehensive system-level constraint simulation
5. **Advanced Cache Policies**: Prefetching and intelligent caching strategies