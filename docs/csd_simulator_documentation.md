# Enhanced RAG-CSD Simulator Documentation

## Overview

The Enhanced RAG-CSD (Retrieval-Augmented Generation with Computational Storage Devices) simulator is a comprehensive system that emulates the complete system-level behavior of computational storage devices for accelerating RAG workloads. This simulator models the entire data flow from DRAM through computational storage to GPU memory with realistic P2P transfers.

## System Architecture

### Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAG-CSD System Architecture                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    PCIe 4.0 x16     ┌─────────────────┐    P2P Transfer      │
│  │    DRAM     │◄──────────────────►│ Computational   │◄─────────────────────┐│
│  │   16GB      │    15.75 GB/s       │ Storage Device  │    12 GB/s           ││
│  │             │                     │     (CSD)       │                      ││
│  └─────────────┘                     │                 │                      ││
│                                      │ ┌─────────────┐ │                      ││
│                                      │ │ L1: 64MB    │ │                      ││
│                                      │ │ L2: 512MB   │ │                      ││
│                                      │ │ L3: 2048MB  │ │                      ││
│                                      │ └─────────────┘ │                      ││
│                                      │                 │                      ││
│                                      │ ┌─────────────┐ │                      ││
│                                      │ │    ERA      │ │                      ││
│                                      │ │  Pipeline   │ │                      ││
│                                      │ │ • Encode    │ │                      ││
│                                      │ │ • Retrieve  │ │                      ││
│                                      │ │ • Augment   │ │                      ││
│                                      │ └─────────────┘ │                      ││
│                                      └─────────────────┘                      ││
│                                                                               ││
│                                                                               ││
│  ┌─────────────┐                                                             ││
│  │     GPU     │◄────────────────────────────────────────────────────────────┘│
│  │    8GB      │                                                              │
│  │             │                                                              │
│  │ ┌─────────┐ │                                                              │
│  │ │Generate │ │                                                              │
│  │ │Pipeline │ │                                                              │
│  │ └─────────┘ │                                                              │
│  └─────────────┘                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Query Processing Data Flow:

┌──────────────┐    ┌─────────────────────────────────────┐    ┌─────────────┐
│    DRAM      │    │         Computational Storage       │    │     GPU     │
│              │    │                                     │    │             │
│ ┌──────────┐ │    │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │    │ ┌─────────┐ │
│ │  Query   │─┼────┼►│ Encode  │►│Retrieve │►│Augment  │─┼────┼►│Generate │ │
│ │   Data   │ │    │ │         │ │         │ │         │ │    │ │         │ │
│ └──────────┘ │    │ └─────────┘ └─────────┘ └─────────┘ │    │ └─────────┘ │
│              │    │                                     │    │             │
│              │    │     System Memory: 4GB CSD         │    │   Memory:   │
│              │    │     Storage: Memory-mapped          │    │    8GB      │
│              │    │     Cache: L1/L2/L3 Hierarchy      │    │             │
└──────────────┘    └─────────────────────────────────────┘    └─────────────┘
       │                                                               │
       │←──────────────── PCIe 4.0 x16 (15.75 GB/s) ─────────────────→│
       │                                                               │
       └─────────────────── P2P Transfer (12 GB/s) ──────────────────→┘

Timing Breakdown:
• DRAM → CSD: PCIe latency + bandwidth constraints
• CSD Processing: Encode (1ms) + Retrieve (5ms) + Augment (0.5ms)  
• CSD → GPU: P2P latency + bandwidth constraints
• GPU Generation: Fixed time (100ms) - not simulated in detail
```

## Core Components

### 1. SystemMemoryManager (`core/system_memory.py`)

**NEW**: Complete system memory management with realistic memory hierarchy simulation.

```
System Memory Architecture:

┌─────────────────────────────────────────────────────────────────────────────┐
│                         SystemMemoryManager                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │ DRAM        │     │ CSD Memory  │     │ GPU Memory  │                   │
│  │ Subsystem   │     │ Subsystem   │     │ Subsystem   │                   │
│  │             │     │             │     │             │                   │
│  │ Capacity:   │     │ Capacity:   │     │ Capacity:   │                   │
│  │ 16GB        │     │ 4GB         │     │ 8GB         │                   │
│  │             │     │             │     │             │                   │
│  │ Bandwidth:  │     │ Bandwidth:  │     │ Bandwidth:  │                   │
│  │ 51.2 GB/s   │     │ 15.75 GB/s  │     │ 900 GB/s    │                   │
│  │             │     │             │     │             │                   │
│  │ LRU Evict   │     │ LRU Evict   │     │ LRU Evict   │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│         │                     │                     │                       │
│         └─────────────────────┼─────────────────────┘                       │
│                               │                                             │
│  ┌─────────────────────────────┼─────────────────────────────────────┐       │
│  │                SystemBus   │                                     │       │
│  │                             │                                     │       │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │       │
│  │  │PCIe 4.0 x16 │     │    P2P      │     │   Memory    │         │       │
│  │  │15.75 GB/s   │     │ 12 GB/s     │     │  Internal   │         │       │
│  │  │2μs latency  │     │1μs latency  │     │ 0.01μs lat. │         │       │
│  │  └─────────────┘     └─────────────┘     └─────────────┘         │       │
│  └─────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Memory Subsystems**: Independent DRAM, CSD, and GPU memory with capacity limits
- **Realistic Transfer Characteristics**: PCIe and P2P bandwidth/latency modeling
- **LRU Memory Management**: Automatic eviction when capacity exceeded
- **Transfer Metrics**: Comprehensive tracking of data movement

### 2. SystemDataFlow (`core/system_data_flow.py`)

**NEW**: Complete system-level data flow orchestration with pipeline stages.

```
SystemDataFlow Processing Pipeline:

Stage 1: DRAM Input
┌─────────────────┐
│ Query arrives   │
│ in DRAM         │
│ • Allocation    │ 
│ • Storage       │
└─────────────────┘
         │
         ▼
Stage 2: DRAM → CSD Transfer  
┌─────────────────┐
│ PCIe Transfer   │
│ • 15.75 GB/s    │
│ • 2μs latency   │
│ • Async option  │
└─────────────────┘
         │
         ▼
Stage 3: CSD Processing (ERA Pipeline)
┌─────────────────┐
│ Encode (1ms)    │
│ ┌─────────────┐ │
│ │Text→Vector  │ │
│ │Embedding    │ │
│ └─────────────┘ │
│                 │
│ Retrieve (5ms)  │
│ ┌─────────────┐ │
│ │Vector Search│ │
│ │Top-K Docs   │ │
│ └─────────────┘ │
│                 │
│ Augment (0.5ms) │
│ ┌─────────────┐ │
│ │Combine Query│ │
│ │+ Retrieved  │ │
│ └─────────────┘ │
└─────────────────┘
         │
         ▼
Stage 4: CSD → GPU P2P Transfer
┌─────────────────┐
│ P2P Transfer    │
│ • 12 GB/s       │
│ • 1μs latency   │
│ • Direct path   │
└─────────────────┘
         │
         ▼
Stage 5: GPU Generation (Simulated)
┌─────────────────┐
│ Generate        │
│ • Fixed 100ms   │
│ • Not detailed  │
│ • Result ready  │
└─────────────────┘
```

**Key Features:**
- **Async/Sync Processing**: Configurable pipeline execution modes
- **Pipeline Parallelism**: Overlapping stage execution for throughput
- **Comprehensive Metrics**: Per-stage timing and data transfer tracking
- **Memory Integration**: Direct integration with SystemMemoryManager

### 3. Enhanced CSD Simulator (`core/csd_emulator.py`)

**ENHANCED**: Now includes complete ERA pipeline processing and system integration.

```
Enhanced CSD Simulator Architecture:

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Enhanced CSD Simulator                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    Cache Hierarchy                                  │     │
│  │                                                                     │     │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │     │
│  │  │ L1 Cache    │    │ L2 Cache    │    │ L3 Cache    │             │     │
│  │  │ 64MB        │    │ 512MB       │    │ 2048MB      │             │     │
│  │  │ Hot Data    │    │ Warm Data   │    │ Cold Data   │             │     │
│  │  │ LRU Evict   │    │ LRU Evict   │    │ LRU Evict   │             │     │
│  │  └─────────────┘    └─────────────┘    └─────────────┘             │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                  ERA Pipeline Processor                            │     │
│  │                                                                     │     │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │     │
│  │  │   Encode    │───►│  Retrieve   │───►│  Augment    │             │     │
│  │  │             │    │             │    │             │             │     │
│  │  │ • Text→Vec  │    │ • Vector    │    │ • Combine   │             │     │
│  │  │ • 1ms       │    │   Search    │    │   Data      │             │     │
│  │  │ • Normalize │    │ • 5ms       │    │ • 0.5ms     │             │     │
│  │  │             │    │ • Top-K     │    │ • Format    │             │     │
│  │  └─────────────┘    └─────────────┘    └─────────────┘             │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                 Memory-Mapped Storage                               │     │
│  │                                                                     │     │
│  │  ┌─────────────┐    ┌─────────────┐                                 │     │
│  │  │Vector Store │    │ Metadata    │                                 │     │
│  │  │Memory-mapped│    │ JSON Store  │                                 │     │
│  │  │Binary Files │    │             │                                 │     │
│  │  │Parallel I/O │    │             │                                 │     │
│  │  └─────────────┘    └─────────────┘                                 │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                  System Integration                                 │     │
│  │                                                                     │     │
│  │  SystemMemoryManager ◄─────► P2P GPU Transfer                      │     │
│  │  Memory Allocation           Direct Storage→GPU                     │     │
│  │  Transfer Coordination       12 GB/s Bandwidth                      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**NEW Features:**
- **ERA Pipeline Processing**: Complete Encode-Retrieve-Augment on CSD
- **P2P GPU Transfer**: Direct storage-to-GPU memory transfer simulation  
- **System Memory Integration**: Coordinated memory management
- **Realistic Timing**: Per-operation latency simulation

### 4. Enhanced RAG Pipeline (`core/pipeline.py`)

**ENHANCED**: Now supports complete system data flow mode alongside traditional processing.

```
Enhanced RAG Pipeline Modes:

Mode 1: Traditional Pipeline (enable_system_data_flow=False)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Encode    │───►│  Retrieve   │───►│  Augment    │───►│  Generate   │
│   Query     │    │ Documents   │    │    Query    │    │  Response   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

Mode 2: System Data Flow (enable_system_data_flow=True)  
┌─────────────┐    ┌─────────────────────────────────────┐    ┌─────────────┐
│    DRAM     │───►│         System Data Flow            │───►│     GPU     │
│   Query     │    │                                     │    │  Generate   │
│             │    │ ┌─────┐   ┌─────┐   ┌─────┐        │    │             │
│             │    │ │  E  │──►│  R  │──►│  A  │        │    │             │
│             │    │ └─────┘   └─────┘   └─────┘        │    │             │
│             │    │           CSD Processing            │    │             │
└─────────────┘    └─────────────────────────────────────┘    └─────────────┘
```

**NEW Configuration:**
```python
# Enable complete system data flow
config = {
    'enable_system_data_flow': True,  # NEW: Full system simulation
    'enable_csd_emulation': True,     # CSD processing
    'enable_pipeline_parallel': True  # Pipeline parallelism
}
```

## Performance Characteristics

### Simulated Hardware Parameters

```
Memory Subsystem Specifications:

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Memory Hierarchy                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DRAM (Host Memory)          │  CSD Memory               │  GPU Memory      │
│  ┌─────────────────────────┐ │  ┌─────────────────────┐   │  ┌─────────────┐ │
│  │ Capacity: 16GB          │ │  │ Capacity: 4GB       │   │  │ Capacity:   │ │
│  │ Bandwidth: 51.2 GB/s    │ │  │ Bandwidth: 2.0 GB/s │   │  │ 8GB         │ │
│  │ Latency: 0.1μs          │ │  │ (SSD: 2.0 GB/s)     │   │  │ Bandwidth:  │ │
│  │ Type: DDR4-3200         │ │  │ (NAND: 0.5 GB/s)    │   │  │ 900 GB/s    │ │
│  └─────────────────────────┘ │  │ Latency: 0.1ms      │   │  │ Latency:    │ │
│                             │  │ Cache: L1/L2/L3      │   │  │ 0.01μs      │ │
│                             │  └─────────────────────┘   │  │ Type: HBM    │ │
│                             │                           │  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

Transfer Characteristics:

┌─────────────────────────────────────────────────────────────────────────────┐
│                            System Bus                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PCIe 4.0 x16               │  P2P Transfer              │  Memory Internal │
│  ┌─────────────────────────┐ │  ┌─────────────────────┐   │  ┌─────────────┐ │
│  │ Bandwidth: 15.75 GB/s   │ │  │ Bandwidth: 12 GB/s  │   │  │ Bandwidth:  │ │
│  │ Latency: 2μs            │ │  │ Latency: 1μs        │   │  │ Native      │ │
│  │ Bidirectional           │ │  │ Direct GPU↔Storage  │   │  │ Latency:    │ │
│  │ Host↔Storage/GPU        │ │  │ Bypass Host         │   │  │ 0.01μs      │ │
│  └─────────────────────────┘ │  └─────────────────────┘   │  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CSD Processing Latencies

- **Encode Stage**: 1.0ms per operation
- **Retrieve Stage**: 5.0ms per operation  
- **Augment Stage**: 0.5ms per operation
- **GPU Generation**: 100ms (fixed, not detailed)

### Cache Performance

- **L1 Cache (64MB)**: Hot embeddings, instant access
- **L2 Cache (512MB)**: Warm embeddings, SSD simulation
- **L3 Cache (2048MB)**: Cold embeddings, disk simulation
- **LRU Eviction**: Automatic capacity management

## Usage Examples

### Basic CSD Simulation

```python
from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline

# Traditional CSD emulation
config = {
    'vector_db_path': './vector_db',
    'storage_path': './storage',
    'enable_csd_emulation': True,
    'enable_system_data_flow': False
}

pipeline = EnhancedRAGPipeline(config)
```

### Complete System Data Flow

```python
# Full system simulation with P2P transfers
config = {
    'vector_db_path': './vector_db', 
    'storage_path': './storage',
    'enable_csd_emulation': True,
    'enable_system_data_flow': True,  # Enable complete system simulation
    'enable_pipeline_parallel': True
}

pipeline = EnhancedRAGPipeline(config)

# Add documents
documents = ["Document 1 content", "Document 2 content"]
result = pipeline.add_documents(documents)

# Query with system data flow
query_result = pipeline.query("What is the main topic?", top_k=5)
print(f"Data flow path: {query_result.get('data_flow_path')}")  # "DRAM→CSD→GPU"

# Get comprehensive statistics
stats = pipeline.get_statistics()
print(f"System memory stats: {stats['system_data_flow_metrics']}")
```

### Performance Analysis

```python
# Get detailed system metrics
stats = pipeline.get_statistics()

# CSD performance
csd_metrics = stats["csd_metrics"]
print(f"Cache hit rate: {csd_metrics['cache_hit_rate']:.2%}")
print(f"Storage usage: {csd_metrics['storage_usage_mb']:.1f}MB")

# System data flow metrics
if "system_data_flow_metrics" in stats:
    sdf_metrics = stats["system_data_flow_metrics"]
    print(f"Average latency: {sdf_metrics['data_flow_metrics']['avg_latency_ms']:.2f}ms")
    print(f"P2P transfers: {sdf_metrics['data_flow_metrics']['p2p_transfers']}")
    print(f"PCIe transfers: {sdf_metrics['data_flow_metrics']['pcie_transfers']}")
```

## System Integration Benefits

### Realistic System Modeling

1. **Complete Memory Hierarchy**: Models actual DRAM, storage, and GPU memory constraints
2. **Realistic Transfer Costs**: Includes PCIe and P2P bandwidth/latency
3. **System Bottlenecks**: Identifies actual performance limitations
4. **P2P Optimization**: Direct storage-to-GPU transfers bypass host

### Performance Insights

1. **End-to-End Latency**: Complete system timing analysis
2. **Memory Utilization**: Track memory pressure across subsystems  
3. **Transfer Efficiency**: P2P vs traditional transfer comparison
4. **Cache Effectiveness**: Multi-level cache performance analysis

### Research Applications

1. **System Design**: Evaluate different memory/storage configurations
2. **Bottleneck Analysis**: Identify system performance limitations
3. **Optimization Strategies**: Compare P2P vs traditional data paths
4. **Scalability Studies**: Model system behavior under different loads

The enhanced simulator now provides a complete and realistic model of the RAG-CSD system architecture, enabling comprehensive performance analysis and system optimization research.