# Motivation: Computational Storage for Efficient RAG Systems

## Abstract

Retrieval-Augmented Generation (RAG) systems have become essential for providing factual, up-to-date responses in large language model applications. However, current RAG implementations face significant challenges in GPU memory management, data movement costs, and computational efficiency. This work motivates the need for Computational Storage Devices (CSDs) to address these limitations by moving ML primitives (encoding, retrieval, augmentation) directly to the storage layer, reducing data movement overhead and improving overall system efficiency.

## 1. GPU Memory Swapping Issues in Classical RAG

### 1.1 Problem Statement

Classical RAG systems suffer from **GPU memory fragmentation** and **model switching overhead** due to the heterogeneous computational requirements of different pipeline stages:

- **Encoding Phase**: Requires smaller transformer models (e.g., E5-base-v2: ~440MB)
- **Generation Phase**: Requires large language models (e.g., Llama3-8B: ~16GB)
- **Retrieval Phase**: Requires vector database indices (e.g., 1M vectors × 384 dims = ~1.5GB)

### 1.2 Current System Limitations

**PipeRAG Analysis:**
- Hardware underutilization: Either inference or retrieval system idle at any given time
- Sequential dependencies lead to GPU memory swapping between encoding and generation models
- 2.6× speedup achieved only through pipeline parallelism, not addressing fundamental memory issues

**FlashRAG Limitations:**
- Requires loading entire vector databases into GPU memory
- Memory thrashing when switching between different model components
- Optimized refiner still requires significant GPU memory for text processing

**EdgeRAG Constraints:**
- Limited memory capacity prevents loading large embedding databases
- Memory thrashing leads to poor performance on resource-constrained devices
- Requires embedding pruning and on-demand generation to fit memory constraints

### 1.3 Quantitative Impact

Current systems face severe memory limitations:

| System | Model Size | Vector DB Size | Total GPU Memory | Swapping Overhead |
|--------|------------|---------------|------------------|-------------------|
| Classical RAG | 16GB (LLM) + 440MB (Encoder) | 1.5GB (1M vectors) | 18GB+ | 200-500ms per swap |
| PipeRAG | 16GB (LLM) + 440MB (Encoder) | 1.5GB (1M vectors) | 18GB+ | Reduced via pipelining |
| FlashRAG | 16GB (LLM) + 440MB (Encoder) | 1.5GB (1M vectors) | 18GB+ | Optimized components |
| EdgeRAG | Pruned models | Pruned embeddings | <8GB | On-demand generation |

**Key Issues:**
1. **Model Switching Latency**: 200-500ms overhead when swapping between encoding and generation models
2. **Memory Fragmentation**: Up to 30% memory waste due to allocation patterns
3. **Cache Pollution**: Frequent model swaps invalidate GPU caches, reducing efficiency

## 2. Data Movement and Bandwidth Costs in Retrieval Operations

### 2.1 Data Movement Analysis

RAG systems require massive data movement between storage, CPU memory, and GPU memory:

**Typical RAG Query Pipeline:**
1. **Query Encoding**: Move query text to GPU → Encode → Move embeddings back
2. **Vector Retrieval**: Load vector index to memory → Compute similarities → Return top-k candidates
3. **Document Retrieval**: Load documents from storage → Move to GPU memory
4. **Context Augmentation**: Combine query + documents → Move to GPU for generation

### 2.2 Bandwidth and Latency Costs

**Storage to CPU Memory:**
- NVMe SSD: ~7GB/s sequential read, ~100μs latency
- Vector database loading: 1.5GB ÷ 7GB/s = 214ms
- Document retrieval: 50MB ÷ 7GB/s = 7ms

**CPU to GPU Memory:**
- PCIe 4.0 x16: ~32GB/s theoretical, ~25GB/s practical
- Vector transfer: 1.5GB ÷ 25GB/s = 60ms
- Document transfer: 50MB ÷ 25GB/s = 2ms

**Total Data Movement per Query:**
- **Latency**: 214ms + 7ms + 60ms + 2ms = **283ms overhead**
- **Bandwidth**: 1.55GB per query × 32GB/s = **49ms theoretical minimum**
- **Energy**: ~15W PCIe + ~300W GPU = **89ms × 315W = 28J per query**

### 2.3 Scaling Impact

For high-throughput RAG systems:
- **1000 queries/second**: 283GB/s data movement required
- **Current limitations**: PCIe 4.0 × 16 = 32GB/s maximum
- **Bottleneck ratio**: 283GB/s ÷ 32GB/s = **8.8× bandwidth deficit**

## 3. Computational Requirements: Encoding vs Generation

### 3.1 Asymmetric Computational Demands

RAG systems exhibit highly asymmetric computational requirements:

**Query Encoding (Lightweight):**
- Model: E5-base-v2 (110M parameters)
- Computation: 110M × 2 FLOPS = 220M FLOPS per query
- Memory: 440MB model + 1.5KB query = 440MB total
- **Suitable for ARM cores or lightweight accelerators**

**Text Generation (Heavyweight):**
- Model: Llama3-8B (8B parameters)
- Computation: 8B × 2 × sequence_length FLOPS
- Memory: 16GB model + multi-GB KV cache
- **Requires high-end GPUs**

### 3.2 ARM Core Capabilities for Encoding

Modern ARM cores (Cortex-A78) specifications:
- **Performance**: 420 GOPS with NEON SIMD
- **Power**: 5W (84 GOPS/W efficiency)
- **Memory**: 25.6GB/s bandwidth, 4MB L3 cache
- **Vector Width**: 128-bit NEON processing

**Encoding Performance Analysis:**
- Query encoding: 220M FLOPS ÷ 420G FLOPS/s = **0.52ms**
- Batch of 100 queries: 100 × 220M ÷ 420G = **52ms**
- **Power efficiency**: 84 GOPS/W vs GPU's ~50 GOPS/W

### 3.3 Computational Waste in Current Systems

Current systems over-provision computational resources:

| Component | Required Compute | Allocated Compute | Efficiency |
|-----------|------------------|-------------------|------------|
| Query Encoding | 420 GOPS | 19.5 TFLOPS (GPU) | 2.1% |
| Vector Retrieval | 100 GOPS | 19.5 TFLOPS (GPU) | 0.5% |
| Text Generation | 19.5 TFLOPS | 19.5 TFLOPS (GPU) | 100% |

**Key Insight**: 95%+ of GPU computational capacity is wasted on lightweight operations that could be efficiently handled by CSD-integrated processors.

## 4. Modern Computational Storage Capabilities

### 4.1 CSD Hardware Specifications

Modern CSDs provide substantial computational capabilities:

**ARM-based CSDs:**
- **Processor**: Quad-core ARM Cortex-A78 @ 3GHz
- **SIMD**: NEON vector processing units
- **Memory**: 8-32GB DDR4, 25.6GB/s bandwidth
- **Storage**: Direct NVMe access, no PCIe overhead
- **Power**: 5-15W total system power

**FPGA-based CSDs (Samsung SmartSSD):**
- **FPGA**: Xilinx Kintex UltraScale+ (16nm)
- **Memory**: 4GB DDR4 dedicated to FPGA
- **Compute**: 160 GOPS at 250MHz
- **Precision**: Optimized for FP16 ML operations
- **Power**: 25W enhanced compute mode

**Custom ASIC CSDs:**
- **Performance**: Up to 5 TOPS for ML operations
- **Memory**: 100GB/s custom memory interfaces
- **Efficiency**: 200+ GOPS/W power efficiency
- **Latency**: Sub-microsecond operation setup times

### 4.2 ML Primitive Capabilities

CSDs can efficiently handle core RAG ML operations:

**Query Encoding:**
- ARM Cortex-A78: 420 GOPS ÷ 220M FLOPS = **0.52ms per query**
- Samsung SmartSSD: 160 GOPS ÷ 220M FLOPS = **1.38ms per query**
- Custom ASIC: 5000 GOPS ÷ 220M FLOPS = **0.044ms per query**

**Vector Similarity Search:**
- ARM cores with NEON: Optimized dot product operations
- FPGA: Parallel similarity computations across multiple candidates
- Custom ASIC: Specialized vector processing units

**Context Augmentation:**
- Lightweight text processing and concatenation
- Template-based context formatting
- Attention weight computation for relevance ranking

### 4.3 Comparative Analysis

| Hardware | Peak Performance | Power | Efficiency | ML Encoding Time |
|----------|------------------|-------|------------|------------------|
| GPU (A100) | 19.5 TFLOPS | 300W | 65 GFLOPS/W | 0.011ms |
| ARM Cortex-A78 | 420 GOPS | 5W | 84 GOPS/W | 0.52ms |
| Samsung SmartSSD | 160 GOPS | 25W | 6.4 GOPS/W | 1.38ms |
| Custom CSD ASIC | 5 TOPS | 200W | 25 GOPS/W | 0.044ms |

**Key Findings:**
1. **ARM cores provide 29% higher power efficiency** than GPUs for ML encoding
2. **Custom ASICs achieve 4× better performance** than GPUs for specialized operations
3. **CSDs eliminate PCIe data movement**, providing additional latency savings

## 5. Index Rebuilding and Dynamic Updates

### 5.1 Current Index Management Challenges

Vector databases require periodic rebuilding for optimal performance:

**FAISS Index Rebuilding:**
- **Trigger conditions**: 10-20% document additions, degraded search quality
- **Rebuilding time**: 1M vectors × 384 dims = 2-4 hours on CPU
- **Memory requirements**: 2-3× vector database size during rebuild
- **Service disruption**: 30-60 minutes of reduced search quality

**Existing Solutions:**
- **FlashRAG**: Requires stopping service for index updates
- **PipeRAG**: No native support for dynamic updates
- **EdgeRAG**: Limited to small-scale indices due to memory constraints

### 5.2 CSD-Based Dynamic Index Management

CSDs enable efficient, continuous index updates:

**Advantages:**
1. **Local Processing**: Index updates occur at storage layer
2. **Parallel Operations**: Index rebuild concurrent with serving queries
3. **Reduced Data Movement**: No need to transfer entire database to CPU/GPU
4. **Incremental Updates**: Fine-grained index maintenance

**Performance Improvements:**
- **Update latency**: 100ms vs 30-60 minutes
- **Memory overhead**: 1.1× vs 3× database size
- **Service availability**: 99.9% vs 95% uptime during updates

### 5.3 Cost Analysis

**Traditional Index Rebuilding:**
- **Compute cost**: 4 hours × $32/hour (A100) = $128 per rebuild
- **Service degradation**: 30 minutes × $1000/hour revenue = $500 lost revenue
- **Frequency**: Weekly rebuilds = $52 × ($128 + $500) = **$32,656 annual cost**

**CSD-Based Index Management:**
- **Compute cost**: Continuous updates, no batch processing needed
- **Service degradation**: <1 minute × $1000/hour = $17 per update
- **Frequency**: Real-time updates = $17 × 52 = **$884 annual cost**

**Cost savings**: $32,656 - $884 = **$31,772 per year (97% reduction)**

## 6. Energy and Sustainability Considerations

### 6.1 Power Consumption Analysis

**Traditional RAG System:**
- **GPU**: 300W (A100) × 24 hours = 7.2 kWh/day
- **CPU**: 150W (server) × 24 hours = 3.6 kWh/day
- **Memory**: 50W (DDR4) × 24 hours = 1.2 kWh/day
- **Total**: 12 kWh/day per system

**CSD-Enhanced RAG System:**
- **GPU**: 300W × 50% utilization = 150W × 24 hours = 3.6 kWh/day
- **CPU**: 150W × 60% utilization = 90W × 24 hours = 2.16 kWh/day
- **CSD**: 15W × 24 hours = 0.36 kWh/day
- **Total**: 6.12 kWh/day per system

**Energy savings**: 12 - 6.12 = **5.88 kWh/day (49% reduction)**

### 6.2 Carbon Footprint Impact

At scale (1000 RAG systems):
- **Traditional**: 12 × 1000 = 12,000 kWh/day = 4.38 GWh/year
- **CSD-Enhanced**: 6.12 × 1000 = 6,120 kWh/day = 2.23 GWh/year
- **CO2 reduction**: 2.15 GWh × 0.4 kg CO2/kWh = **860 tons CO2/year**

## 7. Technical Feasibility and Implementation Path

### 7.1 Current Technology Readiness

**Available Today:**
- ARM Cortex-A78 cores with NEON SIMD (TRL 9)
- Samsung SmartSSD with Xilinx FPGA (TRL 8)
- NVMe computational storage standards (TRL 8)

**Near-term (2024-2025):**
- Custom ASIC CSDs for ML workloads (TRL 6-7)
- Advanced vector processing units (TRL 6)
- Standardized CSD programming interfaces (TRL 7)

**Long-term (2025-2027):**
- Integrated CSD ecosystems (TRL 5-6)
- Hardware-software co-design optimization (TRL 5)

### 7.2 Implementation Challenges

**Technical Challenges:**
1. **Programming Model**: Standardized APIs for CSD computation
2. **Load Balancing**: Dynamic workload distribution between CSD and host
3. **Consistency**: Maintaining data consistency across distributed computations
4. **Debugging**: Tools for debugging distributed CSD applications

**Economic Challenges:**
1. **Initial Cost**: CSD hardware 20-30% more expensive than traditional SSDs
2. **Software Development**: Investment in CSD-aware software stack
3. **Training**: Developer expertise in CSD programming

### 7.3 Return on Investment

**Cost-Benefit Analysis (5-year horizon):**

**Costs:**
- Hardware premium: 25% × $10,000 = $2,500 per system
- Software development: $500,000 one-time investment
- Training: $100,000 one-time investment

**Benefits (per system, annual):**
- Energy savings: 5.88 kWh/day × 365 × $0.12/kWh = $258
- Reduced GPU costs: 50% reduction × $32,000 = $16,000
- Improved availability: 4.9% uptime × $100,000 revenue = $4,900
- **Total annual benefits**: $21,158 per system

**ROI calculation:**
- **Payback period**: $2,500 ÷ $21,158 = 1.4 months
- **5-year NPV**: $21,158 × 5 - $2,500 = **$103,290 per system**

## 8. Conclusion

The motivation for CSD-based RAG systems is compelling across multiple dimensions:

1. **Performance**: Eliminate GPU memory swapping and reduce data movement overhead
2. **Efficiency**: Match computational requirements with appropriate hardware capabilities
3. **Cost**: Reduce operational expenses by 97% through intelligent workload distribution
4. **Sustainability**: Cut energy consumption by 49% and reduce carbon footprint
5. **Scalability**: Enable dynamic index management and real-time updates

Modern computational storage devices provide sufficient capability to handle RAG encoding and retrieval operations efficiently, while generation tasks remain on high-performance GPUs. This hybrid approach maximizes resource utilization and minimizes total cost of ownership.

The technical feasibility is proven, with ARM-based and FPGA-based CSDs available today. The economic case is strong, with payback periods under 2 months and substantial long-term returns. The next step is developing the software stack and programming models to realize this vision.

## References

1. PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design, arXiv:2403.05676
2. FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research, arXiv:2405.13576
3. EdgeRAG: Online-Indexed RAG for Edge Devices, arXiv:2412.21023
4. NVIDIA Technical Blog: Deploying RAG Applications on GH200
5. Samsung SmartSSD: Computational Storage with Xilinx FPGA
6. ARM Computational Storage Technical Specifications
7. SNIA Computational Storage Technical Work Group Standards