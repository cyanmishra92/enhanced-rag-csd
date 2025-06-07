# Comprehensive Public Benchmark Results

**Experiment ID**: `20250607_135609`  
**Date**: June 07, 2025 at 13:56:13  
**Duration**: Complete benchmark suite  
**Status**: ✅ Successfully Completed  

## Benchmark Overview

### Datasets Tested

**Natural Questions Open**  
- Description: Open domain QA benchmark with real Google search queries  
- Questions: 3610  
- Size: ~50MB  

**MS MARCO Passages**  
- Description: Large-scale passage ranking dataset from Bing queries  
- Questions: 6980  
- Size: ~2GB  

**SciFact (BEIR subset)**  
- Description: Scientific fact verification dataset  
- Questions: 300  
- Size: ~10MB  

**TREC-COVID (BEIR subset)**  
- Description: COVID-19 research paper retrieval  
- Questions: 50  
- Size: ~5MB  

### Systems Evaluated
- enhanced_rag_csd, rag_csd, piperag_like, flashrag_like, edgerag_like, vanillarag

## Performance Results

### Overall Performance Rankings

**Latency (Best to Worst):**  
1. flashrag_like, 2. rag_csd, 3. piperag_like, 4. edgerag_like, 5. enhanced_rag_csd, 6. vanillarag

**Throughput (Best to Worst):**  
1. edgerag_like, 2. vanillarag, 3. flashrag_like, 4. piperag_like, 5. rag_csd, 6. enhanced_rag_csd

**Relevance Score (Best to Worst):**  
1. rag_csd, 2. piperag_like, 3. flashrag_like, 4. edgerag_like, 5. vanillarag, 6. enhanced_rag_csd

### Detailed Performance Metrics

| System | Avg Latency (ms) | Throughput (q/s) | Relevance Score | Cache Hit Rate | Error Rate |
|--------|-----------------|------------------|-----------------|----------------|------------|
| rag_csd | 86.7 | 292540.6 | 0.776 | 23.2% | 0.0% |
| flashrag_like | 81.6 | 338222.6 | 0.725 | 22.0% | 0.0% |
| enhanced_rag_csd | 120.0 | 1871.7 | 0.000 | 0.0% | 100.0% |
| vanillarag | 129.3 | 356158.9 | 0.709 | 4.0% | 0.0% |
| piperag_like | 102.0 | 326639.4 | 0.750 | 16.3% | 0.0% |
| edgerag_like | 115.5 | 363878.5 | 0.723 | 31.2% | 0.0% |

## Benchmark-Specific Results

### nq_open

**enhanced_rag_csd:**  
- Latency: 120.0ms (±0.0ms)  
- Throughput: 558.8 queries/second  
- Relevance: 0.000 (±0.000)  
- Cache Hit Rate: 0.0%  

**rag_csd:**  
- Latency: 73.8ms (±8.2ms)  
- Throughput: 203937.0 queries/second  
- Relevance: 0.797 (±0.076)  
- Cache Hit Rate: 26.0%  

**piperag_like:**  
- Latency: 88.3ms (±9.4ms)  
- Throughput: 354248.6 queries/second  
- Relevance: 0.774 (±0.078)  
- Cache Hit Rate: 13.3%  

**flashrag_like:**  
- Latency: 68.7ms (±7.0ms)  
- Throughput: 355449.5 queries/second  
- Relevance: 0.741 (±0.074)  
- Cache Hit Rate: 17.3%  

**edgerag_like:**  
- Latency: 98.7ms (±10.0ms)  
- Throughput: 333410.5 queries/second  
- Relevance: 0.741 (±0.078)  
- Cache Hit Rate: 33.3%  

**vanillarag:**  
- Latency: 110.8ms (±11.5ms)  
- Throughput: 386928.4 queries/second  
- Relevance: 0.726 (±0.079)  
- Cache Hit Rate: 3.3%  

### ms_marco

**enhanced_rag_csd:**  
- Latency: 120.0ms (±0.0ms)  
- Throughput: 2324.3 queries/second  
- Relevance: 0.000 (±0.000)  
- Cache Hit Rate: 0.0%  

**rag_csd:**  
- Latency: 60.3ms (±8.2ms)  
- Throughput: 344501.4 queries/second  
- Relevance: 0.823 (±0.075)  
- Cache Hit Rate: 23.3%  

**piperag_like:**  
- Latency: 69.8ms (±9.4ms)  
- Throughput: 281970.0 queries/second  
- Relevance: 0.796 (±0.079)  
- Cache Hit Rate: 13.3%  

**flashrag_like:**  
- Latency: 56.2ms (±8.6ms)  
- Throughput: 351232.7 queries/second  
- Relevance: 0.778 (±0.089)  
- Cache Hit Rate: 17.5%  

**edgerag_like:**  
- Latency: 77.8ms (±11.3ms)  
- Throughput: 386571.8 queries/second  
- Relevance: 0.774 (±0.081)  
- Cache Hit Rate: 31.7%  

**vanillarag:**  
- Latency: 88.2ms (±12.5ms)  
- Throughput: 368460.1 queries/second  
- Relevance: 0.755 (±0.079)  
- Cache Hit Rate: 4.2%  

### scifact

**enhanced_rag_csd:**  
- Latency: 120.0ms (±0.0ms)  
- Throughput: 2246.0 queries/second  
- Relevance: 0.000 (±0.000)  
- Cache Hit Rate: 0.0%  

**rag_csd:**  
- Latency: 100.3ms (±19.1ms)  
- Throughput: 313319.5 queries/second  
- Relevance: 0.761 (±0.078)  
- Cache Hit Rate: 20.0%  

**piperag_like:**  
- Latency: 115.5ms (±21.0ms)  
- Throughput: 340078.7 queries/second  
- Relevance: 0.727 (±0.087)  
- Cache Hit Rate: 18.7%  

**flashrag_like:**  
- Latency: 89.3ms (±17.1ms)  
- Throughput: 351871.1 queries/second  
- Relevance: 0.702 (±0.079)  
- Cache Hit Rate: 33.3%  

**edgerag_like:**  
- Latency: 128.7ms (±24.5ms)  
- Throughput: 369216.9 queries/second  
- Relevance: 0.710 (±0.069)  
- Cache Hit Rate: 33.3%  

**vanillarag:**  
- Latency: 148.2ms (±29.5ms)  
- Throughput: 377638.4 queries/second  
- Relevance: 0.712 (±0.085)  
- Cache Hit Rate: 5.3%  

### trec_covid

**enhanced_rag_csd:**  
- Latency: 120.0ms (±0.0ms)  
- Throughput: 2357.7 queries/second  
- Relevance: 0.000 (±0.000)  
- Cache Hit Rate: 0.0%  

**rag_csd:**  
- Latency: 112.4ms (±30.8ms)  
- Throughput: 308404.7 queries/second  
- Relevance: 0.723 (±0.074)  
- Cache Hit Rate: 23.3%  

**piperag_like:**  
- Latency: 134.3ms (±33.3ms)  
- Throughput: 330260.2 queries/second  
- Relevance: 0.705 (±0.090)  
- Cache Hit Rate: 20.0%  

**flashrag_like:**  
- Latency: 112.2ms (±28.8ms)  
- Throughput: 294337.1 queries/second  
- Relevance: 0.679 (±0.077)  
- Cache Hit Rate: 20.0%  

**edgerag_like:**  
- Latency: 157.0ms (±43.7ms)  
- Throughput: 366314.8 queries/second  
- Relevance: 0.669 (±0.077)  
- Cache Hit Rate: 26.7%  

**vanillarag:**  
- Latency: 170.1ms (±41.7ms)  
- Throughput: 291608.6 queries/second  
- Relevance: 0.645 (±0.078)  
- Cache Hit Rate: 3.3%  

## Statistical Analysis

- **Confidence Level**: 95.0%
- **Number of Runs**: 3
- **Best Overall System**: flashrag_like

## Research Impact

This comprehensive benchmark demonstrates the performance characteristics of different RAG systems across multiple public datasets, providing insights for:

- **System Selection**: Choose optimal RAG architecture for specific use cases
- **Performance Optimization**: Identify bottlenecks and optimization opportunities  
- **Research Validation**: Validate improvements against established benchmarks
- **Production Deployment**: Understand real-world performance expectations

## Files Generated

- Complete results: `comprehensive_results.json`
- Visualization plots: `plots/` directory
- Raw data: Individual benchmark result files

