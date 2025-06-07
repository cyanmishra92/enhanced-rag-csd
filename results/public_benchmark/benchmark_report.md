# Comprehensive Public Benchmark Results

**Experiment ID**: `20250607_153442`  
**Date**: June 07, 2025 at 15:35:16  
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
- enhanced_rag_csd, piperag_like, flashrag_like, edgerag_like, vanillarag

## Performance Results

### Overall Performance Rankings

**Latency (Best to Worst):**  
1. enhanced_rag_csd, 2. flashrag_like, 3. piperag_like, 4. vanillarag, 5. edgerag_like

**Throughput (Best to Worst):**  
1. enhanced_rag_csd, 2. edgerag_like, 3. flashrag_like, 4. piperag_like, 5. vanillarag

**Relevance Score (Best to Worst):**  
1. enhanced_rag_csd, 2. piperag_like, 3. vanillarag, 4. flashrag_like, 5. edgerag_like

### Detailed Performance Metrics

| System | Avg Latency (ms) | Throughput (q/s) | Relevance Score | Cache Hit Rate | Error Rate |
|--------|-----------------|------------------|-----------------|----------------|------------|
| flashrag_like | 19.1 | 62.3 | 0.700 | 15.1% | 0.0% |
| piperag_like | 21.8 | 60.5 | 0.708 | 16.7% | 0.0% |
| edgerag_like | 27.6 | 101.2 | 0.695 | 16.0% | 0.0% |
| enhanced_rag_csd | 1.0 | 1999.5 | 0.842 | 55.5% | 0.0% |
| vanillarag | 23.8 | 52.1 | 0.704 | 16.3% | 0.0% |

## Benchmark-Specific Results

### nq_open

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.2ms)  
- Throughput: 2059.9 queries/second  
- Relevance: 0.853 (±0.065)  
- Cache Hit Rate: 58.0%  

**piperag_like:**  
- Latency: 14.1ms (±2.1ms)  
- Throughput: 70.3 queries/second  
- Relevance: 0.716 (±0.056)  
- Cache Hit Rate: 17.3%  

**flashrag_like:**  
- Latency: 13.3ms (±1.8ms)  
- Throughput: 74.2 queries/second  
- Relevance: 0.727 (±0.068)  
- Cache Hit Rate: 15.3%  

**edgerag_like:**  
- Latency: 11.4ms (±14.9ms)  
- Throughput: 88.7 queries/second  
- Relevance: 0.716 (±0.062)  
- Cache Hit Rate: 18.7%  

**vanillarag:**  
- Latency: 16.6ms (±4.2ms)  
- Throughput: 60.1 queries/second  
- Relevance: 0.724 (±0.065)  
- Cache Hit Rate: 14.0%  

### ms_marco

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.0ms)  
- Throughput: 2081.5 queries/second  
- Relevance: 0.871 (±0.065)  
- Cache Hit Rate: 52.5%  

**piperag_like:**  
- Latency: 12.6ms (±4.2ms)  
- Throughput: 65.4 queries/second  
- Relevance: 0.748 (±0.067)  
- Cache Hit Rate: 12.5%  

**flashrag_like:**  
- Latency: 13.4ms (±7.7ms)  
- Throughput: 59.1 queries/second  
- Relevance: 0.730 (±0.059)  
- Cache Hit Rate: 13.3%  

**edgerag_like:**  
- Latency: 9.3ms (±12.0ms)  
- Throughput: 88.7 queries/second  
- Relevance: 0.740 (±0.058)  
- Cache Hit Rate: 15.8%  

**vanillarag:**  
- Latency: 13.6ms (±3.3ms)  
- Throughput: 59.3 queries/second  
- Relevance: 0.735 (±0.060)  
- Cache Hit Rate: 14.2%  

### scifact

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.1ms)  
- Throughput: 1952.8 queries/second  
- Relevance: 0.823 (±0.058)  
- Cache Hit Rate: 60.0%  

**piperag_like:**  
- Latency: 18.4ms (±4.7ms)  
- Throughput: 69.5 queries/second  
- Relevance: 0.701 (±0.063)  
- Cache Hit Rate: 12.0%  

**flashrag_like:**  
- Latency: 18.9ms (±4.8ms)  
- Throughput: 69.2 queries/second  
- Relevance: 0.677 (±0.056)  
- Cache Hit Rate: 13.3%  

**edgerag_like:**  
- Latency: 32.3ms (±20.8ms)  
- Throughput: 202.2 queries/second  
- Relevance: 0.677 (±0.061)  
- Cache Hit Rate: 16.0%  

**vanillarag:**  
- Latency: 28.6ms (±11.0ms)  
- Throughput: 46.3 queries/second  
- Relevance: 0.687 (±0.051)  
- Cache Hit Rate: 18.7%  

### trec_covid

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.1ms)  
- Throughput: 1903.9 queries/second  
- Relevance: 0.822 (±0.055)  
- Cache Hit Rate: 51.7%  

**piperag_like:**  
- Latency: 42.2ms (±24.9ms)  
- Throughput: 36.9 queries/second  
- Relevance: 0.669 (±0.054)  
- Cache Hit Rate: 25.0%  

**flashrag_like:**  
- Latency: 30.7ms (±11.5ms)  
- Throughput: 46.9 queries/second  
- Relevance: 0.666 (±0.058)  
- Cache Hit Rate: 18.3%  

**edgerag_like:**  
- Latency: 57.6ms (±23.7ms)  
- Throughput: 25.3 queries/second  
- Relevance: 0.647 (±0.066)  
- Cache Hit Rate: 13.3%  

**vanillarag:**  
- Latency: 36.3ms (±19.2ms)  
- Throughput: 42.8 queries/second  
- Relevance: 0.668 (±0.062)  
- Cache Hit Rate: 18.3%  

## Statistical Analysis

- **Confidence Level**: 95.0%
- **Number of Runs**: 3
- **Best Overall System**: enhanced_rag_csd

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

