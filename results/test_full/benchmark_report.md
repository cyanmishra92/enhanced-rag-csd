# Comprehensive Public Benchmark Results

**Experiment ID**: `20250607_150835`  
**Date**: June 07, 2025 at 15:08:39  
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
- enhanced_rag_csd

## Performance Results

### Overall Performance Rankings

**Latency (Best to Worst):**  
1. enhanced_rag_csd

**Throughput (Best to Worst):**  
1. enhanced_rag_csd

**Relevance Score (Best to Worst):**  
1. enhanced_rag_csd

### Detailed Performance Metrics

| System | Avg Latency (ms) | Throughput (q/s) | Relevance Score | Cache Hit Rate | Error Rate |
|--------|-----------------|------------------|-----------------|----------------|------------|
| enhanced_rag_csd | 1.0 | 1988.2 | 0.839 | 55.0% | 0.0% |

## Benchmark-Specific Results

### nq_open

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.0ms)  
- Throughput: 1843.3 queries/second  
- Relevance: 0.853 (±0.059)  
- Cache Hit Rate: 66.0%  

### ms_marco

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.0ms)  
- Throughput: 1903.8 queries/second  
- Relevance: 0.871 (±0.063)  
- Cache Hit Rate: 65.0%  

### scifact

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.0ms)  
- Throughput: 2163.0 queries/second  
- Relevance: 0.828 (±0.033)  
- Cache Hit Rate: 44.0%  

### trec_covid

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.0ms)  
- Throughput: 2042.6 queries/second  
- Relevance: 0.806 (±0.047)  
- Cache Hit Rate: 45.0%  

## Statistical Analysis

- **Confidence Level**: 95.0%
- **Number of Runs**: 1
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

