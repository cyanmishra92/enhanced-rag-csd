# Comprehensive Public Benchmark Results

**Experiment ID**: `20250607_183624`  
**Date**: June 07, 2025 at 18:36:58  
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
1. enhanced_rag_csd, 2. flashrag_like, 3. edgerag_like, 4. piperag_like, 5. vanillarag

**Relevance Score (Best to Worst):**  
1. enhanced_rag_csd, 2. vanillarag, 3. piperag_like, 4. edgerag_like, 5. flashrag_like

### Detailed Performance Metrics

| System | Avg Latency (ms) | Throughput (q/s) | Relevance Score | Cache Hit Rate | Error Rate |
|--------|-----------------|------------------|-----------------|----------------|------------|
| enhanced_rag_csd | 1.0 | 1887.9 | 0.838 | 59.7% | 0.0% |
| edgerag_like | 26.3 | 57.3 | 0.700 | 14.1% | 0.0% |
| piperag_like | 20.8 | 56.3 | 0.703 | 14.0% | 0.0% |
| flashrag_like | 19.2 | 59.7 | 0.695 | 15.2% | 0.0% |
| vanillarag | 24.0 | 48.5 | 0.705 | 14.9% | 0.0% |

## Benchmark-Specific Results

### nq_open

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.1ms)  
- Throughput: 1965.7 queries/second  
- Relevance: 0.852 (±0.056)  
- Cache Hit Rate: 57.3%  

**piperag_like:**  
- Latency: 17.7ms (±4.3ms)  
- Throughput: 56.2 queries/second  
- Relevance: 0.719 (±0.064)  
- Cache Hit Rate: 9.3%  

**flashrag_like:**  
- Latency: 17.3ms (±3.8ms)  
- Throughput: 57.9 queries/second  
- Relevance: 0.713 (±0.054)  
- Cache Hit Rate: 14.7%  

**edgerag_like:**  
- Latency: 12.7ms (±17.4ms)  
- Throughput: 78.4 queries/second  
- Relevance: 0.718 (±0.063)  
- Cache Hit Rate: 15.3%  

**vanillarag:**  
- Latency: 20.1ms (±5.4ms)  
- Throughput: 50.0 queries/second  
- Relevance: 0.720 (±0.059)  
- Cache Hit Rate: 13.3%  

### ms_marco

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.0ms)  
- Throughput: 1955.1 queries/second  
- Relevance: 0.860 (±0.059)  
- Cache Hit Rate: 60.0%  

**piperag_like:**  
- Latency: 13.4ms (±3.0ms)  
- Throughput: 59.4 queries/second  
- Relevance: 0.741 (±0.063)  
- Cache Hit Rate: 13.3%  

**flashrag_like:**  
- Latency: 12.6ms (±2.7ms)  
- Throughput: 63.2 queries/second  
- Relevance: 0.740 (±0.061)  
- Cache Hit Rate: 13.3%  

**edgerag_like:**  
- Latency: 9.9ms (±12.9ms)  
- Throughput: 84.2 queries/second  
- Relevance: 0.734 (±0.059)  
- Cache Hit Rate: 11.7%  

**vanillarag:**  
- Latency: 15.8ms (±4.0ms)  
- Throughput: 50.8 queries/second  
- Relevance: 0.748 (±0.056)  
- Cache Hit Rate: 17.5%  

### scifact

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.1ms)  
- Throughput: 1885.7 queries/second  
- Relevance: 0.836 (±0.058)  
- Cache Hit Rate: 61.3%  

**piperag_like:**  
- Latency: 25.1ms (±13.4ms)  
- Throughput: 52.9 queries/second  
- Relevance: 0.690 (±0.059)  
- Cache Hit Rate: 20.0%  

**flashrag_like:**  
- Latency: 20.7ms (±4.6ms)  
- Throughput: 61.8 queries/second  
- Relevance: 0.672 (±0.056)  
- Cache Hit Rate: 14.7%  

**edgerag_like:**  
- Latency: 31.9ms (±20.2ms)  
- Throughput: 38.4 queries/second  
- Relevance: 0.689 (±0.066)  
- Cache Hit Rate: 16.0%  

**vanillarag:**  
- Latency: 25.7ms (±6.4ms)  
- Throughput: 51.8 queries/second  
- Relevance: 0.689 (±0.060)  
- Cache Hit Rate: 12.0%  

### trec_covid

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.2ms)  
- Throughput: 1745.0 queries/second  
- Relevance: 0.804 (±0.061)  
- Cache Hit Rate: 60.0%  

**piperag_like:**  
- Latency: 26.8ms (±9.1ms)  
- Throughput: 56.5 queries/second  
- Relevance: 0.662 (±0.061)  
- Cache Hit Rate: 13.3%  

**flashrag_like:**  
- Latency: 26.3ms (±8.1ms)  
- Throughput: 56.2 queries/second  
- Relevance: 0.654 (±0.064)  
- Cache Hit Rate: 18.3%  

**edgerag_like:**  
- Latency: 50.8ms (±15.3ms)  
- Throughput: 28.2 queries/second  
- Relevance: 0.658 (±0.058)  
- Cache Hit Rate: 13.3%  

**vanillarag:**  
- Latency: 34.3ms (±23.9ms)  
- Throughput: 41.5 queries/second  
- Relevance: 0.662 (±0.057)  
- Cache Hit Rate: 16.7%  

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

