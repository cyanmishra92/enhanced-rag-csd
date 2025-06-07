# Comprehensive Public Benchmark Results

**Experiment ID**: `20250607_144240`  
**Date**: June 07, 2025 at 14:42:46  
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
1. enhanced_rag_csd, 2. flashrag_like, 3. rag_csd, 4. piperag_like, 5. edgerag_like, 6. vanillarag

**Throughput (Best to Worst):**  
1. flashrag_like, 2. edgerag_like, 3. rag_csd, 4. vanillarag, 5. piperag_like, 6. enhanced_rag_csd

**Relevance Score (Best to Worst):**  
1. enhanced_rag_csd, 2. rag_csd, 3. piperag_like, 4. flashrag_like, 5. edgerag_like, 6. vanillarag

### Detailed Performance Metrics

| System | Avg Latency (ms) | Throughput (q/s) | Relevance Score | Cache Hit Rate | Error Rate |
|--------|-----------------|------------------|-----------------|----------------|------------|
| rag_csd | 86.4 | 235804.5 | 0.769 | 22.0% | 0.0% |
| enhanced_rag_csd | 1.1 | 1465.3 | 0.851 | 63.4% | 0.0% |
| flashrag_like | 80.6 | 300944.3 | 0.732 | 22.2% | 0.0% |
| piperag_like | 101.6 | 227764.6 | 0.746 | 16.3% | 0.0% |
| vanillarag | 123.8 | 231940.0 | 0.704 | 3.8% | 0.0% |
| edgerag_like | 113.3 | 248950.8 | 0.723 | 31.2% | 0.0% |

## Benchmark-Specific Results

### nq_open

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.1ms)  
- Throughput: 1739.3 queries/second  
- Relevance: 0.867 (±0.052)  
- Cache Hit Rate: 55.3%  

**rag_csd:**  
- Latency: 74.8ms (±6.8ms)  
- Throughput: 277768.5 queries/second  
- Relevance: 0.796 (±0.083)  
- Cache Hit Rate: 20.0%  

**piperag_like:**  
- Latency: 87.9ms (±8.9ms)  
- Throughput: 303641.7 queries/second  
- Relevance: 0.761 (±0.078)  
- Cache Hit Rate: 14.0%  

**flashrag_like:**  
- Latency: 68.0ms (±7.2ms)  
- Throughput: 319363.2 queries/second  
- Relevance: 0.759 (±0.080)  
- Cache Hit Rate: 18.7%  

**edgerag_like:**  
- Latency: 98.3ms (±9.3ms)  
- Throughput: 338250.3 queries/second  
- Relevance: 0.750 (±0.083)  
- Cache Hit Rate: 34.0%  

**vanillarag:**  
- Latency: 110.8ms (±8.8ms)  
- Throughput: 209018.5 queries/second  
- Relevance: 0.722 (±0.077)  
- Cache Hit Rate: 2.7%  

### ms_marco

**enhanced_rag_csd:**  
- Latency: 1.0ms (±0.3ms)  
- Throughput: 1243.8 queries/second  
- Relevance: 0.885 (±0.051)  
- Cache Hit Rate: 64.2%  

**rag_csd:**  
- Latency: 59.3ms (±8.4ms)  
- Throughput: 94095.4 queries/second  
- Relevance: 0.820 (±0.072)  
- Cache Hit Rate: 22.5%  

**piperag_like:**  
- Latency: 69.6ms (±11.0ms)  
- Throughput: 187804.7 queries/second  
- Relevance: 0.788 (±0.074)  
- Cache Hit Rate: 11.7%  

**flashrag_like:**  
- Latency: 55.8ms (±8.9ms)  
- Throughput: 309733.2 queries/second  
- Relevance: 0.787 (±0.078)  
- Cache Hit Rate: 22.5%  

**edgerag_like:**  
- Latency: 79.9ms (±11.4ms)  
- Throughput: 271329.6 queries/second  
- Relevance: 0.778 (±0.080)  
- Cache Hit Rate: 30.0%  

**vanillarag:**  
- Latency: 88.0ms (±12.7ms)  
- Throughput: 322019.5 queries/second  
- Relevance: 0.761 (±0.079)  
- Cache Hit Rate: 4.2%  

### scifact

**enhanced_rag_csd:**  
- Latency: 1.1ms (±0.3ms)  
- Throughput: 1345.3 queries/second  
- Relevance: 0.844 (±0.054)  
- Cache Hit Rate: 64.0%  

**rag_csd:**  
- Latency: 98.3ms (±22.7ms)  
- Throughput: 292353.9 queries/second  
- Relevance: 0.745 (±0.088)  
- Cache Hit Rate: 25.3%  

**piperag_like:**  
- Latency: 112.9ms (±20.8ms)  
- Throughput: 226474.3 queries/second  
- Relevance: 0.737 (±0.092)  
- Cache Hit Rate: 16.0%  

**flashrag_like:**  
- Latency: 91.2ms (±17.2ms)  
- Throughput: 298739.6 queries/second  
- Relevance: 0.700 (±0.084)  
- Cache Hit Rate: 22.7%  

**edgerag_like:**  
- Latency: 128.1ms (±26.2ms)  
- Throughput: 105667.7 queries/second  
- Relevance: 0.697 (±0.080)  
- Cache Hit Rate: 30.7%  

**vanillarag:**  
- Latency: 139.6ms (±27.9ms)  
- Throughput: 178836.2 queries/second  
- Relevance: 0.682 (±0.082)  
- Cache Hit Rate: 6.7%  

### trec_covid

**enhanced_rag_csd:**  
- Latency: 1.1ms (±0.2ms)  
- Throughput: 1532.8 queries/second  
- Relevance: 0.808 (±0.050)  
- Cache Hit Rate: 70.0%  

**rag_csd:**  
- Latency: 113.2ms (±30.2ms)  
- Throughput: 279000.3 queries/second  
- Relevance: 0.716 (±0.093)  
- Cache Hit Rate: 20.0%  

**piperag_like:**  
- Latency: 135.9ms (±30.8ms)  
- Throughput: 193137.6 queries/second  
- Relevance: 0.698 (±0.089)  
- Cache Hit Rate: 23.3%  

**flashrag_like:**  
- Latency: 107.2ms (±29.8ms)  
- Throughput: 275941.1 queries/second  
- Relevance: 0.680 (±0.076)  
- Cache Hit Rate: 25.0%  

**edgerag_like:**  
- Latency: 146.8ms (±36.1ms)  
- Throughput: 280555.5 queries/second  
- Relevance: 0.669 (±0.069)  
- Cache Hit Rate: 30.0%  

**vanillarag:**  
- Latency: 156.9ms (±39.6ms)  
- Throughput: 217885.9 queries/second  
- Relevance: 0.650 (±0.068)  
- Cache Hit Rate: 1.7%  

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

