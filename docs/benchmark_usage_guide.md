# Enhanced RAG-CSD Benchmark Usage Guide

This guide explains how to use the comprehensive benchmark suite for evaluating Enhanced RAG-CSD against public benchmark datasets.

## Quick Start

### 1. Quick 5-Minute Benchmark
```bash
python scripts/run_and_plot_benchmark.py --quick
```

### 2. Standard 10-Minute Benchmark
```bash
python scripts/run_and_plot_benchmark.py --standard
```

### 3. Full Research-Grade Benchmark
```bash
python scripts/run_and_plot_benchmark.py --full
```

## Available Benchmark Configurations

### Quick (3-5 minutes)
- **Systems**: Enhanced-RAG-CSD, RAG-CSD, VanillaRAG
- **Runs**: 2 per system
- **Use case**: Development and quick validation

### Standard (8-12 minutes)
- **Systems**: Enhanced-RAG-CSD, RAG-CSD, PipeRAG-like, EdgeRAG-like, VanillaRAG
- **Runs**: 3 per system
- **Use case**: Regular performance evaluation

### Full (15-25 minutes)
- **Systems**: All 6 baseline systems
- **Runs**: 5 per system
- **Use case**: Comprehensive analysis

### Research (30-45 minutes)
- **Systems**: All 6 baseline systems
- **Runs**: 10 per system
- **Use case**: Publication-quality results with high statistical rigor

## Public Benchmark Datasets

The system evaluates performance across multiple datasets:

### 1. Natural Questions (NQ-Open)
- **Description**: Real Google search queries with Wikipedia answers
- **Domain**: Open domain question answering
- **Questions**: 50 sample questions
- **Difficulty**: Easy to Hard

### 2. MS MARCO Passages
- **Description**: Bing search query logs with passage ranking
- **Domain**: Web search queries
- **Questions**: 40 sample questions
- **Focus**: Factual retrieval and how-to queries

### 3. SciFact (BEIR subset)
- **Description**: Scientific fact verification
- **Domain**: Scientific literature
- **Questions**: 25 sample questions
- **Focus**: Medical research, climate science

### 4. TREC-COVID (BEIR subset)
- **Description**: COVID-19 research paper retrieval
- **Domain**: Medical research
- **Questions**: 20 sample questions
- **Focus**: Pandemic research, public health

## Command Line Options

### List Available Configurations
```bash
python scripts/run_and_plot_benchmark.py --list
```

### Custom System Selection
```bash
python scripts/run_and_plot_benchmark.py --standard --systems Enhanced-RAG-CSD RAG-CSD VanillaRAG
```

### Custom Number of Runs
```bash
python scripts/run_and_plot_benchmark.py --quick --num-runs 5
```

### Generate Plots Only
```bash
python scripts/run_and_plot_benchmark.py --plot-only results/benchmark_standard_20250606_180000/
```

### Specific Plot Types
```bash
python scripts/run_and_plot_benchmark.py --plot-only results/dir/ --plot-types latency throughput accuracy
```

## Output Files

Each benchmark run generates:

### 1. Results Directory Structure
```
results/benchmark_[config]_[timestamp]/
├── comprehensive_results.json      # Complete raw data
├── benchmark_report.md            # Detailed analysis report
├── SUMMARY.md                     # Quick summary
└── plots/                         # Publication-quality figures
    ├── latency_comparison.pdf
    ├── throughput_analysis.pdf
    ├── accuracy_metrics.pdf
    ├── cache_performance.pdf
    ├── system_overview.pdf
    └── benchmark_[dataset]_comparison.pdf (per dataset)
```

### 2. Generated Visualizations
- **Latency Comparison**: Average and P95 latency across systems
- **Throughput Analysis**: Queries per second and memory usage
- **Accuracy Metrics**: Relevance scores and error rates
- **Cache Performance**: Hit rates and cache hierarchy effectiveness
- **System Overview**: Comprehensive radar chart comparison
- **Per-Benchmark Plots**: Dataset-specific performance analysis

### 3. Performance Metrics Collected
- **Latency**: Average, median, P95, P99
- **Throughput**: Queries per second
- **Relevance**: Average relevance scores (0-1)
- **Cache Performance**: Hit rates, miss rates
- **Error Rates**: Failed query percentages
- **Memory Usage**: System memory consumption

## Advanced Usage

### Run Direct Benchmark Script
```bash
python scripts/comprehensive_public_benchmark.py \
    --output-dir results/custom_benchmark \
    --num-runs 5 \
    --systems Enhanced-RAG-CSD RAG-CSD VanillaRAG
```

### Custom Document Corpus
1. Place documents in `data/comprehensive_corpus/`
2. Update metadata in `data/comprehensive_corpus/metadata.json`
3. Run benchmark normally

### Integration with CI/CD
```bash
# Quick validation in CI pipeline
python scripts/run_and_plot_benchmark.py --quick --systems Enhanced-RAG-CSD VanillaRAG
```

## Performance Expectations

### Enhanced RAG-CSD Typical Results
- **Latency**: 20-50ms average
- **Throughput**: 35-45 queries/second
- **Relevance**: 0.85-0.90 average score
- **Cache Hit Rate**: 55-65%
- **Memory Usage**: 400-600 MB

### Baseline Systems Comparison
- **VanillaRAG**: 100-130ms latency, 8-10 q/s
- **EdgeRAG-like**: 90-110ms latency, 9-12 q/s, lowest memory
- **PipeRAG-like**: 80-100ms latency, 10-13 q/s
- **FlashRAG-like**: 65-85ms latency, 12-16 q/s

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure src directory is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python scripts/run_and_plot_benchmark.py --quick
```

#### Memory Issues
```bash
# Run with fewer systems for limited memory
python scripts/run_and_plot_benchmark.py --quick --systems Enhanced-RAG-CSD VanillaRAG
```

#### Permission Errors
```bash
# Ensure write permissions for results directory
mkdir -p results
chmod 755 results
```

### Debugging Failed Runs
1. Check log files in the results directory
2. Verify all dependencies are installed
3. Run with `--quick` first to test setup
4. Check available disk space for results

## Research Applications

### Publication Preparation
1. Run `--research` configuration for maximum statistical rigor
2. Use generated PDF plots directly in papers
3. Cite benchmark datasets appropriately
4. Include confidence intervals and error bars

### System Development
1. Use `--quick` for rapid iteration
2. Monitor performance regressions
3. Validate optimizations with `--standard`
4. Compare against specific baselines

### Production Deployment
1. Run `--full` benchmark on target hardware
2. Evaluate memory and latency requirements
3. Test with domain-specific documents
4. Monitor cache performance characteristics

---

## Example Complete Workflow

```bash
# 1. Quick validation
python scripts/run_and_plot_benchmark.py --quick

# 2. Review results
cat results/benchmark_quick_*/SUMMARY.md

# 3. Full analysis
python scripts/run_and_plot_benchmark.py --standard

# 4. Generate additional plots
python scripts/run_and_plot_benchmark.py --plot-only results/benchmark_standard_* --plot-types latency accuracy

# 5. Research-grade validation
python scripts/run_and_plot_benchmark.py --research --systems Enhanced-RAG-CSD VanillaRAG
```

This comprehensive benchmark suite provides everything needed for rigorous evaluation of RAG systems against established public datasets with publication-quality outputs.