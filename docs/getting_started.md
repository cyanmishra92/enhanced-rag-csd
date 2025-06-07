# Getting Started with Enhanced RAG-CSD

This guide will help you get up and running with Enhanced RAG-CSD in minutes.

## Quick Start in 5 Minutes

### 1. Installation

```bash
# Clone and install from source (recommended)
git clone https://github.com/yourusername/enhanced-rag-csd.git
cd enhanced-rag-csd
pip install -e .

# OR install from PyPI (when available)
pip install enhanced-rag-csd
```

### 2. Your First RAG Pipeline

```python
from enhanced_rag_csd import EnhancedRAGPipeline, PipelineConfig

# Create a simple pipeline
pipeline = EnhancedRAGPipeline(PipelineConfig(
    vector_db_path="./my_first_rag",
    enable_csd_emulation=True
))

# Add some documents
documents = [
    "The Earth is the third planet from the Sun.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science."
]
pipeline.add_documents(documents)

# Ask a question
result = pipeline.query("What is Earth's position in the solar system?")
print(result['augmented_query'])
```

## Detailed Walkthrough

### Step 1: Understanding the Components

Enhanced RAG-CSD consists of three main components:

1. **Document Ingestion**: Processes and stores your documents
2. **Retrieval**: Finds relevant documents for queries
3. **Augmentation**: Combines query with retrieved context

### Step 2: Setting Up Your First Project

Create a project structure:

```bash
mkdir my_rag_project
cd my_rag_project
mkdir data results

# Create a configuration file
cat > config.yaml << EOF
pipeline:
  vector_db_path: "./vectors"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  
csd:
  enable_emulation: true
  max_parallel_ops: 4
  
cache:
  enable: true
  l1_size_mb: 64
EOF
```

### Step 3: Loading Documents

```python
from enhanced_rag_csd import EnhancedRAGPipeline, PipelineConfig
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Initialize pipeline
config = PipelineConfig(**config_dict['pipeline'])
pipeline = EnhancedRAGPipeline(config)

# Load documents from files
import os
documents = []
metadata = []

for filename in os.listdir('data'):
    if filename.endswith('.txt'):
        with open(os.path.join('data', filename), 'r') as f:
            content = f.read()
            documents.append(content)
            metadata.append({'source': filename})

# Add documents with metadata
result = pipeline.add_documents(documents, metadata)
print(f"Processed {result['documents_processed']} documents")
print(f"Created {result['chunks_created']} chunks")
```

### Step 4: Querying Your Knowledge Base

```python
# Single query
query = "What is machine learning?"
result = pipeline.query(query, top_k=3)

print(f"Query: {query}")
print(f"Retrieved {len(result['retrieved_docs'])} documents")
print(f"\nAugmented Response:")
print(result['augmented_query'])

# Batch queries for better performance
queries = [
    "What is artificial intelligence?",
    "How does Python relate to data science?",
    "Explain the solar system"
]

batch_results = pipeline.query_batch(queries)
for query, result in zip(queries, batch_results):
    print(f"\nQ: {query}")
    print(f"A: {result['augmented_query'][:200]}...")
```

### Step 5: Monitoring Performance

```python
# Get performance statistics
stats = pipeline.get_statistics()

print("\nPerformance Metrics:")
print(f"Total vectors: {stats['vector_store']['total_vectors']}")
print(f"Cache hit rate: {stats['metrics'].get('cache_hit_rate', 0):.2%}")
print(f"Average latency: {stats['metrics'].get('avg_latency', 0):.3f}s")

# If CSD emulation is enabled
if 'csd_metrics' in stats:
    print(f"\nCSD Emulator Metrics:")
    print(f"Cache levels: L1={stats['csd_metrics']['l1_size']}MB, "
          f"L2={stats['csd_metrics']['l2_size']}MB")
    print(f"I/O operations: {stats['csd_metrics']['io_ops']}")
```

## Common Use Cases

### 1. Document Q&A System

```python
# Load technical documentation
docs = load_documentation('path/to/docs')
pipeline.add_documents(docs)

# Create Q&A interface
def answer_question(question):
    result = pipeline.query(question, top_k=5)
    return result['augmented_query']

# Use in application
answer = answer_question("How do I configure the system?")
```

### 2. Semantic Search Engine

```python
# Configure for search
search_config = PipelineConfig(
    vector_db_path="./search_index",
    enable_csd_emulation=True,
    retrieval_only=True  # Skip augmentation
)

search_engine = EnhancedRAGPipeline(search_config)

# Index documents
search_engine.add_documents(corpus)

# Search function
def search(query, num_results=10):
    results = search_engine.query(query, top_k=num_results)
    return results['retrieved_docs']
```

### 3. Incremental Knowledge Base

```python
# Enable incremental indexing
config = PipelineConfig(
    vector_db_path="./knowledge_base",
    delta_threshold=1000,  # Create delta index every 1000 docs
    max_delta_indices=5    # Merge after 5 delta indices
)

kb = EnhancedRAGPipeline(config)

# Continuously add documents
for batch in document_stream:
    kb.add_documents(batch)
    
    # Check if reindexing is needed
    stats = kb.get_statistics()
    if stats['vector_store']['drift_detected']:
        print("Index drift detected, optimizing...")
```

## Best Practices

### 1. Document Preparation

- **Chunk Size**: Use 512 tokens for balanced context
- **Overlap**: 10-20% overlap prevents context loss
- **Metadata**: Always include source information

```python
result = pipeline.add_documents(
    documents,
    metadata=[{'source': f, 'date': d} for f, d in zip(files, dates)],
    chunk_size=512,
    chunk_overlap=50
)
```

### 2. Query Optimization

- **Top-K Selection**: Start with 5, increase if needed
- **Caching**: Enable for repeated queries
- **Batch Processing**: Group similar queries

```python
# Optimal configuration for production
config = PipelineConfig(
    enable_caching=True,
    enable_pipeline_parallel=True,
    max_batch_size=32
)
```

### 3. Performance Tuning

```python
# For latency-sensitive applications
low_latency_config = PipelineConfig(
    enable_csd_emulation=True,
    max_parallel_ops=8,
    l1_cache_size_mb=128,  # Larger L1 cache
    prefetch_candidates=True
)

# For high-throughput applications  
high_throughput_config = PipelineConfig(
    enable_csd_emulation=True,
    enable_pipeline_parallel=True,
    max_batch_size=64,
    async_processing=True
)
```

## Troubleshooting Common Issues

### 1. Slow First Query

**Issue**: First query takes much longer than subsequent ones.

**Solution**: Implement model warm-up:

```python
# Warm up the pipeline
pipeline.query("test query", top_k=1)
```

### 2. Out of Memory

**Issue**: Memory errors with large document sets.

**Solution**: Use smaller cache sizes and chunking:

```python
config = PipelineConfig(
    l1_cache_size_mb=32,
    l2_cache_size_mb=256,
    chunk_size=256  # Smaller chunks
)
```

### 3. Poor Retrieval Quality

**Issue**: Retrieved documents aren't relevant.

**Solution**: Tune retrieval parameters:

```python
# Increase top_k for better coverage
results = pipeline.query(query, top_k=10)

# Use different similarity metrics
config = PipelineConfig(
    similarity_metric="cosine",  # or "dot_product", "euclidean"
)
```

## Next Steps

1. **Explore Advanced Features**
   - [Incremental Indexing Guide](incremental_indexing.md)
   - [CSD Emulation Details](csd_emulation.md)
   - [Performance Optimization](optimization.md)

2. **Run Benchmarks**
   ```bash
   python -m enhanced_rag_csd.benchmark --help
   ```

3. **Build Applications**
   - Check [examples/](../examples/) for complete applications
   - Read [API Reference](api.md) for detailed documentation

4. **Join the Community**
   - [GitHub Discussions](https://github.com/yourusername/enhanced-rag-csd/discussions)
   - [Discord Server](https://discord.gg/enhanced-rag-csd)

## Example Applications

Find complete example applications in the `examples/` directory:

- `question_answering.py` - Interactive Q&A system
- `semantic_search.py` - Document search engine
- `knowledge_base.py` - Incremental knowledge management
- `api_server.py` - REST API server
- `streamlit_app.py` - Web interface with Streamlit

Happy building with Enhanced RAG-CSD! 🚀