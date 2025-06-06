# Setup Guide for Enhanced RAG-CSD

This guide provides detailed setup instructions for Enhanced RAG-CSD on various platforms.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Environment Setup](#environment-setup)
4. [Dependency Installation](#dependency-installation)
5. [GPU Setup (Optional)](#gpu-setup-optional)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **OS**: Linux (Ubuntu 18.04+), macOS 10.14+, Windows 10+
- **Python**: 3.8 or higher

### Recommended Requirements
- **CPU**: 8+ cores, 3.0 GHz
- **RAM**: 16 GB or more
- **Storage**: 50 GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional)
- **OS**: Ubuntu 20.04 LTS or newer

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
# Create virtual environment
python -m venv enhanced_rag_env
source enhanced_rag_env/bin/activate  # On Windows: enhanced_rag_env\Scripts\activate

# Install package
pip install enhanced-rag-csd

# Install with GPU support
pip install enhanced-rag-csd[gpu]

# Install with all extras
pip install enhanced-rag-csd[all]
```

### Method 2: Install from Source

```bash
# Clone repository
git clone https://github.com/yourusername/enhanced-rag-csd.git
cd enhanced-rag-csd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Method 3: Using Conda

```bash
# Create conda environment
conda create -n enhanced_rag python=3.9
conda activate enhanced_rag

# Install PyTorch (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install FAISS
conda install -c conda-forge faiss-cpu

# Install the package
pip install enhanced-rag-csd
```

## Environment Setup

### Linux/macOS

```bash
# Set environment variables
export ENHANCED_RAG_HOME=$HOME/.enhanced_rag_csd
export ENHANCED_RAG_CACHE=$ENHANCED_RAG_HOME/cache
export ENHANCED_RAG_DATA=$ENHANCED_RAG_HOME/data

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export ENHANCED_RAG_HOME=$HOME/.enhanced_rag_csd' >> ~/.bashrc
echo 'export ENHANCED_RAG_CACHE=$ENHANCED_RAG_HOME/cache' >> ~/.bashrc
echo 'export ENHANCED_RAG_DATA=$ENHANCED_RAG_HOME/data' >> ~/.bashrc

# Create directories
mkdir -p $ENHANCED_RAG_HOME/{cache,data,logs,models}
```

### Windows

```powershell
# Set environment variables
[Environment]::SetEnvironmentVariable("ENHANCED_RAG_HOME", "$env:USERPROFILE\.enhanced_rag_csd", "User")
[Environment]::SetEnvironmentVariable("ENHANCED_RAG_CACHE", "$env:USERPROFILE\.enhanced_rag_csd\cache", "User")
[Environment]::SetEnvironmentVariable("ENHANCED_RAG_DATA", "$env:USERPROFILE\.enhanced_rag_csd\data", "User")

# Create directories
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.enhanced_rag_csd\cache"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.enhanced_rag_csd\data"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.enhanced_rag_csd\logs"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.enhanced_rag_csd\models"
```

## Dependency Installation

### Core Dependencies

```bash
# Install core dependencies individually if needed
pip install numpy>=1.19.0
pip install faiss-cpu>=1.7.0
pip install sentence-transformers>=2.2.0
pip install torch>=1.9.0
pip install scikit-learn>=0.24.0
```

### NLTK Data

```python
# Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Matplotlib Backend for Headless Systems

```bash
# For systems without display
export MPLBACKEND=Agg

# Add to ~/.bashrc for persistence
echo 'export MPLBACKEND=Agg' >> ~/.bashrc
```

## GPU Setup (Optional)

### NVIDIA GPU Setup

1. **Install CUDA Toolkit**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install appropriate CUDA toolkit (example for CUDA 11.7)
   wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
   sudo sh cuda_11.7.0_515.43.04_linux.run
   ```

2. **Install PyTorch with CUDA**
   ```bash
   # For CUDA 11.7
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

3. **Install FAISS GPU**
   ```bash
   # Uninstall CPU version first
   pip uninstall faiss-cpu
   
   # Install GPU version
   pip install faiss-gpu
   ```

4. **Verify GPU Setup**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0)}")
   
   import faiss
   print(f"FAISS GPU support: {faiss.get_num_gpus()}")
   ```

## Verification

### 1. Basic Import Test

```python
# Test basic imports
import enhanced_rag_csd
print(f"Enhanced RAG-CSD version: {enhanced_rag_csd.__version__}")

from enhanced_rag_csd import EnhancedRAGPipeline, PipelineConfig
print("✓ Core imports successful")
```

### 2. Run Test Script

```bash
# Run verification script
python -m enhanced_rag_csd.verify

# Expected output:
# ✓ Core modules imported successfully
# ✓ FAISS index created successfully
# ✓ Sentence transformer loaded successfully
# ✓ Pipeline initialized successfully
# ✓ All checks passed!
```

### 3. Run Unit Tests

```bash
# Run basic tests
pytest tests/unit/test_imports.py -v

# Run all tests
pytest -v
```

### 4. Run Demo

```bash
# Run standalone performance demo (2 seconds)
python scripts/standalone_demo.py

# Expected output:
# 🎯 Key Demo Results:
#    🚀 4.6x faster query processing
#    ⚡ 4.7x higher throughput
#    🧠 60.0% memory reduction
#    🎯 86.7% relevance accuracy
#    💾 60.0% cache hit rate

# Run quick benchmark validation (3-5 minutes)
python scripts/run_and_plot_benchmark.py --quick

# Run interactive demo
python examples/demo.py --test

# Expected output:
# ✓ Pipeline initialized
# ✓ Documents added: 5
# ✓ Query processed successfully
# ✓ Demo completed!
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'faiss'**
   ```bash
   # Install FAISS
   conda install -c conda-forge faiss-cpu
   # or
   pip install faiss-cpu
   ```

2. **CUDA out of memory**
   ```python
   # Reduce batch size in config
   config = PipelineConfig(
       max_batch_size=16,  # Reduce from default
       embedding_batch_size=32  # Reduce from default
   )
   ```

3. **Matplotlib errors on headless system**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Use non-interactive backend
   import matplotlib.pyplot as plt
   ```

4. **Permission denied errors**
   ```bash
   # Fix permissions
   chmod -R 755 ~/.enhanced_rag_csd
   ```

5. **Memory errors with large datasets**
   ```python
   # Use memory-efficient configuration
   config = PipelineConfig(
       l1_cache_size_mb=32,    # Reduce cache sizes
       l2_cache_size_mb=256,
       l3_cache_size_mb=1024,
       enable_mmap=True        # Use memory-mapped files
   )
   ```

### Platform-Specific Issues

#### macOS
- If using M1/M2 Macs, install ARM64 versions of dependencies
- Use `faiss-cpu` as GPU version is not available

#### Windows
- Use Anaconda for easier dependency management
- Install Visual C++ Build Tools if compilation errors occur

#### WSL2
- Ensure sufficient memory allocation in `.wslconfig`
- GPU passthrough requires Windows 11 or Windows 10 21H2+

### Getting Help

1. Check the [FAQ](docs/faq.md)
2. Search [existing issues](https://github.com/yourusername/enhanced-rag-csd/issues)
3. Join our [Discord community](https://discord.gg/enhanced-rag-csd)
4. Create a [new issue](https://github.com/yourusername/enhanced-rag-csd/issues/new)

## Next Steps

- Read the [Getting Started Guide](docs/getting_started.md)
- Try the [examples](examples/README.md)
- Check the [API documentation](docs/api.md)
- Run [benchmarks](docs/benchmarking.md)