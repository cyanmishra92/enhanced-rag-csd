"""Setup configuration for Enhanced RAG-CSD."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    "numpy>=1.19.0",
    "faiss-cpu>=1.7.0",  # Use faiss-gpu for GPU support
    "sentence-transformers>=2.2.0",
    "torch>=1.9.0",
    "scikit-learn>=0.24.0",
    "scipy>=1.6.0",
    "pandas>=1.2.0",
    "pyyaml>=5.4.0",
    "tqdm>=4.60.0",
    "psutil>=5.8.0",
    "nltk>=3.6.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
]

# Development dependencies
dev_requires = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "pytest-asyncio>=0.15.0",
    "black>=21.5b0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "ipython>=7.25.0",
    "jupyter>=1.0.0",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
]

# Optional dependencies
extras_require = {
    "dev": dev_requires,
    "gpu": ["faiss-gpu>=1.7.0"],
    "viz": ["plotly>=5.0.0", "dash>=2.0.0"],
    "all": dev_requires + ["faiss-gpu>=1.7.0", "plotly>=5.0.0", "dash>=2.0.0"],
}

setup(
    name="enhanced-rag-csd",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-performance RAG system with CSD emulation and incremental indexing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-rag-csd",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/enhanced-rag-csd/issues",
        "Documentation": "https://enhanced-rag-csd.readthedocs.io",
        "Source Code": "https://github.com/yourusername/enhanced-rag-csd",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "enhanced-rag-csd=enhanced_rag_csd.cli:main",
            "erag-benchmark=enhanced_rag_csd.benchmarks.cli:benchmark_main",
            "erag-demo=enhanced_rag_csd.examples.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "enhanced_rag_csd": ["data/*.json", "data/*.yaml"],
    },
    zip_safe=False,
)