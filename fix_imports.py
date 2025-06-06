#!/usr/bin/env python
"""Fix imports in migrated files."""

import os
import re
from pathlib import Path

# Define import replacements
import_replacements = {
    r'from rag_csd\.': 'from enhanced_rag_csd.',
    r'import rag_csd\.': 'import enhanced_rag_csd.',
    r'from \.\.embedding\.encoder': 'from enhanced_rag_csd.core.encoder',
    r'from \.\.retrieval\.incremental_index': 'from enhanced_rag_csd.retrieval.incremental_index',
    r'from \.\.augmentation\.augmentor': 'from enhanced_rag_csd.core.augmentor',
    r'from \.\.csd\.enhanced_simulator': 'from enhanced_rag_csd.core.csd_emulator',
    r'from \.\.utils\.': 'from enhanced_rag_csd.utils.',
    r'from \.\.evaluation\.': 'from enhanced_rag_csd.evaluation.',
    r'from \.\.benchmarks\.': 'from enhanced_rag_csd.benchmarks.',
    r'from rag_csd\.enhanced_pipeline': 'from enhanced_rag_csd.core.pipeline',
    r'from \.enhanced_pipeline': 'from .pipeline',
    r'EnhancedCSDSimulator': 'EnhancedCSDSimulator',
    r'from \.\.baseline_systems': 'from enhanced_rag_csd.benchmarks.baseline_systems',
}

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Apply replacements
    for pattern, replacement in import_replacements.items():
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✓ Fixed imports in {file_path}")
    else:
        print(f"  No changes needed in {file_path}")

def fix_all_imports():
    """Fix imports in all Python files."""
    print("Fixing imports...")
    
    # Find all Python files
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                fix_imports_in_file(file_path)
    
    # Also fix example files
    for file in ["examples/demo.py", "examples/benchmark.py"]:
        if os.path.exists(file):
            fix_imports_in_file(file)
    
    print("\nImport fixing complete!")

if __name__ == "__main__":
    fix_all_imports()