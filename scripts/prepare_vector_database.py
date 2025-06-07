#!/usr/bin/env python
"""
Prepare vector database files for baseline systems.
This script creates the embeddings.npy, chunks.json, and metadata.json files
that baseline systems expect to find in the corpus directory.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def create_vector_database(corpus_path: str, output_path: str = None) -> str:
    """Create vector database files from text corpus."""
    corpus_path = Path(corpus_path)
    if output_path is None:
        output_path = corpus_path
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating vector database from {corpus_path}")
    logger.info(f"Output directory: {output_path}")
    
    # Load sentence transformer model
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Process all text files in the corpus
    chunks = []
    metadata = []
    
    txt_files = list(corpus_path.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files")
    
    for i, txt_file in enumerate(txt_files):
        logger.info(f"Processing {txt_file.name}...")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title and content
        lines = content.split('\n')
        title = lines[0].replace('Title: ', '') if lines[0].startswith('Title: ') else txt_file.stem
        
        # Find content after metadata
        content_start = 0
        for j, line in enumerate(lines):
            if line.strip() == '' and j > 0:
                content_start = j + 1
                break
        
        text_content = '\n'.join(lines[content_start:])
        
        # Create chunks
        text_chunks = chunk_text(text_content, chunk_size=256, overlap=32)
        
        for chunk_idx, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadata.append({
                'id': len(chunks) - 1,
                'document_id': i,
                'document_title': title,
                'document_file': txt_file.name,
                'chunk_index': chunk_idx,
                'chunk_length': len(chunk)
            })
    
    logger.info(f"Created {len(chunks)} chunks from {len(txt_files)} documents")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    
    # Save files
    logger.info("Saving vector database files...")
    
    # Save embeddings
    embeddings_path = output_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings: {embeddings_path}")
    
    # Save chunks
    chunks_path = output_path / "chunks.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved chunks: {chunks_path}")
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata: {metadata_path}")
    
    # Print summary
    logger.info(f"✅ Vector database created successfully!")
    logger.info(f"   Documents: {len(txt_files)}")
    logger.info(f"   Chunks: {len(chunks)}")
    logger.info(f"   Embeddings shape: {embeddings.shape}")
    logger.info(f"   Output directory: {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create vector database for baseline systems")
    parser.add_argument("--corpus-path", required=True, help="Path to text corpus directory")
    parser.add_argument("--output-path", help="Output directory (default: same as corpus)")
    
    args = parser.parse_args()
    
    create_vector_database(args.corpus_path, args.output_path)