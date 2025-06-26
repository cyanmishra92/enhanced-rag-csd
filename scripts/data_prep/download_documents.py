#!/usr/bin/env python
"""
Download public documents for RAG experiments.
This script downloads research papers and articles from public sources.
"""

import os
import sys
import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from urllib.parse import urlparse
import hashlib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def download_file(url: str, output_path: str, max_retries: int = 3) -> bool:
    """Download a file from URL with retries."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {url}... (attempt {attempt + 1})")
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✓ Downloaded: {output_path} ({os.path.getsize(output_path)} bytes)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            
    return False

def download_arxiv_papers() -> List[Dict[str, Any]]:
    """Download ArXiv papers related to RAG, Vector Search, and AI."""
    papers = [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "url": "https://arxiv.org/pdf/2005.11401.pdf",
            "category": "RAG",
            "abstract": "Original RAG paper introducing retrieval-augmented generation"
        },
        {
            "title": "Dense Passage Retrieval for Open-Domain Question Answering", 
            "url": "https://arxiv.org/pdf/2004.04906.pdf",
            "category": "Retrieval",
            "abstract": "Dense retrieval methods for question answering systems"
        },
        {
            "title": "REALM: Retrieval-Augmented Language Model Pre-Training",
            "url": "https://arxiv.org/pdf/2002.08909.pdf", 
            "category": "Language Models",
            "abstract": "Pre-training language models with retrieval augmentation"
        },
        {
            "title": "FiD: Leveraging Passage Retrieval with Generative Models",
            "url": "https://arxiv.org/pdf/2007.01282.pdf",
            "category": "Generation",
            "abstract": "Fusion-in-Decoder approach for retrieval-augmented generation"
        }
    ]
    
    downloaded_papers = []
    
    for paper in papers:
        # Create safe filename
        safe_title = "".join(c for c in paper["title"] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]
        
        output_path = f"data/documents/papers/{safe_title}.pdf"
        
        if download_file(paper["url"], output_path):
            paper["local_path"] = output_path
            paper["size_bytes"] = os.path.getsize(output_path)
            downloaded_papers.append(paper)
    
    return downloaded_papers

def download_wikipedia_articles() -> List[Dict[str, Any]]:
    """Download Wikipedia articles as text files."""
    
    # Wikipedia API for getting article content
    def get_wikipedia_content(title: str) -> str:
        """Get Wikipedia article content via API."""
        api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        
        try:
            response = requests.get(api_url + title.replace(' ', '_'))
            response.raise_for_status()
            data = response.json()
            
            # Get full content
            content_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title.replace(' ', '_')}"
            content_response = requests.get(content_url)
            content_response.raise_for_status()
            content_data = content_response.json()
            
            # Extract text from sections
            text_content = data.get('extract', '')
            
            if 'sections' in content_data:
                for section in content_data['sections']:
                    if 'text' in section:
                        text_content += "\n\n" + section['text']
            
            return text_content
            
        except Exception as e:
            print(f"Failed to get Wikipedia content for {title}: {e}")
            return ""
    
    articles = [
        {
            "title": "Artificial Intelligence",
            "category": "AI",
            "description": "Overview of artificial intelligence field"
        },
        {
            "title": "Machine Learning", 
            "category": "ML",
            "description": "Machine learning concepts and methods"
        },
        {
            "title": "Natural Language Processing",
            "category": "NLP", 
            "description": "Natural language processing techniques"
        },
        {
            "title": "Information Retrieval",
            "category": "IR",
            "description": "Information retrieval systems and methods"
        },
        {
            "title": "Vector Space Model",
            "category": "Vector Search",
            "description": "Vector space model for information retrieval"
        },
        {
            "title": "Question Answering",
            "category": "QA",
            "description": "Automated question answering systems"
        }
    ]
    
    downloaded_articles = []
    
    for article in articles:
        print(f"Downloading Wikipedia article: {article['title']}")
        
        content = get_wikipedia_content(article['title'])
        
        if content:
            safe_title = article['title'].replace(' ', '_')
            output_path = f"data/documents/wikipedia/{safe_title}.txt"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article['title']}\n")
                f.write(f"Category: {article['category']}\n")
                f.write(f"Description: {article['description']}\n")
                f.write("="*80 + "\n\n")
                f.write(content)
            
            article["local_path"] = output_path
            article["size_bytes"] = os.path.getsize(output_path)
            article["content_preview"] = content[:200] + "..."
            downloaded_articles.append(article)
            
            print(f"✓ Saved: {output_path} ({article['size_bytes']} bytes)")
        else:
            print(f"✗ Failed to download: {article['title']}")
    
    return downloaded_articles

def download_public_texts() -> List[Dict[str, Any]]:
    """Download public domain texts from Project Gutenberg."""
    
    texts = [
        {
            "title": "The Time Machine by H.G. Wells",
            "url": "https://www.gutenberg.org/files/35/35-0.txt",
            "category": "Literature",
            "author": "H.G. Wells"
        },
        {
            "title": "Frankenstein by Mary Shelley", 
            "url": "https://www.gutenberg.org/files/84/84-0.txt",
            "category": "Literature", 
            "author": "Mary Shelley"
        },
        {
            "title": "The Art of War by Sun Tzu",
            "url": "https://www.gutenberg.org/files/132/132-0.txt", 
            "category": "Philosophy",
            "author": "Sun Tzu"
        }
    ]
    
    downloaded_texts = []
    
    for text in texts:
        safe_title = "".join(c for c in text["title"] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]
        
        output_path = f"data/documents/literature/{safe_title}.txt"
        
        if download_file(text["url"], output_path):
            text["local_path"] = output_path 
            text["size_bytes"] = os.path.getsize(output_path)
            downloaded_texts.append(text)
    
    return downloaded_texts

def main():
    """Main function to download all documents."""
    print("🔽 Downloading Public Documents for RAG Experiments")
    print("=" * 60)
    
    all_documents = []
    
    # Download ArXiv papers
    print("\n📄 Downloading ArXiv Research Papers...")
    papers = download_arxiv_papers()
    all_documents.extend(papers)
    
    # Download Wikipedia articles
    print("\n📖 Downloading Wikipedia Articles...")
    articles = download_wikipedia_articles()
    all_documents.extend(articles)
    
    # Download public domain texts
    print("\n📚 Downloading Public Domain Literature...")
    texts = download_public_texts()
    all_documents.extend(texts)
    
    # Save metadata
    metadata = {
        "total_documents": len(all_documents),
        "categories": list(set(doc.get("category", "Unknown") for doc in all_documents)),
        "total_size_bytes": sum(doc.get("size_bytes", 0) for doc in all_documents),
        "documents": all_documents
    }
    
    metadata_path = "data/documents/metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Download Complete!")
    print(f"   Total documents: {metadata['total_documents']}")
    print(f"   Total size: {metadata['total_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"   Categories: {', '.join(metadata['categories'])}")
    print(f"   Metadata saved: {metadata_path}")

if __name__ == "__main__":
    main()