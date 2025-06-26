#!/usr/bin/env python
"""
Custom Document Experiment Framework for Enhanced RAG-CSD
This script allows users to run experiments with their own document collections
and custom queries, providing flexible evaluation capabilities.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Set matplotlib backend for headless systems
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline, PipelineConfig
from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class CustomDocumentExperiment:
    """Framework for custom document experiments."""
    
    def __init__(self, output_dir: str = "results/custom_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"custom_experiment_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = self.experiment_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Custom experiment initialized: {self.experiment_dir}")
    
    def load_documents_from_directory(self, documents_dir: str, 
                                    file_extensions: List[str] = ['.txt', '.md', '.pdf']) -> List[Dict[str, Any]]:
        """Load documents from a directory."""
        documents_path = Path(documents_dir)
        if not documents_path.exists():
            raise ValueError(f"Documents directory not found: {documents_dir}")
        
        documents = []
        for ext in file_extensions:
            for file_path in documents_path.rglob(f"*{ext}"):
                try:
                    if ext == '.pdf':
                        # For PDF files, you'd need PyPDF2 or similar
                        logger.warning(f"PDF processing not implemented for {file_path}")
                        continue
                    else:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    
                    documents.append({
                        "title": file_path.stem,
                        "content": content,
                        "source_path": str(file_path),
                        "file_type": ext,
                        "size_bytes": len(content.encode('utf-8'))
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {documents_dir}")
        return documents
    
    def load_documents_from_json(self, json_file: str) -> List[Dict[str, Any]]:
        """Load documents from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and 'documents' in data:
            documents = data['documents']
        else:
            raise ValueError("JSON file should contain a list of documents or a dict with 'documents' key")
        
        # Ensure required fields
        for doc in documents:
            if 'title' not in doc or 'content' not in doc:
                raise ValueError("Each document must have 'title' and 'content' fields")
        
        logger.info(f"Loaded {len(documents)} documents from {json_file}")
        return documents
    
    def load_queries_from_file(self, queries_file: str) -> List[Dict[str, Any]]:
        """Load queries from text or JSON file."""
        queries_path = Path(queries_file)
        
        if queries_path.suffix == '.json':
            with open(queries_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                queries = data
            elif isinstance(data, dict) and 'queries' in data:
                queries = data['queries']
            else:
                raise ValueError("JSON file should contain a list of queries or a dict with 'queries' key")
        
        elif queries_path.suffix in ['.txt', '.tsv']:
            queries = []
            with open(queries_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        if '\t' in line:  # TSV format with metadata
                            parts = line.split('\t')
                            query_text = parts[0]
                            metadata = {}
                            if len(parts) > 1:
                                metadata['category'] = parts[1]
                            if len(parts) > 2:
                                metadata['difficulty'] = parts[2]
                        else:
                            query_text = line
                            metadata = {}
                        
                        queries.append({
                            "id": f"query_{i+1}",
                            "query": query_text,
                            **metadata
                        })
        else:
            raise ValueError("Queries file must be .json, .txt, or .tsv")
        
        logger.info(f"Loaded {len(queries)} queries from {queries_file}")
        return queries
    
    def create_sample_queries(self, domain: str = "general") -> List[Dict[str, Any]]:
        """Create sample queries for testing."""
        if domain == "research":
            queries = [
                {"id": "q1", "query": "What are the main contributions of this research?", "category": "factual"},
                {"id": "q2", "query": "How does this approach compare to existing methods?", "category": "comparison"},
                {"id": "q3", "query": "What are the limitations of this study?", "category": "analysis"},
                {"id": "q4", "query": "What future work is suggested?", "category": "procedural"},
                {"id": "q5", "query": "What datasets were used for evaluation?", "category": "factual"}
            ]
        elif domain == "technical":
            queries = [
                {"id": "q1", "query": "How does this system work?", "category": "factual"},
                {"id": "q2", "query": "What are the performance characteristics?", "category": "analysis"},
                {"id": "q3", "query": "How can this be implemented?", "category": "procedural"},
                {"id": "q4", "query": "What are the system requirements?", "category": "factual"},
                {"id": "q5", "query": "What are common troubleshooting steps?", "category": "procedural"}
            ]
        else:  # general
            queries = [
                {"id": "q1", "query": "What is the main topic discussed?", "category": "factual"},
                {"id": "q2", "query": "What are the key points?", "category": "analysis"},
                {"id": "q3", "query": "How can this information be applied?", "category": "application"},
                {"id": "q4", "query": "What examples are provided?", "category": "factual"},
                {"id": "q5", "query": "What conclusions are drawn?", "category": "analysis"}
            ]
        
        logger.info(f"Created {len(queries)} sample queries for domain: {domain}")
        return queries
    
    def setup_rag_system(self, system_config: Dict[str, Any]) -> EnhancedRAGPipeline:
        """Set up the Enhanced RAG-CSD system."""
        config = PipelineConfig(
            vector_db_path=system_config.get("vector_db_path", "./data/custom_vector_db"),
            enable_csd_emulation=system_config.get("enable_csd_emulation", True),
            enable_pipeline_parallel=system_config.get("enable_pipeline_parallel", True),
            enable_caching=system_config.get("enable_caching", True),
            enable_system_data_flow=system_config.get("enable_system_data_flow", False),
            embedding_model=system_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        pipeline = EnhancedRAGPipeline(config)
        logger.info("Enhanced RAG-CSD system initialized")
        return pipeline
    
    def add_documents_to_system(self, pipeline: EnhancedRAGPipeline, 
                               documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to the RAG system."""
        logger.info(f"Adding {len(documents)} documents to the system...")
        
        start_time = time.time()
        
        # Prepare documents for the pipeline
        doc_texts = [doc["content"] for doc in documents]
        doc_metadata = [
            {
                "title": doc.get("title", f"Document {i}"),
                "source": doc.get("source_path", "unknown"),
                "file_type": doc.get("file_type", "unknown"),
                "doc_id": i
            }
            for i, doc in enumerate(documents)
        ]
        
        # Add documents
        pipeline.add_documents(doc_texts, metadata=doc_metadata)
        
        processing_time = time.time() - start_time
        
        # Get statistics
        stats = pipeline.get_statistics()
        
        result = {
            "processing_time_seconds": processing_time,
            "documents_added": len(documents),
            "total_chunks": stats.get("vector_store", {}).get("total_vectors", 0),
            "storage_path": stats.get("vector_store", {}).get("storage_path", "unknown")
        }
        
        logger.info(f"Documents added successfully in {processing_time:.2f}s")
        return result
    
    def run_query_experiment(self, pipeline: EnhancedRAGPipeline, 
                           queries: List[Dict[str, Any]], 
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """Run query experiment."""
        logger.info(f"Running experiment with {len(queries)} queries...")
        
        results = []
        
        for query_data in queries:
            query_text = query_data["query"]
            query_id = query_data.get("id", f"query_{len(results)}")
            
            try:
                start_time = time.time()
                result = pipeline.query(query_text, top_k=top_k, include_metadata=True)
                processing_time = time.time() - start_time
                
                # Process result
                experiment_result = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_metadata": {k: v for k, v in query_data.items() if k not in ["id", "query"]},
                    "processing_time_ms": processing_time * 1000,
                    "retrieved_documents": len(result.get("retrieved_docs", [])),
                    "top_k": top_k,
                    "augmented_query": result.get("augmented_query", ""),
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "strategy": result.get("strategy", {}),
                    "success": True,
                    "error": None
                }
                
                results.append(experiment_result)
                logger.info(f"Query {query_id}: {processing_time*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {e}")
                experiment_result = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_metadata": {k: v for k, v in query_data.items() if k not in ["id", "query"]},
                    "processing_time_ms": 0,
                    "retrieved_documents": 0,
                    "top_k": top_k,
                    "success": False,
                    "error": str(e)
                }
                results.append(experiment_result)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results."""
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if not successful_results:
            return {
                "success_rate": 0,
                "total_queries": len(results),
                "failed_queries": len(failed_results),
                "error": "No successful queries"
            }
        
        processing_times = [r["processing_time_ms"] for r in successful_results]
        retrieved_docs = [r["retrieved_documents"] for r in successful_results]
        
        # Calculate statistics
        import numpy as np
        
        analysis = {
            "success_rate": len(successful_results) / len(results),
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "failed_queries": len(failed_results),
            "performance_stats": {
                "avg_latency_ms": np.mean(processing_times),
                "min_latency_ms": np.min(processing_times),
                "max_latency_ms": np.max(processing_times),
                "p50_latency_ms": np.percentile(processing_times, 50),
                "p95_latency_ms": np.percentile(processing_times, 95),
                "std_latency_ms": np.std(processing_times)
            },
            "retrieval_stats": {
                "avg_docs_retrieved": np.mean(retrieved_docs),
                "min_docs_retrieved": np.min(retrieved_docs),
                "max_docs_retrieved": np.max(retrieved_docs)
            }
        }
        
        # Analyze by category if available
        categories = set()
        for r in successful_results:
            if "category" in r.get("query_metadata", {}):
                categories.add(r["query_metadata"]["category"])
        
        if categories:
            category_stats = {}
            for category in categories:
                cat_results = [r for r in successful_results 
                             if r.get("query_metadata", {}).get("category") == category]
                if cat_results:
                    cat_times = [r["processing_time_ms"] for r in cat_results]
                    category_stats[category] = {
                        "count": len(cat_results),
                        "avg_latency_ms": np.mean(cat_times),
                        "avg_docs_retrieved": np.mean([r["retrieved_documents"] for r in cat_results])
                    }
            
            analysis["category_breakdown"] = category_stats
        
        return analysis
    
    def save_results(self, documents_info: Dict[str, Any], 
                    queries_info: List[Dict[str, Any]],
                    query_results: List[Dict[str, Any]], 
                    analysis: Dict[str, Any],
                    system_config: Dict[str, Any]) -> None:
        """Save experiment results."""
        
        # Save comprehensive results
        experiment_data = {
            "experiment_info": {
                "timestamp": self.timestamp,
                "experiment_dir": str(self.experiment_dir)
            },
            "system_config": system_config,
            "documents_info": documents_info,
            "queries_info": {
                "total_queries": len(queries_info),
                "sample_queries": queries_info[:5]  # Save first 5 as sample
            },
            "query_results": query_results,
            "analysis": analysis
        }
        
        # Save to JSON (with proper serialization)
        results_file = self.results_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary report
        self._create_summary_report(analysis, system_config)
        
        logger.info(f"Results saved to {results_file}")
    
    def _create_summary_report(self, analysis: Dict[str, Any], system_config: Dict[str, Any]) -> None:
        """Create human-readable summary report."""
        report_file = self.experiment_dir / "EXPERIMENT_SUMMARY.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Custom Document Experiment Results\n\n")
            f.write(f"**Experiment ID:** {self.timestamp}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## System Configuration\n\n")
            for key, value in system_config.items():
                f.write(f"- **{key}:** {value}\n")
            
            f.write("\n## Performance Summary\n\n")
            f.write(f"- **Success Rate:** {analysis['success_rate']*100:.1f}%\n")
            f.write(f"- **Total Queries:** {analysis['total_queries']}\n")
            f.write(f"- **Successful Queries:** {analysis['successful_queries']}\n")
            f.write(f"- **Failed Queries:** {analysis['failed_queries']}\n\n")
            
            if 'performance_stats' in analysis:
                stats = analysis['performance_stats']
                f.write("### Latency Statistics\n\n")
                f.write(f"- **Average Latency:** {stats['avg_latency_ms']:.1f}ms\n")
                f.write(f"- **Median Latency (P50):** {stats['p50_latency_ms']:.1f}ms\n")
                f.write(f"- **95th Percentile (P95):** {stats['p95_latency_ms']:.1f}ms\n")
                f.write(f"- **Standard Deviation:** {stats['std_latency_ms']:.1f}ms\n")
                f.write(f"- **Min Latency:** {stats['min_latency_ms']:.1f}ms\n")
                f.write(f"- **Max Latency:** {stats['max_latency_ms']:.1f}ms\n\n")
            
            if 'retrieval_stats' in analysis:
                stats = analysis['retrieval_stats']
                f.write("### Retrieval Statistics\n\n")
                f.write(f"- **Average Documents Retrieved:** {stats['avg_docs_retrieved']:.1f}\n")
                f.write(f"- **Min Documents Retrieved:** {stats['min_docs_retrieved']}\n")
                f.write(f"- **Max Documents Retrieved:** {stats['max_docs_retrieved']}\n\n")
            
            if 'category_breakdown' in analysis:
                f.write("### Performance by Query Category\n\n")
                f.write("| Category | Count | Avg Latency (ms) | Avg Docs Retrieved |\n")
                f.write("|----------|-------|------------------|--------------------|\n")
                for category, stats in analysis['category_breakdown'].items():
                    f.write(f"| {category} | {stats['count']} | "
                           f"{stats['avg_latency_ms']:.1f} | "
                           f"{stats['avg_docs_retrieved']:.1f} |\n")
            
            f.write("\n## Files Generated\n\n")
            f.write(f"- **Detailed Results:** `{self.results_dir}/experiment_results.json`\n")
            f.write(f"- **Summary Report:** `{report_file.name}`\n")
            f.write(f"- **Logs:** `{self.logs_dir}/`\n")
            
            f.write("\n---\n")
            f.write("*Generated by Enhanced RAG-CSD Custom Document Experiment Framework*\n")
        
        logger.info(f"Summary report saved to {report_file}")
    
    def run_experiment(self, documents_source: str, queries_source: str, 
                      system_config: Dict[str, Any], top_k: int = 5) -> None:
        """Run complete custom experiment."""
        logger.info("🚀 Starting custom document experiment...")
        
        # Load documents
        if documents_source.endswith('.json'):
            documents = self.load_documents_from_json(documents_source)
        else:
            documents = self.load_documents_from_directory(documents_source)
        
        # Load queries
        if queries_source == "sample":
            queries = self.create_sample_queries(system_config.get("domain", "general"))
        else:
            queries = self.load_queries_from_file(queries_source)
        
        # Setup system
        pipeline = self.setup_rag_system(system_config)
        
        # Add documents
        documents_info = self.add_documents_to_system(pipeline, documents)
        
        # Run queries
        query_results = self.run_query_experiment(pipeline, queries, top_k)
        
        # Analyze results
        analysis = self.analyze_results(query_results)
        
        # Save results
        self.save_results(documents_info, queries, query_results, analysis, system_config)
        
        logger.info("✅ Custom experiment completed successfully!")
        logger.info(f"📊 Results saved to: {self.experiment_dir}")


def main():
    """Main function for custom document experiments."""
    parser = argparse.ArgumentParser(description="Enhanced RAG-CSD Custom Document Experiment")
    parser.add_argument("--documents", type=str, required=True,
                       help="Path to documents directory or JSON file")
    parser.add_argument("--queries", type=str, default="sample",
                       help="Path to queries file (.txt, .tsv, .json) or 'sample' for generated queries")
    parser.add_argument("--output-dir", type=str, default="results/custom_experiments",
                       help="Output directory for results")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of documents to retrieve per query")
    parser.add_argument("--enable-system-data-flow", action="store_true",
                       help="Enable system data flow mode")
    parser.add_argument("--embedding-model", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model to use")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Document chunk size")
    parser.add_argument("--domain", type=str, default="general",
                       choices=["general", "research", "technical"],
                       help="Domain for sample queries")
    
    args = parser.parse_args()
    
    # System configuration
    system_config = {
        "vector_db_path": f"./data/custom_vector_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "enable_csd_emulation": True,
        "enable_pipeline_parallel": True,
        "enable_caching": True,
        "enable_system_data_flow": args.enable_system_data_flow,
        "embedding_model": args.embedding_model,
        "domain": args.domain
    }
    
    # Create experiment instance
    experiment = CustomDocumentExperiment(args.output_dir)
    
    # Run experiment
    experiment.run_experiment(
        documents_source=args.documents,
        queries_source=args.queries,
        system_config=system_config,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()