#!/usr/bin/env python3
"""
Comprehensive test suite for new Enhanced RAG-CSD system-level features.
Tests both traditional CSD emulation and new system data flow modes.
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, './src')

from enhanced_rag_csd.core.pipeline import EnhancedRAGPipeline, PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results/logs/test_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_test_data():
    """Setup test documents and queries."""
    documents = [
        "Artificial intelligence is transforming healthcare through advanced diagnostic tools and personalized treatment plans.",
        "Machine learning algorithms are being used to analyze medical images and detect diseases early.",
        "Deep learning models can process large datasets to identify patterns in patient health records.",
        "Natural language processing helps doctors analyze patient symptoms and medical literature.",
        "Computer vision technology assists radiologists in identifying anomalies in X-rays and MRI scans.",
        "Predictive analytics in healthcare can forecast patient outcomes and optimize treatment protocols.",
        "Robotic surgery systems provide precision and minimize invasive procedures for patients.",
        "Telemedicine platforms leverage AI to provide remote consultations and monitoring.",
        "Clinical decision support systems integrate AI to assist doctors in making informed diagnoses.",
        "Electronic health records with AI capabilities improve patient data management and analysis."
    ]
    
    queries = [
        "How is AI being used in medical diagnostics?",
        "What are the applications of machine learning in healthcare?",
        "How does deep learning help with patient data analysis?",
        "What role does NLP play in medical practice?",
        "How is computer vision used in radiology?"
    ]
    
    return documents, queries

def test_traditional_csd_mode():
    """Test traditional CSD emulation mode."""
    logger.info("=" * 60)
    logger.info("TESTING TRADITIONAL CSD MODE")
    logger.info("=" * 60)
    
    config = {
        'vector_db_path': './test_results/traditional_db',
        'storage_path': './test_results/traditional_storage',
        'enable_csd_emulation': True,
        'enable_system_data_flow': False,
        'enable_pipeline_parallel': True,
        'enable_caching': True
    }
    
    pipeline = EnhancedRAGPipeline(config)
    documents, queries = setup_test_data()
    
    results = {}
    
    # Test document addition
    logger.info("Testing document addition...")
    start_time = time.time()
    add_result = pipeline.add_documents(documents)
    add_time = time.time() - start_time
    
    logger.info(f"Added {add_result['documents_processed']} documents in {add_time:.3f}s")
    results['document_addition'] = {
        'time': add_time,
        'documents': add_result['documents_processed'],
        'chunks': add_result['chunks_created']
    }
    
    # Test queries
    logger.info("Testing query processing...")
    query_results = []
    
    for i, query in enumerate(queries):
        start_time = time.time()
        result = pipeline.query(query, top_k=3)
        query_time = time.time() - start_time
        
        query_results.append({
            'query': query,
            'processing_time': query_time,
            'result': result
        })
        
        logger.info(f"Query {i+1}: {query_time:.3f}s")
    
    results['queries'] = query_results
    
    # Get pipeline statistics
    stats = pipeline.get_statistics()
    results['statistics'] = stats
    
    logger.info("Traditional CSD mode statistics:")
    logger.info(f"- CSD emulation: {stats['config']['csd_emulation']}")
    logger.info(f"- Pipeline parallel: {stats['config']['pipeline_parallel']}")
    logger.info(f"- Caching: {stats['config']['caching']}")
    
    if 'csd_metrics' in stats:
        csd_metrics = stats['csd_metrics']
        logger.info(f"- Cache hit rate: {csd_metrics.get('cache_hit_rate', 0):.2%}")
        logger.info(f"- Storage usage: {csd_metrics.get('storage_usage_mb', 0):.1f}MB")
    
    pipeline.shutdown()
    
    return results

def test_system_dataflow_mode():
    """Test new system data flow mode."""
    logger.info("=" * 60)
    logger.info("TESTING SYSTEM DATA FLOW MODE")
    logger.info("=" * 60)
    
    config = {
        'vector_db_path': './test_results/systemdf_db',
        'storage_path': './test_results/systemdf_storage',
        'enable_csd_emulation': True,
        'enable_system_data_flow': True,  # NEW FEATURE
        'enable_pipeline_parallel': True,
        'enable_caching': True
    }
    
    pipeline = EnhancedRAGPipeline(config)
    documents, queries = setup_test_data()
    
    results = {}
    
    # Test document addition
    logger.info("Testing document addition with system data flow...")
    start_time = time.time()
    add_result = pipeline.add_documents(documents)
    add_time = time.time() - start_time
    
    logger.info(f"Added {add_result['documents_processed']} documents in {add_time:.3f}s")
    results['document_addition'] = {
        'time': add_time,
        'documents': add_result['documents_processed'],
        'chunks': add_result['chunks_created']
    }
    
    # Test queries with system data flow
    logger.info("Testing query processing with system data flow...")
    query_results = []
    
    for i, query in enumerate(queries):
        start_time = time.time()
        result = pipeline.query(query, top_k=3)
        query_time = time.time() - start_time
        
        query_results.append({
            'query': query,
            'processing_time': query_time,
            'result': result,
            'data_flow_path': result.get('data_flow_path', 'Unknown')
        })
        
        logger.info(f"Query {i+1}: {query_time:.3f}s, Path: {result.get('data_flow_path', 'N/A')}")
    
    results['queries'] = query_results
    
    # Get comprehensive statistics
    stats = pipeline.get_statistics()
    results['statistics'] = stats
    
    logger.info("System Data Flow mode statistics:")
    logger.info(f"- CSD emulation: {stats['config']['csd_emulation']}")
    logger.info(f"- System data flow: {stats['config']['system_data_flow']}")
    logger.info(f"- Pipeline parallel: {stats['config']['pipeline_parallel']}")
    
    if 'system_data_flow_metrics' in stats:
        sdf_metrics = stats['system_data_flow_metrics']
        df_metrics = sdf_metrics.get('data_flow_metrics', {})
        
        logger.info(f"- Avg latency: {df_metrics.get('avg_latency_ms', 0):.2f}ms")
        logger.info(f"- P2P transfers: {df_metrics.get('p2p_transfers', 0)}")
        logger.info(f"- PCIe transfers: {df_metrics.get('pcie_transfers', 0)}")
        logger.info(f"- Total data transferred: {df_metrics.get('total_data_transferred_mb', 0):.2f}MB")
    
    if 'csd_metrics' in stats:
        csd_metrics = stats['csd_metrics']
        logger.info(f"- CSD cache hit rate: {csd_metrics.get('cache_hit_rate', 0):.2%}")
        
        if 'system_memory' in csd_metrics:
            mem_stats = csd_metrics['system_memory']
            logger.info(f"- System memory utilization: {mem_stats.get('system_summary', {}).get('system_utilization_percent', 0):.1f}%")
    
    pipeline.shutdown()
    
    return results

def test_batch_processing():
    """Test batch processing capabilities."""
    logger.info("=" * 60)
    logger.info("TESTING BATCH PROCESSING")
    logger.info("=" * 60)
    
    # Test with system data flow
    config = {
        'vector_db_path': './test_results/batch_db',
        'storage_path': './test_results/batch_storage',
        'enable_csd_emulation': True,
        'enable_system_data_flow': True,
        'enable_pipeline_parallel': True
    }
    
    pipeline = EnhancedRAGPipeline(config)
    documents, queries = setup_test_data()
    
    # Add documents first
    pipeline.add_documents(documents)
    
    # Test batch query processing
    logger.info("Testing batch query processing...")
    start_time = time.time()
    batch_results = pipeline.query_batch(queries, top_k=3)
    batch_time = time.time() - start_time
    
    logger.info(f"Processed {len(queries)} queries in batch: {batch_time:.3f}s")
    logger.info(f"Average time per query: {batch_time/len(queries):.3f}s")
    
    # Compare with individual processing
    logger.info("Comparing with individual query processing...")
    individual_times = []
    for query in queries:
        start_time = time.time()
        pipeline.query(query, top_k=3)
        individual_times.append(time.time() - start_time)
    
    individual_total = sum(individual_times)
    logger.info(f"Individual processing total: {individual_total:.3f}s")
    logger.info(f"Batch speedup: {individual_total/batch_time:.2f}x")
    
    pipeline.shutdown()
    
    return {
        'batch_time': batch_time,
        'individual_total': individual_total,
        'speedup': individual_total/batch_time,
        'batch_results': batch_results
    }

def generate_performance_plots(traditional_results, systemdf_results, batch_results):
    """Generate performance comparison plots."""
    logger.info("Generating performance plots...")
    
    plt.style.use('seaborn-v0_8')
    
    # Query processing time comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Query processing times
    trad_times = [q['processing_time'] for q in traditional_results['queries']]
    sdf_times = [q['processing_time'] for q in systemdf_results['queries']]
    
    x = np.arange(len(trad_times))
    width = 0.35
    
    ax1.bar(x - width/2, [t*1000 for t in trad_times], width, label='Traditional CSD', alpha=0.8)
    ax1.bar(x + width/2, [t*1000 for t in sdf_times], width, label='System Data Flow', alpha=0.8)
    ax1.set_xlabel('Query Number')
    ax1.set_ylabel('Processing Time (ms)')
    ax1.set_title('Query Processing Time Comparison')
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Q{i+1}' for i in range(len(trad_times))])
    
    # 2. Average processing times
    trad_avg = np.mean(trad_times) * 1000
    sdf_avg = np.mean(sdf_times) * 1000
    
    ax2.bar(['Traditional CSD', 'System Data Flow'], [trad_avg, sdf_avg], alpha=0.8, color=['blue', 'orange'])
    ax2.set_ylabel('Average Processing Time (ms)')
    ax2.set_title('Average Query Processing Time')
    
    # Add value labels on bars
    for i, v in enumerate([trad_avg, sdf_avg]):
        ax2.text(i, v + max(trad_avg, sdf_avg)*0.01, f'{v:.2f}ms', ha='center')
    
    # 3. System resource utilization (if available)
    if 'system_data_flow_metrics' in systemdf_results['statistics']:
        sdf_metrics = systemdf_results['statistics']['system_data_flow_metrics']
        df_metrics = sdf_metrics.get('data_flow_metrics', {})
        
        transfer_types = ['P2P Transfers', 'PCIe Transfers']
        transfer_counts = [
            df_metrics.get('p2p_transfers', 0),
            df_metrics.get('pcie_transfers', 0)
        ]
        
        ax3.pie(transfer_counts, labels=transfer_types, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Data Transfer Distribution (System Data Flow)')
    else:
        ax3.text(0.5, 0.5, 'System Data Flow\nMetrics Not Available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Data Transfer Distribution')
    
    # 4. Batch vs Individual processing
    batch_time = batch_results['batch_time'] * 1000
    individual_time = batch_results['individual_total'] * 1000
    
    ax4.bar(['Batch Processing', 'Individual Processing'], [batch_time, individual_time], 
            alpha=0.8, color=['green', 'red'])
    ax4.set_ylabel('Total Processing Time (ms)')
    ax4.set_title(f'Batch vs Individual Processing\n(Speedup: {batch_results["speedup"]:.2f}x)')
    
    # Add value labels
    for i, v in enumerate([batch_time, individual_time]):
        ax4.text(i, v + max(batch_time, individual_time)*0.01, f'{v:.1f}ms', ha='center')
    
    plt.tight_layout()
    plt.savefig('test_results/plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Memory utilization plot (if available)
    if 'system_data_flow_metrics' in systemdf_results['statistics']:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        sdf_metrics = systemdf_results['statistics']['system_data_flow_metrics']
        memory_metrics = sdf_metrics.get('memory_metrics', {})
        
        if 'memory_subsystems' in memory_metrics:
            subsystems = memory_metrics['memory_subsystems']
            
            names = []
            utilized = []
            capacities = []
            
            for name, stats in subsystems.items():
                names.append(name.upper())
                utilized.append(stats.get('allocated_mb', 0))
                capacities.append(stats.get('capacity_mb', 0))
            
            x = np.arange(len(names))
            width = 0.35
            
            ax.bar(x - width/2, utilized, width, label='Allocated', alpha=0.8)
            ax.bar(x + width/2, capacities, width, label='Total Capacity', alpha=0.8)
            
            ax.set_xlabel('Memory Subsystem')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('System Memory Utilization')
            ax.legend()
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            
            plt.tight_layout()
            plt.savefig('test_results/plots/memory_utilization.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info("Performance plots saved to test_results/plots/")

def save_detailed_metrics(traditional_results, systemdf_results, batch_results):
    """Save detailed metrics to JSON files."""
    logger.info("Saving detailed metrics...")
    
    # Comprehensive test results
    all_results = {
        'test_timestamp': datetime.now().isoformat(),
        'traditional_csd': traditional_results,
        'system_data_flow': systemdf_results,
        'batch_processing': batch_results
    }
    
    # Save complete results
    with open('test_results/metrics/complete_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save summary metrics
    summary = {
        'test_timestamp': datetime.now().isoformat(),
        'traditional_csd': {
            'avg_query_time_ms': np.mean([q['processing_time'] for q in traditional_results['queries']]) * 1000,
            'document_processing_time_s': traditional_results['document_addition']['time'],
            'total_documents': traditional_results['document_addition']['documents'],
            'total_chunks': traditional_results['document_addition']['chunks']
        },
        'system_data_flow': {
            'avg_query_time_ms': np.mean([q['processing_time'] for q in systemdf_results['queries']]) * 1000,
            'document_processing_time_s': systemdf_results['document_addition']['time'],
            'total_documents': systemdf_results['document_addition']['documents'],
            'total_chunks': systemdf_results['document_addition']['chunks']
        },
        'batch_processing': {
            'batch_time_ms': batch_results['batch_time'] * 1000,
            'individual_time_ms': batch_results['individual_total'] * 1000,
            'speedup_factor': batch_results['speedup']
        }
    }
    
    # Add system data flow specific metrics
    if 'system_data_flow_metrics' in systemdf_results['statistics']:
        sdf_metrics = systemdf_results['statistics']['system_data_flow_metrics']
        df_metrics = sdf_metrics.get('data_flow_metrics', {})
        
        summary['system_data_flow'].update({
            'avg_latency_ms': df_metrics.get('avg_latency_ms', 0),
            'p2p_transfers': df_metrics.get('p2p_transfers', 0),
            'pcie_transfers': df_metrics.get('pcie_transfers', 0),
            'total_data_transferred_mb': df_metrics.get('total_data_transferred_mb', 0)
        })
    
    with open('test_results/metrics/test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Metrics saved to test_results/metrics/")

def main():
    """Main test execution function."""
    logger.info("Starting comprehensive Enhanced RAG-CSD system tests...")
    logger.info(f"Test started at: {datetime.now().isoformat()}")
    
    try:
        # Test traditional CSD mode
        traditional_results = test_traditional_csd_mode()
        
        # Test new system data flow mode
        systemdf_results = test_system_dataflow_mode()
        
        # Test batch processing
        batch_results = test_batch_processing()
        
        # Generate plots and save metrics
        generate_performance_plots(traditional_results, systemdf_results, batch_results)
        save_detailed_metrics(traditional_results, systemdf_results, batch_results)
        
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        # Summary statistics
        trad_avg = np.mean([q['processing_time'] for q in traditional_results['queries']])
        sdf_avg = np.mean([q['processing_time'] for q in systemdf_results['queries']])
        
        logger.info(f"Traditional CSD avg query time: {trad_avg*1000:.2f}ms")
        logger.info(f"System Data Flow avg query time: {sdf_avg*1000:.2f}ms")
        logger.info(f"Performance difference: {((sdf_avg - trad_avg)/trad_avg)*100:+.1f}%")
        logger.info(f"Batch processing speedup: {batch_results['speedup']:.2f}x")
        
        logger.info("\nAll tests completed successfully!")
        logger.info("Results saved to:")
        logger.info("- test_results/logs/test_execution.log")
        logger.info("- test_results/plots/performance_comparison.png")
        logger.info("- test_results/metrics/complete_test_results.json")
        logger.info("- test_results/metrics/test_summary.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)