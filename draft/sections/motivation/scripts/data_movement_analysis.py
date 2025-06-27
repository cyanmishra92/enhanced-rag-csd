#!/usr/bin/env python3
"""
Data Movement and Bandwidth Analysis for RAG Systems

This script analyzes data movement costs, bandwidth requirements, and energy consumption
in traditional RAG systems versus CSD-enhanced approaches.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import os

# Set style for academic plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

@dataclass
class DataMovementSpecs:
    """Specifications for data movement between different system components."""
    name: str
    bandwidth_gbps: float
    latency_ms: float
    power_watts: float
    efficiency: float  # Effective bandwidth utilization

@dataclass
class RAGWorkload:
    """Characteristics of a RAG workload."""
    query_size_kb: float
    embedding_size_kb: float
    vector_db_size_gb: float
    document_size_kb: float
    context_size_kb: float
    queries_per_second: int

class RAGDataMovementAnalyzer:
    """Analyze data movement patterns and costs in RAG systems."""
    
    def __init__(self):
        # Data movement interfaces
        self.interfaces = {
            'NVMe SSD': DataMovementSpecs(
                name='NVMe SSD to CPU',
                bandwidth_gbps=7.0,
                latency_ms=0.1,
                power_watts=8.0,
                efficiency=0.85
            ),
            'PCIe 4.0 x16': DataMovementSpecs(
                name='CPU to GPU (PCIe 4.0)',
                bandwidth_gbps=32.0,
                latency_ms=0.05,
                power_watts=15.0,
                efficiency=0.75
            ),
            'PCIe 4.0 x8': DataMovementSpecs(
                name='CPU to GPU (PCIe 4.0 x8)',
                bandwidth_gbps=16.0,
                latency_ms=0.05,
                power_watts=10.0,
                efficiency=0.75
            ),
            'PCIe 4.0 x4': DataMovementSpecs(
                name='CSD to CPU (PCIe 4.0 x4)',
                bandwidth_gbps=8.0,
                latency_ms=0.05,
                power_watts=5.0,
                efficiency=0.80
            ),
            'DDR4-3200': DataMovementSpecs(
                name='CPU Memory Access',
                bandwidth_gbps=25.6,
                latency_ms=0.01,
                power_watts=50.0,
                efficiency=0.90
            ),
            'HBM2': DataMovementSpecs(
                name='GPU Memory Access',
                bandwidth_gbps=900.0,
                latency_ms=0.001,
                power_watts=100.0,
                efficiency=0.95
            ),
            'CSD Direct': DataMovementSpecs(
                name='CSD Direct Access',
                bandwidth_gbps=7.0,
                latency_ms=0.08,
                power_watts=3.0,
                efficiency=0.95
            )
        }
        
        # RAG workload characteristics
        self.workloads = {
            'Light': RAGWorkload(
                query_size_kb=1.5,
                embedding_size_kb=1.5,
                vector_db_size_gb=0.5,
                document_size_kb=10.0,
                context_size_kb=20.0,
                queries_per_second=100
            ),
            'Medium': RAGWorkload(
                query_size_kb=3.0,
                embedding_size_kb=1.5,
                vector_db_size_gb=1.5,
                document_size_kb=25.0,
                context_size_kb=50.0,
                queries_per_second=500
            ),
            'Heavy': RAGWorkload(
                query_size_kb=5.0,
                embedding_size_kb=1.5,
                vector_db_size_gb=5.0,
                document_size_kb=50.0,
                context_size_kb=100.0,
                queries_per_second=1000
            ),
            'Enterprise': RAGWorkload(
                query_size_kb=8.0,
                embedding_size_kb=1.5,
                vector_db_size_gb=20.0,
                document_size_kb=100.0,
                context_size_kb=200.0,
                queries_per_second=2000
            )
        }
        
        self.output_dir = "draft/sections/motivation/figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_data_movement_cost(self, data_size_gb: float, interface: DataMovementSpecs) -> Dict:
        """Calculate the cost of moving data over a specific interface."""
        effective_bandwidth = interface.bandwidth_gbps * interface.efficiency
        
        # Transfer time calculation
        transfer_time_s = data_size_gb / effective_bandwidth
        
        # Total latency including setup
        total_latency_ms = interface.latency_ms + (transfer_time_s * 1000)
        
        # Energy calculation
        energy_joules = interface.power_watts * transfer_time_s
        
        return {
            'transfer_time_ms': total_latency_ms,
            'energy_joules': energy_joules,
            'bandwidth_utilization': min(1.0, data_size_gb / effective_bandwidth),
            'effective_bandwidth_gbps': effective_bandwidth
        }
    
    def analyze_traditional_rag_pipeline(self, workload: RAGWorkload) -> Dict:
        """Analyze data movement in traditional RAG pipeline."""
        
        # Step 1: Load vector database from storage to CPU memory
        vector_db_load = self.calculate_data_movement_cost(
            workload.vector_db_size_gb, self.interfaces['NVMe SSD'])
        
        # Step 2: Transfer vector DB to GPU memory
        vector_db_gpu = self.calculate_data_movement_cost(
            workload.vector_db_size_gb, self.interfaces['PCIe 4.0 x16'])
        
        # Step 3: Query encoding (query to GPU, embedding back)
        query_to_gpu = self.calculate_data_movement_cost(
            workload.query_size_kb / (1024 * 1024), self.interfaces['PCIe 4.0 x16'])
        embedding_from_gpu = self.calculate_data_movement_cost(
            workload.embedding_size_kb / (1024 * 1024), self.interfaces['PCIe 4.0 x16'])
        
        # Step 4: Document retrieval and transfer
        doc_load = self.calculate_data_movement_cost(
            workload.document_size_kb / (1024 * 1024), self.interfaces['NVMe SSD'])
        doc_to_gpu = self.calculate_data_movement_cost(
            workload.context_size_kb / (1024 * 1024), self.interfaces['PCIe 4.0 x16'])
        
        total_latency = (vector_db_load['transfer_time_ms'] + 
                        vector_db_gpu['transfer_time_ms'] +
                        query_to_gpu['transfer_time_ms'] + 
                        embedding_from_gpu['transfer_time_ms'] +
                        doc_load['transfer_time_ms'] + 
                        doc_to_gpu['transfer_time_ms'])
        
        total_energy = (vector_db_load['energy_joules'] + 
                       vector_db_gpu['energy_joules'] +
                       query_to_gpu['energy_joules'] + 
                       embedding_from_gpu['energy_joules'] +
                       doc_load['energy_joules'] + 
                       doc_to_gpu['energy_joules'])
        
        # Calculate per-query costs (amortize vector DB loading)
        vector_db_amortization = 1000  # Load every 1000 queries
        per_query_latency = (total_latency / vector_db_amortization + 
                           query_to_gpu['transfer_time_ms'] + 
                           embedding_from_gpu['transfer_time_ms'] +
                           doc_load['transfer_time_ms'] + 
                           doc_to_gpu['transfer_time_ms'])
        
        per_query_energy = (total_energy / vector_db_amortization + 
                          query_to_gpu['energy_joules'] + 
                          embedding_from_gpu['energy_joules'] +
                          doc_load['energy_joules'] + 
                          doc_to_gpu['energy_joules'])
        
        return {
            'total_latency_ms': total_latency,
            'per_query_latency_ms': per_query_latency,
            'total_energy_joules': total_energy,
            'per_query_energy_joules': per_query_energy,
            'data_moved_gb': (workload.vector_db_size_gb * 2 + 
                            (workload.query_size_kb + workload.embedding_size_kb + 
                             workload.document_size_kb + workload.context_size_kb) / (1024 * 1024)),
            'breakdown': {
                'vector_db_load': vector_db_load,
                'vector_db_gpu': vector_db_gpu,
                'query_processing': query_to_gpu,
                'embedding_return': embedding_from_gpu,
                'document_load': doc_load,
                'context_transfer': doc_to_gpu
            }
        }
    
    def analyze_csd_enhanced_pipeline(self, workload: RAGWorkload) -> Dict:
        """Analyze data movement in CSD-enhanced RAG pipeline."""
        
        # Step 1: Query encoding on CSD (no data movement to GPU)
        query_to_csd = self.calculate_data_movement_cost(
            workload.query_size_kb / (1024 * 1024), self.interfaces['CSD Direct'])
        
        # Step 2: Vector retrieval on CSD (no data movement)
        # Only transfer top-k candidates to CPU/GPU
        candidate_transfer = self.calculate_data_movement_cost(
            0.001,  # 1MB of top candidates
            self.interfaces['PCIe 4.0 x4'])
        
        # Step 3: Document retrieval and context preparation on CSD
        context_to_gpu = self.calculate_data_movement_cost(
            workload.context_size_kb / (1024 * 1024), self.interfaces['PCIe 4.0 x4'])
        
        total_latency = (query_to_csd['transfer_time_ms'] + 
                        candidate_transfer['transfer_time_ms'] +
                        context_to_gpu['transfer_time_ms'])
        
        total_energy = (query_to_csd['energy_joules'] + 
                       candidate_transfer['energy_joules'] +
                       context_to_gpu['energy_joules'])
        
        return {
            'total_latency_ms': total_latency,
            'per_query_latency_ms': total_latency,  # No amortization needed
            'total_energy_joules': total_energy,
            'per_query_energy_joules': total_energy,
            'data_moved_gb': ((workload.query_size_kb + workload.context_size_kb) / (1024 * 1024) + 0.001),
            'breakdown': {
                'query_to_csd': query_to_csd,
                'candidate_transfer': candidate_transfer,
                'context_transfer': context_to_gpu
            }
        }
    
    def plot_latency_comparison(self):
        """Plot latency comparison across workloads and architectures."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        workload_names = list(self.workloads.keys())
        traditional_latencies = []
        csd_latencies = []
        
        for workload_name, workload in self.workloads.items():
            trad_analysis = self.analyze_traditional_rag_pipeline(workload)
            csd_analysis = self.analyze_csd_enhanced_pipeline(workload)
            
            traditional_latencies.append(trad_analysis['per_query_latency_ms'])
            csd_latencies.append(csd_analysis['per_query_latency_ms'])
        
        x = np.arange(len(workload_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, traditional_latencies, width, 
                       label='Traditional RAG', color='#E74C3C', alpha=0.8)
        bars2 = ax1.bar(x + width/2, csd_latencies, width,
                       label='CSD-Enhanced RAG', color='#2ECC71', alpha=0.8)
        
        ax1.set_xlabel('Workload Type')
        ax1.set_ylabel('Data Movement Latency (ms)')
        ax1.set_title('Per-Query Data Movement Latency')
        ax1.set_xticks(x)
        ax1.set_xticklabels(workload_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}ms', ha='center', va='bottom')
        
        # Calculate and plot speedups
        speedups = [trad / csd for trad, csd in zip(traditional_latencies, csd_latencies)]
        bars3 = ax2.bar(x, speedups, color='#3498DB', alpha=0.8)
        
        ax2.set_xlabel('Workload Type')
        ax2.set_ylabel('Speedup (×)')
        ax2.set_title('CSD Data Movement Speedup')
        ax2.set_xticks(x)
        ax2.set_xticklabels(workload_names)
        ax2.grid(True, alpha=0.3)
        
        # Add speedup labels
        for bar, speedup in zip(bars3, speedups):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{speedup:.1f}×', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/data_movement_latency_comparison.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated latency comparison: {self.output_dir}/data_movement_latency_comparison.pdf")
    
    def plot_bandwidth_utilization(self):
        """Plot bandwidth utilization and bottleneck analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Bandwidth requirements vs available bandwidth
        workload_names = list(self.workloads.keys())
        
        # Calculate bandwidth requirements for different query rates
        query_rates = [100, 500, 1000, 2000, 5000]
        
        for i, workload_name in enumerate(workload_names):
            workload = self.workloads[workload_name]
            
            trad_requirements = []
            csd_requirements = []
            
            for qps in query_rates:
                # Traditional RAG bandwidth requirement
                trad_analysis = self.analyze_traditional_rag_pipeline(workload)
                trad_bandwidth = trad_analysis['data_moved_gb'] * qps / 8  # Convert to Gbps
                trad_requirements.append(trad_bandwidth)
                
                # CSD-enhanced bandwidth requirement
                csd_analysis = self.analyze_csd_enhanced_pipeline(workload)
                csd_bandwidth = csd_analysis['data_moved_gb'] * qps / 8  # Convert to Gbps
                csd_requirements.append(csd_bandwidth)
            
            ax1.plot(query_rates, trad_requirements, 'o-', 
                    label=f'Traditional {workload_name}', linewidth=2, markersize=6)
            ax1.plot(query_rates, csd_requirements, 's--', 
                    label=f'CSD {workload_name}', linewidth=2, markersize=6)
        
        # Add bandwidth limit lines
        ax1.axhline(y=32, color='red', linestyle='-', alpha=0.7, label='PCIe 4.0 x16 Limit')
        ax1.axhline(y=16, color='orange', linestyle='-', alpha=0.7, label='PCIe 4.0 x8 Limit')
        ax1.axhline(y=8, color='yellow', linestyle='-', alpha=0.7, label='PCIe 4.0 x4 Limit')
        
        ax1.set_xlabel('Query Rate (queries/sec)')
        ax1.set_ylabel('Required Bandwidth (Gbps)')
        ax1.set_title('Bandwidth Requirements vs Query Rate')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # Bandwidth bottleneck analysis
        interface_names = ['NVMe SSD', 'PCIe 4.0 x16', 'PCIe 4.0 x8', 'PCIe 4.0 x4', 'CSD Direct']
        bandwidths = [self.interfaces[name].bandwidth_gbps * self.interfaces[name].efficiency 
                     for name in interface_names]
        bottleneck_scores = []
        
        # Calculate bottleneck severity (higher = more bottlenecked)
        medium_workload = self.workloads['Medium']
        trad_analysis = self.analyze_traditional_rag_pipeline(medium_workload)
        
        for interface_name in interface_names:
            if interface_name == 'NVMe SSD':
                # Vector DB loading bottleneck
                load_time = medium_workload.vector_db_size_gb / bandwidths[0]
                bottleneck_scores.append(load_time * 1000)  # Convert to ms
            elif 'PCIe' in interface_name:
                # Data transfer bottleneck
                transfer_time = medium_workload.vector_db_size_gb / bandwidths[interface_names.index(interface_name)]
                bottleneck_scores.append(transfer_time * 1000)
            else:  # CSD Direct
                # Minimal bottleneck
                bottleneck_scores.append(1.0)
        
        colors = ['#E74C3C', '#F39C12', '#F1C40F', '#3498DB', '#2ECC71']
        bars = ax2.bar(interface_names, bottleneck_scores, color=colors, alpha=0.8)
        
        ax2.set_xlabel('Interface Type')
        ax2.set_ylabel('Bottleneck Severity (ms)')
        ax2.set_title('Interface Bottleneck Analysis')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add bottleneck labels
        for bar, score in zip(bars, bottleneck_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{score:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bandwidth_utilization_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated bandwidth analysis: {self.output_dir}/bandwidth_utilization_analysis.pdf")
    
    def plot_energy_consumption(self):
        """Plot energy consumption analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        workload_names = list(self.workloads.keys())
        traditional_energy = []
        csd_energy = []
        
        for workload_name, workload in self.workloads.items():
            trad_analysis = self.analyze_traditional_rag_pipeline(workload)
            csd_analysis = self.analyze_csd_enhanced_pipeline(workload)
            
            traditional_energy.append(trad_analysis['per_query_energy_joules'])
            csd_energy.append(csd_analysis['per_query_energy_joules'])
        
        x = np.arange(len(workload_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, traditional_energy, width, 
                       label='Traditional RAG', color='#E74C3C', alpha=0.8)
        bars2 = ax1.bar(x + width/2, csd_energy, width,
                       label='CSD-Enhanced RAG', color='#2ECC71', alpha=0.8)
        
        ax1.set_xlabel('Workload Type')
        ax1.set_ylabel('Energy per Query (Joules)')
        ax1.set_title('Energy Consumption per Query')
        ax1.set_xticks(x)
        ax1.set_xticklabels(workload_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}J', ha='center', va='bottom')
        
        # Calculate energy savings
        energy_savings = [(trad - csd) / trad * 100 for trad, csd in zip(traditional_energy, csd_energy)]
        bars3 = ax2.bar(x, energy_savings, color='#3498DB', alpha=0.8)
        
        ax2.set_xlabel('Workload Type')
        ax2.set_ylabel('Energy Savings (%)')
        ax2.set_title('Energy Savings with CSD Enhancement')
        ax2.set_xticks(x)
        ax2.set_xticklabels(workload_names)
        ax2.grid(True, alpha=0.3)
        
        # Add savings labels
        for bar, saving in zip(bars3, energy_savings):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{saving:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/energy_consumption_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated energy analysis: {self.output_dir}/energy_consumption_analysis.pdf")
    
    def plot_scaling_analysis(self):
        """Plot scaling behavior with query rate."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        query_rates = np.logspace(1, 4, 50)  # 10 to 10,000 QPS
        medium_workload = self.workloads['Medium']
        
        # Traditional RAG scaling
        trad_latencies = []
        trad_energy_rates = []
        trad_bandwidth_req = []
        
        # CSD-Enhanced RAG scaling
        csd_latencies = []
        csd_energy_rates = []
        csd_bandwidth_req = []
        
        for qps in query_rates:
            trad_analysis = self.analyze_traditional_rag_pipeline(medium_workload)
            csd_analysis = self.analyze_csd_enhanced_pipeline(medium_workload)
            
            # Latency (assume some queuing effects)
            trad_base_latency = trad_analysis['per_query_latency_ms']
            csd_base_latency = csd_analysis['per_query_latency_ms']
            
            # Add queuing delay (M/M/1 approximation)
            service_rate_trad = 1000 / trad_base_latency  # queries per second
            service_rate_csd = 1000 / csd_base_latency
            
            if qps < service_rate_trad * 0.9:
                trad_queue_delay = trad_base_latency * (qps / service_rate_trad) / (1 - qps / service_rate_trad)
            else:
                trad_queue_delay = 1000  # System saturated
            
            if qps < service_rate_csd * 0.9:
                csd_queue_delay = csd_base_latency * (qps / service_rate_csd) / (1 - qps / service_rate_csd)
            else:
                csd_queue_delay = 1000  # System saturated
            
            trad_latencies.append(trad_base_latency + trad_queue_delay)
            csd_latencies.append(csd_base_latency + csd_queue_delay)
            
            # Energy rate (Watts)
            trad_energy_rates.append(trad_analysis['per_query_energy_joules'] * qps)
            csd_energy_rates.append(csd_analysis['per_query_energy_joules'] * qps)
            
            # Bandwidth requirements
            trad_bandwidth_req.append(trad_analysis['data_moved_gb'] * qps * 8)  # Gbps
            csd_bandwidth_req.append(csd_analysis['data_moved_gb'] * qps * 8)
        
        # Plot latency scaling
        ax1.plot(query_rates, trad_latencies, 'r-', linewidth=2, label='Traditional RAG')
        ax1.plot(query_rates, csd_latencies, 'g-', linewidth=2, label='CSD-Enhanced RAG')
        ax1.set_xlabel('Query Rate (QPS)')
        ax1.set_ylabel('Average Latency (ms)')
        ax1.set_title('Latency vs Query Rate')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot energy scaling
        ax2.plot(query_rates, trad_energy_rates, 'r-', linewidth=2, label='Traditional RAG')
        ax2.plot(query_rates, csd_energy_rates, 'g-', linewidth=2, label='CSD-Enhanced RAG')
        ax2.set_xlabel('Query Rate (QPS)')
        ax2.set_ylabel('Power Consumption (Watts)')
        ax2.set_title('Power vs Query Rate')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot bandwidth scaling
        ax3.plot(query_rates, trad_bandwidth_req, 'r-', linewidth=2, label='Traditional RAG')
        ax3.plot(query_rates, csd_bandwidth_req, 'g-', linewidth=2, label='CSD-Enhanced RAG')
        ax3.axhline(y=32, color='orange', linestyle='--', alpha=0.7, label='PCIe 4.0 x16 Limit')
        ax3.set_xlabel('Query Rate (QPS)')
        ax3.set_ylabel('Required Bandwidth (Gbps)')
        ax3.set_title('Bandwidth Requirements vs Query Rate')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot efficiency metrics
        efficiency_trad = [min(1.0, 32 / bw) for bw in trad_bandwidth_req]
        efficiency_csd = [min(1.0, 8 / bw) for bw in csd_bandwidth_req]
        
        ax4.plot(query_rates, efficiency_trad, 'r-', linewidth=2, label='Traditional RAG')
        ax4.plot(query_rates, efficiency_csd, 'g-', linewidth=2, label='CSD-Enhanced RAG')
        ax4.set_xlabel('Query Rate (QPS)')
        ax4.set_ylabel('System Efficiency')
        ax4.set_title('System Efficiency vs Query Rate')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/scaling_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated scaling analysis: {self.output_dir}/scaling_analysis.pdf")
    
    def generate_summary_table(self):
        """Generate comprehensive summary table."""
        data = []
        
        for workload_name, workload in self.workloads.items():
            trad_analysis = self.analyze_traditional_rag_pipeline(workload)
            csd_analysis = self.analyze_csd_enhanced_pipeline(workload)
            
            # Calculate improvements
            latency_improvement = (trad_analysis['per_query_latency_ms'] - 
                                 csd_analysis['per_query_latency_ms']) / trad_analysis['per_query_latency_ms'] * 100
            
            energy_improvement = (trad_analysis['per_query_energy_joules'] - 
                                 csd_analysis['per_query_energy_joules']) / trad_analysis['per_query_energy_joules'] * 100
            
            data_reduction = (trad_analysis['data_moved_gb'] - 
                            csd_analysis['data_moved_gb']) / trad_analysis['data_moved_gb'] * 100
            
            data.append({
                'Workload': workload_name,
                'Traditional Latency (ms)': f"{trad_analysis['per_query_latency_ms']:.2f}",
                'CSD Latency (ms)': f"{csd_analysis['per_query_latency_ms']:.2f}",
                'Latency Improvement (%)': f"{latency_improvement:.1f}",
                'Traditional Energy (J)': f"{trad_analysis['per_query_energy_joules']:.3f}",
                'CSD Energy (J)': f"{csd_analysis['per_query_energy_joules']:.3f}",
                'Energy Improvement (%)': f"{energy_improvement:.1f}",
                'Traditional Data (GB)': f"{trad_analysis['data_moved_gb']:.3f}",
                'CSD Data (GB)': f"{csd_analysis['data_moved_gb']:.3f}",
                'Data Reduction (%)': f"{data_reduction:.1f}"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{self.output_dir}/data_movement_summary.csv', index=False)
        
        # Create formatted table
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 2)
        
        # Color-code improvements
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                if 'Improvement' in col or 'Reduction' in col:
                    value = float(df.iloc[i, j])
                    if value > 50:
                        table[(i+1, j)].set_facecolor('#C8E6C9')  # Green for good
                    elif value > 25:
                        table[(i+1, j)].set_facecolor('#FFF9C4')  # Yellow for moderate
                    else:
                        table[(i+1, j)].set_facecolor('#FFCDD2')  # Red for poor
        
        plt.title('Data Movement Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/data_movement_summary_table.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated summary table: {self.output_dir}/data_movement_summary_table.pdf")
        print(f"✅ Generated CSV data: {self.output_dir}/data_movement_summary.csv")
    
    def run_full_analysis(self):
        """Run complete data movement analysis."""
        print("🔍 Starting Data Movement and Bandwidth Analysis...")
        print("=" * 60)
        
        self.plot_latency_comparison()
        self.plot_bandwidth_utilization()
        self.plot_energy_consumption()
        self.plot_scaling_analysis()
        self.generate_summary_table()
        
        print("\n✅ Data Movement Analysis Complete!")
        print(f"📊 All plots saved to: {self.output_dir}/")
        print("📈 Generated plots:")
        print("   - data_movement_latency_comparison.pdf")
        print("   - bandwidth_utilization_analysis.pdf")
        print("   - energy_consumption_analysis.pdf")
        print("   - scaling_analysis.pdf")
        print("   - data_movement_summary_table.pdf")
        print("   - data_movement_summary.csv")

def main():
    """Main function to run the analysis."""
    analyzer = RAGDataMovementAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()