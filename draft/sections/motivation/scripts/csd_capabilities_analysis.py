#!/usr/bin/env python3
"""
Modern CSD Capabilities Analysis

This script analyzes the computational capabilities of modern Computational Storage Devices
for ML operations, comparing different CSD architectures and their suitability for RAG workloads.
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
class CSDSpecs:
    """Specifications for different CSD implementations."""
    name: str
    compute_type: str  # 'ARM', 'FPGA', 'ASIC'
    peak_ops_per_sec: float
    power_watts: float
    storage_capacity_tb: float
    memory_bandwidth_gbps: float
    pcie_bandwidth_gbps: float
    cost_per_tb_usd: int
    ml_optimized: bool
    
    @property
    def ops_per_watt(self) -> float:
        return self.peak_ops_per_sec / self.power_watts
    
    @property
    def total_cost_usd(self) -> int:
        return self.cost_per_tb_usd * self.storage_capacity_tb

@dataclass
class MLOperation:
    """ML operation characteristics for CSD evaluation."""
    name: str
    operation_type: str  # 'encoding', 'retrieval', 'augmentation'
    flops_per_sample: int
    memory_accesses_per_sample: int
    parallelizable: bool
    memory_bound: bool

class CSDCapabilityAnalyzer:
    """Analyze CSD capabilities for ML workloads."""
    
    def __init__(self):
        # Modern CSD specifications based on real products and research
        self.csds = {
            'Samsung SmartSSD': CSDSpecs(
                name='Samsung SmartSSD FP1',
                compute_type='FPGA',
                peak_ops_per_sec=160e9,  # 160 GOPS
                power_watts=25,
                storage_capacity_tb=8.0,
                memory_bandwidth_gbps=14.0,
                pcie_bandwidth_gbps=8.0,  # PCIe 4.0 x4
                cost_per_tb_usd=800,
                ml_optimized=True
            ),
            'SK Hynix CSD': CSDSpecs(
                name='SK Hynix Computational SSD',
                compute_type='ARM',
                peak_ops_per_sec=210e9,  # 210 GOPS
                power_watts=15,
                storage_capacity_tb=4.0,
                memory_bandwidth_gbps=12.8,
                pcie_bandwidth_gbps=8.0,
                cost_per_tb_usd=600,
                ml_optimized=True
            ),
            'Xilinx Alveo CSD': CSDSpecs(
                name='Xilinx Alveo U50 CSD',
                compute_type='FPGA',
                peak_ops_per_sec=3700e9,  # 3.7 TOPS
                power_watts=150,
                storage_capacity_tb=16.0,
                memory_bandwidth_gbps=77.0,  # HBM2
                pcie_bandwidth_gbps=32.0,  # PCIe 4.0 x16
                cost_per_tb_usd=2000,
                ml_optimized=True
            ),
            'AMD Versal CSD': CSDSpecs(
                name='AMD Versal FPGA CSD',
                compute_type='FPGA',
                peak_ops_per_sec=1800e9,  # 1.8 TOPS
                power_watts=75,
                storage_capacity_tb=8.0,
                memory_bandwidth_gbps=58.6,  # HBM2
                pcie_bandwidth_gbps=32.0,
                cost_per_tb_usd=1500,
                ml_optimized=True
            ),
            'Future ARM CSD': CSDSpecs(
                name='Future ARM Cortex-A78 CSD',
                compute_type='ARM',
                peak_ops_per_sec=420e9,  # 420 GOPS
                power_watts=5,
                storage_capacity_tb=2.0,
                memory_bandwidth_gbps=25.6,
                pcie_bandwidth_gbps=16.0,  # PCIe 4.0 x8
                cost_per_tb_usd=300,
                ml_optimized=True
            ),
            'Custom ASIC CSD': CSDSpecs(
                name='Custom ML ASIC CSD',
                compute_type='ASIC',
                peak_ops_per_sec=5000e9,  # 5 TOPS
                power_watts=200,
                storage_capacity_tb=32.0,
                memory_bandwidth_gbps=100.0,
                pcie_bandwidth_gbps=64.0,  # PCIe 5.0 x16
                cost_per_tb_usd=3000,
                ml_optimized=True
            ),
            'Traditional SSD': CSDSpecs(
                name='Traditional NVMe SSD',
                compute_type='None',
                peak_ops_per_sec=0,  # No compute capability
                power_watts=8,
                storage_capacity_tb=8.0,
                memory_bandwidth_gbps=0,
                pcie_bandwidth_gbps=8.0,
                cost_per_tb_usd=100,
                ml_optimized=False
            )
        }
        
        # ML operations for RAG systems
        self.ml_operations = {
            'Query Encoding': MLOperation(
                name='Query Encoding',
                operation_type='encoding',
                flops_per_sample=220_000_000,  # 220M FLOPS per query
                memory_accesses_per_sample=100_000,  # Memory accesses
                parallelizable=True,
                memory_bound=False
            ),
            'Embedding Lookup': MLOperation(
                name='Embedding Lookup',
                operation_type='retrieval',
                flops_per_sample=384,  # Simple memory lookup
                memory_accesses_per_sample=384,  # Vector dimension
                parallelizable=True,
                memory_bound=True
            ),
            'Similarity Compute': MLOperation(
                name='Similarity Computation',
                operation_type='retrieval',
                flops_per_sample=768,  # Dot product + comparison
                memory_accesses_per_sample=768,
                parallelizable=True,
                memory_bound=False
            ),
            'Context Augmentation': MLOperation(
                name='Context Augmentation',
                operation_type='augmentation',
                flops_per_sample=10_000,  # Text processing
                memory_accesses_per_sample=5_000,
                parallelizable=True,
                memory_bound=True
            ),
            'Vector Update': MLOperation(
                name='Vector Database Update',
                operation_type='maintenance',
                flops_per_sample=1_000,  # Index update operations
                memory_accesses_per_sample=2_000,
                parallelizable=False,
                memory_bound=True
            )
        }
        
        self.output_dir = "draft/sections/motivation/figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_operation_performance(self, csd: CSDSpecs, operation: MLOperation, 
                                      batch_size: int = 1000) -> Dict:
        """Calculate performance for a specific operation on a CSD."""
        
        if csd.peak_ops_per_sec == 0:  # Traditional SSD
            return {
                'execution_time_ms': float('inf'),
                'throughput_ops_per_sec': 0,
                'energy_per_operation_j': 0,
                'can_execute': False,
                'bottleneck': 'No compute capability'
            }
        
        # Calculate execution time
        total_flops = operation.flops_per_sample * batch_size
        compute_time_s = total_flops / csd.peak_ops_per_sec
        
        # Calculate memory transfer time
        total_memory_accesses = operation.memory_accesses_per_sample * batch_size
        memory_bytes = total_memory_accesses * 4  # FP32
        memory_time_s = memory_bytes / (csd.memory_bandwidth_gbps * 1e9)
        
        # Determine bottleneck
        if operation.memory_bound or memory_time_s > compute_time_s:
            actual_time_s = memory_time_s
            bottleneck = 'Memory bound'
        else:
            actual_time_s = compute_time_s
            bottleneck = 'Compute bound'
        
        # Calculate metrics
        throughput_ops_per_sec = batch_size / actual_time_s
        energy_per_operation_j = csd.power_watts * actual_time_s / batch_size
        
        return {
            'execution_time_ms': actual_time_s * 1000,
            'throughput_ops_per_sec': throughput_ops_per_sec,
            'energy_per_operation_j': energy_per_operation_j,
            'can_execute': True,
            'bottleneck': bottleneck,
            'compute_utilization': compute_time_s / actual_time_s,
            'memory_utilization': memory_time_s / actual_time_s
        }
    
    def plot_csd_capabilities_overview(self):
        """Plot overview of CSD computational capabilities."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        csd_names = [csd.name for csd in self.csds.values()]
        compute_performance = [csd.peak_ops_per_sec / 1e9 for csd in self.csds.values()]  # GOPS
        power_efficiency = [csd.ops_per_watt / 1e9 for csd in self.csds.values()]  # GOPS/W
        storage_capacity = [csd.storage_capacity_tb for csd in self.csds.values()]
        cost_per_tb = [csd.cost_per_tb_usd for csd in self.csds.values()]
        
        # Color-code by compute type
        colors = []
        for csd in self.csds.values():
            if csd.compute_type == 'FPGA':
                colors.append('#E74C3C')
            elif csd.compute_type == 'ARM':
                colors.append('#2ECC71')
            elif csd.compute_type == 'ASIC':
                colors.append('#9B59B6')
            else:
                colors.append('#BDC3C7')
        
        # 1. Compute Performance
        bars1 = ax1.bar(range(len(csd_names)), compute_performance, color=colors, alpha=0.8)
        ax1.set_xlabel('CSD Type')
        ax1.set_ylabel('Compute Performance (GOPS)')
        ax1.set_title('Peak Computational Performance')
        ax1.set_xticks(range(len(csd_names)))
        ax1.set_xticklabels([name.split()[0] for name in csd_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, compute_performance):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Power Efficiency
        bars2 = ax2.bar(range(len(csd_names)), power_efficiency, color=colors, alpha=0.8)
        ax2.set_xlabel('CSD Type')
        ax2.set_ylabel('Power Efficiency (GOPS/W)')
        ax2.set_title('Power Efficiency Comparison')
        ax2.set_xticks(range(len(csd_names)))
        ax2.set_xticklabels([name.split()[0] for name in csd_names], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, power_efficiency):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Performance vs Cost
        valid_csds = [(perf, cost, name) for perf, cost, name in 
                     zip(compute_performance, cost_per_tb, csd_names) if perf > 0]
        
        if valid_csds:
            perfs, costs, names = zip(*valid_csds)
            scatter = ax3.scatter(costs, perfs, c=colors[:len(valid_csds)], s=100, alpha=0.7)
            
            for i, name in enumerate(names):
                ax3.annotate(name.split()[0], (costs[i], perfs[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Cost per TB ($)')
        ax3.set_ylabel('Compute Performance (GOPS)')
        ax3.set_title('Performance vs Cost Trade-off')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # 4. Storage vs Compute Density
        valid_data = [(stor, perf, name) for stor, perf, name in 
                     zip(storage_capacity, compute_performance, csd_names) if perf > 0]
        
        if valid_data:
            stors, perfs, names = zip(*valid_data)
            scatter = ax4.scatter(stors, perfs, c=colors[:len(valid_data)], s=100, alpha=0.7)
            
            for i, name in enumerate(names):
                ax4.annotate(name.split()[0], (stors[i], perfs[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Storage Capacity (TB)')
        ax4.set_ylabel('Compute Performance (GOPS)')
        ax4.set_title('Storage vs Compute Density')
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        fpga_patch = mpatches.Patch(color='#E74C3C', label='FPGA')
        arm_patch = mpatches.Patch(color='#2ECC71', label='ARM')
        asic_patch = mpatches.Patch(color='#9B59B6', label='ASIC')
        traditional_patch = mpatches.Patch(color='#BDC3C7', label='Traditional')
        ax4.legend(handles=[fpga_patch, arm_patch, asic_patch, traditional_patch])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/csd_capabilities_overview.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated CSD capabilities overview: {self.output_dir}/csd_capabilities_overview.pdf")
    
    def plot_ml_operation_performance(self):
        """Plot performance of different ML operations on CSDs."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for heatmaps
        csd_names = list(self.csds.keys())
        operation_names = list(self.ml_operations.keys())
        
        # Performance matrix (operations per second)
        performance_matrix = np.zeros((len(csd_names), len(operation_names)))
        energy_matrix = np.zeros((len(csd_names), len(operation_names)))
        
        for i, csd_name in enumerate(csd_names):
            for j, op_name in enumerate(operation_names):
                result = self.calculate_operation_performance(
                    self.csds[csd_name], self.ml_operations[op_name])
                
                if result['can_execute']:
                    performance_matrix[i, j] = result['throughput_ops_per_sec']
                    energy_matrix[i, j] = result['energy_per_operation_j']
                else:
                    performance_matrix[i, j] = 0
                    energy_matrix[i, j] = float('inf')
        
        # 1. Throughput heatmap
        # Mask zero values for better visualization
        masked_performance = np.ma.masked_where(performance_matrix == 0, performance_matrix)
        
        im1 = ax1.imshow(masked_performance, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(operation_names)))
        ax1.set_xticklabels([op.replace(' ', '\n') for op in operation_names], rotation=0)
        ax1.set_yticks(range(len(csd_names)))
        ax1.set_yticklabels([name.split()[0] for name in csd_names])
        ax1.set_title('Throughput (Operations/Second)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(csd_names)):
            for j in range(len(operation_names)):
                if performance_matrix[i, j] > 0:
                    text = ax1.text(j, i, f'{performance_matrix[i, j]:.0f}',
                                   ha="center", va="center", color="white", fontweight='bold', fontsize=8)
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. Energy efficiency heatmap
        energy_efficiency = np.where(energy_matrix == float('inf'), 0, 1/energy_matrix)
        masked_efficiency = np.ma.masked_where(energy_efficiency == 0, energy_efficiency)
        
        im2 = ax2.imshow(masked_efficiency, cmap='plasma', aspect='auto')
        ax2.set_xticks(range(len(operation_names)))
        ax2.set_xticklabels([op.replace(' ', '\n') for op in operation_names], rotation=0)
        ax2.set_yticks(range(len(csd_names)))
        ax2.set_yticklabels([name.split()[0] for name in csd_names])
        ax2.set_title('Energy Efficiency (Ops/Joule)', fontweight='bold')
        
        for i in range(len(csd_names)):
            for j in range(len(operation_names)):
                if energy_efficiency[i, j] > 0:
                    text = ax2.text(j, i, f'{energy_efficiency[i, j]:.1f}',
                                   ha="center", va="center", color="white", fontweight='bold', fontsize=8)
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 3. Bottleneck analysis
        bottleneck_data = []
        for csd_name in csd_names:
            compute_bound_count = 0
            memory_bound_count = 0
            
            for op_name in operation_names:
                result = self.calculate_operation_performance(
                    self.csds[csd_name], self.ml_operations[op_name])
                
                if result['can_execute']:
                    if 'Compute' in result['bottleneck']:
                        compute_bound_count += 1
                    else:
                        memory_bound_count += 1
            
            bottleneck_data.append([compute_bound_count, memory_bound_count])
        
        bottleneck_array = np.array(bottleneck_data)
        
        x = np.arange(len(csd_names))
        width = 0.35
        
        bars1 = ax3.bar(x, bottleneck_array[:, 0], width, label='Compute Bound', 
                       color='#3498DB', alpha=0.8)
        bars2 = ax3.bar(x, bottleneck_array[:, 1], width, bottom=bottleneck_array[:, 0],
                       label='Memory Bound', color='#E74C3C', alpha=0.8)
        
        ax3.set_xlabel('CSD Type')
        ax3.set_ylabel('Number of Operations')
        ax3.set_title('Compute vs Memory Bottlenecks')
        ax3.set_xticks(x)
        ax3.set_xticklabels([name.split()[0] for name in csd_names], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Overall suitability score
        suitability_scores = []
        
        for csd_name in csd_names:
            csd = self.csds[csd_name]
            total_score = 0
            
            for op_name in operation_names:
                result = self.calculate_operation_performance(csd, self.ml_operations[op_name])
                
                if result['can_execute']:
                    # Normalize throughput score (0-1)
                    max_throughput = max(performance_matrix[:, list(operation_names).index(op_name)])
                    if max_throughput > 0:
                        throughput_score = result['throughput_ops_per_sec'] / max_throughput
                    else:
                        throughput_score = 0
                    
                    # Power efficiency score
                    efficiency_score = min(1.0, csd.ops_per_watt / 100e9)  # Normalize to 100 GOPS/W
                    
                    # Cost efficiency score  
                    cost_score = min(1.0, 1000 / csd.cost_per_tb_usd)  # Normalize to $1000/TB
                    
                    # Combined score
                    operation_score = (throughput_score * 0.5 + efficiency_score * 0.3 + cost_score * 0.2)
                    total_score += operation_score
            
            suitability_scores.append(total_score)
        
        bars4 = ax4.bar(range(len(csd_names)), suitability_scores, 
                       color=['#E74C3C' if 'FPGA' in name else '#2ECC71' if 'ARM' in name else 
                             '#9B59B6' if 'ASIC' in name else '#BDC3C7' for name in csd_names], 
                       alpha=0.8)
        
        ax4.set_xlabel('CSD Type')
        ax4.set_ylabel('Suitability Score')
        ax4.set_title('Overall RAG Suitability Score')
        ax4.set_xticks(range(len(csd_names)))
        ax4.set_xticklabels([name.split()[0] for name in csd_names], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, score in zip(bars4, suitability_scores):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ml_operation_performance.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated ML operation performance: {self.output_dir}/ml_operation_performance.pdf")
    
    def plot_scaling_analysis(self):
        """Analyze how CSDs scale with workload size."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Test different batch sizes
        batch_sizes = [10, 100, 1000, 10000, 100000]
        
        # Select key CSDs for comparison
        key_csds = ['Samsung SmartSSD', 'SK Hynix CSD', 'Xilinx Alveo CSD', 'Future ARM CSD']
        encoding_operation = self.ml_operations['Query Encoding']
        
        # 1. Throughput scaling
        for csd_name in key_csds:
            throughputs = []
            for batch_size in batch_sizes:
                result = self.calculate_operation_performance(
                    self.csds[csd_name], encoding_operation, batch_size)
                throughputs.append(result['throughput_ops_per_sec'])
            
            ax1.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=6, label=csd_name.split()[0])
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (Operations/Second)')
        ax1.set_title('Throughput Scaling with Batch Size')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Latency scaling
        for csd_name in key_csds:
            latencies = []
            for batch_size in batch_sizes:
                result = self.calculate_operation_performance(
                    self.csds[csd_name], encoding_operation, batch_size)
                latency_per_op = result['execution_time_ms'] / batch_size
                latencies.append(latency_per_op)
            
            ax2.plot(batch_sizes, latencies, 'o-', linewidth=2, markersize=6, label=csd_name.split()[0])
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Latency per Operation (ms)')
        ax2.set_title('Per-Operation Latency vs Batch Size')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Power consumption scaling
        for csd_name in key_csds:
            power_consumptions = []
            for batch_size in batch_sizes:
                result = self.calculate_operation_performance(
                    self.csds[csd_name], encoding_operation, batch_size)
                # Power consumption during operation
                power = self.csds[csd_name].power_watts
                power_consumptions.append(power)
            
            ax3.plot(batch_sizes, power_consumptions, 'o-', linewidth=2, markersize=6, label=csd_name.split()[0])
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Power Consumption (W)')
        ax3.set_title('Power Consumption vs Batch Size')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency at different scales
        for csd_name in key_csds:
            efficiencies = []
            for batch_size in batch_sizes:
                result = self.calculate_operation_performance(
                    self.csds[csd_name], encoding_operation, batch_size)
                # Operations per second per watt
                efficiency = result['throughput_ops_per_sec'] / self.csds[csd_name].power_watts
                efficiencies.append(efficiency)
            
            ax4.plot(batch_sizes, efficiencies, 'o-', linewidth=2, markersize=6, label=csd_name.split()[0])
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Efficiency (Ops/Second/Watt)')
        ax4.set_title('Power Efficiency vs Batch Size')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/csd_scaling_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated CSD scaling analysis: {self.output_dir}/csd_scaling_analysis.pdf")
    
    def generate_summary_table(self):
        """Generate comprehensive CSD capabilities summary."""
        data = []
        encoding_op = self.ml_operations['Query Encoding']
        
        for csd_name, csd in self.csds.items():
            result = self.calculate_operation_performance(csd, encoding_op)
            
            data.append({
                'CSD Type': csd_name,
                'Compute Architecture': csd.compute_type,
                'Peak Performance (GOPS)': f"{csd.peak_ops_per_sec/1e9:.0f}" if csd.peak_ops_per_sec > 0 else "N/A",
                'Power (W)': f"{csd.power_watts:.0f}",
                'Power Efficiency (GOPS/W)': f"{csd.ops_per_watt/1e9:.1f}" if csd.peak_ops_per_sec > 0 else "N/A",
                'Storage (TB)': f"{csd.storage_capacity_tb:.1f}",
                'Memory BW (GB/s)': f"{csd.memory_bandwidth_gbps:.1f}",
                'PCIe BW (GB/s)': f"{csd.pcie_bandwidth_gbps:.1f}",
                'Cost/TB ($)': f"{csd.cost_per_tb_usd:,}",
                'Encoding Throughput (ops/s)': f"{result['throughput_ops_per_sec']:.0f}" if result['can_execute'] else "N/A",
                'ML Optimized': "Yes" if csd.ml_optimized else "No"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{self.output_dir}/csd_capabilities_summary.csv', index=False)
        
        # Create formatted table
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 2)
        
        # Color-code by architecture type
        for i in range(len(df)):
            arch_type = df.iloc[i, 1]  # Compute Architecture column
            if arch_type == 'FPGA':
                color = '#FFCDD2'  # Light red
            elif arch_type == 'ARM':
                color = '#C8E6C9'  # Light green
            elif arch_type == 'ASIC':
                color = '#E1BEE7'  # Light purple
            else:
                color = '#F5F5F5'  # Light gray
            
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(color)
        
        plt.title('Modern CSD Capabilities for ML Workloads', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/csd_capabilities_table.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated capabilities table: {self.output_dir}/csd_capabilities_table.pdf")
        print(f"✅ Generated CSV data: {self.output_dir}/csd_capabilities_summary.csv")
    
    def run_full_analysis(self):
        """Run complete CSD capabilities analysis."""
        print("🔍 Starting Modern CSD Capabilities Analysis...")
        print("=" * 60)
        
        self.plot_csd_capabilities_overview()
        self.plot_ml_operation_performance()
        self.plot_scaling_analysis()
        self.generate_summary_table()
        
        print("\n✅ CSD Capabilities Analysis Complete!")
        print(f"📊 All plots saved to: {self.output_dir}/")
        print("📈 Generated plots:")
        print("   - csd_capabilities_overview.pdf")
        print("   - ml_operation_performance.pdf")
        print("   - csd_scaling_analysis.pdf")
        print("   - csd_capabilities_table.pdf")
        print("   - csd_capabilities_summary.csv")

def main():
    """Main function to run the analysis."""
    analyzer = CSDCapabilityAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()