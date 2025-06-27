#!/usr/bin/env python3
"""
Computational Capability Comparison: GPU vs ARM Cores for RAG Encoding

This script analyzes and compares computational requirements and efficiency
of different hardware platforms for RAG encoding operations.
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
class ProcessorSpecs:
    """Specifications for different processor types."""
    name: str
    peak_ops_per_sec: float  # Operations per second
    power_watts: float
    memory_bandwidth_gbps: float
    cost_usd: int
    precision: str
    vector_width: int
    frequency_mhz: int
    
    @property
    def ops_per_watt(self) -> float:
        return self.peak_ops_per_sec / self.power_watts
    
    @property
    def ops_per_dollar(self) -> float:
        return self.peak_ops_per_sec / self.cost_usd

@dataclass
class EncodingWorkload:
    """Characteristics of different encoding workloads."""
    name: str
    model_parameters: int
    sequence_length: int
    batch_size: int
    embedding_dim: int
    
    @property
    def total_flops(self) -> int:
        # Simplified FLOPS calculation for transformer encoding
        # Attention: O(seq_len^2 * embed_dim) + Linear: O(seq_len * embed_dim * params)
        attention_flops = self.sequence_length ** 2 * self.embedding_dim * self.batch_size
        linear_flops = self.sequence_length * self.embedding_dim * self.model_parameters * self.batch_size
        return attention_flops + linear_flops

class ComputationalAnalyzer:
    """Analyze computational requirements and efficiency across hardware platforms."""
    
    def __init__(self):
        # Hardware specifications based on real products
        self.processors = {
            'A100 GPU': ProcessorSpecs(
                name='NVIDIA A100 (40GB)',
                peak_ops_per_sec=19.5e12,  # 19.5 TFLOPS FP32
                power_watts=300,
                memory_bandwidth_gbps=1555,  # HBM2
                cost_usd=11000,
                precision='fp32',
                vector_width=32,
                frequency_mhz=1410
            ),
            'V100 GPU': ProcessorSpecs(
                name='NVIDIA V100 (32GB)',
                peak_ops_per_sec=14.0e12,  # 14 TFLOPS FP32
                power_watts=250,
                memory_bandwidth_gbps=900,  # HBM2
                cost_usd=8000,
                precision='fp32',
                vector_width=32,
                frequency_mhz=1380
            ),
            'RTX 4090': ProcessorSpecs(
                name='NVIDIA RTX 4090',
                peak_ops_per_sec=83.0e12,  # 83 TFLOPS FP32 (shader ALU)
                power_watts=450,
                memory_bandwidth_gbps=1008,  # GDDR6X
                cost_usd=1600,
                precision='fp32',
                vector_width=32,
                frequency_mhz=2230
            ),
            'ARM Cortex-A78': ProcessorSpecs(
                name='ARM Cortex-A78 (8-core)',
                peak_ops_per_sec=420e9,  # 420 GOPS with NEON
                power_watts=5,
                memory_bandwidth_gbps=25.6,  # LPDDR5
                cost_usd=200,
                precision='fp32',
                vector_width=8,  # NEON 128-bit
                frequency_mhz=3000
            ),
            'ARM Cortex-A76': ProcessorSpecs(
                name='ARM Cortex-A76 (4-core)',
                peak_ops_per_sec=200e9,  # 200 GOPS with NEON
                power_watts=3,
                memory_bandwidth_gbps=17.0,  # LPDDR4X
                cost_usd=150,
                precision='fp32',
                vector_width=8,
                frequency_mhz=2600
            ),
            'Apple M2': ProcessorSpecs(
                name='Apple M2 (8+4 cores)',
                peak_ops_per_sec=800e9,  # 800 GOPS estimated
                power_watts=15,
                memory_bandwidth_gbps=100,  # Unified memory
                cost_usd=400,
                precision='fp32',
                vector_width=16,  # Advanced SIMD
                frequency_mhz=3500
            ),
            'Intel Xeon': ProcessorSpecs(
                name='Intel Xeon Platinum 8380',
                peak_ops_per_sec=1.2e12,  # 1.2 TFLOPS with AVX-512
                power_watts=270,
                memory_bandwidth_gbps=204,  # DDR4-3200
                cost_usd=8000,
                precision='fp32',
                vector_width=16,  # AVX-512
                frequency_mhz=2300
            ),
            'AMD EPYC': ProcessorSpecs(
                name='AMD EPYC 7763 (64-core)',
                peak_ops_per_sec=2.0e12,  # 2.0 TFLOPS estimated
                power_watts=280,
                memory_bandwidth_gbps=204,  # DDR4-3200
                cost_usd=7000,
                precision='fp32',
                vector_width=8,  # AVX2
                frequency_mhz=2450
            )
        }
        
        # Encoding workloads (different model sizes and batch sizes)
        self.workloads = {
            'Light Encoding': EncodingWorkload(
                name='Light Encoding (E5-small)',
                model_parameters=33_000_000,    # 33M parameters
                sequence_length=128,
                batch_size=32,
                embedding_dim=384
            ),
            'Medium Encoding': EncodingWorkload(
                name='Medium Encoding (E5-base)',
                model_parameters=110_000_000,   # 110M parameters
                sequence_length=256,
                batch_size=64,
                embedding_dim=768
            ),
            'Heavy Encoding': EncodingWorkload(
                name='Heavy Encoding (E5-large)',
                model_parameters=335_000_000,   # 335M parameters
                sequence_length=512,
                batch_size=128,
                embedding_dim=1024
            ),
            'Batch Encoding': EncodingWorkload(
                name='Batch Encoding (High Throughput)',
                model_parameters=110_000_000,   # E5-base
                sequence_length=128,
                batch_size=1024,               # Large batch
                embedding_dim=768
            )
        }
        
        self.output_dir = "draft/sections/motivation/figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_encoding_performance(self, processor: ProcessorSpecs, workload: EncodingWorkload) -> Dict:
        """Calculate encoding performance for a specific processor and workload."""
        
        # Calculate execution time
        execution_time_s = workload.total_flops / processor.peak_ops_per_sec
        
        # Memory requirements
        model_memory_gb = workload.model_parameters * 4 / (1024**3)  # FP32 weights
        activation_memory_gb = (workload.batch_size * workload.sequence_length * 
                               workload.embedding_dim * 4) / (1024**3)
        total_memory_gb = model_memory_gb + activation_memory_gb
        
        # Memory bandwidth utilization
        memory_transfer_time_s = total_memory_gb / processor.memory_bandwidth_gbps
        
        # Actual execution time (max of compute and memory bound)
        actual_execution_time_s = max(execution_time_s, memory_transfer_time_s)
        
        # Energy consumption
        energy_joules = processor.power_watts * actual_execution_time_s
        
        # Throughput calculations
        queries_per_second = workload.batch_size / actual_execution_time_s
        
        # Efficiency metrics
        compute_utilization = execution_time_s / actual_execution_time_s
        memory_utilization = memory_transfer_time_s / actual_execution_time_s
        
        return {
            'execution_time_ms': actual_execution_time_s * 1000,
            'energy_joules': energy_joules,
            'queries_per_second': queries_per_second,
            'compute_utilization': compute_utilization,
            'memory_utilization': memory_utilization,
            'memory_requirement_gb': total_memory_gb,
            'is_memory_bound': memory_transfer_time_s > execution_time_s,
            'flops_per_second': workload.total_flops / actual_execution_time_s,
            'cost_per_million_queries': (processor.cost_usd / 
                                       (queries_per_second * 3600 * 24 * 365)) * 1e6  # Annual amortization
        }
    
    def plot_performance_comparison(self):
        """Plot performance comparison across processors and workloads."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        workload_names = list(self.workloads.keys())
        processor_names = list(self.processors.keys())
        
        # Performance matrix (queries per second)
        performance_matrix = np.zeros((len(processor_names), len(workload_names)))
        energy_matrix = np.zeros((len(processor_names), len(workload_names)))
        
        for i, proc_name in enumerate(processor_names):
            for j, workload_name in enumerate(workload_names):
                result = self.calculate_encoding_performance(
                    self.processors[proc_name], self.workloads[workload_name])
                performance_matrix[i, j] = result['queries_per_second']
                energy_matrix[i, j] = result['energy_joules']
        
        # 1. Performance heatmap
        im1 = ax1.imshow(performance_matrix, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(workload_names)))
        ax1.set_xticklabels([w.replace(' ', '\n') for w in workload_names], rotation=0)
        ax1.set_yticks(range(len(processor_names)))
        ax1.set_yticklabels([p.replace(' ', '\n') for p in processor_names])
        ax1.set_title('Throughput (Queries/Second)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(processor_names)):
            for j in range(len(workload_names)):
                text = ax1.text(j, i, f'{performance_matrix[i, j]:.0f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. Energy efficiency comparison
        efficiency_matrix = performance_matrix / energy_matrix  # queries per joule
        im2 = ax2.imshow(efficiency_matrix, cmap='plasma', aspect='auto')
        ax2.set_xticks(range(len(workload_names)))
        ax2.set_xticklabels([w.replace(' ', '\n') for w in workload_names], rotation=0)
        ax2.set_yticks(range(len(processor_names)))
        ax2.set_yticklabels([p.replace(' ', '\n') for p in processor_names])
        ax2.set_title('Energy Efficiency (Queries/Joule)', fontweight='bold')
        
        for i in range(len(processor_names)):
            for j in range(len(workload_names)):
                text = ax2.text(j, i, f'{efficiency_matrix[i, j]:.1f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 3. Performance per dollar
        costs = [self.processors[name].cost_usd for name in processor_names]
        cost_matrix = np.array(costs).reshape(-1, 1)
        cost_efficiency_matrix = performance_matrix / cost_matrix
        
        im3 = ax3.imshow(cost_efficiency_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_xticks(range(len(workload_names)))
        ax3.set_xticklabels([w.replace(' ', '\n') for w in workload_names], rotation=0)
        ax3.set_yticks(range(len(processor_names)))
        ax3.set_yticklabels([p.replace(' ', '\n') for p in processor_names])
        ax3.set_title('Cost Efficiency (Queries/Second/$)', fontweight='bold')
        
        for i in range(len(processor_names)):
            for j in range(len(workload_names)):
                text = ax3.text(j, i, f'{cost_efficiency_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # 4. Power efficiency summary
        processors = list(self.processors.values())
        power_efficiency = [p.ops_per_watt / 1e9 for p in processors]  # GOPS/W
        absolute_performance = [p.peak_ops_per_sec / 1e12 for p in processors]  # TOPS
        
        colors = ['red' if 'GPU' in p.name else 'blue' if 'ARM' in p.name else 'green' 
                 for p in processors]
        
        scatter = ax4.scatter(power_efficiency, absolute_performance, 
                            c=colors, s=100, alpha=0.7)
        
        for i, proc in enumerate(processors):
            ax4.annotate(proc.name.split()[0], 
                        (power_efficiency[i], absolute_performance[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Power Efficiency (GOPS/W)')
        ax4.set_ylabel('Peak Performance (TOPS)')
        ax4.set_title('Performance vs Power Efficiency', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        # Add legend
        gpu_patch = mpatches.Patch(color='red', label='GPU')
        arm_patch = mpatches.Patch(color='blue', label='ARM')
        cpu_patch = mpatches.Patch(color='green', label='CPU')
        ax4.legend(handles=[gpu_patch, arm_patch, cpu_patch])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/computational_comparison.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated computational comparison: {self.output_dir}/computational_comparison.pdf")
    
    def plot_workload_analysis(self):
        """Analyze how different workloads affect processor efficiency."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Focus on key processors for encoding
        key_processors = ['A100 GPU', 'ARM Cortex-A78', 'Apple M2', 'Intel Xeon']
        workload_names = list(self.workloads.keys())
        
        # Calculate utilization patterns
        compute_utilizations = {proc: [] for proc in key_processors}
        memory_utilizations = {proc: [] for proc in key_processors}
        actual_performance = {proc: [] for proc in key_processors}
        
        for workload_name in workload_names:
            workload = self.workloads[workload_name]
            for proc_name in key_processors:
                processor = self.processors[proc_name]
                result = self.calculate_encoding_performance(processor, workload)
                
                compute_utilizations[proc_name].append(result['compute_utilization'])
                memory_utilizations[proc_name].append(result['memory_utilization'])
                actual_performance[proc_name].append(result['queries_per_second'])
        
        # Plot utilization patterns
        x = np.arange(len(workload_names))
        width = 0.2
        
        for i, proc_name in enumerate(key_processors):
            offset = (i - 1.5) * width
            compute_bars = ax1.bar(x + offset, compute_utilizations[proc_name], 
                                 width, alpha=0.8, label=f'{proc_name} (Compute)')
            memory_bars = ax1.bar(x + offset, memory_utilizations[proc_name], 
                                width, bottom=compute_utilizations[proc_name], 
                                alpha=0.6, label=f'{proc_name} (Memory)')
        
        ax1.set_xlabel('Workload Type')
        ax1.set_ylabel('Resource Utilization')
        ax1.set_title('Compute vs Memory Utilization')
        ax1.set_xticks(x)
        ax1.set_xticklabels([w.replace(' ', '\n') for w in workload_names])
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot actual performance comparison
        for i, proc_name in enumerate(key_processors):
            ax2.plot(workload_names, actual_performance[proc_name], 
                    'o-', linewidth=2, markersize=8, label=proc_name)
        
        ax2.set_xlabel('Workload Type')
        ax2.set_ylabel('Throughput (Queries/Second)')
        ax2.set_title('Throughput Across Workloads')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/workload_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated workload analysis: {self.output_dir}/workload_analysis.pdf")
    
    def plot_efficiency_breakdown(self):
        """Break down efficiency metrics across different dimensions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        processors = list(self.processors.values())
        medium_workload = self.workloads['Medium Encoding']
        
        # 1. Operations per Watt comparison
        ops_per_watt = [p.ops_per_watt / 1e9 for p in processors]  # GOPS/W
        processor_names = [p.name for p in processors]
        
        colors = ['#E74C3C' if 'GPU' in name else '#2ECC71' if 'ARM' in name else '#3498DB' 
                 for name in processor_names]
        
        bars1 = ax1.bar(range(len(processor_names)), ops_per_watt, color=colors, alpha=0.8)
        ax1.set_xlabel('Processor')
        ax1.set_ylabel('Power Efficiency (GOPS/W)')
        ax1.set_title('Power Efficiency Comparison')
        ax1.set_xticks(range(len(processor_names)))
        ax1.set_xticklabels([name.split()[0] for name in processor_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, ops_per_watt):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Operations per Dollar
        ops_per_dollar = [p.ops_per_dollar / 1e9 for p in processors]  # GOPS/$
        
        bars2 = ax2.bar(range(len(processor_names)), ops_per_dollar, color=colors, alpha=0.8)
        ax2.set_xlabel('Processor')
        ax2.set_ylabel('Cost Efficiency (GOPS/$)')
        ax2.set_title('Cost Efficiency Comparison')
        ax2.set_xticks(range(len(processor_names)))
        ax2.set_xticklabels([name.split()[0] for name in processor_names], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, ops_per_dollar):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory Bandwidth vs Compute Performance
        memory_bw = [p.memory_bandwidth_gbps for p in processors]
        compute_perf = [p.peak_ops_per_sec / 1e12 for p in processors]
        
        scatter = ax3.scatter(memory_bw, compute_perf, c=colors, s=100, alpha=0.7)
        
        for i, proc in enumerate(processors):
            ax3.annotate(proc.name.split()[0], 
                        (memory_bw[i], compute_perf[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Memory Bandwidth (GB/s)')
        ax3.set_ylabel('Peak Performance (TOPS)')
        ax3.set_title('Memory Bandwidth vs Compute Performance')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # 4. Real-world encoding performance
        encoding_performance = []
        encoding_efficiency = []
        
        for proc in processors:
            result = self.calculate_encoding_performance(proc, medium_workload)
            encoding_performance.append(result['queries_per_second'])
            encoding_efficiency.append(result['queries_per_second'] / proc.power_watts)
        
        bars4 = ax4.bar(range(len(processor_names)), encoding_efficiency, color=colors, alpha=0.8)
        ax4.set_xlabel('Processor')
        ax4.set_ylabel('Encoding Efficiency (Queries/Second/Watt)')
        ax4.set_title('Real-World Encoding Efficiency')
        ax4.set_xticks(range(len(processor_names)))
        ax4.set_xticklabels([name.split()[0] for name in processor_names], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, encoding_efficiency):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/efficiency_breakdown.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated efficiency breakdown: {self.output_dir}/efficiency_breakdown.pdf")
    
    def generate_summary_table(self):
        """Generate comprehensive comparison table."""
        data = []
        medium_workload = self.workloads['Medium Encoding']
        
        for proc_name, processor in self.processors.items():
            result = self.calculate_encoding_performance(processor, medium_workload)
            
            data.append({
                'Processor': proc_name,
                'Peak Performance (TOPS)': f"{processor.peak_ops_per_sec/1e12:.1f}",
                'Power (W)': f"{processor.power_watts:.0f}",
                'Power Efficiency (GOPS/W)': f"{processor.ops_per_watt/1e9:.1f}",
                'Cost ($)': f"{processor.cost_usd:,}",
                'Cost Efficiency (GOPS/$)': f"{processor.ops_per_dollar/1e9:.2f}",
                'Encoding Throughput (Q/s)': f"{result['queries_per_second']:.0f}",
                'Encoding Latency (ms)': f"{result['execution_time_ms']:.1f}",
                'Energy per Query (J)': f"{result['energy_joules']:.3f}",
                'Memory Bound': "Yes" if result['is_memory_bound'] else "No"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{self.output_dir}/computational_comparison.csv', index=False)
        
        # Create formatted table
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 2)
        
        # Color-code by processor type
        for i in range(len(df)):
            proc_name = df.iloc[i, 0]
            if 'GPU' in proc_name:
                color = '#FFCDD2'  # Light red
            elif 'ARM' in proc_name:
                color = '#C8E6C9'  # Light green
            else:
                color = '#E1F5FE'  # Light blue
            
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(color)
        
        plt.title('Computational Capability Comparison for RAG Encoding', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/computational_comparison_table.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated comparison table: {self.output_dir}/computational_comparison_table.pdf")
        print(f"✅ Generated CSV data: {self.output_dir}/computational_comparison.csv")
    
    def run_full_analysis(self):
        """Run complete computational capability analysis."""
        print("🔍 Starting Computational Capability Analysis...")
        print("=" * 60)
        
        self.plot_performance_comparison()
        self.plot_workload_analysis()
        self.plot_efficiency_breakdown()
        self.generate_summary_table()
        
        print("\n✅ Computational Analysis Complete!")
        print(f"📊 All plots saved to: {self.output_dir}/")
        print("📈 Generated plots:")
        print("   - computational_comparison.pdf")
        print("   - workload_analysis.pdf")
        print("   - efficiency_breakdown.pdf")
        print("   - computational_comparison_table.pdf")
        print("   - computational_comparison.csv")

def main():
    """Main function to run the analysis."""
    analyzer = ComputationalAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()