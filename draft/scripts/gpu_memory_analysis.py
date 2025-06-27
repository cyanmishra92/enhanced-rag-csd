#!/usr/bin/env python3
"""
GPU Memory Swapping Analysis for Classical RAG Systems

This script analyzes GPU memory usage patterns, swapping overhead, and fragmentation
in classical RAG systems compared to CSD-enhanced approaches.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass
import os

# Set style for academic plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

@dataclass
class SystemConfig:
    """Configuration for different RAG system architectures."""
    name: str
    llm_model_size_gb: float
    encoder_model_size_gb: float
    vector_db_size_gb: float
    kv_cache_size_gb: float
    total_gpu_memory_gb: float
    swapping_overhead_ms: float
    memory_efficiency: float  # 0.0 to 1.0

class RAGMemoryAnalyzer:
    """Analyze memory usage patterns in different RAG system configurations."""
    
    def __init__(self):
        self.systems = {
            'Classical RAG': SystemConfig(
                name='Classical RAG',
                llm_model_size_gb=16.0,
                encoder_model_size_gb=0.44,
                vector_db_size_gb=1.5,
                kv_cache_size_gb=4.0,
                total_gpu_memory_gb=40.0,
                swapping_overhead_ms=350.0,
                memory_efficiency=0.70
            ),
            'PipeRAG': SystemConfig(
                name='PipeRAG',
                llm_model_size_gb=16.0,
                encoder_model_size_gb=0.44,
                vector_db_size_gb=1.5,
                kv_cache_size_gb=4.0,
                total_gpu_memory_gb=40.0,
                swapping_overhead_ms=200.0,
                memory_efficiency=0.75
            ),
            'FlashRAG': SystemConfig(
                name='FlashRAG',
                llm_model_size_gb=16.0,
                encoder_model_size_gb=0.44,
                vector_db_size_gb=1.5,
                kv_cache_size_gb=3.0,
                total_gpu_memory_gb=40.0,
                swapping_overhead_ms=150.0,
                memory_efficiency=0.80
            ),
            'EdgeRAG': SystemConfig(
                name='EdgeRAG',
                llm_model_size_gb=8.0,  # Pruned model
                encoder_model_size_gb=0.22,  # Pruned encoder
                vector_db_size_gb=0.5,  # Pruned vectors
                kv_cache_size_gb=1.5,
                total_gpu_memory_gb=16.0,
                swapping_overhead_ms=100.0,
                memory_efficiency=0.85
            ),
            'CSD-Enhanced RAG': SystemConfig(
                name='CSD-Enhanced RAG',
                llm_model_size_gb=16.0,
                encoder_model_size_gb=0.0,  # Moved to CSD
                vector_db_size_gb=0.0,  # Moved to CSD
                kv_cache_size_gb=4.0,
                total_gpu_memory_gb=40.0,
                swapping_overhead_ms=50.0,
                memory_efficiency=0.95
            )
        }
        
        self.output_dir = "draft/figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_memory_timeline(self, system_name: str, duration_seconds: int = 60) -> Dict:
        """Simulate memory usage over time for a given system."""
        system = self.systems[system_name]
        timesteps = np.linspace(0, duration_seconds, duration_seconds * 10)
        
        # Simulate different phases of RAG pipeline
        memory_usage = []
        phases = []
        
        for t in timesteps:
            # Simulate cyclical RAG operations
            cycle_time = t % 10  # 10-second cycles
            
            if cycle_time < 2:  # Encoding phase
                if system_name == 'CSD-Enhanced RAG':
                    # No encoder on GPU
                    usage = system.llm_model_size_gb + system.kv_cache_size_gb * 0.3
                else:
                    usage = system.encoder_model_size_gb + system.vector_db_size_gb * 0.5
                phases.append('Encoding')
            elif cycle_time < 4:  # Retrieval phase
                if system_name == 'CSD-Enhanced RAG':
                    # No vector DB on GPU
                    usage = system.llm_model_size_gb + system.kv_cache_size_gb * 0.3
                else:
                    usage = system.vector_db_size_gb + system.encoder_model_size_gb * 0.5
                phases.append('Retrieval')
            elif cycle_time < 8:  # Generation phase
                usage = system.llm_model_size_gb + system.kv_cache_size_gb
                phases.append('Generation')
            else:  # Transition/swapping phase
                if system_name == 'CSD-Enhanced RAG':
                    usage = system.llm_model_size_gb + system.kv_cache_size_gb * 0.5
                else:
                    # Simulate memory fragmentation during swapping
                    usage = (system.llm_model_size_gb + system.encoder_model_size_gb + 
                            system.vector_db_size_gb) * 0.7
                phases.append('Swapping')
            
            # Add noise and efficiency factors
            usage *= (1.0 + np.random.normal(0, 0.05))  # 5% noise
            usage /= system.memory_efficiency
            
            memory_usage.append(min(usage, system.total_gpu_memory_gb))
        
        return {
            'time': timesteps,
            'memory_usage': memory_usage,
            'phases': phases,
            'system': system
        }
    
    def plot_memory_timeline(self):
        """Plot memory usage timeline for different systems."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        
        for i, system_name in enumerate(self.systems.keys()):
            data = self.analyze_memory_timeline(system_name)
            ax = axes[i]
            
            # Plot memory usage
            ax.plot(data['time'], data['memory_usage'], color=colors[i], linewidth=2, alpha=0.8)
            ax.fill_between(data['time'], 0, data['memory_usage'], alpha=0.3, color=colors[i])
            
            # Add memory limit line
            ax.axhline(y=data['system'].total_gpu_memory_gb, color='red', 
                      linestyle='--', alpha=0.7, label='GPU Memory Limit')
            
            # Color-code phases
            phase_colors = {'Encoding': '#FFE4E1', 'Retrieval': '#E1F5FE', 
                           'Generation': '#E8F5E8', 'Swapping': '#FFF3E0'}
            
            for j in range(len(data['time']) - 1):
                if data['phases'][j] == 'Swapping':
                    ax.axvspan(data['time'][j], data['time'][j+1], 
                              alpha=0.2, color='red', zorder=0)
            
            ax.set_title(f'{system_name}\nSwapping Overhead: {data["system"].swapping_overhead_ms:.0f}ms', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('GPU Memory Usage (GB)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(40, data['system'].total_gpu_memory_gb * 1.1))
            
            # Add efficiency annotation
            ax.text(0.02, 0.98, f'Efficiency: {data["system"].memory_efficiency:.0%}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove empty subplot
        if len(self.systems) < 6:
            fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gpu_memory_timeline.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated GPU memory timeline: {self.output_dir}/gpu_memory_timeline.pdf")
    
    def plot_memory_breakdown(self):
        """Plot memory breakdown comparison across systems."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        systems = list(self.systems.keys())
        llm_sizes = [sys.llm_model_size_gb for sys in self.systems.values()]
        encoder_sizes = [sys.encoder_model_size_gb for sys in self.systems.values()]
        vector_sizes = [sys.vector_db_size_gb for sys in self.systems.values()]
        kv_sizes = [sys.kv_cache_size_gb for sys in self.systems.values()]
        
        # Stacked bar chart
        width = 0.6
        x = np.arange(len(systems))
        
        p1 = ax1.bar(x, llm_sizes, width, label='LLM Model', color='#E74C3C', alpha=0.8)
        p2 = ax1.bar(x, encoder_sizes, width, bottom=llm_sizes, label='Encoder Model', color='#3498DB', alpha=0.8)
        p3 = ax1.bar(x, vector_sizes, width, bottom=np.array(llm_sizes) + np.array(encoder_sizes), 
                    label='Vector Database', color='#2ECC71', alpha=0.8)
        p4 = ax1.bar(x, kv_sizes, width, 
                    bottom=np.array(llm_sizes) + np.array(encoder_sizes) + np.array(vector_sizes),
                    label='KV Cache', color='#F39C12', alpha=0.8)
        
        ax1.set_xlabel('RAG System Architecture')
        ax1.set_ylabel('GPU Memory Usage (GB)')
        ax1.set_title('GPU Memory Breakdown by Component')
        ax1.set_xticks(x)
        ax1.set_xticklabels(systems, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory efficiency comparison
        efficiencies = [sys.memory_efficiency for sys in self.systems.values()]
        swapping_overheads = [sys.swapping_overhead_ms for sys in self.systems.values()]
        
        # Normalize swapping overhead for color coding
        normalized_overheads = np.array(swapping_overheads) / max(swapping_overheads)
        colors = plt.cm.RdYlGn_r(normalized_overheads)
        
        bars = ax2.bar(x, efficiencies, width, color=colors, alpha=0.8)
        ax2.set_xlabel('RAG System Architecture')
        ax2.set_ylabel('Memory Efficiency')
        ax2.set_title('Memory Efficiency vs Swapping Overhead')
        ax2.set_xticks(x)
        ax2.set_xticklabels(systems, rotation=45, ha='right')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3)
        
        # Add overhead annotations
        for i, (bar, overhead) in enumerate(zip(bars, swapping_overheads)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{overhead:.0f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/memory_breakdown_comparison.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated memory breakdown: {self.output_dir}/memory_breakdown_comparison.pdf")
    
    def plot_swapping_overhead_analysis(self):
        """Analyze swapping overhead impact on query latency."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Query throughput vs swapping overhead
        query_rates = np.linspace(10, 1000, 100)  # queries per second
        
        systems_data = []
        for system_name, system in self.systems.items():
            # Calculate effective throughput considering swapping overhead
            swapping_delay = system.swapping_overhead_ms / 1000  # convert to seconds
            
            # Assume 20% of queries trigger model swapping
            swapping_frequency = 0.2
            effective_throughput = []
            
            for rate in query_rates:
                # Time spent on swapping per second
                swapping_time_per_sec = rate * swapping_frequency * swapping_delay
                
                # Effective throughput considering swapping overhead
                if swapping_time_per_sec < 1.0:
                    effective_rate = rate * (1.0 - swapping_time_per_sec)
                else:
                    effective_rate = 0  # System saturated
                
                effective_throughput.append(max(0, effective_rate))
            
            systems_data.append({
                'name': system_name,
                'query_rates': query_rates,
                'effective_throughput': effective_throughput,
                'swapping_overhead': system.swapping_overhead_ms
            })
        
        # Plot throughput curves
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        for i, data in enumerate(systems_data):
            ax1.plot(data['query_rates'], data['effective_throughput'], 
                    color=colors[i], linewidth=2, label=data['name'])
        
        ax1.set_xlabel('Target Query Rate (queries/sec)')
        ax1.set_ylabel('Effective Throughput (queries/sec)')
        ax1.set_title('Query Throughput vs Swapping Overhead')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1000)
        
        # Plot swapping overhead impact
        system_names = [data['name'] for data in systems_data]
        overheads = [data['swapping_overhead'] for data in systems_data]
        
        # Calculate throughput degradation at 500 queries/sec
        degradations = []
        for data in systems_data:
            idx = np.argmin(np.abs(np.array(data['query_rates']) - 500))
            degradation = (500 - data['effective_throughput'][idx]) / 500 * 100
            degradations.append(max(0, degradation))
        
        bars = ax2.bar(range(len(system_names)), degradations, color=colors, alpha=0.8)
        ax2.set_xlabel('RAG System Architecture')
        ax2.set_ylabel('Throughput Degradation (%)')
        ax2.set_title('Throughput Degradation at 500 QPS')
        ax2.set_xticks(range(len(system_names)))
        ax2.set_xticklabels(system_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add overhead annotations
        for i, (bar, overhead) in enumerate(zip(bars, overheads)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{overhead:.0f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/swapping_overhead_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated swapping overhead analysis: {self.output_dir}/swapping_overhead_analysis.pdf")
    
    def generate_summary_table(self):
        """Generate summary table of system characteristics."""
        data = []
        for system_name, system in self.systems.items():
            total_memory = (system.llm_model_size_gb + system.encoder_model_size_gb + 
                          system.vector_db_size_gb + system.kv_cache_size_gb)
            
            # Calculate memory fragmentation
            effective_memory = total_memory / system.memory_efficiency
            fragmentation = (effective_memory - total_memory) / total_memory * 100
            
            data.append({
                'System': system_name,
                'LLM Model (GB)': f"{system.llm_model_size_gb:.1f}",
                'Encoder (GB)': f"{system.encoder_model_size_gb:.2f}",
                'Vector DB (GB)': f"{system.vector_db_size_gb:.1f}",
                'KV Cache (GB)': f"{system.kv_cache_size_gb:.1f}",
                'Total Memory (GB)': f"{total_memory:.1f}",
                'Fragmentation (%)': f"{fragmentation:.1f}",
                'Swapping Overhead (ms)': f"{system.swapping_overhead_ms:.0f}",
                'Memory Efficiency (%)': f"{system.memory_efficiency:.0%}"
            })
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        df.to_csv(f'{self.output_dir}/gpu_memory_summary.csv', index=False)
        
        # Create formatted table plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Color-code by performance
        for i in range(len(df)):
            for j in range(len(df.columns)):
                if df.columns[j] == 'Swapping Overhead (ms)':
                    # Red for high overhead, green for low
                    overhead = float(df.iloc[i, j])
                    if overhead > 200:
                        table[(i+1, j)].set_facecolor('#FFCDD2')
                    elif overhead < 100:
                        table[(i+1, j)].set_facecolor('#C8E6C9')
                elif df.columns[j] == 'Memory Efficiency (%)':
                    # Green for high efficiency, red for low
                    efficiency = float(df.iloc[i, j].replace('%', ''))
                    if efficiency > 90:
                        table[(i+1, j)].set_facecolor('#C8E6C9')
                    elif efficiency < 75:
                        table[(i+1, j)].set_facecolor('#FFCDD2')
        
        plt.title('GPU Memory Usage Summary Across RAG Systems', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/gpu_memory_summary_table.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated summary table: {self.output_dir}/gpu_memory_summary_table.pdf")
        print(f"✅ Generated CSV data: {self.output_dir}/gpu_memory_summary.csv")
    
    def run_full_analysis(self):
        """Run complete GPU memory analysis."""
        print("🔍 Starting GPU Memory Swapping Analysis...")
        print("=" * 60)
        
        self.plot_memory_timeline()
        self.plot_memory_breakdown()
        self.plot_swapping_overhead_analysis()
        self.generate_summary_table()
        
        print("\n✅ GPU Memory Analysis Complete!")
        print(f"📊 All plots saved to: {self.output_dir}/")
        print("📈 Generated plots:")
        print("   - gpu_memory_timeline.pdf")
        print("   - memory_breakdown_comparison.pdf") 
        print("   - swapping_overhead_analysis.pdf")
        print("   - gpu_memory_summary_table.pdf")
        print("   - gpu_memory_summary.csv")

def main():
    """Main function to run the analysis."""
    analyzer = RAGMemoryAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()