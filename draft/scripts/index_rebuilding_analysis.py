#!/usr/bin/env python3
"""
Index Rebuilding Cost Analysis for Dynamic Document Addition

This script analyzes the costs and performance implications of index rebuilding
in traditional RAG systems versus CSD-enhanced approaches with dynamic updates.
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
from datetime import datetime, timedelta

# Set style for academic plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

@dataclass
class IndexSpecs:
    """Specifications for vector database index."""
    name: str
    vector_count: int
    vector_dimension: int
    index_type: str  # 'IVF', 'HNSW', 'Flat', 'LSH'
    build_time_hours: float
    memory_requirement_gb: float
    search_latency_ms: float
    update_supported: bool

@dataclass
class SystemConfig:
    """Configuration for different RAG system approaches."""
    name: str
    rebuild_frequency_days: int
    service_downtime_minutes: float
    rebuild_cost_per_hour: float
    search_degradation_percent: float
    supports_online_updates: bool
    update_latency_ms: float

class IndexRebuildingAnalyzer:
    """Analyze index rebuilding costs and dynamic update capabilities."""
    
    def __init__(self):
        # Different index configurations
        self.indices = {
            'Small Index': IndexSpecs(
                name='Small Dataset (100K vectors)',
                vector_count=100_000,
                vector_dimension=384,
                index_type='IVF',
                build_time_hours=0.5,
                memory_requirement_gb=0.15,
                search_latency_ms=2.0,
                update_supported=False
            ),
            'Medium Index': IndexSpecs(
                name='Medium Dataset (1M vectors)',
                vector_count=1_000_000,
                vector_dimension=768,
                index_type='IVF',
                build_time_hours=2.0,
                memory_requirement_gb=3.0,
                search_latency_ms=5.0,
                update_supported=False
            ),
            'Large Index': IndexSpecs(
                name='Large Dataset (10M vectors)',
                vector_count=10_000_000,
                vector_dimension=1024,
                index_type='HNSW',
                build_time_hours=8.0,
                memory_requirement_gb=40.0,
                search_latency_ms=12.0,
                update_supported=True
            ),
            'Enterprise Index': IndexSpecs(
                name='Enterprise Dataset (100M vectors)',
                vector_count=100_000_000,
                vector_dimension=1024,
                index_type='HNSW',
                build_time_hours=24.0,
                memory_requirement_gb=400.0,
                search_latency_ms=25.0,
                update_supported=True
            )
        }
        
        # System configurations
        self.systems = {
            'Traditional RAG': SystemConfig(
                name='Traditional RAG (Batch Rebuild)',
                rebuild_frequency_days=7,
                service_downtime_minutes=30,
                rebuild_cost_per_hour=32,  # A100 GPU cost
                search_degradation_percent=25,
                supports_online_updates=False,
                update_latency_ms=0
            ),
            'FlashRAG': SystemConfig(
                name='FlashRAG (Optimized Rebuild)',
                rebuild_frequency_days=7,
                service_downtime_minutes=15,
                rebuild_cost_per_hour=32,
                search_degradation_percent=15,
                supports_online_updates=False,
                update_latency_ms=0
            ),
            'EdgeRAG': SystemConfig(
                name='EdgeRAG (Pruned Index)',
                rebuild_frequency_days=14,  # Less frequent due to smaller index
                service_downtime_minutes=5,
                rebuild_cost_per_hour=8,   # Edge hardware cost
                search_degradation_percent=10,
                supports_online_updates=True,
                update_latency_ms=50
            ),
            'CSD-Enhanced RAG': SystemConfig(
                name='CSD-Enhanced RAG (Online Updates)',
                rebuild_frequency_days=365,  # Annual full rebuild
                service_downtime_minutes=0,  # No downtime
                rebuild_cost_per_hour=5,   # CSD compute cost
                search_degradation_percent=0,
                supports_online_updates=True,
                update_latency_ms=10
            )
        }
        
        self.output_dir = "draft/figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_annual_costs(self, index: IndexSpecs, system: SystemConfig) -> Dict:
        """Calculate annual costs for index maintenance."""
        
        # Number of rebuilds per year
        rebuilds_per_year = 365 / system.rebuild_frequency_days
        
        # Compute costs
        annual_compute_cost = (rebuilds_per_year * index.build_time_hours * 
                              system.rebuild_cost_per_hour)
        
        # Service degradation costs (assume $1000/hour revenue impact)
        revenue_per_hour = 1000
        downtime_cost_per_rebuild = (system.service_downtime_minutes / 60) * revenue_per_hour
        
        # Search quality degradation cost (between rebuilds)
        # Assume gradual degradation affects revenue
        days_between_rebuilds = system.rebuild_frequency_days
        avg_degradation = system.search_degradation_percent / 2  # Average over time
        degradation_cost_per_rebuild = (days_between_rebuilds * 24 * revenue_per_hour * 
                                       avg_degradation / 100)
        
        annual_downtime_cost = rebuilds_per_year * downtime_cost_per_rebuild
        annual_degradation_cost = rebuilds_per_year * degradation_cost_per_rebuild
        
        # Memory costs (continuous)
        memory_cost_per_gb_per_year = 50  # Cloud memory cost
        annual_memory_cost = index.memory_requirement_gb * memory_cost_per_gb_per_year
        
        # Total cost
        total_annual_cost = (annual_compute_cost + annual_downtime_cost + 
                           annual_degradation_cost + annual_memory_cost)
        
        return {
            'annual_compute_cost': annual_compute_cost,
            'annual_downtime_cost': annual_downtime_cost,
            'annual_degradation_cost': annual_degradation_cost,
            'annual_memory_cost': annual_memory_cost,
            'total_annual_cost': total_annual_cost,
            'rebuilds_per_year': rebuilds_per_year,
            'availability_percent': 100 - (annual_downtime_cost / (365 * 24 * revenue_per_hour) * 100)
        }
    
    def simulate_dynamic_updates(self, days: int = 365) -> Dict:
        """Simulate dynamic document addition and index updates."""
        
        # Document addition patterns (documents per day)
        patterns = {
            'Steady Growth': np.ones(days) * 1000,  # 1000 docs/day
            'Seasonal Spikes': 1000 + 500 * np.sin(np.linspace(0, 4*np.pi, days)),
            'Viral Content': np.concatenate([
                np.ones(300) * 1000,
                np.ones(30) * 10000,  # Viral spike
                np.ones(35) * 5000    # Gradual decline
            ]),
            'Enterprise Batch': np.tile(np.concatenate([
                np.ones(6) * 0,       # No updates for 6 days
                np.ones(1) * 50000,   # Weekly batch
            ]), 52)[:days]  # Repeat for 52 weeks, truncate to days
        }
        
        results = {}
        
        for pattern_name, daily_docs in patterns.items():
            # Traditional system (weekly rebuilds)
            traditional_costs = []
            traditional_quality = []
            traditional_docs_added = 0
            
            # CSD system (online updates)
            csd_costs = []
            csd_quality = []
            
            for day in range(min(days, len(daily_docs))):
                docs_today = daily_docs[day]
                
                # Traditional system
                if day % 7 == 0 and day > 0:  # Weekly rebuild
                    # Rebuild cost
                    rebuild_cost = 2 * 32  # 2 hours at $32/hour
                    traditional_costs.append(rebuild_cost)
                    traditional_quality.append(100)  # Full quality after rebuild
                    traditional_docs_added = 0
                else:
                    traditional_costs.append(0)
                    # Quality degrades as documents accumulate
                    quality_loss = min(25, traditional_docs_added * 0.001)  # 0.1% per 100 docs
                    traditional_quality.append(100 - quality_loss)
                
                traditional_docs_added += docs_today
                
                # CSD system (online updates)
                update_cost = docs_today * 0.0001  # $0.0001 per document update
                csd_costs.append(update_cost)
                csd_quality.append(99.5)  # Slightly reduced quality due to incremental updates
            
            results[pattern_name] = {
                'traditional_total_cost': sum(traditional_costs),
                'csd_total_cost': sum(csd_costs),
                'traditional_avg_quality': np.mean(traditional_quality),
                'csd_avg_quality': np.mean(csd_quality),
                'traditional_costs': traditional_costs,
                'csd_costs': csd_costs,
                'traditional_quality': traditional_quality,
                'csd_quality': csd_quality,
                'daily_docs': daily_docs
            }
        
        return results
    
    def plot_cost_comparison(self):
        """Plot cost comparison across different systems and index sizes."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        index_names = list(self.indices.keys())
        system_names = list(self.systems.keys())
        
        # Prepare cost matrices
        compute_costs = np.zeros((len(system_names), len(index_names)))
        total_costs = np.zeros((len(system_names), len(index_names)))
        availability = np.zeros((len(system_names), len(index_names)))
        
        for i, system_name in enumerate(system_names):
            for j, index_name in enumerate(index_names):
                costs = self.calculate_annual_costs(
                    self.indices[index_name], self.systems[system_name])
                
                compute_costs[i, j] = costs['annual_compute_cost']
                total_costs[i, j] = costs['total_annual_cost']
                availability[i, j] = costs['availability_percent']
        
        # 1. Annual compute costs
        im1 = ax1.imshow(compute_costs, cmap='Reds', aspect='auto')
        ax1.set_xticks(range(len(index_names)))
        ax1.set_xticklabels([name.replace(' ', '\n') for name in index_names])
        ax1.set_yticks(range(len(system_names)))
        ax1.set_yticklabels([name.replace(' ', '\n') for name in system_names])
        ax1.set_title('Annual Compute Costs ($)', fontweight='bold')
        
        for i in range(len(system_names)):
            for j in range(len(index_names)):
                text = ax1.text(j, i, f'${compute_costs[i, j]:,.0f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 2. Total annual costs
        im2 = ax2.imshow(total_costs, cmap='OrRd', aspect='auto')
        ax2.set_xticks(range(len(index_names)))
        ax2.set_xticklabels([name.replace(' ', '\n') for name in index_names])
        ax2.set_yticks(range(len(system_names)))
        ax2.set_yticklabels([name.replace(' ', '\n') for name in system_names])
        ax2.set_title('Total Annual Costs ($)', fontweight='bold')
        
        for i in range(len(system_names)):
            for j in range(len(index_names)):
                text = ax2.text(j, i, f'${total_costs[i, j]:,.0f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 3. System availability
        im3 = ax3.imshow(availability, cmap='Greens', aspect='auto')
        ax3.set_xticks(range(len(index_names)))
        ax3.set_xticklabels([name.replace(' ', '\n') for name in index_names])
        ax3.set_yticks(range(len(system_names)))
        ax3.set_yticklabels([name.replace(' ', '\n') for name in system_names])
        ax3.set_title('System Availability (%)', fontweight='bold')
        
        for i in range(len(system_names)):
            for j in range(len(index_names)):
                text = ax3.text(j, i, f'{availability[i, j]:.1f}%',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # 4. Cost savings with CSD
        traditional_costs = total_costs[0, :]  # Traditional RAG costs
        csd_costs = total_costs[3, :]  # CSD-Enhanced costs
        savings_percent = (traditional_costs - csd_costs) / traditional_costs * 100
        
        bars = ax4.bar(range(len(index_names)), savings_percent, 
                      color=['#2ECC71', '#3498DB', '#E74C3C', '#9B59B6'], alpha=0.8)
        
        ax4.set_xlabel('Index Size')
        ax4.set_ylabel('Cost Savings (%)')
        ax4.set_title('CSD Cost Savings vs Traditional RAG')
        ax4.set_xticks(range(len(index_names)))
        ax4.set_xticklabels([name.replace(' ', '\n') for name in index_names])
        ax4.grid(True, alpha=0.3)
        
        for bar, saving in zip(bars, savings_percent):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{saving:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/index_rebuilding_cost_comparison.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated cost comparison: {self.output_dir}/index_rebuilding_cost_comparison.pdf")
    
    def plot_dynamic_update_analysis(self):
        """Plot analysis of dynamic document addition patterns."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        update_results = self.simulate_dynamic_updates()
        
        # 1. Document addition patterns
        days = np.arange(365)
        for pattern_name, results in update_results.items():
            daily_docs = results['daily_docs']
            plot_days = days[:len(daily_docs)]
            ax1.plot(plot_days, daily_docs, linewidth=2, label=pattern_name, alpha=0.8)
        
        ax1.set_xlabel('Day of Year')
        ax1.set_ylabel('Documents Added per Day')
        ax1.set_title('Document Addition Patterns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative costs
        for pattern_name, results in update_results.items():
            traditional_cumulative = np.cumsum(results['traditional_costs'])
            csd_cumulative = np.cumsum(results['csd_costs'])
            plot_days = days[:len(traditional_cumulative)]
            
            ax2.plot(plot_days, traditional_cumulative, '--', linewidth=2, 
                    label=f'{pattern_name} (Traditional)', alpha=0.7)
            ax2.plot(plot_days, csd_cumulative, '-', linewidth=2, 
                    label=f'{pattern_name} (CSD)', alpha=0.8)
        
        ax2.set_xlabel('Day of Year')
        ax2.set_ylabel('Cumulative Cost ($)')
        ax2.set_title('Cumulative Index Maintenance Costs')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Search quality over time
        for pattern_name, results in update_results.items():
            if pattern_name == 'Steady Growth':  # Show one example for clarity
                trad_quality = results['traditional_quality']
                csd_quality = results['csd_quality']
                plot_days = days[:len(trad_quality)]
                
                ax3.plot(plot_days, trad_quality, 'r--', linewidth=2, 
                        label='Traditional (Weekly Rebuild)', alpha=0.8)
                ax3.plot(plot_days, csd_quality, 'g-', linewidth=2, 
                        label='CSD (Online Updates)', alpha=0.8)
        
        ax3.set_xlabel('Day of Year')
        ax3.set_ylabel('Search Quality (%)')
        ax3.set_title('Search Quality Over Time (Steady Growth Pattern)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(70, 101)
        
        # 4. Cost and quality comparison summary
        pattern_names = list(update_results.keys())
        traditional_costs = [results['traditional_total_cost'] for results in update_results.values()]
        csd_costs = [results['csd_total_cost'] for results in update_results.values()]
        traditional_quality = [results['traditional_avg_quality'] for results in update_results.values()]
        csd_quality = [results['csd_avg_quality'] for results in update_results.values()]
        
        x = np.arange(len(pattern_names))
        width = 0.35
        
        # Cost comparison
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x - width/2, traditional_costs, width, 
                       label='Traditional Cost', color='#E74C3C', alpha=0.8)
        bars2 = ax4.bar(x + width/2, csd_costs, width,
                       label='CSD Cost', color='#2ECC71', alpha=0.8)
        
        # Quality comparison (on secondary axis)
        line1 = ax4_twin.plot(x - width/2, traditional_quality, 'ro-', 
                             label='Traditional Quality', markersize=8, linewidth=2)
        line2 = ax4_twin.plot(x + width/2, csd_quality, 'go-', 
                             label='CSD Quality', markersize=8, linewidth=2)
        
        ax4.set_xlabel('Document Addition Pattern')
        ax4.set_ylabel('Annual Cost ($)', color='black')
        ax4_twin.set_ylabel('Average Search Quality (%)', color='black')
        ax4.set_title('Cost vs Quality Trade-offs')
        ax4.set_xticks(x)
        ax4.set_xticklabels([name.replace(' ', '\n') for name in pattern_names])
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dynamic_update_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated dynamic update analysis: {self.output_dir}/dynamic_update_analysis.pdf")
    
    def plot_scalability_analysis(self):
        """Analyze how index rebuilding costs scale with dataset size."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Generate scaling data
        vector_counts = np.logspace(4, 8, 20)  # 10K to 100M vectors
        vector_dims = [384, 768, 1024]
        
        # 1. Build time scaling
        for dim in vector_dims:
            build_times = []
            for count in vector_counts:
                # Empirical scaling: O(n log n) for HNSW, O(n) for IVF
                if count < 1e6:
                    build_time = count * np.log(count) * dim * 1e-9  # Hours
                else:
                    build_time = count * dim * 2e-9  # Hours (IVF)
                build_times.append(build_time)
            
            ax1.plot(vector_counts, build_times, 'o-', linewidth=2, 
                    label=f'{dim}D vectors', markersize=4)
        
        ax1.set_xlabel('Number of Vectors')
        ax1.set_ylabel('Build Time (Hours)')
        ax1.set_title('Index Build Time Scaling')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory requirements
        for dim in vector_dims:
            memory_reqs = []
            for count in vector_counts:
                # Vector storage + index overhead (1.5x factor)
                memory_gb = count * dim * 4 / (1024**3) * 1.5
                memory_reqs.append(memory_gb)
            
            ax2.plot(vector_counts, memory_reqs, 'o-', linewidth=2, 
                    label=f'{dim}D vectors', markersize=4)
        
        ax2.set_xlabel('Number of Vectors')
        ax2.set_ylabel('Memory Requirement (GB)')
        ax2.set_title('Memory Scaling')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Annual costs comparison
        traditional_costs = []
        csd_costs = []
        
        for count in vector_counts:
            # Traditional: weekly rebuilds
            weekly_build_time = count * np.log(count) * 768 * 1e-9
            weekly_cost = weekly_build_time * 32  # $32/hour
            annual_traditional = weekly_cost * 52 + 50000  # Base operational cost
            
            # CSD: online updates
            daily_updates = count * 0.01  # 1% daily update rate
            update_cost_per_doc = 0.0001
            annual_csd = daily_updates * update_cost_per_doc * 365 + 5000  # Base cost
            
            traditional_costs.append(annual_traditional)
            csd_costs.append(annual_csd)
        
        ax3.plot(vector_counts, traditional_costs, 'r-', linewidth=3, 
                label='Traditional RAG', alpha=0.8)
        ax3.plot(vector_counts, csd_costs, 'g-', linewidth=3, 
                label='CSD-Enhanced RAG', alpha=0.8)
        
        ax3.set_xlabel('Number of Vectors')
        ax3.set_ylabel('Annual Cost ($)')
        ax3.set_title('Annual Cost Scaling')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Break-even analysis
        cost_ratios = np.array(traditional_costs) / np.array(csd_costs)
        
        ax4.semilogx(vector_counts, cost_ratios, 'b-', linewidth=3, alpha=0.8)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax4.fill_between(vector_counts, cost_ratios, 1, where=(cost_ratios > 1), 
                        color='green', alpha=0.3, label='CSD Advantage')
        
        ax4.set_xlabel('Number of Vectors')
        ax4.set_ylabel('Cost Ratio (Traditional/CSD)')
        ax4.set_title('CSD Cost Advantage vs Dataset Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/index_scalability_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated scalability analysis: {self.output_dir}/index_scalability_analysis.pdf")
    
    def generate_summary_table(self):
        """Generate comprehensive summary of index rebuilding analysis."""
        data = []
        
        # Calculate costs for medium index across all systems
        medium_index = self.indices['Medium Index']
        
        for system_name, system in self.systems.items():
            costs = self.calculate_annual_costs(medium_index, system)
            
            data.append({
                'System': system_name,
                'Rebuild Frequency': f"Every {system.rebuild_frequency_days} days",
                'Rebuilds/Year': f"{costs['rebuilds_per_year']:.1f}",
                'Downtime/Rebuild (min)': f"{system.service_downtime_minutes:.0f}",
                'Annual Compute Cost': f"${costs['annual_compute_cost']:,.0f}",
                'Annual Downtime Cost': f"${costs['annual_downtime_cost']:,.0f}",
                'Annual Degradation Cost': f"${costs['annual_degradation_cost']:,.0f}",
                'Total Annual Cost': f"${costs['total_annual_cost']:,.0f}",
                'System Availability': f"{costs['availability_percent']:.1f}%",
                'Online Updates': "Yes" if system.supports_online_updates else "No",
                'Update Latency (ms)': f"{system.update_latency_ms:.0f}" if system.supports_online_updates else "N/A"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{self.output_dir}/index_rebuilding_summary.csv', index=False)
        
        # Create formatted table
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 2)
        
        # Color-code by performance
        for i in range(len(df)):
            system_name = df.iloc[i, 0]
            if 'CSD' in system_name:
                color = '#C8E6C9'  # Light green for best
            elif 'EdgeRAG' in system_name:
                color = '#FFF9C4'  # Light yellow for good
            elif 'FlashRAG' in system_name:
                color = '#FFECB3'  # Light orange for moderate
            else:
                color = '#FFCDD2'  # Light red for traditional
            
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(color)
        
        plt.title('Index Rebuilding Cost Analysis Summary (Medium Dataset)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/index_rebuilding_summary_table.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated summary table: {self.output_dir}/index_rebuilding_summary_table.pdf")
        print(f"✅ Generated CSV data: {self.output_dir}/index_rebuilding_summary.csv")
    
    def run_full_analysis(self):
        """Run complete index rebuilding analysis."""
        print("🔍 Starting Index Rebuilding Cost Analysis...")
        print("=" * 60)
        
        self.plot_cost_comparison()
        self.plot_dynamic_update_analysis()
        self.plot_scalability_analysis()
        self.generate_summary_table()
        
        print("\n✅ Index Rebuilding Analysis Complete!")
        print(f"📊 All plots saved to: {self.output_dir}/")
        print("📈 Generated plots:")
        print("   - index_rebuilding_cost_comparison.pdf")
        print("   - dynamic_update_analysis.pdf")
        print("   - index_scalability_analysis.pdf")
        print("   - index_rebuilding_summary_table.pdf")
        print("   - index_rebuilding_summary.csv")

def main():
    """Main function to run the analysis."""
    analyzer = IndexRebuildingAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()