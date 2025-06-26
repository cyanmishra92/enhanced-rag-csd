#!/usr/bin/env python3
"""
Realistic CSD Hardware Performance Demo

This demo showcases how our Enhanced RAG-CSD system models realistic
computational and communication delays based on actual hardware specifications
from AMD FPGAs, ARM cores, SK-Hynix CSDs, and other commercial solutions.
"""

import sys
import os
import numpy as np
from typing import Dict, Any
# Disable GUI backend for matplotlib in headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_rag_csd.core.hardware_models import (
    CSDHardwareManager, CSDHardwareType, CSDHardwareModel
)
from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


def demonstrate_hardware_models():
    """Demonstrate realistic hardware modeling capabilities."""
    print("🔬 Enhanced RAG-CSD: Realistic Hardware Modeling Demo")
    print("=" * 60)
    
    # Initialize hardware manager
    manager = CSDHardwareManager()
    
    # Add various CSD hardware models based on real specifications
    hardware_configs = [
        ("AMD_Versal_FPGA", CSDHardwareType.AMD_VERSAL_FPGA),
        ("ARM_Cortex_A78", CSDHardwareType.ARM_CORTEX_A78),  
        ("SK_Hynix_CSD", CSDHardwareType.SK_HYNIX_CSD),
        ("Samsung_SmartSSD", CSDHardwareType.SAMSUNG_SMARTSSD),
        ("Xilinx_Alveo", CSDHardwareType.XILINX_ALVEO),
        ("Custom_ASIC", CSDHardwareType.CUSTOM_ASIC)
    ]
    
    for name, hw_type in hardware_configs:
        manager.add_hardware(name, hw_type)
    
    print(f"✅ Initialized {len(hardware_configs)} realistic CSD hardware models")
    print()
    
    return manager


def benchmark_ml_operations(manager: CSDHardwareManager):
    """Benchmark ML operations across different CSD hardware."""
    print("📊 Benchmarking ML Operations on Realistic CSD Hardware")
    print("-" * 60)
    
    # Define representative ML operations for RAG pipelines
    test_operations = [
        ("embedding_lookup", (100, 384), "Encode queries"),
        ("similarity_compute", (1000, 384), "Retrieve candidates"),
        ("attention", (64, 384), "Augment context"),
        ("softmax", (1000,), "Attention weights"),
        ("matrix_multiply", (384, 384), "Linear transform"),
    ]
    
    results = {}
    
    for op_name, data_shape, description in test_operations:
        print(f"\n🔧 {description} [{op_name}] - Shape: {data_shape}")
        
        # Benchmark across all hardware
        hw_results = manager.benchmark_operation(op_name, data_shape)
        results[op_name] = hw_results
        
        # Find optimal hardware
        optimal_hw = manager.get_optimal_hardware(op_name, data_shape)
        
        # Display results
        for hw_name, exec_time in sorted(hw_results.items(), key=lambda x: x[1]):
            status = "⭐ OPTIMAL" if hw_name == optimal_hw else ""
            print(f"  {hw_name:20}: {exec_time*1000:8.3f}ms {status}")
    
    return results


def analyze_communication_overhead(manager: CSDHardwareManager):
    """Analyze communication overhead for different data sizes."""
    print("\n📡 Communication Overhead Analysis")
    print("-" * 60)
    
    # Test different data transfer sizes (typical for RAG workloads)
    data_sizes = [
        (384 * 4, "Single embedding"),      # 1.5KB
        (384 * 100 * 4, "Batch embeddings"), # 150KB  
        (384 * 1000 * 4, "Large batch"),    # 1.5MB
        (384 * 10000 * 4, "Full dataset")   # 15MB
    ]
    
    transfer_types = ["host_to_device", "device_to_host", "p2p_transfer"]
    
    comm_results = {}
    
    for hw_name in manager.hardware_models:
        hw_model = manager.hardware_models[hw_name]
        comm_results[hw_name] = {}
        
        print(f"\n🔗 {hw_name} Communication Performance:")
        
        for size_bytes, description in data_sizes:
            comm_results[hw_name][description] = {}
            print(f"  {description} ({size_bytes/1024:.1f}KB):")
            
            for transfer_type in transfer_types:
                transfer_time = hw_model.calculate_memory_transfer_time(
                    size_bytes, transfer_type
                )
                comm_results[hw_name][description][transfer_type] = transfer_time
                print(f"    {transfer_type:15}: {transfer_time*1000:8.3f}ms")
    
    return comm_results


def demonstrate_thermal_effects(manager: CSDHardwareManager):
    """Demonstrate thermal throttling effects on performance."""
    print("\n🌡️  Thermal Effects and Sustained Performance")
    print("-" * 60)
    
    # Use ARM Cortex-A78 model for thermal demonstration
    arm_model = manager.hardware_models["ARM_Cortex_A78"]
    
    print(f"📊 Simulating sustained workload on {arm_model.hardware_type.value}")
    
    # Simulate sustained high utilization
    utilization_levels = [0.3, 0.5, 0.7, 0.9, 0.95]
    operation = "matrix_multiply"
    data_shape = (256, 256)
    
    thermal_results = []
    
    for util in utilization_levels:
        arm_model.update_utilization(util)
        exec_time = arm_model.calculate_ml_operation_time(operation, data_shape)
        
        thermal_results.append({
            'utilization': util,
            'thermal_state': arm_model.thermal_state,
            'execution_time_ms': exec_time * 1000,
            'performance_degradation': (exec_time * 1000) / (thermal_results[0]['execution_time_ms'] if thermal_results else exec_time * 1000)
        })
        
        print(f"  Utilization: {util:5.1%} | Thermal: {arm_model.thermal_state:5.1%} | "
              f"Time: {exec_time*1000:8.3f}ms | Degradation: {thermal_results[-1]['performance_degradation']:5.2f}x")
    
    return thermal_results


def create_performance_visualization(results: Dict[str, Any], comm_results: Dict[str, Any]):
    """Create performance visualization plots."""
    print("\n📈 Creating Performance Visualization")
    print("-" * 60)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced RAG-CSD: Realistic Hardware Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. ML Operations Performance Comparison
    operations = list(results.keys())
    hardware_names = list(next(iter(results.values())).keys())
    
    # Prepare data for heatmap
    perf_matrix = []
    for hw in hardware_names:
        hw_times = [results[op][hw] * 1000 for op in operations]  # Convert to ms
        perf_matrix.append(hw_times)
    
    im1 = ax1.imshow(perf_matrix, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(len(operations)))
    ax1.set_xticklabels([op.replace('_', '\n') for op in operations], rotation=45)
    ax1.set_yticks(range(len(hardware_names)))
    ax1.set_yticklabels([hw.replace('_', '\n') for hw in hardware_names])
    ax1.set_title('ML Operations Performance (ms)\nLower is Better', fontweight='bold')
    
    # Add text annotations
    for i in range(len(hardware_names)):
        for j in range(len(operations)):
            text = ax1.text(j, i, f'{perf_matrix[i][j]:.2f}', 
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. Communication Bandwidth Comparison
    hw_names_short = [name.replace('_', ' ') for name in hardware_names]
    bandwidth_data = []
    
    for hw_name in hardware_names:
        hw_model = manager.hardware_models[hw_name]
        bandwidth_data.append(hw_model.comm_specs.max_bandwidth_gbps)
    
    bars2 = ax2.bar(hw_names_short, bandwidth_data, color='skyblue', alpha=0.7)
    ax2.set_title('Communication Bandwidth (GB/s)\nHigher is Better', fontweight='bold')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, bandwidth_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Compute Performance vs Power Efficiency
    compute_perf = []
    power_efficiency = []
    
    for hw_name in hardware_names:
        hw_model = manager.hardware_models[hw_name]
        compute_perf.append(hw_model.compute_specs.peak_ops_per_sec / 1e12)  # TOPS
        power_efficiency.append(hw_model.compute_specs.peak_ops_per_sec / hw_model.compute_specs.power_watts / 1e9)  # GOPS/W
    
    scatter = ax3.scatter(power_efficiency, compute_perf, s=100, alpha=0.7, c=range(len(hardware_names)), cmap='viridis')
    
    # Add labels for each point
    for i, hw_name in enumerate(hw_names_short):
        ax3.annotate(hw_name, (power_efficiency[i], compute_perf[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Power Efficiency (GOPS/W)')
    ax3.set_ylabel('Peak Performance (TOPS)')
    ax3.set_title('Performance vs Power Efficiency\nUpper-right is Better', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Hardware Specifications Radar Chart
    # Select key metrics for radar chart
    metrics = ['Compute (TOPS)', 'Memory BW (GB/s)', 'Frequency (GHz)', 'Parallel Units (log10)', 'Power (W)']
    
    # Normalize metrics for radar chart
    normalized_data = []
    for hw_name in hardware_names[:4]:  # Show top 4 hardware
        hw_model = manager.hardware_models[hw_name]
        values = [
            hw_model.compute_specs.peak_ops_per_sec / 1e12,  # TOPS
            hw_model.compute_specs.memory_bandwidth_gbps / 100,  # Normalized to 0-1
            hw_model.compute_specs.frequency_mhz / 3000,  # Normalized to 0-1  
            np.log10(hw_model.compute_specs.parallel_units) / 4,  # Log scale normalized
            hw_model.compute_specs.power_watts / 200  # Normalized to 0-1
        ]
        normalized_data.append(values)
    
    # Create simple bar chart instead of radar for clarity
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (hw_name, values) in enumerate(zip(hardware_names[:4], normalized_data)):
        offset = (i - 1.5) * width
        ax4.bar(x + offset, values, width, label=hw_name.replace('_', ' '), alpha=0.8)
    
    ax4.set_xlabel('Hardware Metrics (Normalized)')
    ax4.set_ylabel('Relative Performance')
    ax4.set_title('Hardware Specifications Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=45, ha='right')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "results/hardware_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/realistic_csd_hardware_analysis.pdf", 
                dpi=300, bbox_inches='tight')
    # plt.show()  # Disabled for headless environment
    
    print(f"📄 Performance visualization saved to: {output_dir}/realistic_csd_hardware_analysis.pdf")


def generate_summary_report(results: Dict[str, Any], manager: CSDHardwareManager):
    """Generate comprehensive summary report."""
    print("\n📋 Hardware Performance Summary Report")
    print("=" * 60)
    
    # Find optimal hardware for each operation
    optimal_recommendations = {}
    for op_name in results:
        optimal_hw = manager.get_optimal_hardware(op_name, (100, 384))
        optimal_recommendations[op_name] = optimal_hw
    
    print("\n🏆 Optimal Hardware Recommendations for RAG Operations:")
    operation_descriptions = {
        "embedding_lookup": "Query Encoding",
        "similarity_compute": "Context Retrieval", 
        "attention": "Context Augmentation",
        "softmax": "Attention Computation",
        "matrix_multiply": "Linear Transformations"
    }
    
    for op_name, optimal_hw in optimal_recommendations.items():
        desc = operation_descriptions.get(op_name, op_name)
        optimal_time = results[op_name][optimal_hw] * 1000
        print(f"  {desc:25}: {optimal_hw:20} ({optimal_time:6.2f}ms)")
    
    print("\n💡 Key Insights:")
    
    # Analyze performance patterns
    fpga_hw = [hw for hw in manager.hardware_models if "FPGA" in hw or "Alveo" in hw]
    arm_hw = [hw for hw in manager.hardware_models if "ARM" in hw]
    ssd_hw = [hw for hw in manager.hardware_models if "Hynix" in hw or "Samsung" in hw]
    
    if fpga_hw:
        fpga_name = fpga_hw[0]
        fpga_model = manager.hardware_models[fpga_name] 
        print(f"  🔥 FPGA Solutions ({fpga_name}):")
        print(f"     - Highest compute performance: {fpga_model.compute_specs.peak_ops_per_sec/1e12:.1f} TOPS")
        print(f"     - Best for: Parallel ML operations, matrix computations")
        print(f"     - Power consumption: {fpga_model.compute_specs.power_watts}W")
    
    if arm_hw:
        arm_name = arm_hw[0] 
        arm_model = manager.hardware_models[arm_name]
        print(f"  📱 ARM Solutions ({arm_name}):")
        print(f"     - Best power efficiency: {arm_model.compute_specs.peak_ops_per_sec/arm_model.compute_specs.power_watts/1e9:.1f} GOPS/W")
        print(f"     - Best for: Edge deployment, mobile CSDs")
        print(f"     - Thermal management: Important for sustained performance")
    
    if ssd_hw:
        ssd_name = ssd_hw[0]
        ssd_model = manager.hardware_models[ssd_name]
        print(f"  💽 Commercial CSD Solutions ({ssd_name}):")
        print(f"     - Balanced performance: {ssd_model.compute_specs.peak_ops_per_sec/1e9:.1f} GOPS")
        print(f"     - Best for: Storage-centric RAG, near-data processing")
        print(f"     - Integration: Standard NVMe interface")
    
    print(f"\n🎯 Performance Summary:")
    print(f"  - Fastest operation: {min(results['embedding_lookup'].values())*1000:.2f}ms (embedding lookup)")
    print(f"  - Most demanding: {max(results['matrix_multiply'].values())*1000:.2f}ms (matrix multiply)")
    print(f"  - Performance range: {max(max(times.values()) for times in results.values())/min(min(times.values()) for times in results.values()):.1f}x across hardware")


def main():
    """Main demo function."""
    print("🚀 Enhanced RAG-CSD: Realistic CSD Hardware Performance Demo")
    print("   Modeling computational and communication delays from real hardware specs")
    print()
    
    try:
        # Initialize hardware models
        global manager
        manager = demonstrate_hardware_models()
        
        # Benchmark ML operations
        results = benchmark_ml_operations(manager)
        
        # Analyze communication overhead  
        comm_results = analyze_communication_overhead(manager)
        
        # Demonstrate thermal effects
        thermal_results = demonstrate_thermal_effects(manager)
        
        # Create visualizations
        create_performance_visualization(results, comm_results)
        
        # Generate summary report
        generate_summary_report(results, manager)
        
        print("\n✅ Demo completed successfully!")
        print("📊 Check the generated plots and analysis for detailed insights.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()