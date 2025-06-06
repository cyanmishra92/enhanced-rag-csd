#!/usr/bin/env python
"""
Run and Plot Benchmark Script for Enhanced RAG-CSD
This script provides a simple interface to run comprehensive benchmarks and generate publication-quality plots.

Usage:
    python scripts/run_and_plot_benchmark.py --quick      # Quick benchmark (fewer systems, fewer runs)
    python scripts/run_and_plot_benchmark.py --full       # Full benchmark (all systems, full statistical rigor)
    python scripts/run_and_plot_benchmark.py --plot-only  # Only generate plots from existing results
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_rag_csd.visualization.research_plots import ResearchPlotter
from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """Easy-to-use benchmark runner with plotting capabilities."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Pre-defined benchmark configurations
        self.configs = {
            "quick": {
                "description": "Quick benchmark for development and testing",
                "systems": ["Enhanced-RAG-CSD", "RAG-CSD", "VanillaRAG"],
                "num_runs": 2,
                "output_suffix": "quick",
                "estimated_time": "3-5 minutes"
            },
            "standard": {
                "description": "Standard benchmark with good statistical rigor",
                "systems": ["Enhanced-RAG-CSD", "RAG-CSD", "PipeRAG-like", "EdgeRAG-like", "VanillaRAG"],
                "num_runs": 3,
                "output_suffix": "standard",
                "estimated_time": "8-12 minutes"
            },
            "full": {
                "description": "Full comprehensive benchmark with all systems",
                "systems": ["Enhanced-RAG-CSD", "RAG-CSD", "PipeRAG-like", "FlashRAG-like", "EdgeRAG-like", "VanillaRAG"],
                "num_runs": 5,
                "output_suffix": "full",
                "estimated_time": "15-25 minutes"
            },
            "research": {
                "description": "Research-grade benchmark with maximum statistical rigor",
                "systems": ["Enhanced-RAG-CSD", "RAG-CSD", "PipeRAG-like", "FlashRAG-like", "EdgeRAG-like", "VanillaRAG"],
                "num_runs": 10,
                "output_suffix": "research",
                "estimated_time": "30-45 minutes"
            }
        }
    
    def list_configurations(self):
        """List available benchmark configurations."""
        print("\n🔧 Available Benchmark Configurations:")
        print("=" * 50)
        
        for name, config in self.configs.items():
            print(f"\n{name.upper()}:")
            print(f"  Description: {config['description']}")
            print(f"  Systems: {', '.join(config['systems'])}")
            print(f"  Runs: {config['num_runs']}")
            print(f"  Estimated Time: {config['estimated_time']}")
    
    def run_benchmark(self, config_name: str, custom_args: Optional[Dict] = None) -> str:
        """Run benchmark with specified configuration."""
        if config_name not in self.configs:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        config = self.configs[config_name].copy()
        
        # Apply custom arguments
        if custom_args:
            config.update(custom_args)
        
        print(f"\n🚀 Running {config_name.upper()} Benchmark")
        print("=" * 50)
        print(f"Description: {config['description']}")
        print(f"Systems: {', '.join(config['systems'])}")
        print(f"Runs per system: {config['num_runs']}")
        print(f"Estimated time: {config['estimated_time']}")
        
        # Prepare output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.results_dir / f"benchmark_{config['output_suffix']}_{timestamp}"
        
        # Build command
        benchmark_script = self.project_root / "scripts" / "comprehensive_public_benchmark.py"
        cmd = [
            sys.executable, str(benchmark_script),
            "--output-dir", str(output_dir),
            "--num-runs", str(config["num_runs"]),
            "--systems"
        ] + config["systems"]
        
        print(f"\n💻 Executing command:")
        print(f"   {' '.join(cmd)}")
        
        # Run benchmark
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                elapsed = time.time() - start_time
                print(f"\n✅ Benchmark completed successfully in {elapsed:.2f} seconds!")
                print(f"📁 Results saved to: {output_dir}")
                
                # Print key output
                if result.stdout:
                    print("\n📊 Benchmark Output:")
                    print("-" * 30)
                    print(result.stdout[-1000:])  # Last 1000 chars
                
                return str(output_dir)
            else:
                print(f"\n❌ Benchmark failed with return code {result.returncode}")
                if result.stderr:
                    print("Error output:")
                    print(result.stderr)
                return ""
                
        except Exception as e:
            print(f"\n❌ Failed to run benchmark: {e}")
            return ""
    
    def generate_plots(self, results_dir: str, plot_types: Optional[List[str]] = None) -> List[str]:
        """Generate plots from benchmark results."""
        results_path = Path(results_dir)
        
        if not results_path.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")
        
        # Find results file
        results_file = None
        for filename in ["comprehensive_results.json", "complete_results.json", "experiment_results.json"]:
            candidate = results_path / filename
            if candidate.exists():
                results_file = candidate
                break
        
        if not results_file:
            raise ValueError(f"No results file found in {results_dir}")
        
        print(f"\n📊 Generating plots from: {results_file}")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Initialize plotter
        plots_dir = results_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        plotter = ResearchPlotter(str(plots_dir))
        
        generated_plots = []
        plot_types = plot_types or ["all"]
        
        try:
            # Generate different types of plots based on available data
            if "all" in plot_types or "latency" in plot_types:
                if "aggregated_results" in results:
                    print("  Creating latency comparison plots...")
                    plot_path = plotter.plot_latency_comparison(results["aggregated_results"])
                    generated_plots.append(plot_path)
            
            if "all" in plot_types or "throughput" in plot_types:
                if "aggregated_results" in results:
                    print("  Creating throughput analysis plots...")
                    plot_path = plotter.plot_throughput_analysis(results["aggregated_results"])
                    generated_plots.append(plot_path)
            
            if "all" in plot_types or "accuracy" in plot_types:
                if "aggregated_results" in results:
                    print("  Creating accuracy metrics plots...")
                    plot_path = plotter.plot_accuracy_metrics(results["aggregated_results"])
                    generated_plots.append(plot_path)
            
            if "all" in plot_types or "cache" in plot_types:
                if "aggregated_results" in results:
                    print("  Creating cache performance plots...")
                    # Extract cache data
                    cache_data = {}
                    for system, metrics in results["aggregated_results"].items():
                        if "avg_cache_hit_rate" in metrics:
                            cache_data[system] = {"cache_hit_rate": metrics["avg_cache_hit_rate"]}
                    
                    if cache_data:
                        plot_path = plotter.plot_cache_performance(cache_data)
                        generated_plots.append(plot_path)
            
            if "all" in plot_types or "overview" in plot_types:
                print("  Creating system overview plots...")
                plot_path = plotter.plot_system_overview(results)
                generated_plots.append(plot_path)
            
            if "all" in plot_types or "benchmark" in plot_types:
                # Benchmark-specific plots
                if "benchmark_results" in results:
                    print("  Creating benchmark-specific plots...")
                    for benchmark_name, benchmark_data in results["benchmark_results"].items():
                        # Create per-benchmark comparison
                        benchmark_aggregated = {}
                        for system, system_data in benchmark_data.items():
                            if "aggregated" in system_data:
                                benchmark_aggregated[system] = system_data["aggregated"]
                        
                        if benchmark_aggregated:
                            plot_path = plotter.plot_benchmark_comparison(
                                benchmark_aggregated, 
                                benchmark_name
                            )
                            generated_plots.append(plot_path)
            
            print(f"\n✅ Generated {len(generated_plots)} plot files:")
            for plot in generated_plots:
                print(f"   📈 {plot}")
            
            return generated_plots
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            print(f"❌ Error generating plots: {e}")
            return generated_plots
    
    def create_summary_report(self, results_dir: str) -> str:
        """Create a summary report with key findings."""
        results_path = Path(results_dir)
        
        # Find results file
        results_file = None
        for filename in ["comprehensive_results.json", "complete_results.json"]:
            candidate = results_path / filename
            if candidate.exists():
                results_file = candidate
                break
        
        if not results_file:
            return ""
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_content = f"""# Benchmark Summary Report

**Generated**: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}  
**Results Directory**: {results_dir}  
**Experiment ID**: {results.get('timestamp', timestamp)}  

## Quick Results

"""
        
        # Add performance summary
        if "aggregated_results" in results:
            systems = results["aggregated_results"]
            
            # Find best system by latency
            best_latency = min(systems.keys(), 
                             key=lambda k: systems[k].get("avg_latency", float('inf')))
            
            # Find best throughput
            best_throughput = max(systems.keys(),
                                key=lambda k: systems[k].get("avg_throughput", 0))
            
            summary_content += f"""### 🏆 Performance Leaders

- **Fastest (Latency)**: {best_latency} ({systems[best_latency].get('avg_latency', 0)*1000:.1f}ms)
- **Highest Throughput**: {best_throughput} ({systems[best_throughput].get('avg_throughput', 0):.1f} q/s)

### 📊 All Systems Performance

| System | Latency (ms) | Throughput (q/s) | Relevance Score |
|--------|-------------|------------------|-----------------|
"""
            
            for system, metrics in systems.items():
                latency = metrics.get("avg_latency", 0) * 1000
                throughput = metrics.get("avg_throughput", 0)
                relevance = metrics.get("avg_relevance_score", 0)
                summary_content += f"| {system} | {latency:.1f} | {throughput:.1f} | {relevance:.3f} |\n"
        
        # Add benchmark breakdown
        if "benchmark_results" in results:
            summary_content += f"""
## Benchmark Breakdown

"""
            for benchmark_name, benchmark_data in results["benchmark_results"].items():
                summary_content += f"""### {benchmark_name}
"""
                for system, system_data in benchmark_data.items():
                    if "aggregated" in system_data:
                        metrics = system_data["aggregated"]
                        latency = metrics.get("avg_latency", 0) * 1000
                        throughput = metrics.get("throughput", 0)
                        summary_content += f"- **{system}**: {latency:.1f}ms, {throughput:.1f} q/s\n"
        
        # Save summary
        summary_path = results_path / "SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        print(f"\n📋 Summary report created: {summary_path}")
        return str(summary_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run and plot Enhanced RAG-CSD benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_and_plot_benchmark.py --quick                    # Quick 5-minute benchmark
  python scripts/run_and_plot_benchmark.py --standard                 # Standard 10-minute benchmark  
  python scripts/run_and_plot_benchmark.py --full                     # Full 20-minute benchmark
  python scripts/run_and_plot_benchmark.py --research                 # Research-grade 45-minute benchmark
  python scripts/run_and_plot_benchmark.py --plot-only results/dir/   # Generate plots only
  python scripts/run_and_plot_benchmark.py --list                     # List available configurations
        """
    )
    
    # Main action arguments
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick benchmark (3-5 minutes)")
    parser.add_argument("--standard", action="store_true",
                       help="Run standard benchmark (8-12 minutes)")
    parser.add_argument("--full", action="store_true",
                       help="Run full benchmark (15-25 minutes)")
    parser.add_argument("--research", action="store_true",
                       help="Run research-grade benchmark (30-45 minutes)")
    parser.add_argument("--list", action="store_true",
                       help="List available benchmark configurations")
    
    # Plot-only mode
    parser.add_argument("--plot-only", type=str, metavar="RESULTS_DIR",
                       help="Generate plots only from existing results directory")
    
    # Customization options
    parser.add_argument("--systems", nargs="+",
                       help="Override systems to test")
    parser.add_argument("--num-runs", type=int,
                       help="Override number of runs")
    parser.add_argument("--plot-types", nargs="+", 
                       choices=["latency", "throughput", "accuracy", "cache", "overview", "benchmark", "all"],
                       default=["all"],
                       help="Types of plots to generate")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    
    # Handle list command
    if args.list:
        runner.list_configurations()
        return
    
    # Handle plot-only mode
    if args.plot_only:
        print(f"\n📊 Generating plots from: {args.plot_only}")
        try:
            plots = runner.generate_plots(args.plot_only, args.plot_types)
            summary = runner.create_summary_report(args.plot_only)
            print(f"✅ Generated {len(plots)} plots and summary report")
        except Exception as e:
            print(f"❌ Failed to generate plots: {e}")
        return
    
    # Determine configuration
    config_name = None
    if args.quick:
        config_name = "quick"
    elif args.standard:
        config_name = "standard"
    elif args.full:
        config_name = "full"
    elif args.research:
        config_name = "research"
    else:
        # Default to quick if no specific config chosen
        print("No configuration specified, using --quick (use --list to see all options)")
        config_name = "quick"
    
    # Prepare custom arguments
    custom_args = {}
    if args.systems:
        custom_args["systems"] = args.systems
    if args.num_runs:
        custom_args["num_runs"] = args.num_runs
    
    print(f"\n🎯 Selected Configuration: {config_name.upper()}")
    
    # Run benchmark
    results_dir = runner.run_benchmark(config_name, custom_args)
    
    if results_dir:
        print(f"\n📊 Generating plots and summary...")
        
        # Generate plots
        plots = runner.generate_plots(results_dir, args.plot_types)
        
        # Create summary report
        summary = runner.create_summary_report(results_dir)
        
        print(f"\n🎉 Benchmark and analysis complete!")
        print(f"📁 Results: {results_dir}")
        print(f"📈 Plots: {len(plots)} files generated")
        print(f"📋 Summary: {summary}")
        print(f"\n💡 Next steps:")
        print(f"   - Review the summary report: {summary}")
        print(f"   - Open plot files for detailed analysis")
        print(f"   - Use results for research papers or presentations")
    else:
        print(f"\n❌ Benchmark failed. Please check the error messages above.")


if __name__ == "__main__":
    main()