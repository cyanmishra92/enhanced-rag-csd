#!/usr/bin/env python
"""
Comprehensive experiment runner for Enhanced RAG-CSD system.
This script provides a unified interface to run various experiments and benchmarks.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_environment():
    """Setup necessary directories and environment."""
    dirs = ['data/experiments', 'results/experiments', 'logs']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set environment variables if not set
    if not os.getenv('ENHANCED_RAG_HOME'):
        os.environ['ENHANCED_RAG_HOME'] = os.path.expanduser('~/.enhanced_rag_csd')
        os.environ['ENHANCED_RAG_CACHE'] = os.path.join(os.environ['ENHANCED_RAG_HOME'], 'cache')
        os.environ['ENHANCED_RAG_DATA'] = os.path.join(os.environ['ENHANCED_RAG_HOME'], 'data')

def run_demo(args):
    """Run the interactive demonstration."""
    print("🚀 Running Enhanced RAG-CSD Demo...")
    
    cmd = [
        sys.executable, "examples/demo.py",
        "--vector-db", args.vector_db or "./data/sample",
        "--output-dir", args.output_dir or "./results/demo",
        "--log-level", args.log_level
    ]
    
    if args.skip_benchmark:
        cmd.append("--skip-benchmark")
    
    return subprocess.run(cmd, capture_output=False)

def run_benchmark(args):
    """Run performance benchmarks."""
    print("📊 Running Performance Benchmarks...")
    
    cmd = [
        sys.executable, "examples/benchmark.py",
        "--vector-db", args.vector_db,
        "--output-dir", args.output_dir or "./results/benchmark",
        "--num-queries", str(args.num_queries),
        "--top-k", str(args.top_k),
        "--runs-per-query", str(args.runs_per_query),
        "--log-level", args.log_level
    ]
    
    if args.skip_accuracy:
        cmd.append("--skip-accuracy")
    
    return subprocess.run(cmd, capture_output=False)

def run_ablation_study(args):
    """Run ablation study to evaluate individual components."""
    print("🔬 Running Ablation Study...")
    
    # Define configurations for ablation study
    configs = [
        {"name": "full_system", "csd": True, "cache": True, "parallel": True},
        {"name": "no_csd", "csd": False, "cache": True, "parallel": True},
        {"name": "no_cache", "csd": True, "cache": False, "parallel": True},
        {"name": "no_parallel", "csd": True, "cache": True, "parallel": False},
        {"name": "minimal", "csd": False, "cache": False, "parallel": False}
    ]
    
    results = {}
    base_output = args.output_dir or "./results/ablation"
    
    for config in configs:
        print(f"\n📋 Testing configuration: {config['name']}")
        
        config_output = os.path.join(base_output, config['name'])
        os.makedirs(config_output, exist_ok=True)
        
        # Create temporary config file
        config_file = os.path.join(config_output, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run benchmark with this configuration
        cmd = [
            sys.executable, "scripts/run_ablation_config.py",
            "--config", config_file,
            "--vector-db", args.vector_db,
            "--output-dir", config_output,
            "--num-queries", str(args.num_queries),
            "--log-level", args.log_level
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        results[config['name']] = {
            "config": config,
            "returncode": result.returncode,
            "stdout": result.stdout if result.returncode == 0 else None,
            "stderr": result.stderr if result.returncode != 0 else None
        }
    
    # Save ablation results
    with open(os.path.join(base_output, "ablation_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Ablation study completed. Results saved to: {base_output}")
    return results

def run_scalability_test(args):
    """Run scalability tests with different dataset sizes."""
    print("📈 Running Scalability Tests...")
    
    # Define test scales
    scales = [100, 500, 1000, 5000, 10000] if not args.quick else [100, 500]
    
    results = {}
    base_output = args.output_dir or "./results/scalability"
    
    for scale in scales:
        print(f"\n📊 Testing with {scale} documents...")
        
        scale_output = os.path.join(base_output, f"scale_{scale}")
        os.makedirs(scale_output, exist_ok=True)
        
        # Generate synthetic dataset
        cmd = [
            sys.executable, "scripts/generate_synthetic_data.py",
            "--num-docs", str(scale),
            "--output-dir", scale_output,
            "--seed", "42"
        ]
        
        gen_result = subprocess.run(cmd, capture_output=True, text=True)
        if gen_result.returncode != 0:
            print(f"❌ Failed to generate data for scale {scale}")
            continue
        
        # Run benchmark
        cmd = [
            sys.executable, "examples/benchmark.py",
            "--vector-db", os.path.join(scale_output, "vectors"),
            "--output-dir", scale_output,
            "--num-queries", str(min(20, args.num_queries)),
            "--runs-per-query", str(max(1, args.runs_per_query // 2)),
            "--log-level", args.log_level
        ]
        
        bench_result = subprocess.run(cmd, capture_output=True, text=True)
        results[scale] = {
            "scale": scale,
            "generation_success": gen_result.returncode == 0,
            "benchmark_success": bench_result.returncode == 0,
            "output_dir": scale_output
        }
    
    # Save scalability results
    with open(os.path.join(base_output, "scalability_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Scalability tests completed. Results saved to: {base_output}")
    return results

def run_comparison_study(args):
    """Run comparison with other RAG systems."""
    print("🥊 Running Comparison Study...")
    
    # Systems to compare
    systems = ["enhanced", "rag_csd", "pipe_rag", "flash_rag", "edge_rag", "vanilla"]
    
    base_output = args.output_dir or "./results/comparison"
    os.makedirs(base_output, exist_ok=True)
    
    cmd = [
        sys.executable, "examples/benchmark.py",
        "--vector-db", args.vector_db,
        "--output-dir", base_output,
        "--num-queries", str(args.num_queries),
        "--top-k", str(args.top_k),
        "--runs-per-query", str(args.runs_per_query),
        "--systems", ",".join(systems),
        "--log-level", args.log_level
    ]
    
    if not args.skip_accuracy:
        cmd.append("--include-accuracy")
    
    result = subprocess.run(cmd, capture_output=False)
    
    print(f"\n✅ Comparison study completed. Results saved to: {base_output}")
    return result

def run_custom_experiment(args):
    """Run custom experiment with user-defined parameters."""
    print("🔧 Running Custom Experiment...")
    
    # Load custom config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    if args.vector_db:
        config['vector_db'] = args.vector_db
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Set defaults
    config.setdefault('vector_db', './data/sample')
    config.setdefault('output_dir', './results/custom')
    config.setdefault('num_queries', args.num_queries)
    config.setdefault('experiment_type', 'benchmark')
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save effective config
    with open(os.path.join(config['output_dir'], 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run experiment based on type
    if config['experiment_type'] == 'demo':
        return run_demo(argparse.Namespace(**config))
    elif config['experiment_type'] == 'benchmark':
        return run_benchmark(argparse.Namespace(**config))
    elif config['experiment_type'] == 'ablation':
        return run_ablation_study(argparse.Namespace(**config))
    elif config['experiment_type'] == 'scalability':
        return run_scalability_test(argparse.Namespace(**config))
    else:
        print(f"❌ Unknown experiment type: {config['experiment_type']}")
        return False

def create_experiment_report(experiment_dirs: List[str], output_dir: str):
    """Create a comprehensive experiment report."""
    print("📝 Generating Experiment Report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
        "summary": {}
    }
    
    # Collect results from each experiment
    for exp_dir in experiment_dirs:
        if not os.path.exists(exp_dir):
            continue
            
        exp_name = os.path.basename(exp_dir)
        report["experiments"][exp_name] = {
            "path": exp_dir,
            "files": os.listdir(exp_dir)
        }
        
        # Try to load specific result files
        result_files = ["combined_report.json", "validation_results.json", "benchmark_results.json"]
        for result_file in result_files:
            result_path = os.path.join(exp_dir, result_file)
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        report["experiments"][exp_name][result_file] = json.load(f)
                except Exception as e:
                    print(f"⚠️  Could not load {result_file}: {e}")
    
    # Generate summary
    total_experiments = len([d for d in experiment_dirs if os.path.exists(d)])
    report["summary"] = {
        "total_experiments": total_experiments,
        "successful_experiments": len(report["experiments"]),
        "experiment_types": list(report["experiments"].keys())
    }
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "experiment_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create markdown summary
    md_lines = [
        "# Enhanced RAG-CSD Experiment Report",
        f"\nGenerated: {report['timestamp']}",
        f"\nTotal Experiments: {report['summary']['total_experiments']}",
        f"Successful: {report['summary']['successful_experiments']}",
        "\n## Experiments\n"
    ]
    
    for exp_name, exp_data in report["experiments"].items():
        md_lines.append(f"### {exp_name}")
        md_lines.append(f"- Path: `{exp_data['path']}`")
        md_lines.append(f"- Files: {len(exp_data['files'])}")
        md_lines.append("")
    
    with open(os.path.join(output_dir, "experiment_report.md"), 'w') as f:
        f.write("\n".join(md_lines))
    
    print(f"📊 Experiment report saved to: {output_dir}")
    return report_path

def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG-CSD Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo
  python run_experiments.py demo

  # Run full benchmark
  python run_experiments.py benchmark --vector-db ./data/sample

  # Run ablation study
  python run_experiments.py ablation --vector-db ./data/sample --quick

  # Run scalability test
  python run_experiments.py scalability --vector-db ./data/sample

  # Run all experiments
  python run_experiments.py all --vector-db ./data/sample --output-base ./results
        """
    )
    
    parser.add_argument("experiment", choices=[
        "demo", "benchmark", "ablation", "scalability", "comparison", "custom", "all"
    ], help="Type of experiment to run")
    
    parser.add_argument("--vector-db", type=str, help="Path to vector database")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--output-base", type=str, default="./results",
                       help="Base output directory (for 'all' experiment)")
    
    # Experiment parameters
    parser.add_argument("--num-queries", type=int, default=20,
                       help="Number of queries for benchmarks")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of documents to retrieve")
    parser.add_argument("--runs-per-query", type=int, default=3,
                       help="Number of runs per query")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    # Flags
    parser.add_argument("--quick", action="store_true",
                       help="Run quick version with reduced parameters")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip benchmarking in demo")
    parser.add_argument("--skip-accuracy", action="store_true",
                       help="Skip accuracy validation")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate comprehensive report after experiments")
    
    args = parser.parse_args()
    
    # Adjust parameters for quick mode
    if args.quick:
        args.num_queries = min(args.num_queries, 10)
        args.runs_per_query = max(1, args.runs_per_query // 2)
    
    # Setup environment
    setup_environment()
    
    print("🚀 Enhanced RAG-CSD Experiment Runner")
    print("=" * 80)
    print(f"Experiment Type: {args.experiment}")
    print(f"Quick Mode: {'Yes' if args.quick else 'No'}")
    
    experiment_dirs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if args.experiment == "demo":
            result = run_demo(args)
            experiment_dirs.append(args.output_dir or "./results/demo")
            
        elif args.experiment == "benchmark":
            if not args.vector_db:
                print("❌ --vector-db is required for benchmark")
                sys.exit(1)
            result = run_benchmark(args)
            experiment_dirs.append(args.output_dir or "./results/benchmark")
            
        elif args.experiment == "ablation":
            if not args.vector_db:
                print("❌ --vector-db is required for ablation study")
                sys.exit(1)
            result = run_ablation_study(args)
            experiment_dirs.append(args.output_dir or "./results/ablation")
            
        elif args.experiment == "scalability":
            result = run_scalability_test(args)
            experiment_dirs.append(args.output_dir or "./results/scalability")
            
        elif args.experiment == "comparison":
            if not args.vector_db:
                print("❌ --vector-db is required for comparison study")
                sys.exit(1)
            result = run_comparison_study(args)
            experiment_dirs.append(args.output_dir or "./results/comparison")
            
        elif args.experiment == "custom":
            result = run_custom_experiment(args)
            experiment_dirs.append(args.output_dir or "./results/custom")
            
        elif args.experiment == "all":
            # Run all experiments
            base_dir = os.path.join(args.output_base, f"full_suite_{timestamp}")
            
            print("\n🔄 Running full experiment suite...")
            
            # Demo
            demo_args = argparse.Namespace(**vars(args))
            demo_args.output_dir = os.path.join(base_dir, "demo")
            run_demo(demo_args)
            experiment_dirs.append(demo_args.output_dir)
            
            # Benchmark (if vector db provided)
            if args.vector_db:
                bench_args = argparse.Namespace(**vars(args))
                bench_args.output_dir = os.path.join(base_dir, "benchmark")
                run_benchmark(bench_args)
                experiment_dirs.append(bench_args.output_dir)
                
                # Ablation
                abl_args = argparse.Namespace(**vars(args))
                abl_args.output_dir = os.path.join(base_dir, "ablation")
                run_ablation_study(abl_args)
                experiment_dirs.append(abl_args.output_dir)
                
                # Comparison
                comp_args = argparse.Namespace(**vars(args))
                comp_args.output_dir = os.path.join(base_dir, "comparison")
                run_comparison_study(comp_args)
                experiment_dirs.append(comp_args.output_dir)
            
            # Scalability
            scale_args = argparse.Namespace(**vars(args))
            scale_args.output_dir = os.path.join(base_dir, "scalability")
            run_scalability_test(scale_args)
            experiment_dirs.append(scale_args.output_dir)
            
            args.output_dir = base_dir
        
        # Generate comprehensive report if requested
        if args.generate_report or args.experiment == "all":
            report_dir = args.output_dir or f"./results/report_{timestamp}"
            create_experiment_report(experiment_dirs, report_dir)
        
        print("\n✅ All experiments completed successfully!")
        if experiment_dirs:
            print("📁 Results saved to:")
            for exp_dir in experiment_dirs:
                print(f"   - {exp_dir}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiments failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()