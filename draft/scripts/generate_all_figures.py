#!/usr/bin/env python3
"""
Master Script to Generate All Motivation Figures for Enhanced RAG-CSD Paper

This script runs all analysis scripts and generates a comprehensive set of figures
and tables for the academic paper motivation section.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path: str, description: str) -> bool:
    """Run a script and return success status."""
    print(f"🔄 Running {description}...")
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 {description} crashed: {e}")
        return False

def main():
    """Generate all motivation figures and analysis."""
    print("🚀 Enhanced RAG-CSD: Academic Paper Figure Generation")
    print("=" * 60)
    print("📊 Generating comprehensive motivation analysis...")
    print()
    
    # Change to the correct directory (from enhanced-rag-csd root)
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent.parent)
    
    # List of analysis scripts to run
    analysis_scripts = [
        ("draft/scripts/gpu_memory_analysis.py", "GPU Memory Swapping Analysis"),
        ("draft/scripts/data_movement_analysis.py", "Data Movement and Bandwidth Analysis"),
        ("draft/scripts/computational_comparison.py", "Computational Capability Comparison"),
        ("draft/scripts/csd_capabilities_analysis.py", "Modern CSD Capabilities Analysis"),
        ("draft/scripts/index_rebuilding_analysis.py", "Index Rebuilding Cost Analysis")
    ]
    
    results = []
    total_scripts = len(analysis_scripts)
    
    for i, (script_path, description) in enumerate(analysis_scripts, 1):
        print(f"📈 [{i}/{total_scripts}] {description}")
        success = run_script(script_path, description)
        results.append((description, success))
        print()
    
    # Summary
    print("📋 Generation Summary")
    print("=" * 60)
    
    successful = sum(1 for _, success in results if success)
    failed = total_scripts - successful
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {description}")
    
    print()
    print(f"📊 Results: {successful}/{total_scripts} scripts completed successfully")
    
    if failed > 0:
        print(f"⚠️  {failed} scripts failed - check error messages above")
        return 1
    
    # Count generated files
    figures_dir = Path("draft/figures")
    if figures_dir.exists():
        pdf_files = list(figures_dir.glob("*.pdf"))
        csv_files = list(figures_dir.glob("*.csv"))
        
        print(f"📄 Generated {len(pdf_files)} PDF figures")
        print(f"📊 Generated {len(csv_files)} CSV data files")
        print()
        print("🎯 All motivation analysis complete!")
        print(f"📁 Files saved to: {figures_dir.absolute()}")
        
        # List key figures for academic paper
        print()
        print("📚 Key Figures for Academic Paper:")
        key_figures = [
            "gpu_memory_timeline.pdf",
            "memory_breakdown_comparison.pdf", 
            "data_movement_latency_comparison.pdf",
            "bandwidth_utilization_analysis.pdf",
            "computational_comparison.pdf",
            "efficiency_breakdown.pdf",
            "csd_capabilities_overview.pdf",
            "ml_operation_performance.pdf",
            "index_rebuilding_cost_comparison.pdf",
            "dynamic_update_analysis.pdf"
        ]
        
        for figure in key_figures:
            if (figures_dir / figure).exists():
                print(f"  ✅ {figure}")
            else:
                print(f"  ❌ {figure} (missing)")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)