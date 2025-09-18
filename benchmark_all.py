#!/usr/bin/env python3
import os
import glob
import subprocess
import json
import fire
from pathlib import Path

from models.utils import pretty_print_results


def main(image_dir, annotation_file, buffer_time=0.0, models_dir="models", output_dir="benchmark_results"):
    """Run all benchmark models and collect outputs into one list."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find and run all benchmark scripts
    scripts = glob.glob(f"{models_dir}/benchmark_*.py")
    all_results = []
    
    for script in scripts:
        # Generate output file name
        script_name = Path(script).stem
        output_file = f"{output_dir}/{script_name}_results.txt"
        
        # Run the script
        try:
            subprocess.run(["python", script, image_dir, annotation_file, str(buffer_time), output_file], check=True)
            
            # Load results
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read().strip()
                
                try:
                    results = json.loads(content)
                except json.JSONDecodeError:
                    results = [line for line in content.split('\n') if line.strip()]
                
                # Add to combined list
                all_results.extend(results)
            else:
                raise ValueError(f"Output file {output_file} does not exist")
                    
        except subprocess.CalledProcessError:
            print(f"Failed to run {script}")
    
    # Save and print results
    combined_file = f"{output_dir}/combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    pretty_print_results(all_results)


if __name__ == "__main__":
    fire.Fire(main)