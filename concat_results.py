#!/usr/bin/env python3
"""
Simple script to concatenate CSV results from different experimental settings.
"""

import os
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import glob

def find_csv_files(root_dir: str, csv_names: List[str] = None) -> Dict[str, List[str]]:
    """
    Find CSV files in the directory tree.
    
    Args:
        root_dir: Root directory to search
        csv_names: Specific CSV filenames to look for (e.g., ['results.csv', 'acc_results.csv'])
        
    Returns:
        Dictionary mapping CSV filename to list of file paths
    """
    if csv_names is None:
        csv_names = ['results.csv', 'acc_results.csv', 'sentence_similarity_results.csv', 
                    'accumulated_similarity_results.csv']
    
    csv_files = {}
    
    for csv_name in csv_names:
        csv_files[csv_name] = []
        
        # Search for files with this name
        pattern = os.path.join(root_dir, "**", csv_name)
        found_files = glob.glob(pattern, recursive=True)
        csv_files[csv_name].extend(found_files)
    
    # Also search for any CSV files
    all_csv_pattern = os.path.join(root_dir, "**", "*.csv")
    all_csv_files = glob.glob(all_csv_pattern, recursive=True)
    
    for csv_file in all_csv_files:
        filename = os.path.basename(csv_file)
        if filename not in csv_files:
            csv_files[filename] = []
        if csv_file not in csv_files[filename]:
            csv_files[filename].append(csv_file)
    
    # Remove empty entries
    csv_files = {k: v for k, v in csv_files.items() if v}
    
    print(f"Found CSV files:")
    for csv_name, files in csv_files.items():
        print(f"  {csv_name}: {len(files)} files")
    
    return csv_files

def extract_experiment_info(file_path: str) -> Dict[str, str]:
    """
    Extract experiment information from the file path.
    """
    path_parts = Path(file_path).parts
    
    info = {
        'dataset': 'unknown',
        'solver_type': 'unknown',
        'model_config': 'unknown',
        'experiment_name': 'unknown'
    }
    
    # Look for common patterns in the path
    for part in path_parts:
        if part in ['mmlu', 'mmlu_pro', 'gsm8k', 'bigbench']:
            info['dataset'] = part
        elif 'multi_agent' in part or 'single_agent' in part:
            info['solver_type'] = part
        elif any(model in part.lower() for model in ['claude', 'gpt', 'llama', 'qwen']):
            info['model_config'] = part
    
    # Create experiment name from relevant path parts
    relevant_parts = []
    for part in path_parts:
        if part not in ['src', 'sycophancy_multi_agent', 'output', 'outputs', 'eval']:
            relevant_parts.append(part)
    
    if len(relevant_parts) > 1:
        info['experiment_name'] = '/'.join(relevant_parts[:-1])  # Exclude filename
    else:
        info['experiment_name'] = relevant_parts[0] if relevant_parts else 'unknown'
    
    return info

def concatenate_csv_files(csv_file_groups: Dict[str, List[str]], output_dir: str) -> None:
    """
    Concatenate CSV files from different experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for csv_name, file_paths in csv_file_groups.items():
        if not file_paths:
            continue
            
        print(f"\nProcessing {csv_name} ({len(file_paths)} files)...")
        
        combined_data = []
        
        for file_path in file_paths:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                if df.empty:
                    print(f"  ⚠️  Empty file: {file_path}")
                    continue
                
                # Extract experiment info from the path
                experiment_info = extract_experiment_info(file_path)
                
                # Add experiment metadata columns
                for key, value in experiment_info.items():
                    df[key] = value
                
                # Add source file path for reference
                df['source_file'] = file_path
                
                combined_data.append(df)
                print(f"  ✅ Added {len(df)} rows from {experiment_info['experiment_name']}")
                
            except Exception as e:
                print(f"  ❌ Error reading {file_path}: {e}")
        
        if combined_data:
            # Concatenate all dataframes
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Save the combined CSV
            output_file = os.path.join(output_dir, f"combined_{csv_name}")
            combined_df.to_csv(output_file, index=False)
            print(f"  💾 Saved to: {output_file}")
            print(f"  📊 Total rows: {len(combined_df)}, Experiments: {len(combined_data)}")
            
            # Create a summary by experiment
            if 'experiment_name' in combined_df.columns:
                summary = combined_df.groupby('experiment_name').size().reset_index(name='row_count')
                summary_file = os.path.join(output_dir, f"summary_{csv_name}")
                summary.to_csv(summary_file, index=False)
                print(f"  📋 Summary saved to: {summary_file}")
        else:
            print(f"  ⚠️  No valid data found for {csv_name}")

def create_master_summary(csv_file_groups: Dict[str, List[str]], output_dir: str) -> None:
    """
    Create a master summary of all experiments and their CSV files.
    """
    summary_data = []
    
    all_experiments = set()
    for file_paths in csv_file_groups.values():
        for file_path in file_paths:
            experiment_info = extract_experiment_info(file_path)
            all_experiments.add(experiment_info['experiment_name'])
    
    for experiment in all_experiments:
        exp_data = {
            'experiment_name': experiment,
            'dataset': 'unknown',
            'solver_type': 'unknown',
            'model_config': 'unknown'
        }
        
        # Find any file from this experiment to get metadata
        for file_paths in csv_file_groups.values():
            for file_path in file_paths:
                exp_info = extract_experiment_info(file_path)
                if exp_info['experiment_name'] == experiment:
                    exp_data.update(exp_info)
                    break
        
        # Count available CSV types for this experiment
        csv_types = []
        for csv_name, file_paths in csv_file_groups.items():
            for file_path in file_paths:
                exp_info = extract_experiment_info(file_path)
                if exp_info['experiment_name'] == experiment:
                    csv_types.append(csv_name)
                    break
        
        exp_data['available_csv_files'] = ', '.join(csv_types)
        exp_data['csv_count'] = len(csv_types)
        
        summary_data.append(exp_data)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "master_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\n📋 Master summary saved to: {summary_file}")
        
        # Print statistics
        print(f"\n📊 Master Summary:")
        print(f"  Total experiments: {len(summary_data)}")
        print(f"  Datasets: {summary_df['dataset'].unique().tolist()}")
        print(f"  Solver types: {summary_df['solver_type'].unique().tolist()}")
        print(f"  CSV file types found: {list(csv_file_groups.keys())}")

def main():
    parser = argparse.ArgumentParser(description="Concatenate CSV results from different experimental settings")
    parser.add_argument("--root-dir", type=str, default=".", 
                       help="Root directory to search for CSV files (default: current directory)")
    parser.add_argument("--output-dir", type=str, default="concatenated_results",
                       help="Output directory for concatenated results")
    parser.add_argument("--csv-names", nargs='+', 
                       help="Specific CSV filenames to look for (e.g., results.csv acc_results.csv)")
    
    args = parser.parse_args()
    
    print(f"🔍 Searching for CSV files in {args.root_dir}...")
    
    # Find all CSV files
    csv_file_groups = find_csv_files(args.root_dir, args.csv_names)
    
    if not csv_file_groups:
        print("❌ No CSV files found")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    print(f"\n🔗 Concatenating CSV files...")
    
    # Concatenate CSV files
    concatenate_csv_files(csv_file_groups, output_dir)
    
    # Create master summary
    create_master_summary(csv_file_groups, output_dir)
    
    print(f"\n✅ Concatenation complete!")
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()
