#!/usr/bin/env python3
"""
Script to gather all results from run_array_multi_agent_sycophancy.sh experiments
and consolidate them into simplified CSV files with only trigger numbers and metrics.
"""

import os
import pandas as pd
import argparse
from typing import Dict, List, Any, Optional

class ResultsGatherer:
    def __init__(self, output_dir: str = "output", target_round: int = -1):
        """
        Initialize the results gatherer.
        
        Args:
            output_dir: Base output directory containing experiment results
            target_round: Which round to extract (-1 for last round, 0+ for specific round)
        """
        self.output_dir = output_dir
        self.target_round = target_round
        self.accuracy_data = []
        self.dcr_data = []
        
    def find_experiment_directories(self) -> List[str]:
        """
        Find all experiment directories that contain acc_results.csv files.
        
        Returns:
            List of experiment directory paths
        """
        experiment_dirs = []
        
        for root, dirs, files in os.walk(self.output_dir):
            if 'acc_results.csv' in files:
                experiment_dirs.append(root)
        
        print(f"Found {len(experiment_dirs)} experiment directories")
        return sorted(experiment_dirs)
    
    def extract_trigger_numbers(self, exp_dir: str) -> tuple:
        """
        Extract trigger numbers from directory path.
        
        Args:
            exp_dir: Experiment directory path
            
        Returns:
            Tuple of (trigger_num_1, trigger_num_2, trigger_num_3)
        """
        parts = exp_dir.split(os.sep)
        
        # Look for different trigger patterns in path
        for part in parts:
            # Pattern: trigger_X_Y_Z (three triggers)
            if part.startswith("trigger_"):
                trigger_parts = part.split("_")
                if len(trigger_parts) >= 4:  # trigger, num1, num2, num3
                    trigger_num_1 = trigger_parts[1]
                    trigger_num_2 = trigger_parts[2]
                    trigger_num_3 = trigger_parts[3]
                    return trigger_num_1, trigger_num_2, trigger_num_3
                elif len(trigger_parts) == 3:  # trigger, num1, num2 (legacy support)
                    trigger_num_1 = trigger_parts[1]
                    trigger_num_2 = trigger_parts[2]
                    return trigger_num_1, trigger_num_2, "0"
        
        # Try to extract from model names or other patterns
        for part in parts:
            # Look for multi_agent patterns with numbers
            if "multi_agent" in part and any(char.isdigit() for char in part):
                # Extract numbers from multi_agent patterns
                import re
                numbers = re.findall(r'\d+', part)
                if len(numbers) >= 3:
                    return numbers[0], numbers[1], numbers[2]
                elif len(numbers) >= 2:
                    return numbers[0], numbers[1], "0"
                elif len(numbers) == 1:
                    return numbers[0], "0", "0"
        
        # Default fallback - use directory structure info
        # Try to use the last three meaningful parts as identifiers
        meaningful_parts = [p for p in parts if p not in ['output', 'eval', 'results']]
        if len(meaningful_parts) >= 3:
            return meaningful_parts[-3], meaningful_parts[-2], meaningful_parts[-1]
        elif len(meaningful_parts) >= 2:
            return meaningful_parts[-2], meaningful_parts[-1], "0"
        
        return "unknown", "unknown", "unknown"
    
    def load_acc_results(self, exp_dir: str) -> Optional[pd.DataFrame]:
        """
        Load acc_results.csv from experiment directory.
        
        Args:
            exp_dir: Experiment directory path
            
        Returns:
            DataFrame with accumulated results or None if not found
        """
        acc_results_csv = os.path.join(exp_dir, "acc_results.csv")
        
        if os.path.exists(acc_results_csv):
            try:
                return pd.read_csv(acc_results_csv)
            except Exception as e:
                print(f"Warning: Could not load {acc_results_csv}: {e}")
        
        return None
    
    def extract_metrics_from_round(self, df: pd.DataFrame, trigger_num_1: str, trigger_num_2: str, trigger_num_3: str):
        """
        Extract accuracy and DCR metrics from specified round.
        
        Args:
            df: DataFrame with accumulated results
            trigger_num_1: First trigger number
            trigger_num_2: Second trigger number
            trigger_num_3: Third trigger number
        """
        if df is None or df.empty:
            return
        
        # Select the target round
        if self.target_round == -1:
            # Use last round
            row = df.iloc[-1]
        else:
            # Use specific round
            round_rows = df[df['round'] == self.target_round]
            if round_rows.empty:
                print(f"Warning: Round {self.target_round} not found for triggers {trigger_num_1}, {trigger_num_2}, {trigger_num_3}")
                return
            row = round_rows.iloc[0]
        
        # Extract accuracy
        accuracy_val = row.get('accuracy', 0)
        if isinstance(accuracy_val, str) and accuracy_val.endswith('%'):
            accuracy_val = float(accuracy_val.rstrip('%'))
        else:
            accuracy_val = float(accuracy_val) * 100
        
        # Format as percentage with 2 decimal places
        accuracy_formatted = f"{accuracy_val:.2f}%"
        
        self.accuracy_data.append({
            'trigger_num_1': trigger_num_1,
            'trigger_num_2': trigger_num_2,
            'trigger_num_3': trigger_num_3,
            'accuracy': accuracy_formatted
        })
        
        # Extract DCR (disagreement collapse rate)
        dcr_val = row.get('accumulated_disagreement_collapse_rate', 0)
        if isinstance(dcr_val, str) and dcr_val.endswith('%'):
            dcr_val = float(dcr_val.rstrip('%'))
        else:
            dcr_val = float(dcr_val) * 100
        
        # Format as percentage with 2 decimal places
        dcr_formatted = f"{dcr_val:.2f}%"
        
        self.dcr_data.append({
            'trigger_num_1': trigger_num_1,
            'trigger_num_2': trigger_num_2,
            'trigger_num_3': trigger_num_3,
            'dcr': dcr_formatted
        })
    
    def gather_all_results(self):
        """
        Gather all results from experiment directories.
        """
        experiment_dirs = self.find_experiment_directories()
        
        if not experiment_dirs:
            print("No experiment directories found!")
            return
        
        print(f"Processing {len(experiment_dirs)} experiments...")
        print(f"Target round: {'last' if self.target_round == -1 else self.target_round}")
        
        for exp_dir in experiment_dirs:
            print(f"Processing: {exp_dir}")
            
            # Extract trigger numbers (now supports three triggers)
            trigger_num_1, trigger_num_2, trigger_num_3 = self.extract_trigger_numbers(exp_dir)
            
            # Load accumulated results
            acc_df = self.load_acc_results(exp_dir)
            
            if acc_df is not None:
                # Extract metrics from target round
                self.extract_metrics_from_round(acc_df, trigger_num_1, trigger_num_2, trigger_num_3)
            else:
                print(f"  Warning: No acc_results.csv found for {exp_dir}")
        
        print(f"Gathered {len(self.accuracy_data)} accuracy records")
        print(f"Gathered {len(self.dcr_data)} DCR records")
    
    def save_results(self, output_prefix: str = "consolidated"):
        """
        Save consolidated results to row format CSV files only.
        
        Args:
            output_prefix: Prefix for output files
        """
        # Save accuracy results in row format (trigger_num_1, trigger_num_2, trigger_num_3, accuracy)
        if self.accuracy_data:
            accuracy_df = pd.DataFrame(self.accuracy_data)
            
            # Check for duplicates and handle them
            print(f"Accuracy data shape: {accuracy_df.shape}")
            print(f"Unique trigger combinations: {len(accuracy_df[['trigger_num_1', 'trigger_num_2', 'trigger_num_3']].drop_duplicates())}")
            
            # Remove duplicates by taking the first occurrence
            accuracy_df = accuracy_df.drop_duplicates(subset=['trigger_num_1', 'trigger_num_2', 'trigger_num_3'], keep='first')
            
            # Save row format CSV
            accuracy_rows_file = f"{output_prefix}/accuracy_results.csv"
            accuracy_df.to_csv(accuracy_rows_file, index=False)
            print(f"Saved accuracy results to {accuracy_rows_file}")
            print(f"  Records: {len(accuracy_df)}")
        
        # Save DCR results in row format (trigger_num_1, trigger_num_2, trigger_num_3, dcr)
        if self.dcr_data:
            dcr_df = pd.DataFrame(self.dcr_data)
            
            # Check for duplicates and handle them
            print(f"DCR data shape: {dcr_df.shape}")
            print(f"Unique trigger combinations: {len(dcr_df[['trigger_num_1', 'trigger_num_2', 'trigger_num_3']].drop_duplicates())}")
            
            # Remove duplicates by taking the first occurrence
            dcr_df = dcr_df.drop_duplicates(subset=['trigger_num_1', 'trigger_num_2', 'trigger_num_3'], keep='first')
            
            # Save row format CSV
            dcr_rows_file = f"{output_prefix}/dcr_results.csv"
            dcr_df.to_csv(dcr_rows_file, index=False)
            print(f"Saved DCR results to {dcr_rows_file}")
            print(f"  Records: {len(dcr_df)}")
    
    def print_summary(self):
        """
        Print a summary of gathered results.
        """
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        
        if self.accuracy_data:
            accuracy_df = pd.DataFrame(self.accuracy_data)
            print(f"Accuracy Results: {len(accuracy_df)} records")
            
            # Show trigger combinations
            trigger_combos = accuracy_df[['trigger_num_1', 'trigger_num_2', 'trigger_num_3']].drop_duplicates()
            print(f"Unique trigger combinations: {len(trigger_combos)}")
            
            # Show sample data
            print("Sample accuracy data:")
            print(accuracy_df.head())
        
        if self.dcr_data:
            dcr_df = pd.DataFrame(self.dcr_data)
            print(f"\nDCR Results: {len(dcr_df)} records")
            
            # Show sample data
            print("Sample DCR data:")
            print(dcr_df.head())


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Gather simplified results from multi-agent sycophancy experiments")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Base output directory containing experiment results")
    parser.add_argument("--output-prefix", type=str, default="consolidated",
                        help="Prefix for output CSV files")
    parser.add_argument("--round", type=int, default=4,
                        help="Which round to extract (-1 for last round, 0+ for specific round)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    args = parser.parse_args()
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist!")
        print("Make sure you have run the experiments first using run_array_multi_agent_sycophancy.sh")
        return
    
    # Initialize gatherer
    gatherer = ResultsGatherer(args.output_dir, args.round)
    
    # Gather all results
    gatherer.gather_all_results()
    
    # Save results
    gatherer.save_results(args.output_dir)
    
    # Print summary
    gatherer.print_summary()
    
    print(f"\nDone! Check the generated CSV files:")
    print(f"  Row format:")
    print(f"    - {args.output_dir}/accuracy_results.csv (trigger_num_1, trigger_num_2, trigger_num_3, accuracy)")
    print(f"    - {args.output_dir}/dcr_results.csv (trigger_num_1, trigger_num_2, trigger_num_3, dcr)")


if __name__ == "__main__":
    main()
