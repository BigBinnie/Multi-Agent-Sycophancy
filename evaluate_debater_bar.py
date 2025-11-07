#!/usr/bin/env python3
"""
Script to generate blind_agreement_results.csv with average blind agreement rates
by round for each agent across the dataset.
"""

import argparse
import json
import pandas as pd
import os
from typing import Dict, List, Any
from blind_agreement_evaluator import BlindAgreementEvaluator

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load results from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing results.
    
    Returns:
        List of result dictionaries.
    """
    print(f"Loading results from {file_path}...")
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Check if the loaded data is a list or a single result
    if not isinstance(results, list):
        results = [results]
    
    print(f"Loaded {len(results)} results.")
    return results

def convert_json_to_evaluator_format(json_results):
    """Convert JSON results to the format expected by BlindAgreementEvaluator."""
    evaluator = BlindAgreementEvaluator()
    evaluator.start_evaluation()
    
    for result in json_results:
        if 'dialogue_histories' not in result:
            continue
        question = result['question']
        correct_answer = result['correct_answer']
        dialogue_histories = result['dialogue_histories']
        answers_by_round = result['answers_by_round']
        
        # Convert string keys to integers for rounds
        converted_answers_by_round = {}
        for agent_id, rounds_data in answers_by_round.items():
            converted_answers_by_round[agent_id] = {}
            for round_str, answer in rounds_data.items():
                converted_answers_by_round[agent_id][int(round_str)] = answer
        
        evaluator.add_dialogue_result(
            question=question,
            correct_answer=correct_answer,
            dialogue_histories=dialogue_histories,
            answers_by_round=converted_answers_by_round
        )
    
    return evaluator

def evaluate_results(results: List[Dict[str, Any]], output_dir: str, prefix: str):
    """
    Evaluate results using the BlindAgreementEvaluator class.
    
    Args:
        results: List of result dictionaries.
        output_dir: Directory to save evaluation results.
        prefix: Prefix for output files.
    """
    print(f"Evaluating {len(results)} results...")
    
    # Convert to evaluator format
    evaluator = convert_json_to_evaluator_format(results)
    
    # Calculate blind agreement scores
    print("Calculating blind agreement scores...")
    evaluator.calculate_blind_agreement()
    
    # Save results
    output_csv = os.path.join(output_dir, "blind_agreement_results.csv")
    claude_responses_file = os.path.join(output_dir, "blind_agreement_gpt_responses.json")
    blind_scores_file = os.path.join(output_dir, "blind_agreement_scores.json")
    accumulated_csv = os.path.join(output_dir, "accumulated_blind_agreement_results.csv")
    
    # Save all results
    evaluator.save_consolidated_results(output_csv)
    evaluator.save_gpt_responses(claude_responses_file)
    evaluator.save_blind_agreement_scores(blind_scores_file)
    evaluator.calculate_accumulated_blind_agreement(accumulated_csv)
    
    print(f"\n✅ Blind agreement results saved to: {output_csv}")
    print(f"✅ Claude responses saved to: {claude_responses_file}")
    print(f"✅ Raw blind agreement scores saved to: {blind_scores_file}")
    print(f"✅ Accumulated blind agreement results saved to: {accumulated_csv}")

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Evaluate blind agreement from a JSON file.")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the JSON file containing results.")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results.")
    parser.add_argument("--prefix", type=str, default="eval",
                        help="Prefix for output files.")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.input)
    
    # Evaluate results
    evaluate_results(results, args.output_dir, args.prefix)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
