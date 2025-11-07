"""
Script to evaluate results from a JSON file using the MathEvaluator class.
"""

import argparse
import json
import os
from typing import Dict, List, Any
from evaluate import MathEvaluator

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

def evaluate_results(results: List[Dict[str, Any]], output_dir: str, prefix: str):
    """
    Evaluate results using the MathEvaluator class.
    
    Args:
        results: List of result dictionaries.
        output_dir: Directory to save evaluation results.
        prefix: Prefix for output files.
    """
    print(f"Evaluating {len(results)} results...")
    
    # Create evaluator
    evaluator = MathEvaluator()
    evaluator.start_evaluation()
    
    # Add each result to the evaluator
    for result in results:
        # Extract required fields, with defaults if not present
        question = result.get('question', '')
        correct_answer = result.get('correct_answer', '')
        predicted_answer = result.get('predicted_answer', '')
        solver_response = result.get('solver_response', '')
        processing_time = result.get('processing_time', 0.0)
        individual_answers = result.get('individual_answers', [])
        answers_by_round = result.get('answers_by_round', {})
        confidence_by_round = result.get('confidence_by_round', {})
        system_prompts = result.get('system_prompts', {})
        dialogue_histories = result.get('dialogue_histories', {})
        
        # Add result to evaluator
        evaluator.add_result(
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted_answer,
            solver_response=solver_response,
            processing_time=processing_time,
            individual_answers=individual_answers,
            answers_by_round=answers_by_round,
            confidence_by_round=confidence_by_round,
            system_prompts=system_prompts,
            dialogue_histories=dialogue_histories
        )
    
    # Save evaluation results
    evaluator.save_results(output_dir, prefix)
    
    # Get and print metrics
    metrics = evaluator.get_metrics()
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Evaluate results from a JSON file.")
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
    
    print(f"Evaluation complete. Results saved to {args.output_dir}/{args.prefix}")

if __name__ == "__main__":
    main()
