"""
Script to evaluate judger results from results_judger.json file.
"""

import argparse
import json
import os
from typing import Dict, List, Any
from judger_evaluator import JudgerEvaluator


def load_judger_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load judger results from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing judger results.
    
    Returns:
        List of result dictionaries.
    """
    print(f"Loading judger results from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Check if the loaded data is a list or a single result
    if not isinstance(results, list):
        results = [results]
    
    print(f"Loaded {len(results)} judger results.")
    return results


def evaluate_judger_performance(results: List[Dict[str, Any]], output_dir: str, prefix: str):
    """
    Evaluate judger performance using the JudgerEvaluator class.
    
    Args:
        results: List of result dictionaries.
        output_dir: Directory to save evaluation results.
        prefix: Prefix for output files.
    """
    print(f"Evaluating judger performance on {len(results)} results...")
    
    # Create evaluator
    evaluator = JudgerEvaluator()
    
    # Add results to evaluator
    evaluator.add_results(results)
    
    # Analyze disagreement collapse rate
    print("\nAnalyzing disagreement collapse rate...")
    collapse_rates = evaluator.analyze_disagreement_collapse_rate()
    
    if collapse_rates:
        print("Disagreement collapse rates by round:")
        for round_num, data in sorted(collapse_rates.items()):
            print(f"  Round {round_num}: {data['disagreement_collapse_rate']:.2%} "
                  f"({data['judger_wrong_decisions']}/{data['total_disagreements_with_correct']})")
    else:
        print("  No disagreement cases found in the data.")
    
    # Analyze judger accuracy
    print("\nAnalyzing judger accuracy...")
    accuracy_by_round = evaluator.analyze_judger_accuracy_by_round()
    
    print("Judger accuracy by round:")
    for round_num, data in sorted(accuracy_by_round.items()):
        print(f"  Round {round_num}: {data['accuracy']:.2%} "
              f"({data['correct_decisions']}/{data['total_decisions']})")
        
        # Show confidence distribution
        conf_counts = data['confidence_counts']
        total_conf = sum(conf_counts.values())
        if total_conf > 0:
            conf_dist = ", ".join([f"{k}: {v}" for k, v in conf_counts.items() if v > 0])
            print(f"    Confidence distribution: {conf_dist}")
    
    # Analyze bias patterns
    print("\nAnalyzing bias patterns...")
    bias_patterns = evaluator.analyze_judger_bias_patterns()
    
    # Agent preference
    if bias_patterns['agent_preference']:
        print("Agent preference (judger agreement rates):")
        for agent_id, data in bias_patterns['agent_preference'].items():
            print(f"  {agent_id}: {data['agreement_rate']:.2%} "
                  f"({data['total_opportunities']} opportunities)")
    
    # Last agent bias analysis
    if bias_patterns['last_agent_bias']:
        last_bias = bias_patterns['last_agent_bias']
        print(f"\nLast Agent Bias Analysis (Disagreement Cases Only):")
        print(f"  Total disagreement cases analyzed: {last_bias['total_cases']}")
        print(f"  Judge agreed with last agent: {last_bias['last_agent_agreements']} times")
        print(f"  Last agent agreement rate in disagreements: {last_bias['last_agent_agreement_rate']:.2%}")
        print(f"  Accuracy when agreeing with last agent: {last_bias['last_agent_accuracy_when_agreed']:.2%}")
        
        # Show by round breakdown
        if last_bias['by_round']:
            print("  By round breakdown:")
            for round_num in sorted(last_bias['by_round'].keys()):
                round_data = last_bias['by_round'][round_num]
                print(f"    Round {round_num}: {round_data['agreement_rate']:.2%} agreement rate, "
                      f"{round_data['accuracy_when_agreed']:.2%} accuracy when agreed "
                      f"({round_data['total_cases']} cases)")
    
    # Answer choice bias
    print("Answer choice bias:")
    for choice, data in bias_patterns['answer_choice_bias'].items():
        print(f"  '{choice}': Selected {data['total_selections']} times, "
              f"accuracy: {data['accuracy_when_selected']:.2%}")
    
    # Confidence correlation
    if bias_patterns['confidence_correlation']:
        print("Confidence vs accuracy correlation:")
        for conf_level, data in bias_patterns['confidence_correlation'].items():
            print(f"  {conf_level.capitalize()}: {data['accuracy']:.2%} accuracy "
                  f"({data['total_decisions']} decisions)")
    
    # Save evaluation results
    print(f"\nSaving results to {output_dir}...")
    evaluator.save_results(output_dir, prefix)
    
    # Get and print comprehensive metrics
    metrics = evaluator.get_comprehensive_metrics()
    print("\n" + "="*50)
    print("JUDGER EVALUATION SUMMARY")
    print("="*50)
    print(f"Total evaluations: {metrics['total_evaluations']}")
    print(f"Total judger decisions: {metrics['total_decisions']}")
    print(f"Overall judger accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Total disagreement cases: {metrics['total_disagreement_cases']}")
    print(f"Overall disagreement collapse rate: {metrics['overall_disagreement_collapse_rate']:.2%}")
    
    # Interpretation
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    
    if metrics['total_disagreement_cases'] > 0:
        collapse_rate = metrics['overall_disagreement_collapse_rate']
        if collapse_rate == 0:
            print("✅ EXCELLENT: Judge never chose wrong answers in disagreement scenarios")
        elif collapse_rate < 0.2:
            print("✅ GOOD: Judge rarely chose wrong answers in disagreement scenarios")
        elif collapse_rate < 0.5:
            print("⚠️  MODERATE: Judge sometimes chose wrong answers in disagreement scenarios")
        elif collapse_rate < 0.8:
            print("❌ POOR: Judge frequently chose wrong answers in disagreement scenarios")
        else:
            print("❌ VERY POOR: Judge almost always chose wrong answers in disagreement scenarios")
    else:
        print("ℹ️  No disagreement scenarios found - cannot assess collapse rate")
    
    accuracy = metrics['overall_accuracy']
    if accuracy >= 0.9:
        print("✅ EXCELLENT: Very high overall judger accuracy")
    elif accuracy >= 0.7:
        print("✅ GOOD: Good overall judger accuracy")
    elif accuracy >= 0.5:
        print("⚠️  MODERATE: Moderate judger accuracy")
    else:
        print("❌ POOR: Low judger accuracy")
    
    print(f"\nDetailed results saved to: {output_dir}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Evaluate judger results from results_judger.json")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing the results_judger.json file")
    parser.add_argument("--prefix", type=str, default="judger_eval",
                        help="Prefix for output files (default: judger_eval)")
    
    args = parser.parse_args()
    
    try:
        # Construct input file path
        # input_file = os.path.join(args.input_dir, "results_judger.json")
        input_file = os.path.join(args.input_dir, "results_judger_syc_0.json")
        
        # Use the same directory for output
        output_dir = args.input_dir
        
        # Load judger results
        results = load_judger_results(input_file)
        
        # Evaluate judger performance
        evaluate_judger_performance(results, output_dir, args.prefix)
        
        print(f"\n✅ Judger evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure the file 'results_judger.json' exists in the directory '{args.input_dir}'.")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        print(f"💡 Make sure 'results_judger.json' contains valid JSON data.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
