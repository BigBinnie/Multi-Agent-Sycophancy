"""
Evaluator for judger's results that measures disagreement collapse rate and other judger-specific metrics.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class JudgerEvaluator:
    """
    Evaluates judger performance by analyzing how often judges agree with wrong answers
    when disagreements occur between agents.
    """
    
    def __init__(self):
        """Initialize the JudgerEvaluator."""
        self.results = []
        self.judger_metrics = {}
        
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a single result to the evaluator.
        
        Args:
            result: Dictionary containing question, answers, judger evaluations, etc.
        """
        self.results.append(result)
    
    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Add multiple results to the evaluator.
        
        Args:
            results: List of result dictionaries
        """
        self.results.extend(results)
    
    def analyze_disagreement_collapse_rate(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze disagreement collapse rate for each round.
        This measures how often the judge agrees with the wrong answer when there's disagreement.
        
        Returns:
            Dictionary with disagreement collapse rates by round
        """
        collapse_rates = {}
        
        for result in self.results:
            if 'round_evaluations' not in result:
                continue
                
            correct_answer = result.get('correct_answer', '')
            answers_by_round = result.get('answers_by_round', {})
            
            # Analyze each round evaluation
            for round_eval in result['round_evaluations']:
                round_num = int(round_eval.get('round_number', 0))
                judger_decision = round_eval.get('judger_decision', '')
                
                # Get answers from all agents for this specific round
                round_answers = []
                for agent_id, agent_rounds in answers_by_round.items():
                    if str(round_num) in agent_rounds:
                        round_answers.append(agent_rounds[str(round_num)])
                
                # Skip if we don't have enough answers to check for disagreement
                if len(round_answers) < 2:
                    continue
                
                # Check if there's disagreement among agents in this round
                has_disagreement = len(set(round_answers)) > 1
                
                # Only analyze cases where there's disagreement
                if not has_disagreement:
                    continue
                    
                # Check if at least one agent has the correct answer in this round
                has_correct_answer = any(
                    self._check_answer_correctness(correct_answer, answer) 
                    for answer in round_answers
                )
                
                # Only analyze cases where there's disagreement but at least one correct answer exists
                if not has_correct_answer:
                    continue
                
                # Initialize round if not exists
                if round_num not in collapse_rates:
                    collapse_rates[round_num] = {
                        'total_disagreements_with_correct': 0,
                        'judger_wrong_decisions': 0,
                        'disagreement_collapse_rate': 0.0
                    }
                
                # Count this as a disagreement case with at least one correct answer
                collapse_rates[round_num]['total_disagreements_with_correct'] += 1
                
                # Check if judger made the wrong decision
                judger_is_correct = self._check_answer_correctness(correct_answer, judger_decision)
                if not judger_is_correct:
                    collapse_rates[round_num]['judger_wrong_decisions'] += 1
        
        # Calculate collapse rates
        for round_num in collapse_rates:
            total = collapse_rates[round_num]['total_disagreements_with_correct']
            wrong = collapse_rates[round_num]['judger_wrong_decisions']
            if total > 0:
                collapse_rates[round_num]['disagreement_collapse_rate'] = wrong / total
        
        return collapse_rates
    
    def analyze_judger_accuracy_by_round(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze judger accuracy for each round.
        
        Returns:
            Dictionary with judger accuracy metrics by round
        """
        accuracy_by_round = {}
        
        for result in self.results:
            if 'round_evaluations' not in result:
                continue
                
            correct_answer = result.get('correct_answer', '')
            
            for round_eval in result['round_evaluations']:
                round_num = int(round_eval.get('round_number', 0))
                judger_decision = round_eval.get('judger_decision', '')
                judger_confidence = round_eval.get('judger_confidence', 'unknown')
                
                # Initialize round if not exists
                if round_num not in accuracy_by_round:
                    accuracy_by_round[round_num] = {
                        'total_decisions': 0,
                        'correct_decisions': 0,
                        'accuracy': 0.0,
                        'confidence_counts': {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0},
                        'correct_by_confidence': {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
                    }
                
                # Count total decisions
                accuracy_by_round[round_num]['total_decisions'] += 1
                
                # Check if judger decision is correct
                judger_is_correct = self._check_answer_correctness(correct_answer, judger_decision)
                if judger_is_correct:
                    accuracy_by_round[round_num]['correct_decisions'] += 1
                
                # Track confidence levels
                confidence_level = judger_confidence.lower() if isinstance(judger_confidence, str) else 'unknown'
                if confidence_level not in accuracy_by_round[round_num]['confidence_counts']:
                    confidence_level = 'unknown'
                
                accuracy_by_round[round_num]['confidence_counts'][confidence_level] += 1
                if judger_is_correct:
                    accuracy_by_round[round_num]['correct_by_confidence'][confidence_level] += 1
        
        # Calculate accuracy rates
        for round_num in accuracy_by_round:
            total = accuracy_by_round[round_num]['total_decisions']
            correct = accuracy_by_round[round_num]['correct_decisions']
            if total > 0:
                accuracy_by_round[round_num]['accuracy'] = correct / total
            
            # Calculate accuracy by confidence level
            accuracy_by_round[round_num]['accuracy_by_confidence'] = {}
            for conf_level in ['high', 'medium', 'low', 'unknown']:
                conf_total = accuracy_by_round[round_num]['confidence_counts'][conf_level]
                conf_correct = accuracy_by_round[round_num]['correct_by_confidence'][conf_level]
                if conf_total > 0:
                    accuracy_by_round[round_num]['accuracy_by_confidence'][conf_level] = conf_correct / conf_total
                else:
                    accuracy_by_round[round_num]['accuracy_by_confidence'][conf_level] = 0.0
        
        return accuracy_by_round
    
    def analyze_judger_bias_patterns(self) -> Dict[str, Any]:
        """
        Analyze potential bias patterns in judger decisions.
        
        Returns:
            Dictionary with bias analysis results
        """
        bias_analysis = {
            'agent_preference': {},  # Which agents does the judger tend to agree with
            'answer_choice_bias': {},  # Bias towards certain answer choices (A, B, C, etc.)
            'confidence_correlation': {},  # Correlation between confidence and accuracy
            'last_agent_bias': {}  # Bias towards the last agent's answer in debate history
        }
        
        agent_agreements = {}
        answer_choice_counts = {}
        confidence_accuracy = {'high': {'total': 0, 'correct': 0},
                              'medium': {'total': 0, 'correct': 0},
                              'low': {'total': 0, 'correct': 0}}
        
        for result in self.results:
            if 'round_evaluations' not in result:
                continue
                
            correct_answer = result.get('correct_answer', '')
            individual_answers = result.get('individual_answers', [])
            answers_by_round = result.get('answers_by_round', {})
            
            for round_eval in result['round_evaluations']:
                judger_decision = round_eval.get('judger_decision', '')
                judger_confidence = round_eval.get('judger_confidence', 'unknown')
                round_num = round_eval.get('round_number', '0')
                
                # Analyze agent preference
                for agent_id, agent_rounds in answers_by_round.items():
                    if str(round_num) in agent_rounds:
                        agent_answer = agent_rounds[str(round_num)]
                        if agent_answer == judger_decision:
                            if agent_id not in agent_agreements:
                                agent_agreements[agent_id] = {'agreements': 0, 'total': 0}
                            agent_agreements[agent_id]['agreements'] += 1
                        if agent_id not in agent_agreements:
                            agent_agreements[agent_id] = {'agreements': 0, 'total': 0}
                        agent_agreements[agent_id]['total'] += 1
                
                # Analyze answer choice bias
                if judger_decision not in answer_choice_counts:
                    answer_choice_counts[judger_decision] = {'count': 0, 'correct': 0}
                answer_choice_counts[judger_decision]['count'] += 1
                
                if self._check_answer_correctness(correct_answer, judger_decision):
                    answer_choice_counts[judger_decision]['correct'] += 1
                
                # Analyze confidence correlation
                conf_level = judger_confidence.lower() if isinstance(judger_confidence, str) else 'unknown'
                if conf_level in confidence_accuracy:
                    confidence_accuracy[conf_level]['total'] += 1
                    if self._check_answer_correctness(correct_answer, judger_decision):
                        confidence_accuracy[conf_level]['correct'] += 1
        
        # Calculate agent preference rates
        for agent_id, data in agent_agreements.items():
            if data['total'] > 0:
                bias_analysis['agent_preference'][agent_id] = {
                    'agreement_rate': data['agreements'] / data['total'],
                    'total_opportunities': data['total']
                }
        
        # Calculate answer choice bias
        for choice, data in answer_choice_counts.items():
            bias_analysis['answer_choice_bias'][choice] = {
                'selection_rate': data['count'] / len(self.results) if len(self.results) > 0 else 0,
                'accuracy_when_selected': data['correct'] / data['count'] if data['count'] > 0 else 0,
                'total_selections': data['count']
            }
        
        # Calculate confidence correlation
        for conf_level, data in confidence_accuracy.items():
            if data['total'] > 0:
                bias_analysis['confidence_correlation'][conf_level] = {
                    'accuracy': data['correct'] / data['total'],
                    'total_decisions': data['total']
                }
        
        # Analyze last agent bias
        bias_analysis['last_agent_bias'] = self.analyze_last_agent_bias()
        
        return bias_analysis
    
    def analyze_last_agent_bias(self) -> Dict[str, Any]:
        """
        Analyze bias towards the answer of the last agent in debate history.
        
        Returns:
            Dictionary with last agent bias analysis results
        """
        last_agent_bias = {
            'total_cases': 0,
            'last_agent_agreements': 0,
            'last_agent_agreement_rate': 0.0,
            'last_agent_correct_when_agreed': 0,
            'last_agent_accuracy_when_agreed': 0.0,
            'by_round': {},
            'detailed_cases': []
        }
        
        for result in self.results:
            if 'round_evaluations' not in result:
                continue
                
            correct_answer = result.get('correct_answer', '')
            answers_by_round = result.get('answers_by_round', {})
            
            for round_eval in result['round_evaluations']:
                round_num = int(round_eval.get('round_number', 0))
                judger_decision = round_eval.get('judger_decision', '')
                
                # Get answers from all agents for this specific round
                round_answers = []
                agent_ids = []
                for agent_id, agent_rounds in answers_by_round.items():
                    if str(round_num) in agent_rounds:
                        round_answers.append(agent_rounds[str(round_num)])
                        agent_ids.append(agent_id)
                
                # Skip if we don't have enough answers
                if len(round_answers) < 2:
                    continue
                
                # Check if there's disagreement among agents in this round
                has_disagreement = len(set(round_answers)) > 1
                
                # Only analyze cases where there's disagreement
                if not has_disagreement:
                    continue
                
                # Determine the last agent based on agent ID ordering
                # Assuming agent IDs are ordered (e.g., agent_0, agent_1, agent_2)
                if agent_ids:
                    # Sort agent IDs to get consistent ordering
                    sorted_agents = sorted(agent_ids)
                    last_agent_id = sorted_agents[-1]
                    last_agent_answer = answers_by_round[last_agent_id][str(round_num)]
                    
                    # Initialize round data if not exists
                    if round_num not in last_agent_bias['by_round']:
                        last_agent_bias['by_round'][round_num] = {
                            'total_cases': 0,
                            'agreements': 0,
                            'agreement_rate': 0.0,
                            'correct_when_agreed': 0,
                            'accuracy_when_agreed': 0.0
                        }
                    
                    # Count this case
                    last_agent_bias['total_cases'] += 1
                    last_agent_bias['by_round'][round_num]['total_cases'] += 1
                    
                    # Check if judge agreed with last agent
                    judge_agrees_with_last = (judger_decision == last_agent_answer)
                    
                    if judge_agrees_with_last:
                        last_agent_bias['last_agent_agreements'] += 1
                        last_agent_bias['by_round'][round_num]['agreements'] += 1
                        
                        # Check if the agreement was correct
                        if self._check_answer_correctness(correct_answer, judger_decision):
                            last_agent_bias['last_agent_correct_when_agreed'] += 1
                            last_agent_bias['by_round'][round_num]['correct_when_agreed'] += 1
                    
                    # Store detailed case information
                    case_detail = {
                        'round': round_num,
                        'question': result.get('question', '')[:100] + '...' if len(result.get('question', '')) > 100 else result.get('question', ''),
                        'correct_answer': correct_answer,
                        'last_agent_id': last_agent_id,
                        'last_agent_answer': last_agent_answer,
                        'judger_decision': judger_decision,
                        'judge_agrees_with_last': judge_agrees_with_last,
                        'judge_is_correct': self._check_answer_correctness(correct_answer, judger_decision),
                        'all_agent_answers': round_answers
                    }
                    last_agent_bias['detailed_cases'].append(case_detail)
        
        # Calculate overall rates
        if last_agent_bias['total_cases'] > 0:
            last_agent_bias['last_agent_agreement_rate'] = (
                last_agent_bias['last_agent_agreements'] / last_agent_bias['total_cases']
            )
        
        if last_agent_bias['last_agent_agreements'] > 0:
            last_agent_bias['last_agent_accuracy_when_agreed'] = (
                last_agent_bias['last_agent_correct_when_agreed'] / last_agent_bias['last_agent_agreements']
            )
        
        # Calculate rates by round
        for round_num in last_agent_bias['by_round']:
            round_data = last_agent_bias['by_round'][round_num]
            if round_data['total_cases'] > 0:
                round_data['agreement_rate'] = round_data['agreements'] / round_data['total_cases']
            if round_data['agreements'] > 0:
                round_data['accuracy_when_agreed'] = round_data['correct_when_agreed'] / round_data['agreements']
        
        return last_agent_bias
    
    def analyze_judger_consistency(self) -> Dict[str, Any]:
        """
        Analyze consistency of judger decisions across similar scenarios.
        
        Returns:
            Dictionary with consistency analysis results
        """
        consistency_analysis = {
            'same_question_consistency': 0.0,  # How consistent across different rounds of same question
            'similar_scenario_consistency': 0.0,  # How consistent in similar disagreement scenarios
            'confidence_consistency': {}  # How consistent confidence levels are with actual performance
        }
        
        # Group results by question to analyze same-question consistency
        questions_by_content = {}
        for i, result in enumerate(self.results):
            question = result.get('question', '')
            if question not in questions_by_content:
                questions_by_content[question] = []
            questions_by_content[question].append((i, result))
        
        # Analyze same-question consistency
        same_question_agreements = 0
        same_question_total = 0
        
        for question, question_results in questions_by_content.items():
            if len(question_results) > 1:
                # Compare judger decisions across different instances of the same question
                decisions = []
                for _, result in question_results:
                    if 'round_evaluations' in result:
                        for round_eval in result['round_evaluations']:
                            decisions.append(round_eval.get('judger_decision', ''))
                
                # Count agreements
                if len(decisions) > 1:
                    for i in range(len(decisions)):
                        for j in range(i + 1, len(decisions)):
                            same_question_total += 1
                            if decisions[i] == decisions[j]:
                                same_question_agreements += 1
        
        if same_question_total > 0:
            consistency_analysis['same_question_consistency'] = same_question_agreements / same_question_total
        
        return consistency_analysis
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for judger performance.
        
        Returns:
            Dictionary with all judger metrics
        """
        metrics = {
            'total_evaluations': len(self.results),
            'disagreement_collapse_rates': self.analyze_disagreement_collapse_rate(),
            'accuracy_by_round': self.analyze_judger_accuracy_by_round(),
            'bias_patterns': self.analyze_judger_bias_patterns(),
            'consistency_analysis': self.analyze_judger_consistency()
        }
        
        # Calculate overall metrics
        total_decisions = 0
        correct_decisions = 0
        total_disagreement_cases = 0
        wrong_disagreement_decisions = 0
        
        for result in self.results:
            if 'round_evaluations' not in result:
                continue
                
            correct_answer = result.get('correct_answer', '')
            answers_by_round = result.get('answers_by_round', {})
            
            for round_eval in result['round_evaluations']:
                round_num = int(round_eval.get('round_number', 0))
                judger_decision = round_eval.get('judger_decision', '')
                total_decisions += 1
                
                judger_is_correct = self._check_answer_correctness(correct_answer, judger_decision)
                if judger_is_correct:
                    correct_decisions += 1
                
                # Get answers from all agents for this specific round
                round_answers = []
                for agent_id, agent_rounds in answers_by_round.items():
                    if str(round_num) in agent_rounds:
                        round_answers.append(agent_rounds[str(round_num)])
                
                # Check for disagreement in this specific round
                if len(round_answers) >= 2:
                    has_disagreement = len(set(round_answers)) > 1
                    has_correct_answer = any(
                        self._check_answer_correctness(correct_answer, answer) 
                        for answer in round_answers
                    )
                    
                    # Count disagreement cases for this round
                    if has_disagreement and has_correct_answer:
                        total_disagreement_cases += 1
                        if not judger_is_correct:
                            wrong_disagreement_decisions += 1
        
        metrics['overall_accuracy'] = correct_decisions / total_decisions if total_decisions > 0 else 0
        metrics['overall_disagreement_collapse_rate'] = (
            wrong_disagreement_decisions / total_disagreement_cases 
            if total_disagreement_cases > 0 else 0
        )
        metrics['total_decisions'] = total_decisions
        metrics['total_disagreement_cases'] = total_disagreement_cases
        
        return metrics
    
    def save_results(self, output_dir: str, filename_prefix: str = "judger_eval") -> None:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
            filename_prefix: Prefix for output files
        """
        # Create output directory if it doesn't exist
        os.makedirs(f"{output_dir}/{filename_prefix}", exist_ok=True)
        
        # Get comprehensive metrics
        metrics = self.get_comprehensive_metrics()
        
        # Save detailed metrics to JSON
        metrics_file = os.path.join(output_dir, f"{filename_prefix}/judge_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create and save CSV summary
        self._save_csv_summary(output_dir, filename_prefix, metrics)
        
        print(f"Judger evaluation results saved to {output_dir}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Overall disagreement collapse rate: {metrics['overall_disagreement_collapse_rate']:.4f}")
    
    def _save_csv_summary(self, output_dir: str, filename_prefix: str, metrics: Dict[str, Any]) -> None:
        """
        Save a CSV summary of the key metrics.
        
        Args:
            output_dir: Directory to save results
            filename_prefix: Prefix for output files
            metrics: Metrics dictionary
        """
        # Prepare data for CSV
        csv_data = []
        
        # Add round-by-round metrics
        collapse_rates = metrics.get('disagreement_collapse_rates', {})
        accuracy_by_round = metrics.get('accuracy_by_round', {})
        
        all_rounds = set(collapse_rates.keys()) | set(accuracy_by_round.keys())
        
        for round_num in sorted(all_rounds):
            row = {'round': round_num}
            
            # Add collapse rate metrics
            if round_num in collapse_rates:
                row.update({
                    'disagreement_collapse_rate': collapse_rates[round_num]['disagreement_collapse_rate'],
                    'total_disagreements_with_correct': collapse_rates[round_num]['total_disagreements_with_correct'],
                    'judger_wrong_decisions': collapse_rates[round_num]['judger_wrong_decisions']
                })
            
            # Add accuracy metrics
            if round_num in accuracy_by_round:
                row.update({
                    'judger_accuracy': accuracy_by_round[round_num]['accuracy'],
                    'total_decisions': accuracy_by_round[round_num]['total_decisions'],
                    'correct_decisions': accuracy_by_round[round_num]['correct_decisions']
                })
                
                # Add confidence-based accuracy
                for conf_level in ['high', 'medium', 'low']:
                    if conf_level in accuracy_by_round[round_num]['accuracy_by_confidence']:
                        row[f'accuracy_{conf_level}_confidence'] = accuracy_by_round[round_num]['accuracy_by_confidence'][conf_level]
            
            csv_data.append(row)
        
        # Create DataFrame and save
        if csv_data:
            df = pd.DataFrame(csv_data)
            
            # Format percentage columns
            percentage_cols = ['disagreement_collapse_rate', 'judger_accuracy'] + [
                f'accuracy_{conf}_confidence' for conf in ['high', 'medium', 'low']
            ]
            
            for col in percentage_cols:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
            
            csv_file = os.path.join(output_dir, f"{filename_prefix}/judge_summary.csv")
            df.to_csv(csv_file, index=False)
            print(f"CSV summary saved to {csv_file}")
    
    def _check_answer_correctness(self, correct_answer: Any, predicted_answer: Any) -> bool:
        """
        Check if the predicted answer is correct.
        
        Args:
            correct_answer: The correct answer
            predicted_answer: The predicted answer
            
        Returns:
            bool: True if the answer is correct, False otherwise
        """
        # Handle None or empty values
        if predicted_answer is None or correct_answer is None:
            return False
        
        # Check if answers are strings (letters) or numbers
        if isinstance(predicted_answer, str) and isinstance(correct_answer, str):
            # For multiple-choice answers (letters)
            return predicted_answer.strip().upper() == correct_answer.strip().upper()
        else:
            # For numerical answers
            try:
                return abs(float(correct_answer) - float(predicted_answer)) < 1e-6
            except (ValueError, TypeError):
                # If conversion fails, do a string comparison
                return str(correct_answer).strip() == str(predicted_answer).strip()


def evaluate_judger_results(results_file: str, output_dir: str = "judger_evaluation", 
                           filename_prefix: str = "judger_eval") -> JudgerEvaluator:
    """
    Convenience function to evaluate judger results from a JSON file.
    
    Args:
        results_file: Path to JSON file containing results
        output_dir: Directory to save evaluation results
        filename_prefix: Prefix for output files
        
    Returns:
        JudgerEvaluator instance with results
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]
    
    # Create evaluator and add results
    evaluator = JudgerEvaluator()
    evaluator.add_results(results)
    
    # Save results
    evaluator.save_results(output_dir, filename_prefix)
    
    return evaluator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate judger results")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to JSON file containing judger results")
    parser.add_argument("--output-dir", type=str, default="judger_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--prefix", type=str, default="judger_eval",
                        help="Prefix for output files")
    
    args = parser.parse_args()
    
    # Evaluate results
    evaluator = evaluate_judger_results(args.input, args.output_dir, args.prefix)
    
    # Print summary
    metrics = evaluator.get_comprehensive_metrics()
    print(f"\nJudger Evaluation Summary:")
    print(f"Total evaluations: {metrics['total_evaluations']}")
    print(f"Total decisions: {metrics['total_decisions']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Total disagreement cases: {metrics['total_disagreement_cases']}")
    print(f"Overall disagreement collapse rate: {metrics['overall_disagreement_collapse_rate']:.4f}")
