import json
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from round_analyzer import RoundAnalyzer

class MathEvaluator:
    def __init__(self):
        self.results = []
        self.start_time = None
        
    def start_evaluation(self):
        self.start_time = time.time()
        
    def add_result(self, question: str, correct_answer: any, predicted_answer: any, 
                   solver_response: str, processing_time: float, individual_answers: List,
                   answers_by_round: Dict = None, confidence_by_round: Dict = None,
                   system_prompts: Dict = None, dialogue_histories: Dict = None,
                   judger_evaluation: Dict = None):
        """
        Add a single evaluation result
        """
        is_correct = self._check_answer_correctness(correct_answer, predicted_answer)
        
        result = {
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'solver_response': solver_response,
            'processing_time': processing_time,
            'individual_answers': individual_answers,
            'answers_by_round': answers_by_round or {},
            'confidence_by_round': confidence_by_round or {},
            'system_prompts': system_prompts or {},
            'dialogue_histories': dialogue_histories or {},
            'judger_evaluation': judger_evaluation or {}
        }
        self.results.append(result)
    
    def _check_answer_correctness(self, correct_answer: Any, predicted_answer: Any) -> bool:
        """
        Check if the predicted answer is correct
        
        Args:
            correct_answer: The correct answer
            predicted_answer: The predicted answer
            
        Returns:
            bool: True if the answer is correct, False otherwise
        """
        # Use RoundAnalyzer's static method for consistency
        return RoundAnalyzer.check_answer_correctness(correct_answer, predicted_answer)
    
    def _analyze_individual_answers_consistency(self, result: Dict) -> Tuple[bool, bool]:
        """
        Analyze if individual answers are consistent
        
        Args:
            result: A single result dictionary
            
        Returns:
            Tuple[bool, bool]: (has_inconsistent_answers, is_correct_and_consistent)
        """
        has_inconsistent_answers = False
        is_correct_and_consistent = False
        
        if "individual_answers" in result and len(result["individual_answers"]) > 1:
            # Check if all answers in individual_answers are the same
            first_answer = result["individual_answers"][0]
            for answer in result["individual_answers"][1:]:
                if answer != first_answer:
                    has_inconsistent_answers = True
                    break
        
        # Only count as correct if the answer is correct AND individual answers are consistent
        if result['is_correct'] and not has_inconsistent_answers:
            is_correct_and_consistent = True
            
        return has_inconsistent_answers, is_correct_and_consistent
        
    def get_metrics(self) -> Dict:
        """
        Calculate evaluation metrics
        """
        total_time = time.time() - self.start_time
        total_count = len(self.results)
        
        # Initialize counters
        correct_count = 0  # Correct answers that are consistent
        inconsistent_count = 0  # Correct answers that are inconsistent
        total_inconsistent_count = 0  # Track all inconsistent answers regardless of correctness
        
        # Initialize round-based metrics
        answer_changes_by_round = {}
        
        for r in self.results:
            # Check if individual answers are consistent
            has_inconsistent_answers, _ = self._analyze_individual_answers_consistency(r)
            
            # Track all inconsistent answers regardless of correctness
            if has_inconsistent_answers:
                total_inconsistent_count += 1
                
            # Count correct answers based on consistency
            if r['is_correct']:
                if has_inconsistent_answers:
                    inconsistent_count += 1
                else:
                    correct_count += 1
            
            # Analyze answers by round
            self._analyze_answer_changes_by_round(r, answer_changes_by_round)
        
        # Calculate average answer changes by round
        avg_answer_changes = {
            solver_id: sum(changes) / len(changes) if changes else 0
            for solver_id, changes in answer_changes_by_round.items()
        }
        
        metrics = {
            'accuracy': correct_count / total_count if total_count > 0 else 0,
            'total_questions': total_count,
            'correct_answers': correct_count,
            'inconsistent_rate': total_inconsistent_count / total_count if total_count > 0 else 0,
            'total_inconsistent_answers': total_inconsistent_count,
            'average_time_per_question': np.mean([r['processing_time'] for r in self.results]),
            'total_processing_time': total_time,
            'avg_answer_changes_by_solver': avg_answer_changes
        }
        return metrics
    
    def _analyze_answer_changes_by_round(self, result: Dict, answer_changes_by_round: Dict) -> None:
        """
        Analyze answer changes by round for a single result
        
        Args:
            result: A single result dictionary
            answer_changes_by_round: Dictionary to store answer changes by round
        """
        if 'answers_by_round' in result and result['answers_by_round']:
            for solver_id, rounds_data in result['answers_by_round'].items():
                # Convert string keys to integers if needed
                rounds_data = RoundAnalyzer.convert_string_keys_to_int(rounds_data)
                
                # Count answer changes across rounds
                if len(rounds_data) > 1:
                    rounds = sorted(rounds_data.keys())
                    changes = 0
                    for i in range(1, len(rounds)):
                        if rounds_data[rounds[i-1]] != rounds_data[rounds[i]]:
                            changes += 1
                    
                    # Record changes for this solver
                    solver_key = f"{solver_id}"
                    if solver_key not in answer_changes_by_round:
                        answer_changes_by_round[solver_key] = []
                    answer_changes_by_round[solver_key].append(changes)
    
    def save_results(self, out_dir, filename_prefix: str):
        """
        Save evaluation results and metrics to files
        """

        # Create output directory if it doesn't exist
        output_path = f"{out_dir}/{filename_prefix}"
        os.makedirs(output_path, exist_ok=True)
        # Save detailed results
        self._save_detailed_results(output_path)
        
        # Get metrics and save metrics summary
        metrics = self.get_metrics()
        self._save_metrics_summary(output_path, metrics)
        
        # Create and save metrics DataFrame
        self._save_metrics_dataframe(output_path)
        
        # Print summary statistics
        # self._print_summary_statistics(metrics, output_path)
    
    def _save_detailed_results(self, output_path: str) -> None:
        """
        Save detailed results to a JSON file
        
        Args:
            output_path: Path to save the results
        """
        results_file = os.path.join(output_path, "results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _save_metrics_summary(self, output_path: str, metrics: Dict) -> None:
        """
        Save metrics summary to a JSON file
        
        Args:
            output_path: Path to save the metrics
            metrics: Metrics dictionary
        """
        metrics_file = os.path.join(output_path, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _save_metrics_dataframe(self, output_path: str) -> None:
        """
        Create and save metrics DataFrame with round-specific metrics
        
        Args:
            output_path: Path to save the metrics
        """
        # Create a RoundAnalyzer instance
        analyzer = RoundAnalyzer(self.results)
        
        # Get round-by-round metrics
        metrics_by_round = analyzer.analyze_metrics_by_round()
        
        # Get solver metrics
        all_solver_metrics = analyzer.combine_all_metrics()
        
        # Get all rounds and solvers
        all_solvers = list(all_solver_metrics.keys())
        
        # Create rounds data with metrics
        rounds_data = self._create_rounds_data(metrics_by_round, all_solvers)
        
        # Create DataFrame and save to CSV
        if rounds_data:
            # Create DataFrame
            metrics_df = pd.DataFrame(rounds_data)
            
            # Add agreement rate metrics to the DataFrame
            self._add_agreement_rate_metrics(metrics_df, analyzer, rounds_data, all_solvers)
            
            # Add confidence metrics to the DataFrame
            self._add_confidence_metrics(metrics_df, metrics_by_round, all_solvers)
            
            # Format numeric columns as percentages
            self._format_percentage_columns(metrics_df)
            
            # Save to CSV
            results_csv = os.path.join(output_path, "results.csv")
            metrics_df.to_csv(results_csv, index=False)
            print(f"Key metrics saved to {results_csv}")
            
            # Save accumulated metrics to CSV
            self._save_accumulated_metrics(output_path, analyzer)
            
            # Perform sentence similarity analysis
            self._perform_sentence_similarity_analysis(output_path, analyzer)
    
    def _save_accumulated_metrics(self, output_path: str, analyzer: RoundAnalyzer) -> None:
        """
        Save accumulated metrics to a CSV file using RoundAnalyzer's method
        
        Args:
            output_path: Path to save the metrics
            analyzer: RoundAnalyzer instance
        """
        # Extract the directory name from the output path for the filename prefix
        filename_prefix = os.path.basename(output_path)
        parent_dir = os.path.dirname(output_path)
        
        # Use RoundAnalyzer's method to save accumulated metrics
        analyzer.save_accumulated_metrics_to_csv(parent_dir, filename_prefix)
    
    
    def _create_rounds_data(self, metrics_by_round: Dict, all_solvers: List[str]) -> List[Dict]:
        """
        Create rounds data with basic metrics
        
        Args:
            metrics_by_round: Dictionary with inconsistent rate by round
            all_solvers: List of all solvers
            
        Returns:
            List[Dict]: List of dictionaries with round metrics
        """
        rounds_data = []
        for round_num, data in sorted(metrics_by_round.items()):
            # For the metrics CSV
            round_metrics = {
                'round': round_num,
                'accuracy': data['accuracy'],
                'not_consistent_rate': data['inconsistent_rate'],
            }
            
            # Add disagreement collapse rate if available
            if 'disagreement_collapse_rate' in data:
                round_metrics['disagreement_collapse_rate'] = data['disagreement_collapse_rate']
            
            # Append the metrics for this round to the list
            rounds_data.append(round_metrics)
            
        return rounds_data
    
    def _add_agreement_rate_metrics(self, metrics_df: pd.DataFrame, analyzer: RoundAnalyzer, 
                                   rounds_data: List[Dict], all_solvers: List[str]) -> None:
        """
        Add agent disagreement collapse rate metrics to the DataFrame
        
        Args:
            metrics_df: DataFrame to add metrics to
            analyzer: RoundAnalyzer instance
            rounds_data: List of dictionaries with round metrics
            all_solvers: List of all solvers
        """
        # Get the all_metrics again to ensure we have the latest data
        all_metrics = analyzer.combine_all_metrics()
        
        # For each round, add only the agent disagreement collapse rate for each solver
        for round_idx, round_data in enumerate(rounds_data):
            round_num = round_data['round']
            for solver_id in all_solvers:
                # Add agent disagreement collapse rate
                if solver_id in all_metrics and 'agent_disagreement_collapse_rate_by_round' in all_metrics[solver_id]:
                    round_key = f'round_{round_num}'
                    if round_key in all_metrics[solver_id]['agent_disagreement_collapse_rate_by_round']:
                        agent_collapse_rate = all_metrics[solver_id]['agent_disagreement_collapse_rate_by_round'][round_key]
                        metrics_df.loc[round_idx, f'{solver_id}_agent_disagreement_collapse_rate'] = agent_collapse_rate
    
    def _add_confidence_metrics(self, metrics_df: pd.DataFrame, metrics_by_round: Dict, 
                               all_solvers: List[str]) -> None:
        """
        Add confidence metrics to the DataFrame
        
        Args:
            metrics_df: DataFrame to add metrics to
            metrics_by_round: Dictionary with inconsistent rate by round
            all_solvers: List of all solvers
        """
        for round_num, data in sorted(metrics_by_round.items()):
            if round_num < len(metrics_df):
                # Get confidence data for this round
                for solver_id in all_solvers:
                    # Find the average confidence for this solver in this round
                    confidence_values = []
                    for r in self.results:
                        if ('confidence_by_round' in r and r['confidence_by_round'] and 
                            solver_id in r['confidence_by_round']):
                            confidence_data = r['confidence_by_round'][solver_id]
                            # Convert string keys to integers if needed
                            confidence_data = RoundAnalyzer.convert_string_keys_to_int(confidence_data)
                            if round_num in confidence_data and confidence_data[round_num] is not None:
                                try:
                                    confidence_values.append(float(confidence_data[round_num]))
                                except (ValueError, TypeError):
                                    pass
                    
                    # Calculate average and add to the DataFrame if we have confidence data
                    if confidence_values:
                        avg_confidence = sum(confidence_values) / len(confidence_values)
                        col_name = f'{solver_id}_confidence'
                        metrics_df.loc[metrics_df['round'] == round_num, col_name] = avg_confidence
                        print(f"Round {round_num}, Solver {solver_id}: Avg confidence {avg_confidence:.2f}% from {len(confidence_values)} values")
    
    def _format_percentage_columns(self, metrics_df: pd.DataFrame) -> None:
        """
        Format numeric columns as percentages with 2 decimal places
        
        Args:
            metrics_df: DataFrame to format
        """
        for col in metrics_df.columns:
            if col != 'round':  # Skip the round column
                # For confidence columns, keep as numeric values (don't format as strings)
                if 'confidence' in col:
                    # Keep confidence as numeric values for CSV compatibility
                    metrics_df[col] = metrics_df[col].apply(
                        lambda x: round(x, 2) if pd.notna(x) else None
                    )
                else:
                    # For other metrics, convert to percentage format with 2 decimal places
                    metrics_df[col] = metrics_df[col].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
                    )
    
    def _print_summary_statistics(self, metrics: Dict, output_path: str) -> None:
        """
        Print summary statistics
        
        Args:
            metrics: Metrics dictionary
            output_path: Path where results were saved
        """
        print(f"Evaluation complete. Results saved to {output_path}")
        print(f"Overall statistics:")
        print(f"  Total questions in evaluation: {len(self.results)}")
        print(f"  Correct answers: {metrics.get('correct_answers', 0)}")
        print(f"  Total inconsistent answers: {metrics.get('total_inconsistent_answers', 0)}")
        print(f"  Inconsistent rate: {metrics.get('inconsistent_rate', 0) * 100:.2f}%")
        print(f"  Overall accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
        
        # Print answer changes by solver if available
        if 'avg_answer_changes_by_solver' in metrics and metrics['avg_answer_changes_by_solver']:
            print("\nAnswer changes by solver:")
            for solver_id, avg_changes in metrics['avg_answer_changes_by_solver'].items():
                print(f"  {solver_id}: {avg_changes:.2f} average changes per question")
    
    def _perform_sentence_similarity_analysis(self, output_path: str, analyzer: RoundAnalyzer) -> None:
        """
        Perform sentence similarity analysis for blind agreement detection
        
        Args:
            output_path: Path to save the similarity analysis results
            analyzer: RoundAnalyzer instance
        """
        try:
            print("\nPerforming sentence similarity analysis for blind agreement detection...")
            
            # Extract the directory name from the output path for the filename prefix
            filename_prefix = os.path.basename(output_path)
            parent_dir = os.path.dirname(output_path)
            
            # Call the analyze_sentence_similarity_blind_agreement function
            similarity_evaluator = analyzer.analyze_sentence_similarity_blind_agreement(
                out_dir=parent_dir,
                filename_prefix=filename_prefix
            )
            
            print("✅ Sentence similarity analysis completed successfully!")
            
            # Print summary of similarity analysis if available
            if hasattr(similarity_evaluator, 'results') and similarity_evaluator.results:
                print(f"  Processed {len(similarity_evaluator.results)} dialogue results for similarity analysis")
                
                # Call save_round0_similarity_results for different agent pairs
                print("\nSaving round 0 similarity results...")
                try:
                    # Get available agents from the first result
                    first_result = similarity_evaluator.results[0]
                    if 'dialogue_histories' in first_result:
                        agents = list(first_result['dialogue_histories'].keys())
                        
                        # Save round 0 similarity results for all agent pairs
                        for i, agent1 in enumerate(agents):
                            for j, agent2 in enumerate(agents):
                                if i < j:  # Avoid duplicate pairs and self-comparison
                                    output_file = os.path.join(parent_dir, f"{filename_prefix}/round0_similarity.json")
                                    similarity_evaluator.save_round0_similarity_results(agent1, agent2, output_file)
                    else:
                        print("No dialogue histories found in similarity evaluator results")
                        
                except Exception as e:
                    print(f"⚠️  Warning: Could not save round 0 similarity results: {e}")
            
        except ImportError as e:
            print(f"⚠️  Warning: Could not perform sentence similarity analysis - missing dependencies: {e}")
            print("   Install required packages: pip install sentence-transformers")
        except Exception as e:
            print(f"⚠️  Warning: Sentence similarity analysis failed: {e}")
            print("   Continuing with standard evaluation metrics...")
