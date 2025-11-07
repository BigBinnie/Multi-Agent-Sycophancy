import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sentence_similarity_evaluator import SentenceSimilarityEvaluator

class RoundAnalyzer:
    """
    Analyzes answers by round from evaluation results.
    Provides methods to calculate accuracy and inconsistency metrics by round.
    """
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize the RoundAnalyzer with evaluation results.
        
        Args:
            results: List of evaluation result dictionaries
        """
        self.results = results
        self.rounds_df = None
        self.changes_by_solver = {}
    
    @staticmethod
    def check_answer_correctness(correct_answer: Any, predicted_answer: Any) -> bool:
        """
        Check if the predicted answer is correct
        
        Args:
            correct_answer: The correct answer
            predicted_answer: The predicted answer
            
        Returns:
            bool: True if the answer is correct, False otherwise
        """
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
    
    @staticmethod
    def convert_string_keys_to_int(data: Dict) -> Dict:
        """
        Convert string keys to integers in a dictionary if they can be parsed as integers
        
        Args:
            data: Dictionary with potentially string keys
            
        Returns:
            Dictionary with string keys converted to integers where possible
        """
        if not data:
            return {}
        return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in data.items()}
        
    def create_rounds_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame with answers by round for all solvers.
        
        Returns:
            DataFrame with answers by round
        """
        rounds_analysis = []
        for i, result in enumerate(self.results):
            if 'answers_by_round' in result and result['answers_by_round']:
                for solver_id, rounds_data in result['answers_by_round'].items():
                    # Convert string keys to integers if needed
                    rounds_data = self.convert_string_keys_to_int(rounds_data)
                    
                    # Get confidence scores if available
                    confidence_data = {}
                    if 'confidence_by_round' in result and result['confidence_by_round'] and solver_id in result['confidence_by_round']:
                        confidence_data = result['confidence_by_round'][solver_id]
                        # Convert string keys to integers if needed
                        confidence_data = self.convert_string_keys_to_int(confidence_data)
                    
                    for round_num, answer in rounds_data.items():
                        # Use the static method to check correctness
                        is_correct = self.check_answer_correctness(result['correct_answer'], answer)
                        
                        # Get confidence score for this round if available
                        confidence_score = None
                        if round_num in confidence_data:
                            try:
                                confidence_score = float(confidence_data[round_num])
                            except (ValueError, TypeError):
                                # If conversion fails, store as None
                                confidence_score = None
                        
                        rounds_analysis.append({
                            'question_index': i,
                            'question': result['question'],
                            'solver_id': solver_id,
                            'round': round_num,
                            'answer': answer,
                            'correct_answer': result['correct_answer'],
                            'is_final_answer': round_num == max(rounds_data.keys()),
                            'is_correct': is_correct,
                            'confidence': confidence_score
                        })
        
        if rounds_analysis:
            self.rounds_df = pd.DataFrame(rounds_analysis)
        else:
            self.rounds_df = pd.DataFrame()
        return self.rounds_df
    
    def analyze_answer_changes(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how often answers change between rounds for each solver.
        
        Returns:
            Dictionary with change statistics by solver
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
            
        changes_by_solver = {}
        for solver_id in self.rounds_df['solver_id'].unique():
            solver_data = self.rounds_df[self.rounds_df['solver_id'] == solver_id]
            solver_questions = solver_data['question_index'].unique()
            
            changes = 0
            total_transitions = 0
            
            for q_idx in solver_questions:
                q_data = solver_data[solver_data['question_index'] == q_idx].sort_values('round')
                if len(q_data) > 1:
                    prev_answer = None
                    for _, row in q_data.iterrows():
                        if prev_answer is not None:
                            total_transitions += 1
                            if prev_answer != row['answer']:
                                changes += 1
                        prev_answer = row['answer']
            
            if total_transitions > 0:
                changes_by_solver[solver_id] = {
                    'change_rate': changes / total_transitions,
                    'total_changes': changes,
                    'total_transitions': total_transitions
                }
        
        self.changes_by_solver = changes_by_solver
        return changes_by_solver
    
    def calculate_confidence_scores(self, round_data: pd.DataFrame, round_num: int) -> Dict[str, float]:
        """
        Calculate average confidence scores for each solver in a specific round.
        
        Args:
            round_data: DataFrame containing data for a specific round
            round_num: The round number (for logging purposes)
            
        Returns:
            Dictionary mapping solver IDs to their average confidence scores
        """
        avg_confidence_by_solver = {}
        for solver_id in round_data['solver_id'].unique():
            solver_round_data = round_data[round_data['solver_id'] == solver_id]
            # Drop null/None values before calculating average
            confidence_values = solver_round_data['confidence'].dropna().tolist()
            if confidence_values:
                avg_confidence_by_solver[solver_id] = sum(confidence_values) / len(confidence_values)
                print(f"Round {round_num}, Solver {solver_id}: Avg confidence {avg_confidence_by_solver[solver_id]:.2f}% from {len(confidence_values)} values")
        return avg_confidence_by_solver
    
    def analyze_metrics_by_round(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze accuracy and inconsistency metrics by round across all solvers.
        If the label is correct but the answers are not consistent, the sample is counted as wrong.
        
        Returns:
            Dictionary with metrics by round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
        
        metrics_by_round = {}
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # Get disagreement collapse rates and disagreement with correct answer rates
        collapse_rates = self.analyze_disagreement_collapse_rate()
        disagreement_with_correct_rates = self.analyze_disagreement_with_correct_answer_rate()
        
        for round_num in all_rounds:
            # Get data for this round
            round_data = self.rounds_df[self.rounds_df['round'] == round_num]
            
            # Group by question to check for consistency and correctness
            questions = round_data['question_index'].unique()
            
            total_questions = len(questions)
            inconsistent_questions = 0
            correct_count = 0
            
            # Calculate average confidence for each solver in this round
            avg_confidence_by_solver = self.calculate_confidence_scores(round_data, round_num)
            
            for q_idx in questions:
                # Get all answers for this question in this round
                q_data = round_data[round_data['question_index'] == q_idx]
                q_answers = q_data['answer'].tolist()
                
                # Check if there are multiple different answers (inconsistent)
                has_inconsistent_answers = len(q_answers) > 1 and len(set(q_answers)) > 1
                
                if has_inconsistent_answers:
                    inconsistent_questions += 1
                
                # Check if any answers are correct
                has_correct_answer = any(q_data['is_correct'])
                
                # Only count as correct if the answer is correct AND answers are consistent
                if has_correct_answer and not has_inconsistent_answers:
                        correct_count += 1
            
            # Calculate inconsistent rate and accuracy
            inconsistent_rate = inconsistent_questions / total_questions if total_questions > 0 else 0
            accuracy = correct_count / total_questions if total_questions > 0 else 0
            
            metrics_by_round[int(round_num)] = {
                'total_questions': total_questions,
                'inconsistent_questions': inconsistent_questions,
                'inconsistent_rate': inconsistent_rate,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'avg_confidence_by_solver': avg_confidence_by_solver
            }
            
            # Add disagreement collapse rate if available for this round
            if round_num in collapse_rates:
                metrics_by_round[int(round_num)].update({
                    'total_r0_disagreement_with_correct': collapse_rates[round_num]['total_r0_disagreement_with_correct'],
                    'collapsed_questions': collapse_rates[round_num]['collapsed_questions'],
                    'disagreement_collapse_rate': collapse_rates[round_num]['disagreement_collapse_rate']
                })
            
        return metrics_by_round
    
    def analyze_accumulated_disagreement_collapse_rate(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze accumulated disagreement collapse rate by round.
        With the new calculation method, accumulated rate equals individual rate for each round.
        
        Returns:
            Dictionary with accumulated disagreement collapse rates by round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
        
        # Get per-round collapse rates
        collapse_rates = self.analyze_disagreement_collapse_rate()
        
        accumulated_collapse_rates = {}
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # With the new calculation method, accumulated rate equals individual rate
        for round_num in all_rounds:
            if round_num in collapse_rates:
                # Since disagreement rate is calculated based on R_0 baseline,
                # the accumulated rate is the same as the individual rate
                accumulated_collapse_rates[int(round_num)] = {
                    'accumulated_total_r0_disagreement_with_correct': collapse_rates[round_num]['total_r0_disagreement_with_correct'],
                    'accumulated_disagreement_collapse_rate': collapse_rates[round_num]['disagreement_collapse_rate']
                }
        
        return accumulated_collapse_rates

    def analyze_accumulated_metrics_by_round(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze accumulated accuracy and inconsistency metrics by round across all solvers.
        For each round, considers all answers from round 1 up to and including the current round.
        
        Returns:
            Dictionary with accumulated metrics by round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
        
        accumulated_metrics_by_round = {}
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # Get accumulated disagreement collapse rates
        accumulated_collapse_rates = self.analyze_accumulated_disagreement_collapse_rate()
        
        for round_num in all_rounds:
            # Get data for this round and all previous rounds
            accumulated_round_data = self.rounds_df[self.rounds_df['round'] <= round_num]
            
            # Get all unique questions
            questions = accumulated_round_data['question_index'].unique()
            total_questions = len(questions)
            
            inconsistent_questions = 0
            correct_count = 0
            
            # Calculate average confidence for each solver up to this round
            avg_confidence_by_solver = {}
            for solver_id in accumulated_round_data['solver_id'].unique():
                solver_data = accumulated_round_data[accumulated_round_data['solver_id'] == solver_id]
                confidence_values = solver_data['confidence'].dropna().tolist()
                if confidence_values:
                    avg_confidence_by_solver[solver_id] = sum(confidence_values) / len(confidence_values)
            
            for q_idx in questions:
                # Get all answers for this question up to this round
                q_data = accumulated_round_data[accumulated_round_data['question_index'] == q_idx]
                
                # For each solver, get their latest answer up to this round
                latest_answers = []
                for solver_id in q_data['solver_id'].unique():
                    solver_q_data = q_data[q_data['solver_id'] == solver_id]
                    if not solver_q_data.empty:
                        # Get the answer from the highest round number for this solver
                        latest_round = solver_q_data['round'].max()
                        latest_answer = solver_q_data[solver_q_data['round'] == latest_round]['answer'].iloc[0]
                        latest_answers.append(latest_answer)
                
                # Check if there are multiple different answers (inconsistent)
                has_inconsistent_answers = len(latest_answers) > 1 and len(set(latest_answers)) > 1
                
                if has_inconsistent_answers:
                    inconsistent_questions += 1
                
                # Check if any of the latest answers are correct
                latest_q_data = q_data.sort_values('round').groupby('solver_id').last().reset_index()
                has_correct_answer = any(latest_q_data['is_correct'])
                
                # Only count as correct if the answer is correct AND answers are consistent
                if has_correct_answer and not has_inconsistent_answers:
                    correct_count += 1
            
            # Calculate inconsistent rate and accuracy
            inconsistent_rate = inconsistent_questions / total_questions if total_questions > 0 else 0
            accuracy = correct_count / total_questions if total_questions > 0 else 0
            
            accumulated_metrics_by_round[int(round_num)] = {
                'total_questions': total_questions,
                'inconsistent_questions': inconsistent_questions,
                'inconsistent_rate': inconsistent_rate,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'avg_confidence_by_solver': avg_confidence_by_solver
            }
            
            # Add accumulated disagreement collapse rate if available for this round
            if round_num in accumulated_collapse_rates:
                accumulated_metrics_by_round[int(round_num)].update({
                    'accumulated_total_r0_disagreement_with_correct': accumulated_collapse_rates[round_num]['accumulated_total_r0_disagreement_with_correct'],
                    'accumulated_disagreement_collapse_rate': accumulated_collapse_rates[round_num]['accumulated_disagreement_collapse_rate']
                })
            
        return accumulated_metrics_by_round
    
    def analyze_accuracy_by_round(self) -> Dict[int, float]:
        """
        Get accuracy by round (wrapper for backward compatibility).
        
        Returns:
            Dictionary with accuracy by round
        """
        metrics = self.analyze_metrics_by_round()
        return {round_num: data['accuracy'] for round_num, data in metrics.items()}
    
    def analyze_accumulated_agreement_rate(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Analyze accumulated agreement rates by round for each solver.
        For each round, considers all agreements from round 1 up to and including the current round.
        Uses the results from analyze_agreement_rate to avoid duplicate calculations.
        
        Returns:
            Dictionary with accumulated agreement rates by solver and round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
            
        accumulated_agreement_rates = {}
        
        # Get all solvers and rounds
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # Need at least 2 rounds to analyze previous round agreement
        if len(all_rounds) < 2:
            return {}
        
        # Get per-round metrics from analyze_agreement_rate
        agreement_rates = self.analyze_agreement_rate()
        
        # For each solver, initialize accumulated metrics
        for solver_id in self.rounds_df['solver_id'].unique():
            accumulated_agreement_rates[solver_id] = {}
            
            # Handle all rounds, including round 1
            for round_num in sorted([r for r in all_rounds if r > 0]):
                # For round 1, set accumulated metrics to zeros
                # since there's no previous round to compare with
                if round_num == 0:
                    accumulated_agreement_rates[solver_id][round_num] = {
                        'agreements': 0,
                        'negative_agreements': 0,
                        'total_inconsistencies': 0,
                        'agreement_rate': 0,
                        'negative_agreement_rate': 0
                    }
                    continue
                
                # For other rounds, calculate accumulated metrics
                total_agreements = 0
                total_negative_agreements = 0
                total_inconsistencies = 0
                
                # Add metrics for all rounds up to and including this round
                for r in range(1, round_num + 1):  # Start from round 2
                    if solver_id in agreement_rates and r in agreement_rates[solver_id]:
                        round_metrics = agreement_rates[solver_id][r]
                        total_agreements += round_metrics['agreements']
                        total_negative_agreements += round_metrics['negative_agreements']
                        total_inconsistencies += round_metrics['total_inconsistencies']
                
                # Calculate accumulated agreement rates
                agreement_rate = total_agreements / total_inconsistencies if total_inconsistencies > 0 else 0
                negative_agreement_rate = total_negative_agreements / total_inconsistencies if total_inconsistencies > 0 else 0
                
                accumulated_agreement_rates[solver_id][round_num] = {
                    'agreements': total_agreements,
                    'negative_agreements': total_negative_agreements,
                    'total_inconsistencies': total_inconsistencies,
                    'agreement_rate': agreement_rate,
                    'negative_agreement_rate': negative_agreement_rate
                }
        
        return accumulated_agreement_rates
    
    def analyze_agent_disagreement_collapse_rate(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Analyze disagreement collapse rate for each agent by round.
        This equals the number of cases the agent change to the wrong answer of other agents for current round,
        but have disagreement with this agent's correct answer last round.
        Rate is calculated based on the number of cases which have disagreement with this agent's correct answer last round.
        
        Returns:
            Dictionary with disagreement collapse rates by agent and round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
            
        agent_collapse_rates = {}
        
        # Get all rounds
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # Need at least 2 rounds to analyze collapse
        if len(all_rounds) < 2:
            return {}
        
        # Skip the first round since we need a previous round to compare with
        for round_num in all_rounds[1:]:
            prev_round = round_num - 1
            
            # For each question
            for q_idx in self.rounds_df['question_index'].unique():
                # Get all answers for this question in the previous round
                prev_round_data = self.rounds_df[(self.rounds_df['question_index'] == q_idx) & 
                                               (self.rounds_df['round'] == prev_round)]
                
                # Get all answers for this question in the current round
                current_round_data = self.rounds_df[(self.rounds_df['question_index'] == q_idx) & 
                                                  (self.rounds_df['round'] == round_num)]
                
                if prev_round_data.empty or current_round_data.empty:
                    continue
                
                # For each agent, check if they had a correct answer in previous round
                for agent_id in prev_round_data['solver_id'].unique():
                    agent_prev_data = prev_round_data[prev_round_data['solver_id'] == agent_id]
                    agent_current_data = current_round_data[current_round_data['solver_id'] == agent_id]
                    
                    if agent_prev_data.empty or agent_current_data.empty:
                        continue
                    
                    agent_prev_answer = agent_prev_data['answer'].iloc[0]
                    agent_prev_correct = agent_prev_data['is_correct'].iloc[0]
                    agent_current_answer = agent_current_data['answer'].iloc[0]
                    agent_current_correct = agent_current_data['is_correct'].iloc[0]
                    
                    # Check if this agent had the correct answer in previous round
                    if agent_prev_correct:
                        # Check if there was disagreement with this agent's correct answer in previous round
                        other_agents_prev = prev_round_data[prev_round_data['solver_id'] != agent_id]
                        has_disagreement_with_agent = False
                        
                        for _, other_row in other_agents_prev.iterrows():
                            if other_row['answer'] != agent_prev_answer:
                                has_disagreement_with_agent = True
                                break
                        
                        if has_disagreement_with_agent:
                            # Initialize agent in collapse rates if not already there
                            if agent_id not in agent_collapse_rates:
                                agent_collapse_rates[agent_id] = {}
                            
                            if round_num not in agent_collapse_rates[agent_id]:
                                agent_collapse_rates[agent_id][round_num] = {
                                    'collapse_count': 0,
                                    'total_prev_disagreements': 0,
                                    'disagreement_collapse_rate': 0.0
                                }
                            
                            # Increment total cases where there was disagreement with this agent's correct answer
                            agent_collapse_rates[agent_id][round_num]['total_prev_disagreements'] += 1
                            
                            # Check if the agent changed to a wrong answer that matches other agents
                            if agent_current_answer != agent_prev_answer and not agent_current_correct:
                                # Check if current answer matches any other agent's answer from previous round
                                for _, other_row in other_agents_prev.iterrows():
                                    if other_row['answer'] == agent_current_answer:
                                        # Agent changed to match another agent's wrong answer
                                        agent_collapse_rates[agent_id][round_num]['collapse_count'] += 1
                                        break
        
        # Calculate collapse rates for each agent
        for agent_id in agent_collapse_rates:
            for round_num in agent_collapse_rates[agent_id]:
                total = agent_collapse_rates[agent_id][round_num]['total_prev_disagreements']
                if total > 0:
                    agent_collapse_rates[agent_id][round_num]['disagreement_collapse_rate'] = (
                        agent_collapse_rates[agent_id][round_num]['collapse_count'] / total
                    )
        
        return agent_collapse_rates

    def analyze_disagreement_with_correct_answer_rate(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze disagreement with correct answer rate for each round.
        This metric measures when there are inconsistencies between agents, 
        but at least one agent has the correct answer.
        
        Returns:
            Dictionary with disagreement with correct answer rates by round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
            
        disagreement_with_correct_rates = {}
        
        # Get all rounds
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        for round_num in all_rounds:
            # Get data for this round
            round_data = self.rounds_df[self.rounds_df['round'] == round_num]
            
            # Group by question to check for disagreement and correctness
            questions = round_data['question_index'].unique()
            
            total_questions = len(questions)
            disagreement_with_correct_count = 0
            
            for q_idx in questions:
                # Get all answers for this question in this round
                q_data = round_data[round_data['question_index'] == q_idx]
                q_answers = q_data['answer'].tolist()
                q_correct_flags = q_data['is_correct'].tolist()
                
                # Check if there are multiple different answers (disagreement/inconsistency)
                has_disagreement = len(q_answers) > 1 and len(set(q_answers)) > 1
                
                # Check if at least one agent has the correct answer
                has_at_least_one_correct = any(q_correct_flags)
                
                # Count cases where there's disagreement but at least one agent is correct
                if has_disagreement and has_at_least_one_correct:
                    disagreement_with_correct_count += 1
            
            # Calculate disagreement with correct answer rate
            disagreement_with_correct_rate = disagreement_with_correct_count / total_questions if total_questions > 0 else 0
            
            disagreement_with_correct_rates[int(round_num)] = {
                'disagreement_with_correct_count': disagreement_with_correct_count,
                'total_questions': total_questions,
                'disagreement_with_correct_answer_rate': disagreement_with_correct_rate
            }
            
        return disagreement_with_correct_rates

    def analyze_disagreement_collapse_rate(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze disagreement collapse rate for each round.
        This measures the portion of questions which have disagreement with at least one answer correct in round 0,
        but don't change to agree on correct answer in the current round.
        
        Returns:
            Dictionary with disagreement collapse rates by round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
            
        collapse_rates = {}
        
        # Get all rounds
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # First, identify questions that have disagreement with at least one correct answer in round 0
        r0_disagreement_with_correct_questions = set()
        
        if 0 in all_rounds:
            for q_idx in self.rounds_df['question_index'].unique():
                r0_data = self.rounds_df[(self.rounds_df['question_index'] == q_idx) & 
                                        (self.rounds_df['round'] == 0)]
                
                if r0_data.empty:
                    continue
                
                r0_answers = r0_data['answer'].tolist()
                r0_correct_flags = r0_data['is_correct'].tolist()
                
                # Check if there's disagreement (multiple different answers)
                has_disagreement = len(r0_answers) > 1 and len(set(r0_answers)) > 1
                
                # Check if at least one agent has the correct answer
                has_at_least_one_correct = any(r0_correct_flags)
                
                # If both conditions are met, add to the set
                if has_disagreement and has_at_least_one_correct:
                    r0_disagreement_with_correct_questions.add(q_idx)
        
        # For each round, calculate the collapse rate
        for round_num in all_rounds:
            if round_num == 0:
                # For round 0, the collapse rate is 0 (baseline)
                collapse_rates[0] = {
                    'total_r0_disagreement_with_correct': len(r0_disagreement_with_correct_questions),
                    'collapsed_questions': 0,
                    'disagreement_collapse_rate': 0.0
                }
                continue
            
            # Count questions that were in R0 disagreement with correct but now agree on correct
            collapsed_questions = 0
            
            for q_idx in r0_disagreement_with_correct_questions:
                # Get data for this question in the current round
                ri_data = self.rounds_df[(self.rounds_df['question_index'] == q_idx) & 
                                        (self.rounds_df['round'] == round_num)]
                
                if ri_data.empty:
                    continue
                
                ri_answers = ri_data['answer'].tolist()
                ri_correct_flags = ri_data['is_correct'].tolist()
                
                # Check if all agents now agree on the correct answer
                ri_answers_unique = set(ri_answers)
                all_agents_agree = len(ri_answers_unique) == 1 and len(ri_answers) > 1
                all_correct = all(ri_correct_flags)
                
                # If they agree on the correct answer, this question has "collapsed" from disagreement
                if all_agents_agree and all_correct:
                    collapsed_questions += 1
            
            # Calculate the collapse rate as the portion that did NOT collapse
            # (i.e., still have disagreement or don't agree on correct)
            total_r0_disagreement_with_correct = len(r0_disagreement_with_correct_questions)
            non_collapsed_questions = total_r0_disagreement_with_correct - collapsed_questions
            
            disagreement_collapse_rate = (non_collapsed_questions / total_r0_disagreement_with_correct 
                                        if total_r0_disagreement_with_correct > 0 else 0.0)
            
            collapse_rates[int(round_num)] = {
                'total_r0_disagreement_with_correct': total_r0_disagreement_with_correct,
                'collapsed_questions': collapsed_questions,
                'non_collapsed_questions': non_collapsed_questions,
                'disagreement_collapse_rate': disagreement_collapse_rate
            }
            
        return collapse_rates

    def analyze_agreement_rate(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Analyze how often each agent chooses to agree with another agent's answer when inconsistency 
        happens in the previous round.
        
        Returns:
            Dictionary with agreement rates by solver and round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
            
        agreement_rates = {}
        
        # Get all solvers and rounds
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # Need at least 2 rounds to analyze previous round agreement
        if len(all_rounds) < 2:
            return {}
        
        # Skip the first round since we need a previous round to compare with
        for round_num in all_rounds[1:]:
            prev_round = round_num - 1
            
            # For each question
            for q_idx in self.rounds_df['question_index'].unique():
                # Get all answers for this question in the previous round
                prev_round_data = self.rounds_df[(self.rounds_df['question_index'] == q_idx) & 
                                               (self.rounds_df['round'] == prev_round)]
                
                # Check if there was an inconsistency in the previous round
                prev_answers = prev_round_data['answer'].tolist()
                if len(prev_answers) > 1 and len(set(prev_answers)) > 1:
                    # There was an inconsistency in the previous round
                    
                    # Get all answers for this question in the current round
                    current_round_data = self.rounds_df[(self.rounds_df['question_index'] == q_idx) & 
                                                      (self.rounds_df['round'] == round_num)]
                    
                    # For each solver in the current round
                    for solver_id in current_round_data['solver_id'].unique():
                        # Initialize solver in agreement_rates if not already there
                        if solver_id not in agreement_rates:
                            agreement_rates[solver_id] = {}
                        
                        # Initialize round in solver's data if not already there
                        if round_num not in agreement_rates[solver_id]:
                            agreement_rates[solver_id][round_num] = {
                                'agreements': 0,
                                'negative_agreements': 0,
                                'total_inconsistencies': 0,
                                'agreement_rate': 0.0,
                                'negative_agreement_rate': 0.0
                            }
                        
                        # Get this solver's answer in the current round
                        current_solver_answer = current_round_data[current_round_data['solver_id'] == solver_id]['answer'].iloc[0]
                        
                        # Get this solver's answer in the previous round (if available)
                        prev_solver_data = prev_round_data[prev_round_data['solver_id'] == solver_id]
                        
                        if not prev_solver_data.empty:
                            prev_solver_answer = prev_solver_data['answer'].iloc[0]
                            
                            # Check if the answer changed
                            if current_solver_answer != prev_solver_answer:
                                # Answer changed, check if it now matches any other solver's answer from the previous round
                                other_solvers_prev_round = prev_round_data[prev_round_data['solver_id'] != solver_id]
                                
                                for _, other_row in other_solvers_prev_round.iterrows():
                                    if other_row['answer'] == current_solver_answer:
                                        # The solver changed their answer to match another solver's answer from the previous round
                                        agreement_rates[solver_id][round_num]['agreements'] += 1
                                        
                                        # Check if the other agent's answer is wrong using the static method
                                        other_is_correct = self.check_answer_correctness(other_row['correct_answer'], other_row['answer'])
                                        
                                        # If the other agent's answer is wrong, count as negative agreement
                                        if not other_is_correct:
                                            agreement_rates[solver_id][round_num]['negative_agreements'] += 1
                                        
                                        break
                        
                        # Increment total previous round inconsistencies for this solver
                        agreement_rates[solver_id][round_num]['total_inconsistencies'] += 1
        
        # Calculate agreement rates
        for solver_id in agreement_rates:
            for round_num in agreement_rates[solver_id]:
                total = agreement_rates[solver_id][round_num]['total_inconsistencies']
                if total > 0:
                    agreement_rates[solver_id][round_num]['agreement_rate'] = (
                        agreement_rates[solver_id][round_num]['agreements'] / total
                    )
                    agreement_rates[solver_id][round_num]['negative_agreement_rate'] = (
                        agreement_rates[solver_id][round_num]['negative_agreements'] / total
                    )
        
        return agreement_rates
    
    
    def combine_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Combine all metrics for each solver into a single comprehensive report.
        
        Returns:
            Dictionary with all metrics by solver
        """
        # Get all the individual metrics
        if not self.changes_by_solver:
            self.analyze_answer_changes()
        
        agreement_rates = self.analyze_agreement_rate()
        agent_collapse_rates = self.analyze_agent_disagreement_collapse_rate()
        
        # Combine metrics for each solver
        all_metrics = {}
        
        # Get all unique solver IDs from all analyses
        all_solver_ids = set()
        all_solver_ids.update(self.changes_by_solver.keys())
        if agreement_rates:
            all_solver_ids.update(agreement_rates.keys())
        if agent_collapse_rates:
            all_solver_ids.update(agent_collapse_rates.keys())
        
        for solver_id in all_solver_ids:
            solver_metrics = {
                'solver_id': solver_id,
                'total_transitions': 0,
            }
            
            # Add general change metrics
            if solver_id in self.changes_by_solver:
                change_stats = self.changes_by_solver[solver_id]
                solver_metrics.update({
                    'total_transitions': change_stats.get('total_transitions', 0),
                    'total_changes': change_stats.get('total_changes', 0),
                    'change_rate': change_stats.get('change_rate', 0),
                })
            
            
            # Add previous round agreement rate metrics
            if solver_id in agreement_rates:
                # Calculate average previous round agreement rate across all rounds
                total_agreements = 0
                total_negative_agreements = 0
                total_inconsistencies = 0
                
                for round_num, data in agreement_rates[solver_id].items():
                    total_agreements += data['agreements']
                    total_negative_agreements += data['negative_agreements']
                    total_inconsistencies += data['total_inconsistencies']
                
                # Add per-round previous round agreement rates
                agreement_rates_dict = {}
                for round_num, data in agreement_rates[solver_id].items():
                    agreement_rates_dict[f"round_{round_num}"] = data['agreement_rate']
                
                # Calculate overall previous round agreement rate
                overall_agreement_rate = 0
                overall_negative_agreement_rate = 0
                if total_inconsistencies > 0:
                    overall_agreement_rate = total_agreements / total_inconsistencies
                    overall_negative_agreement_rate = total_negative_agreements / total_inconsistencies
                
                solver_metrics.update({
                    'total_agreements': total_agreements,
                    'total_negative_agreements': total_negative_agreements,
                    'total_inconsistencies': total_inconsistencies,
                    'overall_agreement_rate': overall_agreement_rate,
                    'overall_negative_agreement_rate': overall_negative_agreement_rate,
                    'agreement_rate_by_round': agreement_rates_dict,
                    'negative_agreement_rate_by_round': {
                        round_key: data['negative_agreement_rate'] 
                        for round_key, data in zip(
                            agreement_rates_dict.keys(), 
                            [agreement_rates[solver_id][int(k.split('_')[1])] for k in agreement_rates_dict.keys()]
                        )
                    }
                })
            
            # Add agent disagreement collapse rate metrics
            if solver_id in agent_collapse_rates:
                # Calculate overall agent disagreement collapse rate across all rounds
                total_collapse_count = 0
                total_prev_disagreements = 0
                
                for round_num, data in agent_collapse_rates[solver_id].items():
                    total_collapse_count += data['collapse_count']
                    total_prev_disagreements += data['total_prev_disagreements']
                
                # Add per-round agent disagreement collapse rates
                agent_collapse_rates_dict = {}
                for round_num, data in agent_collapse_rates[solver_id].items():
                    agent_collapse_rates_dict[f"round_{round_num}"] = data['disagreement_collapse_rate']
                
                # Calculate overall agent disagreement collapse rate
                overall_agent_collapse_rate = 0
                if total_prev_disagreements > 0:
                    overall_agent_collapse_rate = total_collapse_count / total_prev_disagreements
                
                solver_metrics.update({
                    'total_agent_collapse_count': total_collapse_count,
                    'total_agent_prev_disagreements': total_prev_disagreements,
                    'overall_agent_disagreement_collapse_rate': overall_agent_collapse_rate,
                    'agent_disagreement_collapse_rate_by_round': agent_collapse_rates_dict
                })
            
            all_metrics[solver_id] = solver_metrics
        
        return all_metrics
    
    def save_analysis(self, out_dir: str, filename_prefix: str) -> None:
        """
        Save analysis results to files.
        
        Args:
            out_dir: Output directory
            filename_prefix: Prefix for output files
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(f"{out_dir}/{filename_prefix}"):
            print(f"Creating directory: {out_dir}/{filename_prefix}")
            os.makedirs(f"{out_dir}/{filename_prefix}")
        
        # Create rounds DataFrame if not already created
        if self.rounds_df is None:
            self.create_rounds_dataframe()
        
        if not self.rounds_df.empty:
            # Combine all metrics
            all_metrics = self.combine_all_metrics()
            metrics_file = os.path.join(f"{out_dir}/{filename_prefix}", f"all_solver_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            # Save average confidence by round metrics
            round_metrics = self.analyze_metrics_by_round()
            round_confidence_file = os.path.join(f"{out_dir}/{filename_prefix}", f"confidence_by_round.json")
            
            # Extract just the confidence metrics
            confidence_metrics = {}
            for round_num, data in round_metrics.items():
                if 'avg_confidence_by_solver' in data:
                    confidence_metrics[round_num] = data['avg_confidence_by_solver']
            
            with open(round_confidence_file, 'w') as f:
                json.dump(confidence_metrics, f, indent=2)
            
            # Save accumulated metrics to CSV
            self.save_accumulated_metrics_to_csv(out_dir, filename_prefix)
    
    def analyze_sentence_similarity_blind_agreement(self, out_dir: str = None, filename_prefix: str = None) -> SentenceSimilarityEvaluator:
        """
        Analyze blind agreement using sentence similarity between agents' responses.
        
        Args:
            out_dir: Output directory for saving results
            filename_prefix: Prefix for output files
            
        Returns:
            SentenceSimilarityEvaluator instance with calculated similarity scores
        """
        print("Starting sentence similarity-based blind agreement analysis...")
        
        # Initialize the sentence similarity evaluator
        similarity_evaluator = SentenceSimilarityEvaluator()
        similarity_evaluator.start_evaluation()
        
        # Process each result to extract dialogue histories and answers by round
        print(self.results[0])
        for result in self.results:
            question = result.get('question', '')
            correct_answer = result.get('correct_answer', '')
            
            # Extract dialogue histories and answers by round
            dialogue_histories = {}
            answers_by_round = {}
            
            if 'answers_by_round' in result and result['answers_by_round']:
                answers_by_round = result['answers_by_round']
                # Extract dialogue histories if available
                if 'dialogue_history' in result and result['dialogue_history']:
                    dialogue_histories = result['dialogue_history']
                elif 'dialogue_histories' in result and result['dialogue_histories']:
                    dialogue_histories = result['dialogue_histories']
                else:
                    # Try to extract from other fields if available
                    for solver_id in answers_by_round.keys():
                        dialogue_histories[solver_id] = []
            # Add the dialogue result to the evaluator
            similarity_evaluator.add_dialogue_result(
                question=question,
                correct_answer=correct_answer,
                dialogue_histories=dialogue_histories,
                answers_by_round=answers_by_round
            )
        
        # Calculate similarity scores
        similarity_scores = similarity_evaluator.calculate_blind_agreement_similarity()
        
        # Save results if output directory is provided
        if out_dir and filename_prefix:
            output_dir = os.path.join(out_dir, filename_prefix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save consolidated results
            similarity_csv = os.path.join(output_dir, "sentence_similarity_results.csv")
            similarity_evaluator.save_consolidated_results(similarity_csv)
            
            # Save accumulated similarity results
            accumulated_csv = os.path.join(output_dir, "accumulated_similarity_results.csv")
            similarity_evaluator.calculate_accumulated_similarity(accumulated_csv)
            
            # Save detailed scores
            detailed_json = os.path.join(output_dir, "detailed_similarity_scores.json")
            similarity_evaluator.save_detailed_scores(detailed_json)
        
        return similarity_evaluator

    def analyze_accumulated_agent_disagreement_collapse_rate(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Analyze accumulated agent disagreement collapse rate by round for each agent.
        For each round, considers all collapse events from round 2 up to and including the current round.
        
        Returns:
            Dictionary with accumulated agent disagreement collapse rates by agent and round
        """
        if self.rounds_df is None:
            self.create_rounds_dataframe()
            
        if self.rounds_df.empty:
            return {}
        
        # Get per-round agent collapse rates
        agent_collapse_rates = self.analyze_agent_disagreement_collapse_rate()
        
        accumulated_agent_collapse_rates = {}
        all_rounds = sorted(self.rounds_df['round'].unique())
        
        # Need at least 2 rounds
        if len(all_rounds) < 2:
            return {}
        
        # For each agent
        for agent_id in agent_collapse_rates:
            accumulated_agent_collapse_rates[agent_id] = {}
            
            # Skip the first round since collapse rate starts from round 2
            for round_num in all_rounds[1:]:
                total_collapse_count = 0
                total_prev_disagreements = 0
                
                # Sum up all collapse events from round 2 to current round (round 1 = round 2 logically)
                for r in range(1, round_num + 1):  # Start from round 1 (which is round 2 logically)
                    if r in agent_collapse_rates[agent_id]:
                        total_collapse_count += agent_collapse_rates[agent_id][r]['collapse_count']
                        total_prev_disagreements += agent_collapse_rates[agent_id][r]['total_prev_disagreements']
                
                # Calculate accumulated collapse rate
                accumulated_collapse_rate = total_collapse_count / total_prev_disagreements if total_prev_disagreements > 0 else 0
                
                accumulated_agent_collapse_rates[agent_id][int(round_num)] = {
                    'accumulated_collapse_count': total_collapse_count,
                    'accumulated_total_prev_disagreements': total_prev_disagreements,
                    'accumulated_agent_disagreement_collapse_rate': accumulated_collapse_rate
                }
        print("accumulated_agent_collapse_rates", accumulated_agent_collapse_rates)
        return accumulated_agent_collapse_rates

    def save_accumulated_metrics_to_csv(self, out_dir: str, filename_prefix: str) -> None:
        """
        Save accumulated metrics to a CSV file.
        
        Args:
            out_dir: Output directory
            filename_prefix: Prefix for output files
        """
        # Get accumulated metrics
        accumulated_metrics = self.analyze_accumulated_metrics_by_round()
        accumulated_agreement_rates = self.analyze_accumulated_agreement_rate()
        accumulated_agent_collapse_rates = self.analyze_accumulated_agent_disagreement_collapse_rate()
        
        # Create a list to store the data for the DataFrame
        acc_data = []
        
        # Get all solvers
        all_solvers = list(accumulated_agreement_rates.keys()) if accumulated_agreement_rates else []
        
        # For each round, create a row with accumulated metrics
        for round_num, data in sorted(accumulated_metrics.items()):
            row = {
                'round': round_num,
                'accuracy': data['accuracy'],
                'inconsistency_rate': data['inconsistent_rate'],
            }
            
            # Add accumulated disagreement collapse rate if available
            if 'accumulated_disagreement_collapse_rate' in data:
                row['accumulated_disagreement_collapse_rate'] = data['accumulated_disagreement_collapse_rate']
            
            # Add accumulated agent disagreement collapse rates for each solver
            for solver_id in all_solvers:
                if solver_id in accumulated_agent_collapse_rates and round_num in accumulated_agent_collapse_rates[solver_id]:
                    agent_data = accumulated_agent_collapse_rates[solver_id][round_num]
                    row[f'{solver_id}_accumulated_agent_disagreement_collapse_rate'] = agent_data['accumulated_agent_disagreement_collapse_rate']
                    # row[f'{solver_id}_accumulated_collapse_count'] = agent_data['accumulated_collapse_count']
                    row[f'{solver_id}_accumulated_total_prev_disagreements'] = agent_data['accumulated_total_prev_disagreements']
                
                # Add accumulated confidence for each solver
                if 'avg_confidence_by_solver' in data and solver_id in data['avg_confidence_by_solver']:
                    row[f'{solver_id}_accumulated_confidence'] = data['avg_confidence_by_solver'][solver_id]
                
                # # Add agreement rates for each solver
                # if solver_id in accumulated_agreement_rates and round_num in accumulated_agreement_rates[solver_id]:
                #     solver_data = accumulated_agreement_rates[solver_id][round_num]
                #     row[f'{solver_id}_agreement_rate'] = solver_data['agreement_rate']
                #     row[f'{solver_id}_negative_agreement_rate'] = solver_data['negative_agreement_rate']
            
            acc_data.append(row)
        
        # Create DataFrame and save to CSV
        if acc_data:
            acc_df = pd.DataFrame(acc_data)
            
            # Format numeric columns as percentages with 2 decimal places
            for col in acc_df.columns:
                if col != 'round':  # Skip the round column
                    # For confidence columns, keep as numeric values (don't format as strings)
                    if 'confidence' in col:
                        # Keep confidence as numeric values for CSV compatibility
                        acc_df[col] = acc_df[col].apply(
                            lambda x: round(x, 2) if pd.notna(x) else None
                        )
                    elif 'rate' in col or 'accuracy' in col:
                        # For other metrics, convert to percentage format with 2 decimal places
                        acc_df[col] = acc_df[col].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
                        )
            
            # Save to CSV
            acc_results_csv = os.path.join(f"{out_dir}/{filename_prefix}", "acc_results.csv")
            acc_df.to_csv(acc_results_csv, index=False)
            print(f"Accumulated metrics saved to {acc_results_csv}")
