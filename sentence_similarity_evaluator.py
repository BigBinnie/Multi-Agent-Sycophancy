import json
import os
import re
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm


class SentenceSimilarityEvaluator:
    """
    Evaluates blind agreement between agents based on sentence similarity.
    Compares each agent's response to others' responses from the previous round.
    """

    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.results = []
        self.similarity_scores = []

    def _load_model(self, model_name: str):
        try:
            print(f"Loading sentence transformer model: {model_name}")
            model = SentenceTransformer(model_name)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Failed to load model '{model_name}', falling back to default: {e}")
            fallback = 'all-MiniLM-L6-v2'
            try:
                model = SentenceTransformer(fallback)
                print(f"Fallback model '{fallback}' loaded.")
                return model
            except Exception as e2:
                raise RuntimeError(f"Could not load any model. Error: {e2}")

    def start_evaluation(self):
        print("Starting blind agreement evaluation using sentence similarity...")

    def add_dialogue_result(self, question: str, correct_answer: Any,
                            dialogue_histories: Dict[str, List[Dict]],
                            answers_by_round: Dict[str, Dict[int, Any]]):
        self.results.append({
            'question': question,
            'correct_answer': correct_answer,
            'dialogue_histories': dialogue_histories,
            'answers_by_round': answers_by_round,
            'timestamp': datetime.now().isoformat()
        })

    def extract_response_text(self, history: List[Any], round_num: int) -> str:
        if not history:
            return ""

        try:
            round_num = int(round_num)
            
            # Handle list of dictionaries with 'source' and 'content' fields
            if all(isinstance(turn, dict) and 'source' in turn and 'content' in turn for turn in history):
                # Find agent responses (non-user messages) for the specified round
                agent_responses = [turn for turn in history if turn.get('source') != 'user']
                
                # Round numbers are 0-indexed, so we need the round_num-th agent response (if it exists)
                if 0 <= round_num < len(agent_responses):
                    content = agent_responses[round_num].get('content', '')
                    # print(f"Found agent response for round {round_num}: {content[:100]}...")
                else:
                    print(f"No agent response found for round {round_num}. Available rounds: 0-{len(agent_responses)-1}")
                    return ""
            
            # Handle simple list of strings
            elif all(isinstance(turn, str) for turn in history):
                idx = round_num
                content = history[idx] if 0 <= idx < len(history) else ""
                # print(f"Found string response for round {round_num}: {content[:100]}...")
            
            # Handle mixed or other dictionary formats
            elif all(isinstance(turn, dict) for turn in history):
                # Try to find the agent response for this round
                agent_responses = []
                for turn in history:
                    if isinstance(turn, dict) and turn.get('source') != 'user':
                        agent_responses.append(turn)
                
                if 0 <= round_num < len(agent_responses):
                    content = agent_responses[round_num].get("content", "")
                    # print(f"Found dict response for round {round_num}: {content[:100]}...")
                else:
                    print(f"No dict response found for round {round_num}. Available rounds: 0-{len(agent_responses)-1}")
                    return ""
            
            # Fallback for mixed formats
            else:
                print(f"Mixed format detected, trying fallback for round {round_num}")
                if round_num < len(history):
                    item = history[round_num]
                    content = item.get("content", "") if isinstance(item, dict) else str(item)
                    # print(f"Found fallback response for round {round_num}: {content[:100]}...")
                else:
                    return ""

            # Clean the content - remove answer format markers like {{A}}, {{B}}, etc.
            cleaned_content = re.sub(r'\{\{[A-J]\}\}', '', str(content)).strip()
            # Remove "The answer is " patterns including the period
            cleaned_content = re.sub(r'The answer is\s*\.?', '', cleaned_content, flags=re.IGNORECASE).strip()
            return cleaned_content
            
        except Exception as e:
            print(f"Failed to extract response for round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def calculate_sentence_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            # print(f"Comparing:\nCurrent: {text1}\nOther: {text2}")
            embeddings = self.model.encode([text1, text2])
            score = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0

    def calculate_blind_agreement_similarity(self) -> List[Dict]:
        if not self.results:
            return []

        print("Calculating blind agreement scores...")
        for result_idx, result in tqdm(enumerate(self.results), total=len(self.results)):
            q = result['question']
            histories = result['dialogue_histories']
            answers = result['answers_by_round']
            agents = list(answers.keys())
            rounds = sorted({int(k) for a in answers.values() for k in a})
            for r in rounds:
                prev_r = r - 1

                prev_ans = [
                    answers[a].get(prev_r, answers[a].get(str(prev_r)))
                    for a in agents
                    if prev_r in answers[a] or str(prev_r) in answers[a]
                ]

                inconsistent = len(prev_ans) > 1 and len(set(prev_ans)) > 1

                for agent in agents:
                    if r not in answers[agent] and str(r) not in answers[agent]:
                        continue

                    current_resp = self.extract_response_text(histories.get(agent, []), r)
                    if not current_resp:
                        self.similarity_scores.append({
                            'question_idx': result_idx,
                            'question': q,
                            'agent': agent,
                            'round': r,
                            'similarity_score': 0.0,
                            'compared_agents': [],
                            'individual_similarities': {}
                        })
                        continue

                    if r == 0 or not inconsistent:
                        self.similarity_scores.append({
                            'question_idx': result_idx,
                            'question': q,
                            'agent': agent,
                            'round': r,
                            'similarity_score': None if r > 0 else 0.0,
                            'compared_agents': [],
                            'individual_similarities': {}
                        })
                        continue

                    similarities = {}
                    for other_agent in agents:
                        if other_agent == agent:
                            continue
                        other_resp = self.extract_response_text(histories.get(other_agent, []), prev_r)
                        if other_resp:
                            sim = self.calculate_sentence_similarity(current_resp, other_resp)
                            similarities[other_agent] = sim

                    avg_sim = np.mean(list(similarities.values())) if similarities else 0.0
                    self.similarity_scores.append({
                        'question_idx': result_idx,
                        'question': q,
                        'agent': agent,
                        'round': r,
                        'similarity_score': float(avg_sim),
                        'compared_agents': list(similarities.keys()),
                        'individual_similarities': similarities
                    })
        return self.similarity_scores

    def save_detailed_scores(self, output_file: str = "detailed_similarity_scores.json"):
        if self.similarity_scores:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.similarity_scores, f, indent=2, ensure_ascii=False)
            print(f"Detailed scores saved to: {output_file}")

    def save_results(self, output_file: str = "sentence_similarity_results.csv"):
        if not self.similarity_scores:
            print("No similarity scores to save.")
            return

        df = pd.DataFrame(self.similarity_scores)
        summary = (
            df[df['similarity_score'].notna()]
            .groupby(['agent', 'round'])['similarity_score']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_similarity_score', 'count': 'num_questions'})
        )

        pivot = summary.pivot(index='round', columns='agent', values='avg_similarity_score').reset_index()
        pivot.to_csv(output_file, index=False)
        print(f"Summary results saved to: {output_file}")
        
        return summary

    def calculate_accumulated_similarity(self, output_file: str = "accumulated_similarity_results.csv"):
        if not self.similarity_scores:
            print("No similarity scores found.")
            return

        df = pd.DataFrame(self.similarity_scores)
        # Filter out None values but keep round 0 (which should be 0.0)
        df = df[df['similarity_score'].notna()]
        
        if df.empty:
            print("No valid similarity scores found.")
            return
            
        df['round'] = df['round'].astype(int)

        # Group by agent and round, then calculate mean similarity and count per round
        round_stats = df.groupby(['agent', 'round'])['similarity_score'].agg(['mean', 'count']).reset_index()
        round_stats.columns = ['agent', 'round', 'round_avg_similarity', 'round_count']

        # Calculate accumulated averages for each agent
        agent_accumulated = {}
        for agent in round_stats['agent'].unique():
            agent_df = round_stats[round_stats['agent'] == agent].sort_values('round')
            
            accumulated_scores = []
            accumulated_sum = 0
            accumulated_count = 0
            
            for i, row in agent_df.iterrows():
                # For accumulated calculations, exclude round 0 from averaging
                if row['round'] > 0:
                    accumulated_sum += row['round_avg_similarity'] * row['round_count']
                    accumulated_count += row['round_count']
                
                cumulative_avg = accumulated_sum / accumulated_count if accumulated_count > 0 else 0
                
                accumulated_scores.append({
                    'round': int(row['round']),
                    'current_round_avg_similarity': round(row['round_avg_similarity'], 4),
                    'current_round_count': int(row['round_count']),
                    'accumulated_avg_similarity': round(cumulative_avg, 4),
                    'accumulated_count': int(accumulated_count)
                })
            
            agent_accumulated[agent] = accumulated_scores

        # Create pivot table format - one row per round
        all_rounds = sorted(set(round_stats['round']))
        all_agents = sorted(round_stats['agent'].unique())
        
        pivot_data = []
        for round_num in all_rounds:
            row_data = {'round': round_num}
            
            # Add current round averages and counts for each agent
            for agent in all_agents:
                agent_data = agent_accumulated.get(agent, [])
                round_data = next((item for item in agent_data if item['round'] == round_num), None)
                if round_data:
                    row_data[f'{agent}_current_round_avg_similarity'] = round_data['current_round_avg_similarity']
                    row_data[f'{agent}_current_round_count'] = round_data['current_round_count']
                    row_data[f'{agent}_accumulated_avg_similarity'] = round_data['accumulated_avg_similarity']
                    row_data[f'{agent}_accumulated_count'] = round_data['accumulated_count']
                else:
                    row_data[f'{agent}_current_round_avg_similarity'] = None
                    row_data[f'{agent}_current_round_count'] = None
                    row_data[f'{agent}_accumulated_avg_similarity'] = None
                    row_data[f'{agent}_accumulated_count'] = None
            
            pivot_data.append(row_data)

        result_df = pd.DataFrame(pivot_data)
        result_df.to_csv(output_file, index=False)
        print(f"Accumulated similarity saved to: {output_file}")
        return result_df

    def calculate_round0_agent_similarity(self, agent1: str, agent2: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Calculate the average response similarity between two specific agents in round 0 across the dataset.
        Only calculates similarity when the two agents have different/inconsistent answers.
        
        Args:
            agent1: Name of the first agent
            agent2: Name of the second agent
            save_results: Whether to automatically save results to files (default: True)
            
        Returns:
            Dictionary containing:
            - avg_similarity: Average similarity score across all questions
            - individual_similarities: List of similarity scores for each question
            - valid_comparisons: Number of questions where both agents had valid responses and different answers
            - total_questions: Total number of questions in the dataset
            - skipped_consistent: Number of questions skipped due to consistent answers
        """
        if not self.results:
            print("No results available for similarity calculation.")
            return {
                'avg_similarity': 0.0,
                'individual_similarities': [],
                'valid_comparisons': 0,
                'total_questions': 0,
                'skipped_consistent': 0
            }
        
        individual_similarities = []
        valid_comparisons = 0
        skipped_consistent = 0
        
        print(f"Calculating round 0 similarity between {agent1} and {agent2} (only for inconsistent answers)...")
        
        for result_idx, result in enumerate(self.results):
            question = result['question']
            histories = result['dialogue_histories']
            answers = result['answers_by_round']
            
            # Check if both agents exist in the data
            if agent1 not in histories or agent2 not in histories:
                print(f"Question {result_idx}: One or both agents not found in dialogue histories")
                continue
                
            if agent1 not in answers or agent2 not in answers:
                print(f"Question {result_idx}: One or both agents not found in answers")
                continue
            
            # Get round 0 answers for both agents
            agent1_answer = answers[agent1].get(0, answers[agent1].get('0'))
            agent2_answer = answers[agent2].get(0, answers[agent2].get('0'))
            
            # Skip if either answer is missing
            if agent1_answer is None or agent2_answer is None:
                print(f"Question {result_idx}: Missing round 0 answer from one or both agents")
                continue
            
            # Only calculate similarity if the answers are different/inconsistent
            if agent1_answer == agent2_answer:
                skipped_consistent += 1
                print(f"Question {result_idx}: Agents have consistent answers ({agent1_answer}), skipping similarity calculation")
                continue
            
            # Extract round 0 responses for both agents
            agent1_response = self.extract_response_text(histories.get(agent1, []), 0)
            agent2_response = self.extract_response_text(histories.get(agent2, []), 0)

            # Skip if either response is empty
            if not agent1_response.strip() or not agent2_response.strip():
                print(f"Question {result_idx}: Empty response from one or both agents")
                continue
            
            # Calculate similarity between the two responses
            similarity = self.calculate_sentence_similarity(agent1_response, agent2_response)
            individual_similarities.append({
                'question_idx': result_idx,
                'question': question,
                'agent1_answer': agent1_answer,
                'agent2_answer': agent2_answer,
                'agent1_response': agent1_response,
                'agent2_response': agent2_response,
                'similarity': similarity
            })
            valid_comparisons += 1
        
        # Calculate average similarity
        if valid_comparisons > 0:
            avg_similarity = np.mean([item['similarity'] for item in individual_similarities])
        else:
            avg_similarity = 0.0
        
        result = {
            'agent1': agent1,
            'agent2': agent2,
            'avg_similarity': float(avg_similarity),
            'individual_similarities': individual_similarities,
            'valid_comparisons': valid_comparisons,
            'total_questions': len(self.results),
            'skipped_consistent': skipped_consistent
        }
        
        print(f"Round 0 similarity between {agent1} and {agent2}:")
        print(f"  Average similarity: {avg_similarity:.4f}")
        print(f"  Valid comparisons (inconsistent answers): {valid_comparisons}/{len(self.results)}")
        print(f"  Skipped (consistent answers): {skipped_consistent}/{len(self.results)}")
        
        return result
    
    def save_round0_similarity_results(self, agent1: str, agent2: str, output_file: str = None):
        """
        Calculate and save round 0 similarity results between two agents.
        
        Args:
            agent1: Name of the first agent
            agent2: Name of the second agent
            output_file: Output file path (optional, will generate default name if not provided)
        
        Returns:
            Dictionary containing similarity results
        """
        if output_file is None:
            output_file = f"round0_similarity_{agent1}_{agent2}.json"
        
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        similarity_result = self.calculate_round0_agent_similarity(agent1, agent2, save_results=False)
        
        # Save detailed results to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(similarity_result, f, indent=2, ensure_ascii=False)
        
        print(f"Round 0 similarity results saved to: {output_file}")
        
        # Also create a summary CSV
        csv_file = output_file.replace('.json', '_summary.csv')
        
        # Ensure the directory exists for CSV file too
        csv_dir = os.path.dirname(csv_file)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        
        summary_data = [{
            'agent1': similarity_result['agent1'],
            'agent2': similarity_result['agent2'],
            'avg_similarity': similarity_result['avg_similarity'],
            'valid_comparisons': similarity_result['valid_comparisons'],
            'total_questions': similarity_result['total_questions'],
            'skipped_consistent': similarity_result['skipped_consistent'],
            'coverage_rate': similarity_result['valid_comparisons'] / similarity_result['total_questions'] if similarity_result['total_questions'] > 0 else 0.0,
            'inconsistent_rate': similarity_result['valid_comparisons'] / (similarity_result['valid_comparisons'] + similarity_result['skipped_consistent']) if (similarity_result['valid_comparisons'] + similarity_result['skipped_consistent']) > 0 else 0.0
        }]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        print(f"Summary saved to: {csv_file}")
        
        return similarity_result

    def save_consolidated_results(self, output_file: str = "sentence_similarity_results.csv"):
        return self.save_results(output_file)
