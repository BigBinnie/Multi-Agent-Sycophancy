import json
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import argparse
import pandas as pd
import numpy as np

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_max_rounds(results):
    max_rounds = 0
    for result in results:
        if 'answers_by_round' in result:
            for agent, rounds in result['answers_by_round'].items():
                if not rounds:
                    continue
                try:
                    max_round = max(int(r) for r in rounds.keys())
                    max_rounds = max(max_rounds, max_round + 1)
                except ValueError:
                    continue
    return max_rounds

def analyze_agent_performance_evolution(results, max_rounds):
    """Analyze how each agent's performance evolves across rounds"""
    agent_performance = defaultdict(lambda: {i: {'correct': 0, 'total': 0} for i in range(max_rounds)})
    
    for result in results:
        if 'answers_by_round' not in result or 'correct_answer' not in result:
            continue
            
        correct_answer = result['correct_answer']
        agents = list(result['answers_by_round'].keys())
        
        for agent in agents:
            for round_num in range(max_rounds):
                if str(round_num) in result['answers_by_round'][agent]:
                    agent_answer = result['answers_by_round'][agent][str(round_num)]
                    agent_performance[agent][round_num]['total'] += 1
                    if agent_answer == correct_answer:
                        agent_performance[agent][round_num]['correct'] += 1
    
    return agent_performance

def analyze_agent_influence(results, max_rounds):
    """Analyze how agents influence each other's answers"""
    influence_matrix = defaultdict(lambda: defaultdict(int))
    convergence_patterns = defaultdict(int)
    
    for result in results:
        if 'answers_by_round' not in result:
            continue
            
        agents = list(result['answers_by_round'].keys())
        if len(agents) < 2:
            continue
            
        # Track when agents change their answers to match others
        for round_num in range(1, max_rounds):
            prev_round = str(round_num - 1)
            curr_round = str(round_num)
            
            # Get all answers for previous and current round
            prev_answers = {}
            curr_answers = {}
            
            for agent in agents:
                if prev_round in result['answers_by_round'][agent]:
                    prev_answers[agent] = result['answers_by_round'][agent][prev_round]
                if curr_round in result['answers_by_round'][agent]:
                    curr_answers[agent] = result['answers_by_round'][agent][curr_round]
            
            # Check for convergence patterns
            if len(set(curr_answers.values())) == 1 and len(set(prev_answers.values())) > 1:
                convergence_patterns['converged_to_agreement'] += 1
            elif len(set(curr_answers.values())) > 1 and len(set(prev_answers.values())) == 1:
                convergence_patterns['diverged_from_agreement'] += 1
            
            # Track individual agent changes
            for agent in agents:
                if agent in prev_answers and agent in curr_answers:
                    if prev_answers[agent] != curr_answers[agent]:
                        # Agent changed their answer - check if they matched someone else's previous answer
                        for other_agent in agents:
                            if other_agent != agent and other_agent in prev_answers:
                                if curr_answers[agent] == prev_answers[other_agent]:
                                    influence_matrix[other_agent][agent] += 1
    
    return influence_matrix, convergence_patterns

def create_performance_evolution_chart(agent_performance, output_html_path, output_png_path=None):
    """Create line chart showing agent performance evolution"""
    
    agents = list(agent_performance.keys())
    max_rounds = len(agent_performance[agents[0]]) if agents else 3
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, agent in enumerate(agents):
        accuracy_scores = []
        for round_num in range(max_rounds):
            total = agent_performance[agent][round_num]['total']
            correct = agent_performance[agent][round_num]['correct']
            accuracy = (correct / total * 100) if total > 0 else 0
            accuracy_scores.append(accuracy)
        
        fig.add_trace(go.Scatter(
            x=[f'Round {r}' for r in range(max_rounds)],
            y=accuracy_scores,
            mode='lines+markers',
            name=f'Agent {agent}',
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Agent Performance Evolution Across Rounds',
        xaxis_title='Round',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 100]),
        width=1000,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Save the chart
    evolution_html_path = output_html_path.replace('.html', '_evolution.html')
    fig.write_html(evolution_html_path)
    print(f"Performance evolution chart saved as '{evolution_html_path}'")
    
    if output_png_path:
        try:
            evolution_png_path = output_png_path.replace('.png', '_evolution.png')
            fig.write_image(evolution_png_path, width=1000, height=600, scale=2)
            print(f"Evolution PNG saved as '{evolution_png_path}'")
        except Exception as e:
            print(f"PNG export not available for evolution chart: {e}")
    
    fig.show()
    return fig

def create_influence_heatmap(influence_matrix, output_html_path, output_png_path=None):
    """Create heatmap showing agent influence patterns"""
    
    agents = list(influence_matrix.keys())
    if not agents:
        print("No influence data found")
        return None
    
    # Create matrix for heatmap
    all_agents = sorted(set(agents) | set().union(*[influence_matrix[a].keys() for a in agents]))
    matrix = []
    
    for influencer in all_agents:
        row = []
        for influenced in all_agents:
            if influencer in influence_matrix and influenced in influence_matrix[influencer]:
                row.append(influence_matrix[influencer][influenced])
            else:
                row.append(0)
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f'Agent {a}' for a in all_agents],
        y=[f'Agent {a}' for a in all_agents],
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="Influence Count")
    ))
    
    fig.update_layout(
        title='Agent Influence Matrix<br><sub>Rows: Influencer, Columns: Influenced</sub>',
        xaxis_title='Influenced Agent',
        yaxis_title='Influencing Agent',
        width=800,
        height=800
    )
    
    # Save the heatmap
    influence_html_path = output_html_path.replace('.html', '_influence.html')
    fig.write_html(influence_html_path)
    print(f"Influence heatmap saved as '{influence_html_path}'")
    
    if output_png_path:
        try:
            influence_png_path = output_png_path.replace('.png', '_influence.png')
            fig.write_image(influence_png_path, width=800, height=800, scale=2)
            print(f"Influence PNG saved as '{influence_png_path}'")
        except Exception as e:
            print(f"PNG export not available for influence heatmap: {e}")
    
    fig.show()
    return fig

def analyze_agent_agreement_patterns(results, max_rounds):
    """Analyze patterns of agreement between specific agent pairs"""
    agent_pairs = defaultdict(lambda: {'agree': 0, 'disagree': 0})
    
    for result in results:
        if 'answers_by_round' not in result:
            continue
            
        agents = list(result['answers_by_round'].keys())
        if len(agents) < 2:
            continue
        
        for round_num in range(max_rounds):
            round_str = str(round_num)
            round_answers = {}
            
            for agent in agents:
                if round_str in result['answers_by_round'][agent]:
                    round_answers[agent] = result['answers_by_round'][agent][round_str]
            
            # Compare all pairs of agents
            agent_list = list(round_answers.keys())
            for i in range(len(agent_list)):
                for j in range(i + 1, len(agent_list)):
                    agent1, agent2 = agent_list[i], agent_list[j]
                    pair_key = tuple(sorted([agent1, agent2]))
                    
                    if round_answers[agent1] == round_answers[agent2]:
                        agent_pairs[pair_key]['agree'] += 1
                    else:
                        agent_pairs[pair_key]['disagree'] += 1
    
    return agent_pairs

def print_summary_statistics(agent_performance, influence_matrix, convergence_patterns, agent_pairs):
    """Print comprehensive summary statistics"""
    print("\n" + "="*60)
    print("AGENT DYNAMICS SUMMARY STATISTICS")
    print("="*60)
    
    # Performance summary
    print("\n1. AGENT PERFORMANCE SUMMARY:")
    print("-" * 40)
    agents = list(agent_performance.keys())
    max_rounds = len(agent_performance[agents[0]]) if agents else 0
    
    for agent in sorted(agents):
        print(f"\nAgent {agent}:")
        for round_num in range(max_rounds):
            total = agent_performance[agent][round_num]['total']
            correct = agent_performance[agent][round_num]['correct']
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"  Round {round_num}: {correct}/{total} ({accuracy:.1f}%)")
    
    # Influence summary
    print("\n2. INFLUENCE PATTERNS:")
    print("-" * 40)
    if influence_matrix:
        for influencer in sorted(influence_matrix.keys()):
            total_influence = sum(influence_matrix[influencer].values())
            if total_influence > 0:
                print(f"Agent {influencer} influenced others {total_influence} times")
                for influenced, count in sorted(influence_matrix[influencer].items()):
                    if count > 0:
                        print(f"  -> Agent {influenced}: {count} times")
    else:
        print("No clear influence patterns detected")
    
    # Convergence patterns
    print("\n3. CONVERGENCE PATTERNS:")
    print("-" * 40)
    for pattern, count in convergence_patterns.items():
        print(f"{pattern.replace('_', ' ').title()}: {count} times")
    
    # Agreement patterns
    print("\n4. AGENT PAIR AGREEMENT:")
    print("-" * 40)
    for pair, stats in sorted(agent_pairs.items()):
        total = stats['agree'] + stats['disagree']
        if total > 0:
            agreement_rate = (stats['agree'] / total * 100)
            print(f"Agents {pair[0]} & {pair[1]}: {agreement_rate:.1f}% agreement ({stats['agree']}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare agent performance and interaction patterns")
    parser.add_argument("--input", required=True, help="Path to the input JSON results file")
    parser.add_argument("--output-html", default="agent_comparison.html", help="Output HTML file path")
    parser.add_argument("--output-png", help="Optional output PNG file path")
    args = parser.parse_args()

    results = load_results(args.input)
    max_rounds = get_max_rounds(results)
    
    # Perform all analyses
    agent_performance = analyze_agent_performance_evolution(results, max_rounds)
    influence_matrix, convergence_patterns = analyze_agent_influence(results, max_rounds)
    agent_pairs = analyze_agent_agreement_patterns(results, max_rounds)
    
    # Create visualizations
    create_performance_evolution_chart(agent_performance, args.output_html, args.output_png)
    create_influence_heatmap(influence_matrix, args.output_html, args.output_png)
    
    # Print summary statistics
    print_summary_statistics(agent_performance, influence_matrix, convergence_patterns, agent_pairs)
