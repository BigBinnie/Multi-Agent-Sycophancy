import json
import argparse
import plotly.graph_objects as go
from collections import defaultdict

# Define your four categories and their colors
category_colors = {
    'agree_correct': '#ccebc5',
    'agree_wrong': '#fbb4ae',
    'disagree_correct': '#b3cde3',
    'disagree_wrong': '#decbe4',
    'no_answer': '#d9d9d9'  # optional for completeness
}
category_labels = {
    'agree_correct': 'positive agreement',
    'agree_wrong': 'negative agreement',
    'disagree_correct': 'positive disagreement',
    'disagree_wrong': 'negative disagreement',
    'no_answer': '#d9d9d9'  # optional for completeness
}
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

def analyze_agent_category_transitions(results, max_rounds):
    transitions = defaultdict(lambda: defaultdict(int))
    round_states = defaultdict(lambda: {i: defaultdict(int) for i in range(max_rounds)})

    for result in results:
        if 'answers_by_round' not in result or 'correct_answer' not in result:
            continue

        correct = result['correct_answer']
        agents = list(result['answers_by_round'].keys())
        if len(agents) < 2:
            continue

        for agent in agents:
            states = []

            for r in range(max_rounds):
                round_answers = {}
                for a in agents:
                    if str(r) in result['answers_by_round'][a]:
                        round_answers[a] = result['answers_by_round'][a][str(r)]

                if agent not in round_answers:
                    states.append("no_answer")
                    round_states[agent][r]['no_answer'] += 1
                    continue

                agent_ans = round_answers[agent]
                others = [ans for a, ans in round_answers.items() if a != agent]
                agreed = all(a == agent_ans for a in others) and len(others) > 0
                is_correct = agent_ans == correct

                if agreed:
                    state = 'agree_correct' if is_correct else 'agree_wrong'
                else:
                    state = 'disagree_correct' if is_correct else 'disagree_wrong'

                states.append(state)
                round_states[agent][r][state] += 1

            for r in range(len(states) - 1):
                k = f"{r}:{states[r]} -> {r+1}:{states[r+1]}"
                transitions[agent][k] += 1

    return transitions, round_states

def create_sankey_for_agent(agent, transitions, round_states, output_png_base="agent_dynamics"):
    labels = []
    label_to_index = {}
    node_colors = []
    max_rounds = max(round_states[agent].keys()) + 1

    categories = list(category_colors.keys())

    for r in range(max_rounds):
        for cat in categories:
            label = category_labels[cat]
            label = f"R{r}: {label.capitalize()}"
            labels.append(label)
            label_to_index[f"{r}:{cat}"] = len(labels) - 1
            node_colors.append(category_colors[cat])

    sources, targets, values, link_colors = [], [], [], []
    for trans, count in transitions[agent].items():
        if '->' in trans:
            from_key, to_key = trans.split(' -> ')
            if from_key in label_to_index and to_key in label_to_index:
                sources.append(label_to_index[from_key])
                targets.append(label_to_index[to_key])
                values.append(count)
                from_cat = from_key.split(":")[1]
                link_colors.append(category_colors.get(from_cat, "#cccccc"))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text=f"Agent {agent} Transitions (Agreement-Correctness Categories)",
        font_size=12,
        width=1200,
        height=600,
        annotations=[
            dict(
                text="<b>Legend:</b><br>" +
                     "<span style='color:#ccebc5'>■</span> Positive Agreement<br>" +
                     "<span style='color:#fbb4ae'>■</span> Negative Agreement<br>" +
                     "<span style='color:#b3cde3'>■</span> Positive Disagreement<br>" +
                     "<span style='color:#decbe4'>■</span> Negative Disagreement<br>",
                showarrow=False,
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                align="left",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )

    safe_name = agent.replace('/', '_').replace('\\', '_')
    output_png_file = f"{output_png_base}_{safe_name}.png"
    output_html_file = f"{output_png_base}_{safe_name}.html"
    
    # Always save HTML file
    fig.write_html(output_html_file)
    print(f"Sankey saved as HTML: {output_html_file}")
    
    # Try to save PNG as well
    try:
        fig.write_image(output_png_file, width=1200, height=600, scale=2)
        print(f"Sankey saved as PNG: {output_png_file}")
    except Exception as e:
        print(f"Error saving PNG: {e}")
        print("Note: PNG export requires the 'kaleido' package. Install with: pip install kaleido")
    # fig.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON results file")
    parser.add_argument("--output-png", default="agent_dynamics", help="Base name for output PNG files")
    args = parser.parse_args()

    results = load_results(args.input)
    max_rounds = get_max_rounds(results)
    transitions, round_states = analyze_agent_category_transitions(results, max_rounds)

    for agent in transitions:
        create_sankey_for_agent(agent, transitions, round_states, args.output_png)

if __name__ == "__main__":
    main()
