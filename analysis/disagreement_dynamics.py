import json
import plotly.graph_objects as go
from collections import defaultdict
import argparse
import os

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

def analyze_correctness_agreement_patterns(results, max_rounds):
    round_categories = {i: defaultdict(int) for i in range(max_rounds)}
    transitions = defaultdict(int)
    detailed_cases = []

    for i, result in enumerate(results):
        if 'answers_by_round' not in result or 'correct_answer' not in result:
            continue

        agents = list(result['answers_by_round'].keys())
        if len(agents) < 2:
            continue

        correct_answer = result['correct_answer']
        round_cats_for_question = []

        for round_num in range(max_rounds):
            round_answers = [
                result['answers_by_round'][agent][str(round_num)]
                for agent in agents
                if str(round_num) in result['answers_by_round'][agent]
            ]

            if not round_answers:
                continue

            agents_agree = len(set(round_answers)) == 1
            has_correct = correct_answer in round_answers

            if agents_agree:
                category = 'agreement_correct' if round_answers[0] == correct_answer else 'agreement_wrong'
            else:
                category = 'disagreement_with_correct' if has_correct else 'disagreement_without_correct'

            round_categories[round_num][category] += 1
            round_cats_for_question.append(category)

        for r in range(len(round_cats_for_question) - 1):
            from_cat = round_cats_for_question[r]
            to_cat = round_cats_for_question[r + 1]
            transitions[f"{r}:{from_cat} -> {r+1}:{to_cat}"] += 1

        detailed_cases.append({
            'question_id': i,
            'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
            'correct_answer': correct_answer,
            'final_prediction': result.get('predicted_answer', ''),
            'is_correct': result.get('is_correct', False),
            'round_categories': round_cats_for_question,
            'answers_by_round': result['answers_by_round']
        })

    return round_categories, transitions, detailed_cases

def create_correctness_agreement_sankey(round_categories, transitions, output_html_path, output_png_path=None):
    import os

    category_short = {
        'agreement_correct': 'Positive Agreement',
        'agreement_wrong': 'Negative Agreement',
        'disagreement_with_correct': 'Positive Disagreement',
        'disagreement_without_correct': 'Negative Disagreement'
    }

    category_colors = {
        'agreement_correct': '#ccebc5',
        'agreement_wrong': '#fbb4ae',
        'disagreement_with_correct': '#b3cde3',
        'disagreement_without_correct': '#decbe4'
    }

    max_rounds = len(round_categories)
    category_order = ['agreement_correct', 'disagreement_with_correct', 'agreement_wrong', 'disagreement_without_correct']

    labels = []
    node_colors = []
    label_to_index = {}

    for r in range(max_rounds):
        for category in category_order:
            label = f"R{r}: {category_short[category]}"
            labels.append(label)
            label_to_index[f"{r}:{category}"] = len(labels) - 1
            node_colors.append(category_colors[category])

    sources, targets, values, link_colors = [], [], [], []
    for transition, count in transitions.items():
        if '->' in transition:
            from_part, to_part = transition.split(' -> ')
            if from_part in label_to_index and to_part in label_to_index:
                sources.append(label_to_index[from_part])
                targets.append(label_to_index[to_part])
                values.append(count)
                from_cat = from_part.split(":")[1]
                link_colors.append(category_colors.get(from_cat, "#cccccc"))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        ))])

    # Add layout details
    fig.update_layout(
        title_text="Correctness-Agreement Transitions Across Rounds",
        font_size=12,
        width=1200,
        height=600,
        annotations=[
            dict(
                text="<b>Color Legend:</b><br>" +
                     "<span style='color:#ccebc5'>■</span> Positive Agreement <br>" +
                     "<span style='color:#b3cde3'>■</span> Positive Disagreement<br>" +
                     "<span style='color:#fbb4ae'>■</span> Negative Agreement<br>" +
                     "<span style='color:#decbe4'>■</span> Negative Disagreement",
                showarrow=False,
                x=0.98,
                y=0.98,
                xref="paper",
                yref="paper",
                align="left",
                font=dict(size=18),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )

    # Save the figure
    fig.write_html(output_html_path)
    print(f"Sankey diagram saved as '{output_html_path}'")

    # Try to save PNG with high resolution
    if output_png_path:
        try:
            os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
            fig.write_image(output_png_path, width=1200, height=600, scale=2)
            print(f"PNG version saved as '{output_png_path}'")
        except Exception as e:
            print(f"PNG export not available (install kaleido for PNG export): {e}")

    # Show interactive figure
    # fig.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze correctness-agreement transitions in multi-agent results")
    parser.add_argument("--input", required=True, help="Path to the input JSON results file")
    parser.add_argument("--output-html", default="correctness_agreement_sankey.html", help="Output HTML file path")
    parser.add_argument("--output-png", help="Optional output PNG file path")
    args = parser.parse_args()

    results = load_results(args.input)
    max_rounds = get_max_rounds(results)
    round_categories, transitions, _ = analyze_correctness_agreement_patterns(results, max_rounds)
    create_correctness_agreement_sankey(round_categories, transitions, args.output_html, args.output_png)
