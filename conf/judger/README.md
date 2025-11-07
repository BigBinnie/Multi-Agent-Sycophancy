# MAD Judger Configuration

This directory contains configuration files for the Multi-Agent Debate (MAD) Judger component.

## Overview

The MAD Judger evaluates debates between multiple AI agents and makes final decisions based on the quality of arguments presented. It analyzes logical consistency, evidence quality, counterargument handling, reasoning depth, and mathematical accuracy.

## Configuration Files

### `default.yaml`
Standard judger configuration with balanced evaluation criteria and moderate timeouts.
- **Use case**: General purpose evaluation
- **Evaluation time**: ~60 seconds
- **Prompt style**: Standard

### `strict.yaml`
High-standards evaluation with detailed analysis and stricter confidence thresholds.
- **Use case**: Critical evaluations requiring thorough analysis
- **Evaluation time**: ~120 seconds
- **Prompt style**: Detailed
- **Features**: Self-reflection enabled, considers more debate rounds

### `fast.yaml`
Optimized for speed with concise evaluation and shorter timeouts.
- **Use case**: Quick evaluations or high-throughput scenarios
- **Evaluation time**: ~30 seconds
- **Prompt style**: Concise
- **Features**: Simplified processing, fewer rounds considered

### `math_focused.yaml`
Specialized for mathematical reasoning with custom mathematical evaluation prompts.
- **Use case**: Mathematical problem solving and reasoning tasks
- **Evaluation time**: ~90 seconds
- **Prompt style**: Custom mathematical evaluation
- **Features**: Higher weight on mathematical accuracy

## Configuration Structure

### Core Settings

```yaml
# Enable/disable judger functionality
enabled: true

# Model configuration
model:
  model_name: "claude_haiku"
  model_type: "bedrock"

# Evaluation criteria weights (must sum to 1.0)
evaluation:
  criteria_weights:
    logical_consistency: 0.25
    evidence_quality: 0.25
    counterargument_handling: 0.20
    reasoning_depth: 0.20
    mathematical_accuracy: 0.10
```

### Prompt Configuration

```yaml
prompts:
  prompt_style: "standard"  # Options: standard, detailed, concise
  custom_system_prompt: null  # Override with custom prompt
```

### Advanced Settings

```yaml
advanced:
  detailed_history: true          # Include detailed debate formatting
  include_confidence: true        # Include agent confidence scores
  track_agent_performance: true   # Track individual agent performance
  max_rounds_considered: 10       # Maximum debate rounds to analyze
  enable_self_reflection: false   # Enable judger self-reflection
```

## Usage

### Basic Usage

Use the default judger configuration:

```bash
python mad_main.py judger=default
```

### Specialized Configurations

For mathematical problems:
```bash
python mad_main.py judger=math_focused
```

For quick evaluation:
```bash
python mad_main.py judger=fast
```

For thorough analysis:
```bash
python mad_main.py judger=strict
```

### Custom Configuration

You can override specific settings:

```bash
python mad_main.py judger=default judger.evaluation.criteria_weights.mathematical_accuracy=0.3
```

### Disabling Judger

To use majority voting instead of judger evaluation:

```bash
python mad_main.py solver.use_judger=false
```

## Evaluation Criteria

### Logical Consistency (default: 25%)
- Coherence of arguments across debate rounds
- Internal consistency within each agent's reasoning
- Logical flow from premises to conclusions

### Evidence Quality (default: 25%)
- Relevance and strength of supporting evidence
- Use of appropriate examples and citations
- Quality of factual claims and data

### Counterargument Handling (default: 20%)
- How well agents address opposing viewpoints
- Ability to refute counterarguments effectively
- Acknowledgment and integration of valid opposing points

### Reasoning Depth (default: 20%)
- Thoroughness of analysis and explanation
- Consideration of multiple perspectives
- Depth of understanding demonstrated

### Mathematical Accuracy (default: 10%)
- Correctness of calculations and formulas
- Proper use of mathematical concepts
- Accuracy of quantitative reasoning

## Fallback Behavior

When the judger fails or times out, the system falls back to:
- `majority_vote`: Use the most common answer among agents
- `first_response`: Use the first valid response received
- `random`: Randomly select from available responses

## Performance Considerations

- **Fast configuration**: Best for high-throughput scenarios
- **Default configuration**: Balanced performance and accuracy
- **Strict configuration**: Best accuracy but slower processing
- **Math-focused configuration**: Optimized for mathematical problems

## Troubleshooting

### Common Issues

1. **Timeout errors**: Increase `evaluation_timeout` in configuration
2. **Poor judger decisions**: Try `strict` configuration or enable `self_reflection`
3. **Slow evaluation**: Use `fast` configuration or reduce `max_rounds_considered`
4. **Mathematical errors**: Use `math_focused` configuration

### Debugging

Enable detailed logging by setting `fallback.enable_logging: true` to track when fallback mechanisms are used.
