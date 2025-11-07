#!/bin/bash

# Default input directory - modify this to match your experiment output
input_dir=output/mmlu_pro/multi_agent_r5/claude_haiku_claude_sonnet_3_7

# system dynamics
python analysis/disagreement_dynamics.py \
--input $input_dir/results.json \
--output-html $input_dir/disagreement.html \
--output-png $input_dir/disagreement.png

# agent dynamics
python analysis/agent_dynamics.py \
--input $input_dir/results.json \
--output-png $input_dir/agent_dynamics.png