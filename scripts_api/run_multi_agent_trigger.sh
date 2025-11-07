#!/bin/bash
# solver_2 is the trigger
dataset=gsm8k
solver=multi_agent_trigger
solver_1=claude_haiku
solver_2=claude_sonnet_3_7
# solver_1=llama33_70B
# solver_2=qwen3_14b
# solver_1=gpt_4o_mini
# solver_2=gpt_o4_mini
type_1=bedrock
type_2=qwen
type_3=llama
type_4=openai
round=5
partition_type=index
start_idx=0
end_idx=500
experiment_name=$dataset/${solver}_r${round}_t0_7/${solver_1}_${solver_2}

mkdir -p output/$experiment_name
rm output/$experiment_name/log.txt
# sleep 180m

python main.py \
output_prefix=$experiment_name \
measure_confidence=false \
data.partition.partition_type=$partition_type \
data.partition.start_idx=$start_idx \
data.partition.end_idx=$end_idx \
solver.max_round=$round \
solver.solvers.solver_a.model=$solver_1 \
solver.solvers.solver_b.model=$solver_2 \
solver.solvers.solver_a.model_type=$type_1 \
solver.solvers.solver_b.model_type=$type_1 \
data.dataset=$dataset \
data.split=test \
solver=$solver \
data.num_samples=500 \
prompts=$dataset \
>> output/$experiment_name/log.txt