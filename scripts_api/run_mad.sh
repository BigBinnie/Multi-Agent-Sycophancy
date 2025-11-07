#!/bin/bash
dataset=mmlu_pro
solver=multi_agent
solver_1=claude_haiku
solver_2=claude_sonnet
judger=claude_sonnet_3_7
# solver_1=gpt_4o_mini
# solver_2=gpt_o4_mini
type_1=bedrock
type_2=qwen
type_3=llama
type_4=openai
round=5
partition_type=index
start_idx=0
end_idx=100  
experiment_name=$dataset/mad/${solver}_r${round}/${solver_1}_${solver_2}_${judger}

mkdir -p output/$experiment_name
rm output/$experiment_name/log.txt

python mad_main.py \
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
judger.model.model_name=$judger \
data.dataset=$dataset \
data.split=test \
solver=$solver \
data.num_samples=500 \
prompts=$dataset \
>> output/$experiment_name/log.txt
