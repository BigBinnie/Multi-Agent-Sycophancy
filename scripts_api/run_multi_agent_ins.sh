#!/bin/bash
# solver_1 is the trigger
dataset=mmlu_pro
solver=multi_agent_ins
solver_1=claude_sonnet_3_7
solver_2=claude_haiku
# solver_1=llama33_70B
# solver_2=qwen3_14b
type_1=bedrock
type_2=qwen
round=3
partition_type=index
start_idx=0
end_idx=500
experiment_name=$dataset/${solver}_r${round}/ins_3/${solver_1}_${solver_2}

sleep 4h
mkdir -p output/$experiment_name
rm output/$experiment_name/log.txt

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
prompts=mmlu \
>> output/$experiment_name/log.txt