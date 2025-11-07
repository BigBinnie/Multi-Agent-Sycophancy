#!/bin/bash
dataset=commonsenseqa
solver=multi_agent
solver_1=llama3_3_70b
solver_2=qwen3_32b
# solver_2=qwen3_32b
judger=qwen3_32b

type_1=bedrock
type_2=qwen
type_3=llama
type_4=openai
round=5
partition_type=index
start_idx=0
end_idx=1221
batch_size=256  
# Use sycophancy_system_prompt_number from environment variable, default to 0 if not set
sycophancy_system_prompt_number=${sycophancy_system_prompt_number:-0}
experiment_name=$dataset/batch/${solver}_r${round}/${solver_1}_${solver_2}

mkdir -p output/$experiment_name
rm output/$experiment_name/judger_log.txt

python accelerate_batch_judger.py \
output_prefix=$experiment_name \
measure_confidence=false \
+batch_size=$batch_size \
data.partition.partition_type=$partition_type \
data.partition.start_idx=$start_idx \
data.partition.end_idx=$end_idx \
solver.max_round=$round \
solver.solvers.solver_a.model=$solver_1 \
solver.solvers.solver_b.model=$solver_2 \
solver.solvers.solver_a.model_type=$type_2 \
solver.solvers.solver_b.model_type=$type_2 \
judger.model.model_name=$judger \
judger.model.model_type=$type_2 \
judger.prompts.sycophancy_system_prompt_number=$sycophancy_system_prompt_number \
data.dataset=$dataset \
data.split=validation \
solver=$solver \
data.num_samples=1221 \
prompts=$dataset \
>> output/$experiment_name/judger_log.txt
