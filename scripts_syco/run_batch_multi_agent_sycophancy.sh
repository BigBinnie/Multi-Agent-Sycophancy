#!/bin/bash
# Dataset and solver configuration
dataset=commonsenseqa
solver=multi_agent_sycophancy
solver_1=llama3_3_70b
solver_2=llama3_3_70b
type_1=llama
type_2=qwen
round=5
partition_type=index
start_idx=0
end_idx=1221  
batch_size=256  # Set batch size for processing
model_path=models
# Accept trigger numbers as command line arguments
trigger_num_1=${1:-0}
trigger_num_2=${2:-0}

# Experiment naming
experiment_name=$dataset/batch/${solver}_r${round}/${solver_1}_${solver_2}/trigger_${trigger_num_1}_${trigger_num_2}

mkdir -p output/$experiment_name
rm -f output/$experiment_name/log.txt

python accelerate_batch_main.py \
output_prefix=$experiment_name \
measure_confidence=true \
+batch_size=$batch_size \
+local_model_path=$model_path \
data.partition.partition_type=$partition_type \
data.partition.start_idx=$start_idx \
data.partition.end_idx=$end_idx \
solver.max_round=$round \
solver.solvers.solver_a.model=$solver_1 \
solver.solvers.solver_b.model=$solver_2 \
solver.solvers.solver_a.sycophancy_system_prompt_number=$trigger_num_1 \
solver.solvers.solver_b.sycophancy_system_prompt_number=$trigger_num_2 \
solver.solvers.solver_a.model_type=$type_2 \
solver.solvers.solver_b.model_type=$type_2 \
data.dataset=$dataset \
data.split=validation \
solver=$solver \
data.num_samples=1221 \
prompts=$dataset \
>> output/$experiment_name/log.txt
