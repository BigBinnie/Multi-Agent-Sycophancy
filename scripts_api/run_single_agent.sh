#!/bin/bash
dataset=mmlu_pro
solver=single_agent
model_id=qwen3_14b
type_1=bedrock
type_2=qwen
type_3=llama
type_4=openai
partition_type=index
start_idx=0
end_idx=1000  
experiment_name=$dataset/$solver/$model_id

# Create output directory if it doesn't exist
mkdir -p output/$experiment_name
rm output/$experiment_name/log.txt

# Run the mad_bedrock.py script with MMLU Pro dataset configuration
# Using the single solver type for multiple-choice questions
python main.py \
output_prefix=$experiment_name \
data.partition.partition_type=$partition_type \
data.partition.start_idx=$start_idx \
data.partition.end_idx=$end_idx \
solver.solvers.solver_a.model=$model_id \
solver.solvers.solver_a.model_type=$type_2 \
data.dataset=$dataset \
data.split=test \
data.num_samples=1000 \
solver=$solver \
prompts=$dataset \
>> output/$experiment_name/log.txt