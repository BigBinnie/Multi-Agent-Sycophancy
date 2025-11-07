#!/bin/bash
dataset=mmlu_pro
solver=single_agent
solver_1=llama3_3_70b
# solver_1=qwen3_14b
type_1=llama
type_2=qwen
round=1
partition_type=index
start_idx=0
end_idx=1000  
batch_size=256 # Set batch size for processing
experiment_name=$dataset/batch/${solver}/${solver_1}

mkdir -p output/$experiment_name
rm -f output/$experiment_name/log.txt

python accelerate_batch_main.py \
output_prefix=$experiment_name \
measure_confidence=true \
+batch_size=$batch_size \
data.partition.partition_type=$partition_type \
data.partition.start_idx=$start_idx \
data.partition.end_idx=$end_idx \
solver.solvers.solver_a.model=$solver_1 \
solver.solvers.solver_a.model_type=$type_2 \
data.dataset=$dataset \
data.split=test \
solver=$solver \
data.num_samples=1000 \
prompts=$dataset \
>> output/$experiment_name/log.txt
