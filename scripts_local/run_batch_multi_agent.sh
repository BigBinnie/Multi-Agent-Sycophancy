#!/bin/bash
dataset=gsm8k
solver=multi_agent
# llama3_3_70b
solver_1=llama3_3_70b
solver_2=llama3_3_70b
# solver_1=claude_haiku
# solver_2=claude_sonnet_3_7
type_1=llama
type_2=qwen
round=5
partition_type=index
start_idx=0
end_idx=1000  
batch_size=256  # Set batch size for processing
experiment_name=$dataset/${solver}_batch_r${round}/${solver_1}_${solver_2}

mkdir -p output/$experiment_name
rm -f output/$experiment_name/log.txt

python accelerate_batch_main.py \
output_prefix=$experiment_name \
measure_confidence=true \
+batch_size=$batch_size \
data.partition.partition_type=$partition_type \
data.partition.start_idx=$start_idx \
data.partition.end_idx=$end_idx \
solver.max_round=$round \
solver.solvers.solver_a.model=$solver_1 \
solver.solvers.solver_b.model=$solver_2 \
solver.solvers.solver_a.model_type=$type_2 \
solver.solvers.solver_b.model_type=$type_2 \
data.dataset=$dataset \
data.split=test \
solver=$solver \
data.num_samples=1000 \
prompts=$dataset \
>> output/$experiment_name/log.txt
