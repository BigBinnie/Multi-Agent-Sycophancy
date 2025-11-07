#!/bin/bash
dataset=commonsenseqa
solver=multi_agent
# llama3_3_70b
solver_1=qwen3_32b
solver_2=qwen3_32b
# solver_1=claude_haiku
# solver_2=claude_sonnet_3_7
type_1=llama
type_2=qwen
round=5
partition_type=index
start_idx=0
end_idx=1221  
batch_size=256  # Set batch size for processing

# Global steering configuration (fallback)
enable_steering_global=false
steering_vector_path_global="persona_vectors/persona_vectors/Qwen3-32B/sycophantic_response_avg_diff.pt"
steering_layer_global=15
steering_coefficient_global=1.0
steering_type_global="response"

# Per-agent steering configuration
# Agent A (solver_a) steering
enable_steering_a=true
steering_vector_path_a="persona_vectors/persona_vectors/Qwen3-32B/sycophantic_response_avg_diff.pt"
steering_layer_a=15
steering_coefficient_a=1.0
steering_type_a="response"

# Agent B (solver_b) steering  
enable_steering_b=false
steering_vector_path_b="persona_vectors/persona_vectors/Qwen3-32B/sycophantic_response_avg_diff.pt"
steering_layer_b=15
steering_coefficient_b=-1.0  # Negative coefficient for opposite behavior
steering_type_b="response"

experiment_name=$dataset/${solver}_steering_batch_r${round}/${solver_1}_${solver_2}_a${steering_coefficient_a}_b${steering_coefficient_b}

mkdir -p output/$experiment_name
rm -f output/$experiment_name/log.txt

python accelerate_batch_steering_main.py \
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
data.num_samples=1221 \
prompts=$dataset \
+steering.enable_steering=$enable_steering_global \
+steering.vector_path=$steering_vector_path_global \
+steering.layer=$steering_layer_global \
+steering.coefficient=$steering_coefficient_global \
+steering.type=$steering_type_global \
+steering.agents.solver_a.enable_steering=$enable_steering_a \
+steering.agents.solver_a.vector_path=$steering_vector_path_a \
+steering.agents.solver_a.layer=$steering_layer_a \
+steering.agents.solver_a.coefficient=$steering_coefficient_a \
+steering.agents.solver_a.type=$steering_type_a \
+steering.agents.solver_b.enable_steering=$enable_steering_b \
+steering.agents.solver_b.vector_path=$steering_vector_path_b \
+steering.agents.solver_b.layer=$steering_layer_b \
+steering.agents.solver_b.coefficient=$steering_coefficient_b \
+steering.agents.solver_b.type=$steering_type_b \
>> output/$experiment_name/log.txt
