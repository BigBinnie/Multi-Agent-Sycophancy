#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <trigger_num_1> <trigger_num_2> <trigger_num_3>"
    echo "Example: $0 0 2 4"
    exit 1
fi

# Parse command line arguments
trigger_num_1=${1:-0}
trigger_num_2=${2:-0}
trigger_num_3=${3:-0}

# Dataset and solver configuration
dataset=commonsenseqa
solver=multi_agent_3_sycophancy
solver_1=qwen3_32b
solver_2=qwen3_32b
solver_3=qwen3_32b
type_1=qwen
type_2=qwen
type_3=qwen
round=5
partition_type=index
start_idx=0
end_idx=1221  
batch_size=256  # Set batch size for processing
model_path=models

# Experiment naming
experiment_name=$dataset/batch/${solver}_r${round}/${solver_1}_${solver_2}_${solver_3}/trigger_${trigger_num_1}_${trigger_num_2}_${trigger_num_3}/

echo "Starting experiment: $experiment_name"
echo "Trigger numbers: $trigger_num_1, $trigger_num_2, $trigger_num_3"
echo "Solvers: $solver_1 ($type_1), $solver_2 ($type_2), $solver_3 ($type_3)"

# Create output directory
mkdir -p output/$experiment_name
rm -f output/$experiment_name/log.txt

# Log experiment details
echo "Experiment started at: $(date)" >> output/$experiment_name/log.txt
echo "Trigger num 1: $trigger_num_1" >> output/$experiment_name/log.txt
echo "Trigger num 2: $trigger_num_2" >> output/$experiment_name/log.txt
echo "Trigger num 3: $trigger_num_3" >> output/$experiment_name/log.txt
echo "Solver 1: $solver_1 (type: $type_1)" >> output/$experiment_name/log.txt
echo "Solver 2: $solver_2 (type: $type_2)" >> output/$experiment_name/log.txt
echo "Solver 3: $solver_3 (type: $type_3)" >> output/$experiment_name/log.txt
echo "Dataset: $dataset" >> output/$experiment_name/log.txt
echo "Rounds: $round" >> output/$experiment_name/log.txt
echo "Sample range: $start_idx to $end_idx" >> output/$experiment_name/log.txt
echo "Batch size: $batch_size" >> output/$experiment_name/log.txt
echo "----------------------------------------" >> output/$experiment_name/log.txt

# Run the accelerated batch main script
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
solver.solvers.solver_c.model=$solver_3 \
solver.solvers.solver_a.sycophancy_system_prompt_number=$trigger_num_1 \
solver.solvers.solver_b.sycophancy_system_prompt_number=$trigger_num_2 \
solver.solvers.solver_c.sycophancy_system_prompt_number=$trigger_num_3 \
solver.solvers.solver_a.model_type=$type_1 \
solver.solvers.solver_b.model_type=$type_2 \
solver.solvers.solver_c.model_type=$type_3 \
data.dataset=$dataset \
data.split=validation \
solver=$solver \
data.num_samples=1221 \
prompts=$dataset \
>> output/$experiment_name/log.txt 2>&1

# Log completion
echo "----------------------------------------" >> output/$experiment_name/log.txt
echo "Experiment completed at: $(date)" >> output/$experiment_name/log.txt

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo "Experiment $experiment_name completed successfully"
else
    echo "Experiment $experiment_name failed with exit code $?"
fi
