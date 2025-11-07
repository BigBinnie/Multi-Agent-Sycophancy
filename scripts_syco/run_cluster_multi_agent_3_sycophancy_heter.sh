#!/bin/bash

# Create a parameter file with all valid combinations of trigger numbers for three solvers
param_file="three_solvers_trigger_job_params.txt"
rm -f "$param_file"

echo "Generating trigger number combinations for three solvers..."
job_count=0

# Generate all combinations of trigger numbers (0-8) for three solvers
# Following the pattern from the original script where trigger_num_2 >= trigger_num_1
# For three solvers, we'll use: trigger_num_3 >= trigger_num_2 >= trigger_num_1
for trigger_num_1 in {0..8}; do
    for trigger_num_2 in {0..8}; do
        for trigger_num_3 in {0..8}; do
            if [ "$trigger_num_3" -ge "$trigger_num_2" ]; then
                echo "$trigger_num_1 $trigger_num_2 $trigger_num_3" >> "$param_file"
                ((job_count++))
            fi
        done
    done
done

echo "Generated $job_count parameter combinations in $param_file"

# Submit job array with limited concurrent jobs
# Reduced to avoid Hugging Face API rate limits
max_concurrent_jobs=2  # Limit to 2 concurrent jobs to avoid rate limits

echo "Submitting job array with max $max_concurrent_jobs concurrent jobs..."

sbatch --array=1-${job_count}%${max_concurrent_jobs} \
       -p p4 \
       -c 16 \
       --exclusive \
       --constrain p4d.24xlarge \
       --job-name "three_solvers_trigger" \
       -e "output/log/three_solvers_trigger_%A/%a_err.log" \
       -o "output/log/three_solvers_trigger_%A/%a_std.log" \
       --wrap="
# Read parameters for this array task
params=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $param_file)
trigger_num_1=\$(echo \$params | cut -d' ' -f1)
trigger_num_2=\$(echo \$params | cut -d' ' -f2)
trigger_num_3=\$(echo \$params | cut -d' ' -f3)

echo \"Running job \${SLURM_ARRAY_TASK_ID}/${job_count}: trigger_num_1=\$trigger_num_1, trigger_num_2=\$trigger_num_2, trigger_num_3=\$trigger_num_3\"

# Add a random delay to avoid simultaneous API calls
delay=\$((RANDOM % 60 + 30))  # Random delay between 30-90 seconds
echo \"Waiting \$delay seconds to avoid API rate limits...\"
sleep \$delay

# Run the batch script with these parameters
bash scripts/run_batch_multi_agent_3_sycophancy.sh \$trigger_num_1 \$trigger_num_2 \$trigger_num_3
"

echo "Job array submitted successfully!"
echo "Total jobs: $job_count"
echo "Max concurrent jobs: $max_concurrent_jobs"
echo "Parameter file: $param_file"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all array jobs with: scancel -u \$USER --name='three_solvers_trigger'"
