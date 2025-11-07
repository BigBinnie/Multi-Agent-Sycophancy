#!/bin/bash

# Create a parameter file with all sycophancy system prompt numbers (0-8)
param_file="mad_sycophancy_job_params.txt"
rm -f "$param_file"

echo "Generating sycophancy system prompt numbers..."
job_count=0
for sycophancy_num in {0..8}; do
    echo "$sycophancy_num" >> "$param_file"
    ((job_count++))
done

echo "Generated $job_count parameter combinations in $param_file"

# Submit job array with limited concurrent jobs
# Reduced to avoid API rate limits
max_concurrent_jobs=9  # Limit to 2 concurrent jobs to avoid rate limits

echo "Submitting job array with max $max_concurrent_jobs concurrent jobs..."

sbatch --array=1-${job_count}%${max_concurrent_jobs} \
       -p p4 \
       -c 16 \
       --exclusive \
       --constrain p4d.24xlarge \
       --job-name "mad_syco_array" \
       -e "output/log/mad_syco_array_%A/%a_err.log" \
       -o "output/log/mad_syco_array_%A/%a_std.log" \
       --wrap="
# Read parameters for this array task
sycophancy_system_prompt_number=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $param_file)

echo \"Running job \${SLURM_ARRAY_TASK_ID}/${job_count}: sycophancy_system_prompt_number=\$sycophancy_system_prompt_number\"

# Add a random delay to avoid simultaneous API calls
delay=\$((RANDOM % 60 + 30))  # Random delay between 30-90 seconds
echo \"Waiting \$delay seconds to avoid API rate limits...\"
sleep \$delay

# Export the sycophancy_system_prompt_number for the batch script
export sycophancy_system_prompt_number=\$sycophancy_system_prompt_number

# Run the batch MAD script with this parameter
bash scripts/run_batch_mad.sh
"

echo "Job array submitted successfully!"
echo "Total jobs: $job_count"
echo "Max concurrent jobs: $max_concurrent_jobs"
echo "Parameter file: $param_file"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all array jobs with: scancel -u \$USER --name='mad_syco_array'"
