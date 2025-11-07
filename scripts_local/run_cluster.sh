#!/bin/bash
job_name="d1_llama_single"

# Submit the job with sbatch
sbatch -p p4 -c 16 --exclusive --constrain p4d.24xlarge \
        --job-name "$job_name" \
        -e "output/log/${job_name}_err.log" \
        -o "output/log/${job_name}_std.log" \
        scripts/run_batch_single_agent.sh

#!/bin/bash
job_name="d1_llama_single"

# Submit the job with sbatch
sbatch -p p4 -c 16 --exclusive --constrain p4d.24xlarge \
        --job-name "$job_name" \
        -e "output/log/${job_name}_err.log" \
        -o "output/log/${job_name}_std.log" \
        scripts/run_batch_single_agent.sh

echo "All jobs submitted successfully!"
