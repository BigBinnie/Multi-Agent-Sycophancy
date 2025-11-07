dir=output/commonsenseqa/batch/multi_agent_r5/llama3_3_70b_qwen3_32b/eval
# python evaluate_results.py --input $dir/results.json --output $dir
python evaluate_debater.py  --input $dir/results.json --output $dir

# python extract_dialogue_history.py $dir/log.txt
# python evaluate_results.py --input $dir/combined_results.json --output $dir