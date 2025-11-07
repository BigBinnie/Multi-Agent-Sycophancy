dataset=commonsenseqa
python evaluate_judge.py --input-dir output/$dataset/batch/multi_agent_sycophancy_r5/qwen3_32b_qwen3_32b/trigger_0_0/qwen
python evaluate_judge.py --input-dir output/$dataset/batch/multi_agent_sycophancy_r5/llama3_3_70b_qwen3_32b/trigger_0_0/qwen
python evaluate_judge.py --input-dir output/$dataset/batch/multi_agent_sycophancy_r5/llama3_3_70b_llama3_3_70b/trigger_0_0/qwen
python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_sycophancy_r5/qwen3_32b_qwen3_32b_qwen3_32b/trigger_0_0_0/qwen
python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_sycophancy_r5/llama3_3_70b_qwen3_32b_qwen3_32b/trigger_0_0_0/qwen
python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_sycophancy_r5/llama3_3_70b_llama3_3_70b_qwen3_32b/trigger_0_0_0/qwen
python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_sycophancy_r5/llama3_3_70b_llama3_3_70b_llama3_3_70b/trigger_0_0_0/qwen
# python blind_agreement_evaluator_judge.py \
# --input-dir output/$dataset/batch/multi_agent_r5/qwen3_32b_qwen3_32b/ \
# --output-dir output/$dataset/batch/multi_agent_r5/qwen3_32b_qwen3_32b/
# python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_r5/llama3_3_70b_qwen3_32b/
# python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_r5/llama3_3_70b_llama3_3_70b/
# python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_r5/qwen3_32b_qwen3_32b_qwen3_32b/
# python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_r5/llama3_3_70b_qwen3_32b_qwen3_32b/
# python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_r5/llama3_3_70b_llama3_3_70b_qwen3_32b/
# python evaluate_judger.py --input-dir output/dataset/batch/multi_agent_3_r5/llama3_3_70b_llama3_3_70b_llama3_3_70b/