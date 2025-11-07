# Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate

This project implements a multi-agent debate system for understanding how sycophancy dynamics shape the system performance.

# 1. Environment Setup
Create a virtual enviroment and install dependencies.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Download models from Huggingface to run experiments in parallel
```bash
bash ./model_download/download_all_models.sh [model_dir]
```
Before downloading the Llama model, you need to apply for the permission on Huggingface first if you haven't applied before, and then log in by    huggingface-cli login.
# 2. Standard Debate
## Test by calling APIs of OpenAI or Bedrock
For OpenAI, you need to set your API key first
```bash
export OPENAI_API_KEY="YOUR_KEY"
```
Single Agent Testing
The usage case for mmlu pro is at `scripts_api/run_single_agent.sh`, and the use case for commonsenseqa is at `scripts_api/run_multi_agent.sh`
```bash
bash scripts_api/run_single_agent.sh
```
Multi Agent Testing by Decentralized Structure
```bash
# 2 agent
bash scripts_api/run_multi_agent.sh

#3 agent
bash scripts_api/run_multi_agent_3.sh
```
Multi Agent Testing by Centralized Structure
```bash
bash scripts_api/run_mad.sh
```
## Test by Batch Inference on Local GPUs (8*40G A100s)
Single Agent Testing
```bash
bash scripts_local/run_batch_single_agent.sh
```
Multi Agent Testing by Decentralized Structure
```bash
## 2 agent
bash scripts_local/run_batch_multi_agent.sh

## 3 agent
bash scripts_local/run_batch_multi_agent_3.sh
```
Multi Agent Testing by Centralized Structure
```bash
bash scripts_local/run_batch_mad.sh
```

Submit job to the cluster by
```
bash scripts_local/run_cluster.sh
```

# 3. Control Agent Sycophancy by System Prompts
## Test by Batch Inference on Local GPUs (8*40G A100s)
Multi Agent Testing by Decentralized Structure
```bash
## 2 agent
bash scripts_syco/run_batch_multi_agent_sycophancy.sh

## 3 agent
bash scripts_syco/run_batch_multi_agent_3_sycophancy.sh
```
Multi Agent Testing by Centralized Structure
```bash
bash scripts_local/run_batch_mad.sh
```

Submit job to the cluster to test different sycophancy combinations by
```bash
bash scripts_syco/run_cluster_multi_agent_sycophancy_homo.sh
bash scripts_syco/run_cluster_multi_agent_sycophancy_heter.sh

bash scripts_syco/run_cluster_multi_agent_3_sycophancy_homo.sh
bash scripts_syco/run_cluster_multi_agent_3_sycophancy_heter.sh

bash scripts_syco/run_cluster_batch_mad.sh  
```

Gather results from different combinations
```bash
python gather_results.py --output-dir output_dir
```
# 4. Control Agent Sycophancy by Persona Vectors

Set up the persona_vectors at scripts scripts_syco/run_batch_steering_multi_agent.sh
Run the multi-agent testing by
```bash
bash scripts_syco/run_batch_steering_multi_agent.sh
```

# 5. Evaluation and Analysis
Evaluation
```bash
bash scripts_eval/run_evaluate_debater.sh
bash scripts_eval/run_evaluate_judge.sh
```
Analysis
```bash
bash scripts_eval/run_analyze.sh
```