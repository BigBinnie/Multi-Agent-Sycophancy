## Folder Structure

This file describes the organization and purpose of each folder under the `src/sycophancy_multi_agent/` directory:

### `/analysis/`
Contains tools for analyzing agent behavior and debate dynamics:
- **`agent_comparison.py`**: Compares agent performance and analyzes interaction patterns with performance evolution charts and influence heatmaps
- **`agent_dynamics.py`**: Analyzes individual agent behavior patterns across debate rounds using Sankey diagrams and consistency analysis
- **`disagreement_dynamics.py`**: Analyzes overall disagreement patterns and correctness-agreement transitions in the multi-agent system
- **`README.md`**: Comprehensive documentation for all analysis tools, metrics, and visualization types

### `/clients/`
Contains client implementations for different AI model providers:
- **`BedrockClient.py`**: AWS Bedrock API client for accessing Claude and other Bedrock models
- **`LlamaClient.py`**: Client for Llama model interactions (local or API-based)
- **`OpenAIClient.py`**: OpenAI API client for GPT models
- **`QwenClient.py`**: Client for Qwen model interactions
- **`__init__.py`**: Package initialization file

### `/conf/`
Configuration files organized by component and model type:
- **`config.yaml`**: Main configuration file with solver, data, and evaluation settings
- **`accelerate_config.yaml`**: Configuration for distributed training/inference using Accelerate
- **`sycophancy_system_prompt_judge.json`**: System prompts specifically for judge agents in sycophancy experiments
- **`sycophancy_system_prompts.json`**: System prompts for sycophancy-related agent behaviors
- **`/bedrock/`**: Bedrock-specific model configurations (default.yaml)
- **`/judger/`**: Judge agent configurations with different evaluation strategies (default, fast, math_focused, strict)
- **`/llama/`**: Llama model configurations
- **`/openai/`**: OpenAI model configurations
- **`/prompts/`**: Dataset-specific prompt templates (ai2_arc_challenge, bigbench, commonsenseqa, gsm8k, mmlu, mmlu_pro)
- **`/qwen/`**: Qwen model configurations
- **`/solver/`**: Solver agent configurations for different multi-agent setups (single agent, multi-agent, sycophancy variants)

### `/model_download/`
Scripts and utilities for downloading and managing models:
- **`download_all_models.sh`**: Shell script to download all required models
- **`download_models.py`**: Python script for programmatic model downloading
- **`models_list.txt`**: List of models to be downloaded

### `/utils/`
Utility functions and data structures:
- **`dataclass.py`**: Data classes and structures used throughout the system
- **`dataloader.py`**: Data loading utilities for different datasets and formats

## Root Level Files

- **`main.py`**: Main entry point for running multi-agent debate experiments
- **`mad_main.py`**: Specific entry point for Multi-Agent Debate (MAD) experiments
- **`accelerate_batch_main.py`**: Main script for distributed batch processing
- **`accelerate_batch_judger.py`**: Distributed judger evaluation
- **`accelerate_batch_steering_main.py`**: Distributed steering experiments
- **`evaluate.py`**: General evaluation utilities
- **`evaluate_debater_all.py`**: Comprehensive debater evaluation
- **`evaluate_debater_bar.py`**: BAR debater evaluation
- **`evaluate_judge_all.py`**: Comprehensive judge evaluation
- **`judger_evaluator.py`**: Judge-specific evaluation logic
- **`blind_agreement_evaluator.py`**: Evaluator for blind agreement experiments
- **`blind_agreement_evaluator_judge.py`**: Judge-focused blind agreement evaluation
- **`sentence_similarity_evaluator.py`**: Sentence similarity-based evaluation
- **`round_analyzer.py`**: Analysis of debate rounds and transitions
- **`gather_results.py`**: Utility for collecting and aggregating experimental results
- **`config_schema.py`**: Configuration schema validation
