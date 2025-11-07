# Configuration Management with Hydra and OmegaConf

This project uses [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management. This README explains how to use these tools to configure and run the application.

## Configuration Structure

The configuration is organized in a hierarchical structure under `src/sycophancy_multi_agent/conf/`:

### Core Configuration Files
- **`config.yaml`**: Main configuration file that orchestrates all other configurations
- **`accelerate_config.yaml`**: Configuration for distributed training/inference using Accelerate
- **`sycophancy_system_prompt_judge.json`**: System prompts specifically for judge agents in sycophancy experiments
- **`sycophancy_system_prompts.json`**: General sycophancy prompts for influencing agent behavior

### Configuration Folders

**`/bedrock/`**: AWS Bedrock model configurations
- Contains default Bedrock settings and model-specific configurations for Claude, DeepSeek, Llama, Mistral, and Qwen models accessible through Bedrock

**`/judger/`**: Judge agent configurations
- Different evaluation strategies including default, fast, math-focused, and strict judging approaches for evaluating agent responses

**`/llama/`**: Llama model configurations  
- Default Llama settings and specific configurations for different Llama model variants (3.1, 3.2, 3.3) in various sizes

**`/openai/`**: OpenAI model configurations
- Default OpenAI settings and specific configurations for GPT models including GPT-3.5, GPT-4, and GPT-4o variants

**`/prompts/`**: Dataset-specific prompt templates
- Customized prompts for different datasets including GSM8K, MMLU, ARC Challenge, BigBench, and CommonsenseQA

**`/qwen/`**: Qwen model configurations
- Default Qwen settings and specific configurations for different Qwen model variants (Qwen2, Qwen3) in various sizes

**`/solver/`**: Solver agent configurations
- Different multi-agent debate setups including single agent, multi-agent, sycophancy variants, and trigger-based configurations

## Running the Application

To run the application with the default configuration:

```bash
python src/sycophancy_multi_agent/sparse_mad_bedrock.py
```

## Command-line Overrides

Hydra allows you to override configuration values from the command line:

```bash
# Override the number of solvers
python src/sycophancy_multi_agent/main.py solver.num_solvers=3

# Override the dataset and number of samples
python src/sycophancy_multi_agent/main.py data.dataset=gsm8k data.num_samples=100

# Use specific model configurations
python src/sycophancy_multi_agent/main.py bedrock/models=claude_sonnet_3_7
python src/sycophancy_multi_agent/main.py openai/models=gpt_4o
python src/sycophancy_multi_agent/main.py llama/models=llama3_3_70b

# Override model temperature for specific providers
python src/sycophancy_multi_agent/main.py bedrock.models.claude_haiku.config.temperature=0.5

# Use different solver configurations
python src/sycophancy_multi_agent/main.py solver=multi_agent_3_sycophancy
python src/sycophancy_multi_agent/main.py solver=single_agent

# Use different judge configurations
python src/sycophancy_multi_agent/main.py judger=strict
python src/sycophancy_multi_agent/main.py judger=math_focused

# Use dataset-specific prompts
python src/sycophancy_multi_agent/main.py prompts=gsm8k
python src/sycophancy_multi_agent/main.py prompts=mmlu_pro

# Toggle between sycophancy trigger and system prompt
python src/sycophancy_multi_agent/main.py solver.sycophancy.use_trigger=false

# Override the sycophancy system prompt
python src/sycophancy_multi_agent/main.py solver.sycophancy.system_prompt="Your custom system prompt here"
```

## Configuration Composition

You can select different configuration groups at runtime:

### Model Provider Selection
```bash
# Use different model providers
python src/sycophancy_multi_agent/main.py bedrock=default
python src/sycophancy_multi_agent/main.py openai=default
python src/sycophancy_multi_agent/main.py llama=default
python src/sycophancy_multi_agent/main.py qwen=default
```

### Specific Model Selection
```bash
# Use specific Bedrock models
python src/sycophancy_multi_agent/main.py bedrock/models=claude_opus_4
python src/sycophancy_multi_agent/main.py bedrock/models=deepseek_r1

# Use specific OpenAI models
python src/sycophancy_multi_agent/main.py openai/models=gpt_4o_mini
python src/sycophancy_multi_agent/main.py openai/models=gpt_4_1_mini

# Use specific Llama models
python src/sycophancy_multi_agent/main.py llama/models=llama3_1_70b
python src/sycophancy_multi_agent/main.py llama/models=llama3_2_3b

# Use specific Qwen models
python src/sycophancy_multi_agent/main.py qwen/models=qwen3_32b
python src/sycophancy_multi_agent/main.py qwen/models=qwen2_7b
```

### Solver Configuration Selection
```bash
# Use different solver setups
python src/sycophancy_multi_agent/main.py solver=single_agent
python src/sycophancy_multi_agent/main.py solver=multi_agent_3
python src/sycophancy_multi_agent/main.py solver=multi_agent_sycophancy
python src/sycophancy_multi_agent/main.py solver=multi_agent_trigger
```

### Judge Configuration Selection
```bash
# Use different judge configurations
python src/sycophancy_multi_agent/main.py judger=default
python src/sycophancy_multi_agent/main.py judger=fast
python src/sycophancy_multi_agent/main.py judger=strict
python src/sycophancy_multi_agent/main.py judger=math_focused
```

### Prompt Template Selection
```bash
# Use dataset-specific prompts
python src/sycophancy_multi_agent/main.py prompts=gsm8k
python src/sycophancy_multi_agent/main.py prompts=mmlu
python src/sycophancy_multi_agent/main.py prompts=ai2_arc_challenge
python src/sycophancy_multi_agent/main.py prompts=commonsenseqa
``
