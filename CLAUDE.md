# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Speechless repository is a comprehensive LLM training and inference framework focused on building specialized code generation models. It includes fine-tuning capabilities, evaluation pipelines, and model serving infrastructure.

## Core Architecture

### Main Components

- **Model Training**: Full fine-tuning, LoRA, QLoRA, FSDP, and specialized methods like GRPO, PPO
- **Model Evaluation**: HumanEval, MultiPL-E, lm-evaluation-harness, CodeQwen evaluation
- **Model Serving**: OpenAI-compatible API servers (vLLM, Llama.cpp, MLX, SGLang)
- **Data Processing**: Dataset generation, cleaning, and formatting for various domains
- **Quantization**: GGUF support, AWQ, GPTQ conversion utilities
- **Tool Integration**: Framework for tool-calling and agent capabilities

### Key Directories

- `speechless/finetune/` - Core training pipeline with LoRA/QLoRA support
- `speechless/eval/` - Evaluation frameworks (HumanEval, MultiPL-E, etc.)
- `speechless/api/` - Model serving APIs
- `speechless/generate/` - Model inference utilities
- `speechless/agents/` - Tool-calling and agent framework
- `speechless/quant/` - Model quantization utilities

## Commands

### Setup & Installation

```bash
# Install dependencies
pip install speechless
# or
poetry install

# Set environment variables
export SPEECHLESS_ROOT=/path/to/speechless
```

### Model Training

```bash
# Initialize new training task
python -m speechless.finetune init --task_name my_task

# Run training
python -m speechless.finetune run --task_name my_task

# Merge LoRA adapters
python -m speechless.finetune merge --task_name my_task

# Alternative Makefile commands
make finetune_mistral_7b
make finetune_34b
```

### Model Evaluation

```bash
# HumanEval evaluation
export TEST_MODEL_PATH=/path/to/your/model
make humaneval

# MultiPL-E evaluation
make multiple_gen TEST_MODEL_PATH=/path/to/your/model
make multiple TEST_MODEL_PATH=/path/to/your/model

# lm-evaluation-harness
make lmeval TEST_MODEL_PATH=/path/to/your/model
```

### Model Serving

```bash
# Start API server (vLLM backend)
python -m speechless.api.server start --model /path/to/model --backbone vllm --port 5001

# Alternative Makefile command
make api_server TEST_MODEL_PATH=/path/to/model
```

### Data Processing

```bash
# Generate synthetic datasets
python scripts/generate_speechless_datasets/generate_speechless_coding_dataset.py

# Convert HF to GGUF
python -m speechless.quant llamacpp --model_path /path/to/hf/model --llamacpp_quant_type q4_k_m
```

## Environment Configuration

Key environment variables to set:

```bash
export MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
export SPEECHLESS_DATA_DIR=/opt/local/datasets/speechless_data
export SPEECHLESS_ROOT=/path/to/speechless
```

## Training Configuration

Training tasks are configured through task-specific directories containing:
- `task.env` - Environment variables and hyperparameters
- `run_finetune.sh` - Training script
- `merge.sh` - Model merging script
- `dataset/` - Training data location

## Evaluation Metrics

- **Code Generation**: HumanEval, MultiPL-E pass rates
- **General benchmarks**: lm-evaluation-harness results
- **Tool-calling**: ToolBench datasets performance
- **SQL Generation**: SQLEval benchmark

## Model Formats

Supported model backends:
- Hugging Face Transformers
- GGUF (GGML quantized)
- AWQ quantized
- MLX (for Apple Silicon)
- vLLM (high-performance serving)

## Common Workflows

1. **Training Pipeline**: `init → run → merge → eval → deploy`
2. **Evaluation Pipeline**: `generate → evaluate → analyze results`
3. **Model Serving**: `train → quantize → serve via API`

## Technical Stack

- **Training**: PyTorch Lightning, Transformers, PEFT, DeepSpeed
- **Optimization**: LoRA, QLoRA, FSDP, Gradient Checkpointing
- **Evaluation**: Custom eval frameworks, benchmarks
- **Serving**: FastAPI, vLLM, Llama.cpp
- **Data**: JSONL format, conversation-style datasets

This framework emphasizes reproducibility and extensibility for LLM research and applied AI systems.