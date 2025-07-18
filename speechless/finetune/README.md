# LightLLM: Lightweight Framework for LLM Fine-Tuning

A modular and efficient framework for fine-tuning large language models using various training methods and adaptation techniques.

## Features

- **Multiple Training Methods**:
  - Supervised Fine-Tuning (SFT)
  - Proximal Policy Optimization (PPO)
  - Generalized Reward-weighted Policy Optimization (GRPO)

- **Adaptation Techniques**:
  - LoRA (Low-Rank Adaptation)
  - Full Fine-Tuning

- **Key Capabilities**:
  - Distributed training via Accelerate
  - Mixed precision training (FP16/BF16)
  - Logging with Weights & Biases
  - Flexible configuration system
  - Optimized inference
  - LoRA weight merging

## Installation

```bash
# Install from the repository
pip install -e .
```

## Usage Examples

### Supervised Fine-Tuning with LoRA

```bash
python -m speechless.finetune \
  --mode train \
  --model_name_or_path "facebook/opt-1.3b" \
  --train_file "data/train.json" \
  --eval_file "data/eval.json" \
  --training_method "sft" \
  --adaptation_method "lora" \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir "./output/sft-lora"
```

### PPO with Full Fine-Tuning

```bash
python -m speechless.finetune \
  --mode train \
  --model_name_or_path "facebook/opt-1.3b" \
  --train_file "data/ppo_data.json" \
  --training_method "ppo" \
  --adaptation_method "full" \
  --num_train_epochs 5 \
  --output_dir "./output/ppo-full"
```

### Merging LoRA Weights

```bash
python -m speechless.finetune \
  --mode merge_lora \
  --model_name_or_path "facebook/opt-1.3b" \
  --lora_model_path "./output/sft-lora/final" \
  --merged_model_path "./output/merged-model"
```

### Inference

```bash
python -m speechless.finetune \
  --mode inference \
  --model_name_or_path "./output/merged-model" \
  --prompt "Hello, how are you?" \
  --max_new_tokens 100 \
  --temperature 0.7
```

## Data Format

### SFT Format

```json
[
  {
    "input": "Human: What is the capital of France?\n",
    "output": "Assistant: The capital of France is Paris."
  }
]
```

### RL Format (PPO/GRPO)

```json
[
  {
    "prompt": "Human: What is the capital of France?\n",
    "response": "Assistant: The capital of France is Paris.",
    "reward": 0.8
  }
]
```

## Configuration Options

The framework provides a wide range of configuration options:

- **Model Settings**: model path, tokenizer path
- **Data Settings**: train file, eval file, sequence length
- **Training Settings**: batch size, learning rate, epochs, etc.
- **LoRA Settings**: rank, alpha, dropout, target modules
- **PPO/GRPO Settings**: KL coefficient, clip range, etc.
- **System Settings**: precision (FP16/BF16), distributed training

## Development

### Running Tests

```bash
python -m unittest speechless.finetune.test_lightllm
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.
