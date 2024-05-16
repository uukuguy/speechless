#!/usr/bin/env python
import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import datasets

model_root_dir = "/opt/local/llm_models/huggingface.co"
seed = 18341
max_seq_length = 2048  # Supports RoPE Scaling interally, so choose any!
eval_size = 1000
train_file = "/opt/local/datasets/alpaca_gpt4/alpaca_gpt4_data.json"

# ==================== Utils ====================
def show_gpu_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory

def show_trainer_stats(trainer_stats):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ==================== Model & Tokenizer ====================
def load_model_and_tokenizer(model_name, max_seq_length):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer

# ==================== Dataset ====================
full_dataset = load_dataset("json", data_files=train_file, split="train")
data_module = full_dataset.train_test_split(test_size=eval_size, shuffle=True, seed=seed)
train_dataset = data_module['train']
eval_dataset = data_module['test']


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)

# ==================== Trainer ====================

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        # max_steps=60,
        num_train_epochs=3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        warmup_steps=10,
        logging_steps=1,
        # save_strategy="steps",
        # save_steps=100,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# ==================== Do Training ====================
start_gpu_memory, max_memory = show_gpu_stats()
trainer_stats = trainer.train()
show_trainer_stats(trainer_stats)


# ==================== Pipeline ====================
args = []
model, tokenizer = load_model_and_tokenizer(model_name=args.model_name, max_seq_length=args.max_seq_length)
