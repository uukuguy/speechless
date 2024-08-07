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

# Get LAION dataset
# url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
# dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

train_file = "/opt/local/datasets/alpaca_gpt4/alpaca_gpt4_data.json"
full_dataset = load_dataset("json", data_files=train_file, split="train")

data_module = full_dataset.train_test_split(test_size=eval_size, shuffle=True, seed=seed)
train_dataset = data_module['train']
eval_dataset = data_module['test']

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",  # [NEW] 15 Trillion token Llama-3
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/llama-3-70b-Instruct-bnb-4bit",
    "unsloth/mistral-7b-v0.2-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/codegemma-7b-bnb-4bit",
    "unsloth/yi-34b-bnb-4bit",
    "unsloth/yi-34b-chat-bnb-4bit",
    "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit",
    "unsloth/zephyr-sft-bnb-4bit",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",  # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",  # Instruct version of Gemma 2b
    "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/llama-3-8b-bnb-4bit",
    model_name=os.path.join(model_root_dir, "unsloth/mistral-7b-v0.2-bnb-4bit"),
    # model_name=os.path.join(model_root_dir, "mistral/Mistral-7B-v0.2-hf"),
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Do model patching and add fast LoRA weights
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
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

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

start_gpu_memory, max_memory = show_gpu_stats()
trainer_stats = trainer.train()
show_trainer_stats(trainer_stats)


# Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Cutomized chat templates
