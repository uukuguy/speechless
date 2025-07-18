#!/usr/bin/env python
"""
# LightLLM: Lightweight Framework for LLM Fine-Tuning

A modular and efficient framework for fine-tuning large language models using various
training methods and adaptation techniques.

## Training Methods

- **Supervised Fine-Tuning (SFT)**: Standard instruction tuning
- **Proximal Policy Optimization (PPO)**: Reinforcement learning from human feedback
- **Generalized Reward-weighted Policy Optimization (GRPO)**: Alternative to PPO

## Adaptation Methods

- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
- **Full Fine-Tuning**: Complete model parameter updates

## Key Features

- Integrated with 🤗 Transformers and PEFT libraries
- Distributed training support via Accelerate
- Mixed precision training (FP16/BF16)
- Logging with Weights & Biases
- Flexible configuration system
- Optimized inference capabilities
- LoRA weight merging utilities

## Usage Example

```bash
# SFT with LoRA
python -m speechless.finetune \
  --model_name_or_path "facebook/opt-1.3b" \
  --train_file "data/train.json" \
  --eval_file "data/eval.json" \
  --training_method "sft" \
  --adaptation_method "lora" \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir "./output/sft-lora"

# PPO with full fine-tuning
python -m speechless.finetune \
  --model_name_or_path "facebook/opt-1.3b" \
  --train_file "data/ppo_data.json" \
  --training_method "ppo" \
  --adaptation_method "full" \
  --num_train_epochs 5 \
  --output_dir "./output/ppo-full"
```
"""

import os
import logging
import json
import math
import gc
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, TypeVar, Generic, cast
from dataclasses import dataclass, field
from contextlib import nullcontext
from enum import Enum, auto
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm.auto import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    TextIteratorStreamer,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Type variables for generic typing
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=PreTrainedModel)


class TrainingMethod(Enum):
    """Supported training methods."""
    SFT = auto()
    PPO = auto()
    GRPO = auto()
    
    @classmethod
    def from_string(cls, value: str) -> 'TrainingMethod':
        """Convert string to enum value with validation."""
        try:
            return {
                "sft": cls.SFT,
                "ppo": cls.PPO,
                "grpo": cls.GRPO
            }[value.lower()]
        except KeyError:
            valid_values = ", ".join([f"'{k}'" for k in ["sft", "ppo", "grpo"]])
            raise ValueError(f"Invalid training method: '{value}'. Valid values are: {valid_values}")


class AdaptationMethod(Enum):
    """Supported model adaptation methods."""
    LORA = auto()
    FULL = auto()
    
    @classmethod
    def from_string(cls, value: str) -> 'AdaptationMethod':
        """Convert string to enum value with validation."""
        try:
            return {
                "lora": cls.LORA,
                "full": cls.FULL
            }[value.lower()]
        except KeyError:
            valid_values = ", ".join([f"'{k}'" for k in ["lora", "full"]])
            raise ValueError(f"Invalid adaptation method: '{value}'. Valid values are: {valid_values}")

@dataclass
class TrainingConfig:
    """Configuration for training a language model.
    
    This class contains all parameters needed to configure the training process,
    including model settings, data settings, training hyperparameters, and system
    configuration.
    """
    # Model settings
    model_name_or_path: str = "facebook/opt-350m"
    tokenizer_name_or_path: Optional[str] = None  # If None, uses model_name_or_path
    
    # Data settings
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    max_seq_length: int = 512
    
    # Training settings
    training_method: str = "sft"  # One of: sft, ppo, grpo
    adaptation_method: str = "lora"  # One of: lora, full
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1  # Ratio of total training steps for warmup
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # PPO/GRPO settings
    kl_coef: float = 0.1
    clip_range: float = 0.2
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    num_ppo_epochs: int = 4
    
    # Logging and saving
    output_dir: str = "./output"
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    use_wandb: bool = False
    wandb_project: str = "llm-finetuning"
    
    # System settings
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    local_rank: int = -1
    use_deepspeed: bool = False
    use_fsdp: bool = False
    
    def __post_init__(self) -> None:
        """Validate and initialize configuration after creation."""
        # Set tokenizer path if not provided
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
        
        # Validate training method
        try:
            self.training_method_enum = TrainingMethod.from_string(self.training_method)
        except ValueError as e:
            raise ValueError(f"Invalid training method: {self.training_method}") from e
        
        # Validate adaptation method
        try:
            self.adaptation_method_enum = AdaptationMethod.from_string(self.adaptation_method)
        except ValueError as e:
            raise ValueError(f"Invalid adaptation method: {self.adaptation_method}") from e
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check for wandb availability if use_wandb is True
        if self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("Weights & Biases (wandb) is not installed. Setting use_wandb to False.")
            self.use_wandb = False


class DataFormat(Enum):
    """Supported data formats for training."""
    SFT = auto()  # Supervised fine-tuning format with input/output pairs
    RL = auto()   # Reinforcement learning format with prompt/response/reward

class TextDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
    ):
        super(TextDataset, self).__init__()
        
        logger.info(f"Loading data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # For SFT, we expect "input" and "output" fields
        if "input" in item and "output" in item:
            # Format: <s>input</s>output</s>
            input_text = item["input"]
            output_text = item["output"]
            text = input_text + output_text
            
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            input_ids = encodings.input_ids[0]
            attention_mask = encodings.attention_mask[0]
            
            # Create labels for SFT (shifted input_ids)
            labels = input_ids.clone()
            
            # Create masks for input and output parts
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            input_length = input_encoding.input_ids.shape[1]
            
            # Set labels to -100 for input part (we don't want to compute loss for it)
            labels[:input_length] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        # For PPO/GRPO, we expect "prompt", "response", and "reward" fields
        elif "prompt" in item and "response" in item and "reward" in item:
            prompt_text = item["prompt"]
            response_text = item["response"]
            reward = item["reward"]
            
            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            response_encoding = self.tokenizer(
                response_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            return {
                "prompt_input_ids": prompt_encoding.input_ids[0],
                "prompt_attention_mask": prompt_encoding.attention_mask[0],
                "response_input_ids": response_encoding.input_ids[0],
                "response_attention_mask": response_encoding.attention_mask[0],
                "reward": torch.tensor(reward, dtype=torch.float),
            }
        
        else:
            raise ValueError(f"Invalid data format: {item}")

class ModelAdapter(ABC):
    """Abstract base class for model adaptation methods."""
    
    @staticmethod
    @abstractmethod
    def prepare_model(model: PreTrainedModel, config: TrainingConfig) -> PreTrainedModel:
        """Prepare model for fine-tuning.
        
        Args:
            model: The pre-trained model to adapt
            config: Training configuration
            
        Returns:
            Adapted model ready for fine-tuning
        """
        pass
    
    @staticmethod
    @abstractmethod
    def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str) -> None:
        """Save model and tokenizer.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_dir: Directory to save to
        """
        pass


class LoRAAdapter(ModelAdapter):
    """Adapter for LoRA fine-tuning."""
    
    @staticmethod
    def prepare_model(model: PreTrainedModel, config: TrainingConfig) -> PreTrainedModel:
        """Prepare model for LoRA fine-tuning.
        
        Args:
            model: The pre-trained model to adapt
            config: Training configuration
            
        Returns:
            Model with LoRA adapters attached
        """
        # For 4-bit or 8-bit quantized models
        if hasattr(model, "is_quantized") and model.is_quantized:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Get LoRA model
        model = get_peft_model(model, lora_config)
        
        # Log trainable parameters
        trainable_params, all_params = LoRAAdapter._count_parameters(model)
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%} of {all_params:,} total)")
        
        return model
    
    @staticmethod
    def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str) -> None:
        """Save LoRA model and adapter.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_dir: Directory to save to
        """
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapter weights
        model.save_pretrained(output_dir)
        
        logger.info(f"LoRA model saved to {output_dir}")
    
    @staticmethod
    def _count_parameters(model: PreTrainedModel) -> Tuple[int, int]:
        """Count trainable and total parameters in the model.
        
        Args:
            model: The model to analyze
            
        Returns:
            Tuple of (trainable_params, all_params)
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        return trainable_params, all_params


class FullAdapter(ModelAdapter):
    """Adapter for full fine-tuning."""
    
    @staticmethod
    def prepare_model(model: PreTrainedModel, config: TrainingConfig) -> PreTrainedModel:
        """Prepare model for full fine-tuning.
        
        Args:
            model: The pre-trained model to adapt
            config: Training configuration
            
        Returns:
            Model ready for full fine-tuning
        """
        # No special preparation needed for full fine-tuning
        # Log trainable parameters
        trainable_params, all_params = FullAdapter._count_parameters(model)
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%} of {all_params:,} total)")
        
        return model
    
    @staticmethod
    def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str) -> None:
        """Save fully fine-tuned model and tokenizer.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_dir: Directory to save to
        """
        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Full model saved to {output_dir}")
    
    @staticmethod
    def _count_parameters(model: PreTrainedModel) -> Tuple[int, int]:
        """Count trainable and total parameters in the model.
        
        Args:
            model: The model to analyze
            
        Returns:
            Tuple of (trainable_params, all_params)
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        return trainable_params, all_params

class BaseTrainer(ABC, Generic[ModelType]):
    """Abstract base class for all trainers.
    
    This class provides common functionality for all training methods.
    """
    
    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set seed for reproducibility
        self._set_seed(config.seed)
        
        # Initialize tokenizer
        self.tokenizer = self._initialize_tokenizer()
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            fp16=config.fp16,
            bf16=config.bf16,
        )
        
        # Initialize wandb if needed
        if config.use_wandb and WANDB_AVAILABLE and self.accelerator.is_local_main_process:
            wandb.init(project=config.wandb_project)
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        """Initialize the tokenizer.
        
        Returns:
            Initialized tokenizer
        """
        logger.info(f"Loading tokenizer from {self.config.tokenizer_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path,
            use_fast=True,
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            logger.info("Pad token not found, using EOS token instead")
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _get_model_adapter(self) -> ModelAdapter:
        """Get the appropriate model adapter based on the configuration.
        
        Returns:
            Model adapter instance
        """
        if self.config.adaptation_method_enum == AdaptationMethod.LORA:
            return LoRAAdapter
        else:  # AdaptationMethod.FULL
            return FullAdapter
    
    def _create_optimizer(self, model: ModelType) -> Optimizer:
        """Create optimizer for the model.
        
        Args:
            model: The model to optimize
            
        Returns:
            Configured optimizer
        """
        return AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
    
    def _create_scheduler(self, optimizer: Optimizer, num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler.
        
        Args:
            optimizer: The optimizer
            num_training_steps: Total number of training steps
            
        Returns:
            Learning rate scheduler
        """
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        logger.info(f"Using {num_warmup_steps} warmup steps ({self.config.warmup_ratio:.1%} of {num_training_steps} total steps)")
        
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, epoch: Optional[float] = None) -> None:
        """Log metrics to console and wandb if enabled.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step (optional)
            epoch: Current epoch (optional)
        """
        # Log to console
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        if step is not None:
            metrics_str = f"Step {step}: {metrics_str}"
        if epoch is not None:
            metrics_str = f"Epoch {epoch:.2f}: {metrics_str}"
        
        logger.info(metrics_str)
        
        # Log to wandb
        if self.config.use_wandb and WANDB_AVAILABLE and self.accelerator.is_local_main_process:
            wandb_metrics = {k: v for k, v in metrics.items()}
            if step is not None:
                wandb_metrics["step"] = step
            if epoch is not None:
                wandb_metrics["epoch"] = epoch
            
            wandb.log(wandb_metrics)
    
    def save_model(self, output_dir: str) -> None:
        """Save model and tokenizer.
        
        Args:
            output_dir: Directory to save to
        """
        if not self.accelerator.is_local_main_process:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Use the appropriate adapter to save the model
        adapter = self._get_model_adapter()
        adapter.save_model(unwrapped_model, self.tokenizer, output_dir)
    
    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass


class SupervisedTrainer(BaseTrainer[PreTrainedModel]):
    """Trainer for supervised fine-tuning (SFT)."""
    
    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the SFT trainer.
        
        Args:
            config: Training configuration
            
        Raises:
            ValueError: If train_file is not provided
        """
        super().__init__(config)
        
        if not config.train_file:
            raise ValueError("train_file must be provided for SFT")
        
        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
        )
        
        # Apply LoRA or prepare for full fine-tuning
        if config.adaptation_method == "lora":
            self.model = LoRAAdapter.prepare_model(self.model, config)
        else:  # Full fine-tuning
            # No special preparation needed for full fine-tuning
            pass
        
        # Set up datasets and data loaders
        train_dataset = TextDataset(
            config.train_file,
            self.tokenizer,
            config.max_seq_length,
        )
        
        eval_dataset = None
        if config.eval_file:
            eval_dataset = TextDataset(
                config.eval_file,
                self.tokenizer,
                config.max_seq_length,
            )
        
        # Set up accelerator
        self.accelerator = Accelerator(
            fp16=config.fp16,
            bf16=config.bf16,
        )
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
            )
        
        # Create optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / config.gradient_accumulation_steps
        )
        max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * max_train_steps),
            num_training_steps=max_train_steps,
        )
        
        # Prepare everything with the accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
        )
        
        self.max_train_steps = max_train_steps
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        
        # Initialize wandb if needed
        if config.use_wandb:
            wandb.init(project=config.wandb_project)
    
    def train(self) -> None:
        """Train the model using SFT."""
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        
        # Initialize progress bar
        progress_bar = tqdm(range(self.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        
        # Training loop
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            total_loss = 0
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    self.accelerator.backward(loss)
                    
                    # Clip gradients
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Update model
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Log loss
                total_loss += loss.detach().float()
                
                # Update progress bar
                if step % self.config.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    completed_steps += 1
                    progress_bar.update(1)
                    
                    # Log metrics periodically
                    if completed_steps % self.config.logging_steps == 0:
                        avg_loss = total_loss.item() / self.config.logging_steps
                        lr = self.lr_scheduler.get_last_lr()[0]
                        current_epoch = epoch + step / len(self.train_dataloader)
                        
                        self._log_metrics(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                            },
                            step=completed_steps,
                            epoch=current_epoch,
                        )
                        total_loss = 0
                    
                    # Evaluate periodically
                    if self.eval_dataloader is not None and completed_steps % self.config.eval_steps == 0:
                        self.evaluate()
                    
                    # Save checkpoint periodically
                    if completed_steps % self.config.save_steps == 0:
                        self.save_model(f"{self.config.output_dir}/checkpoint-{completed_steps}")
                
                if completed_steps >= self.max_train_steps:
                    break
        
        # Final evaluation
        if self.eval_dataloader is not None:
            self.evaluate()
        
        # Save final model
        self.save_model(f"{self.config.output_dir}/final")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("***** Running evaluation *****")
        self.model.eval()
        eval_loss = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                eval_loss += loss.detach().float()
        
        eval_loss = eval_loss / len(self.eval_dataloader)
        perplexity = torch.exp(eval_loss)
        
        metrics = {
            "eval/loss": eval_loss.item(),
            "eval/perplexity": perplexity.item(),
        }
        
        self._log_metrics(metrics)
        
        return metrics
    
    def save_model(self, output_dir):
        """Save model and tokenizer."""
        
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model and tokenizer
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            if self.config.adaptation_method == "lora":
                LoRAAdapter.save_model(unwrapped_model, self.tokenizer, output_dir)
            else:  # Full fine-tuning
                unwrapped_model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Model saved to {output_dir}")

class PPOTrainer(BaseTrainer[PreTrainedModel]):
    """Trainer for Proximal Policy Optimization (PPO)."""
    
    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the PPO trainer.
        
        Args:
            config: Training configuration
            
        Raises:
            ValueError: If train_file is not provided
        """
        super().__init__(config)
        
        if not config.train_file:
            raise ValueError("train_file must be provided for PPO")
        
        # Initialize models
        self.policy_model, self.ref_model, self.value_model, self.value_head = self._initialize_models()
        
        # Set up datasets and data loaders
        self.train_dataloader = self._setup_dataloaders()
        
        # Create optimizers
        self.policy_optimizer, self.value_optimizer = self._setup_optimization()
        
        # Prepare everything with the accelerator
        (
            self.policy_model,
            self.value_model,
            self.value_head,
            self.ref_model,
            self.policy_optimizer,
            self.value_optimizer,
            self.train_dataloader,
        ) = self.accelerator.prepare(
            self.policy_model,
            self.value_model,
            self.value_head,
            self.ref_model,
            self.policy_optimizer,
            self.value_optimizer,
            self.train_dataloader,
        )
    
    def _initialize_models(self) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedModel, nn.Linear]:
        """Initialize policy, reference, and value models.
        
        Returns:
            Tuple of (policy_model, ref_model, value_model, value_head)
        """
        logger.info(f"Loading models from {self.config.model_name_or_path}")
        
        # Initialize policy model
        policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.fp16 else (
                torch.bfloat16 if self.config.bf16 else torch.float32
            ),
        )
        
        # Initialize reference model (frozen copy of policy model)
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.fp16 else (
                torch.bfloat16 if self.config.bf16 else torch.float32
            ),
        )
        
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # Apply adaptation method to policy model
        adapter = self._get_model_adapter()
        policy_model = adapter.prepare_model(policy_model, self.config)
        
        # Initialize value model
        # In practice, value models can be separate or share parameters with policy model
        # Here we create a separate model for simplicity
        value_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.fp16 else (
                torch.bfloat16 if self.config.bf16 else torch.float32
            ),
        )
        
        # Modify the value model to output a scalar value
        value_head = nn.Linear(value_model.config.hidden_size, 1)
        
        # Apply adaptation method to value model
        if self.config.adaptation_method_enum == AdaptationMethod.LORA:
            value_model = adapter.prepare_model(value_model, self.config)
        
        return policy_model, ref_model, value_model, value_head
    
    def _setup_dataloaders(self) -> DataLoader:
        """Set up training data loader.
        
        Returns:
            Training data loader
        """
        # Create training dataset
        train_dataset = TextDataset(
            self.config.train_file,
            self.tokenizer,
            self.config.max_seq_length,
        )
        
        # Verify dataset format
        if not hasattr(train_dataset, 'data_format') or train_dataset.data_format != DataFormat.RL:
            logger.warning("Dataset may not be in the expected RL format (prompt/response/reward). Check your data.")
        
        # Create training dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        
        return train_dataloader
        
    def _setup_optimization(self) -> Tuple[Optimizer, Optimizer]:
        """Set up optimizers for policy and value models.
        
        Returns:
            Tuple of (policy_optimizer, value_optimizer)
        """
        # Create policy optimizer
        policy_optimizer = self._create_optimizer(self.policy_model)
        
        # Create value optimizer (includes value head parameters)
        value_optimizer = AdamW(
            list(self.value_model.parameters()) + list(self.value_head.parameters()),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        return policy_optimizer, value_optimizer
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: Reward values for each step
            values: Value estimates for each step
            masks: Masks indicating valid steps (1.0) or padding (0.0)
            gamma: Discount factor for future rewards
            lam: GAE lambda parameter for advantage weighting
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            last_gae_lam = delta + gamma * lam * masks[t] * last_gae_lam
            advantages[t] = last_gae_lam
        
        returns = advantages + values
        
        return advantages, returns
    
    def get_logprobs_and_values(
        self,
        model: PreTrainedModel,
        value_head: nn.Linear,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get log probabilities and values for the given inputs.
        
        Args:
            model: The model to get outputs from
            value_head: Linear layer for value prediction
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (log_probs, values)
            
        Raises:
            ValueError: If model outputs don't contain hidden states for value computation
        """
        # Get model outputs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        logits = outputs.logits
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get values
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("Model outputs don't contain hidden states needed for value computation")
            
        hidden_states = outputs.hidden_states[-1]  # Last hidden state
        values = value_head(hidden_states).squeeze(-1)
        
        return log_probs, values
    
    def train(self) -> None:
        """Train the model using PPO."""
        
        logger.info("***** Running PPO training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  PPO epochs per batch = {self.config.num_ppo_epochs}")
        logger.info(f"  KL coefficient = {self.config.kl_coef}")
        logger.info(f"  Value loss coefficient = {self.config.value_loss_coef}")
        logger.info(f"  Entropy coefficient = {self.config.entropy_coef}")
        logger.info(f"  Clip range = {self.config.clip_range}")
        
        # Training loop
        for epoch in range(self.config.num_train_epochs):
            self.policy_model.train()
            self.value_model.train()
            self.value_head.train()
            
            total_policy_loss = 0
            total_value_loss = 0
            total_kl_loss = 0
            total_entropy = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Get prompt and response data
                prompt_input_ids = batch["prompt_input_ids"]
                prompt_attention_mask = batch["prompt_attention_mask"]
                response_input_ids = batch["response_input_ids"]
                response_attention_mask = batch["response_attention_mask"]
                rewards = batch["reward"]
                
                # Generate outputs with the policy model
                with torch.no_grad():
                    policy_log_probs, policy_values = self.get_logprobs_and_values(
                        self.policy_model,
                        self.value_head,
                        response_input_ids,
                        response_attention_mask,
                    )
                    
                    # Get reference model log probabilities for KL penalty
                    ref_log_probs, _ = self.get_logprobs_and_values(
                        self.ref_model,
                        self.value_head,
                        response_input_ids,
                        response_attention_mask,
                    )
                
                # Compute advantages and returns
                masks = response_attention_mask.float()
                advantages, returns = self.compute_advantages(rewards, policy_values, masks)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO update loop
                for ppo_epoch in range(self.config.num_ppo_epochs):
                    # Forward pass
                    new_log_probs, values = self.get_logprobs_and_values(
                        self.policy_model,
                        self.value_head,
                        response_input_ids,
                        response_attention_mask,
                    )
                    
                    # Compute entropy
                    entropy = -torch.sum(torch.exp(new_log_probs) * new_log_probs, dim=-1)
                    entropy = torch.mean(entropy)
                    
                    # Compute KL divergence
                    kl = torch.sum(
                        torch.exp(ref_log_probs) * (ref_log_probs - new_log_probs),
                        dim=-1,
                    )
                    kl = torch.mean(kl)
                    
                    # Compute policy loss
                    log_ratio = new_log_probs - policy_log_probs
                    ratio = torch.exp(log_ratio)
                    
                    policy_loss1 = -advantages * ratio
                    policy_loss2 = -advantages * torch.clamp(
                        ratio,
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range,
                    )
                    policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                    
                    # Add KL penalty
                    policy_loss += self.config.kl_coef * kl
                    
                    # Compute value loss
                    value_loss = F.mse_loss(values, returns)
                    
                    # Total loss
                    loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
                    
                    # Backward pass and optimization
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    
                    self.accelerator.backward(loss)
                    
                    # Clip gradients
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
                        self.accelerator.clip_grad_norm_(
                            list(self.value_model.parameters()) + list(self.value_head.parameters()),
                            self.config.max_grad_norm,
                        )
                    
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                
                # Update metrics
                total_policy_loss += policy_loss.detach().float()
                total_value_loss += value_loss.detach().float()
                total_kl_loss += kl.detach().float()
                total_entropy += entropy.detach().float()
                
                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "kl": kl.item(),
                        "entropy": entropy.item(),
                    }
                )
            
            # Log epoch metrics
            avg_policy_loss = total_policy_loss / len(self.train_dataloader)
            avg_value_loss = total_value_loss / len(self.train_dataloader)
            avg_kl_loss = total_kl_loss / len(self.train_dataloader)
            avg_entropy = total_entropy / len(self.train_dataloader)
            
            metrics = {
                "train/policy_loss": avg_policy_loss.item(),
                "train/value_loss": avg_value_loss.item(),
                "train/kl": avg_kl_loss.item(),
                "train/entropy": avg_entropy.item(),
            }
            
            self._log_metrics(metrics, epoch=epoch + 1)
            
            # Save model periodically
            if (epoch + 1) % 5 == 0 or epoch == self.config.num_train_epochs - 1:
                self.save_model(f"{self.config.output_dir}/checkpoint-{epoch+1}")
        
        # Save final model
        self.save_model(f"{self.config.output_dir}/final")
    
    def save_model(self, output_dir: str) -> None:
        """Save policy model and tokenizer.
        
        Args:
            output_dir: Directory to save to
        """
        if not self.accelerator.is_local_main_process:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
        
        # Use the appropriate adapter to save the model
        adapter = self._get_model_adapter()
        adapter.save_model(unwrapped_model, self.tokenizer, output_dir)

class GRPOTrainer(BaseTrainer[PreTrainedModel]):
    """Trainer for Generalized Reward-weighted Policy Optimization (GRPO).
    
    GRPO is similar to PPO, but with a different objective function.
    It uses a generalized form of the policy gradient that includes
    reward, KL, and entropy terms.
    """
    
    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the GRPO trainer.
        
        Args:
            config: Training configuration
            
        Raises:
            ValueError: If train_file is not provided
        """
        super().__init__(config)
        
        if not config.train_file:
            raise ValueError("train_file must be provided for GRPO")
        
        # Initialize models
        self.policy_model, self.ref_model = self._initialize_models()
        
        # Set up datasets and data loaders
        self.train_dataloader = self._setup_dataloaders()
        
        # Create optimizer
        self.optimizer = self._setup_optimization()
        
        # Prepare everything with the accelerator
        (
            self.policy_model,
            self.ref_model,
            self.optimizer,
            self.train_dataloader,
        ) = self.accelerator.prepare(
            self.policy_model,
            self.ref_model,
            self.optimizer,
            self.train_dataloader,
        )
    
    def _initialize_models(self) -> Tuple[PreTrainedModel, PreTrainedModel]:
        """Initialize policy and reference models.
        
        Returns:
            Tuple of (policy_model, ref_model)
        """
        logger.info(f"Loading models from {self.config.model_name_or_path}")
        
        # Initialize policy model
        policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.fp16 else (
                torch.bfloat16 if self.config.bf16 else torch.float32
            ),
        )
        
        # Initialize reference model (frozen copy of policy model)
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.fp16 else (
                torch.bfloat16 if self.config.bf16 else torch.float32
            ),
        )
        
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # Apply adaptation method to policy model
        adapter = self._get_model_adapter()
        policy_model = adapter.prepare_model(policy_model, self.config)
        
        return policy_model, ref_model
    
    def _setup_dataloaders(self) -> DataLoader:
        """Set up training data loader.
        
        Returns:
            Training data loader
        """
        # Create training dataset
        train_dataset = TextDataset(
            self.config.train_file,
            self.tokenizer,
            self.config.max_seq_length,
        )
        
        # Verify dataset format
        if not hasattr(train_dataset, 'data_format') or train_dataset.data_format != DataFormat.RL:
            logger.warning("Dataset may not be in the expected RL format (prompt/response/reward). Check your data.")
        
        # Create training dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        
        return train_dataloader
    
    def _setup_optimization(self) -> Optimizer:
        """Set up optimizer for policy model.
        
        Returns:
            Optimizer for policy model
        """
        # Create policy optimizer
        optimizer = self._create_optimizer(self.policy_model)
        
        return optimizer
        
    
    def get_logprobs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get log probabilities for the given inputs.
        
        Args:
            model: The model to get outputs from
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Log probabilities tensor of shape [batch_size, seq_len, vocab_size]
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def train(self) -> None:
        """Train the model using GRPO.
        
        GRPO uses a reward-weighted KL divergence as the objective function.
        Higher rewards encourage the policy to move away from the reference model,
        while lower rewards encourage it to stay close.
        """
        logger.info("***** Running GRPO training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  KL coefficient = {self.config.kl_coef}")
        logger.info(f"  Entropy coefficient = {self.config.entropy_coef}")
        
        # Training loop
        for epoch in range(self.config.num_train_epochs):
            self.policy_model.train()
            
            total_loss = 0
            total_kl_loss = 0
            total_entropy = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Get prompt and response data
                prompt_input_ids = batch["prompt_input_ids"]
                prompt_attention_mask = batch["prompt_attention_mask"]
                response_input_ids = batch["response_input_ids"]
                response_attention_mask = batch["response_attention_mask"]
                rewards = batch["reward"]
                
                # Generate outputs with the reference model
                with torch.no_grad():
                    ref_log_probs = self.get_logprobs(
                        self.ref_model,
                        response_input_ids,
                        response_attention_mask,
                    )
                
                # Get policy log probabilities
                policy_log_probs = self.get_logprobs(
                    self.policy_model,
                    response_input_ids,
                    response_attention_mask,
                )
                
                # 1. Compute the log ratio between policy and reference
                log_ratio = policy_log_probs - ref_log_probs
                ratio = torch.exp(log_ratio)
                
                # 2. Apply reward weighting
                # Higher rewards should encourage the policy to move more from the reference
                # Lower or negative rewards should encourage the policy to stay close to the reference
                reward_weights = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                reward_weights = torch.exp(reward_weights)  # Exponentiate to make all weights positive
                
                # 3. Compute the reward-weighted KL divergence
                kl = torch.sum(
                    torch.exp(ref_log_probs) * (ref_log_probs - policy_log_probs),
                    dim=-1,
                )
                weighted_kl = kl * reward_weights
                kl_loss = torch.mean(weighted_kl)
                
                # 4. Compute entropy for exploration
                entropy = -torch.sum(torch.exp(policy_log_probs) * policy_log_probs, dim=-1)
                entropy = torch.mean(entropy)
                
                # 5. Total loss
                loss = kl_loss - self.config.entropy_coef * entropy
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                
                # Clip gradients
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.detach().float()
                total_kl_loss += kl_loss.detach().float()
                total_entropy += entropy.detach().float()
                
                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": loss.item(),
                        "kl": kl_loss.item(),
                        "entropy": entropy.item(),
                    }
                )
            
            # Log epoch metrics
            avg_loss = total_loss / len(self.train_dataloader)
            avg_kl_loss = total_kl_loss / len(self.train_dataloader)
            avg_entropy = total_entropy / len(self.train_dataloader)
            
            metrics = {
                "train/loss": avg_loss.item(),
                "train/kl": avg_kl_loss.item(),
                "train/entropy": avg_entropy.item(),
            }
            
            self._log_metrics(metrics, epoch=epoch + 1)
            
            # Save model periodically
            if (epoch + 1) % 5 == 0 or epoch == self.config.num_train_epochs - 1:
                self.save_model(f"{self.config.output_dir}/checkpoint-{epoch+1}")
        
        # Save final model
        self.save_model(f"{self.config.output_dir}/final")
    
    def save_model(self, output_dir: str) -> None:
        """Save policy model and tokenizer.
        
        Args:
            output_dir: Directory to save to
        """
        # Use the BaseTrainer's implementation
        super().save_model(output_dir)

def main() -> None:
    """Main function to run the training.
    
    This function parses command-line arguments and runs the appropriate
    operation mode (training, LoRA merging, or inference).
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LightLLM")
    
    # Main operation mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "merge_lora", "inference"], 
                       help="Operation mode: train, merge_lora, or run inference")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or identifier from huggingface.co/models")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Path to pre-trained tokenizer or identifier from huggingface.co/models")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default=None, help="Path to training data (JSON file)")
    parser.add_argument("--eval_file", type=str, default=None, help="Path to evaluation data (JSON file)")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--training_method", type=str, default="sft", choices=["sft", "ppo", "grpo"], help="Training method")
    parser.add_argument("--adaptation_method", type=str, default="lora", choices=["lora", "full"], help="Adaptation method")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj", help="Comma-separated list of target modules for LoRA")
    
    # PPO/GRPO arguments
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL divergence coefficient")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_loss_coef", type=float, default=0.1, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--num_ppo_epochs", type=int, default=4, help="Number of PPO epochs")
    
    # Logging and saving arguments
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X updates steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="llm-finetuning", help="Weights & Biases project name")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--use_fsdp", action="store_true", help="Use FSDP (Fully Sharded Data Parallel)")
    
    # LoRA merge arguments
    parser.add_argument("--lora_model_path", type=str, default=None, help="Path to LoRA model for merging")
    parser.add_argument("--merged_model_path", type=str, default=None, help="Path to save merged model")
    
    # Inference arguments
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy generation")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "4bit", "8bit"], help="Quantization method for inference")
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.mode == "train":
        if args.train_file is None:
            raise ValueError("--train_file is required for training mode")
        
        # Convert arguments to config
        config = TrainingConfig(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            train_file=args.train_file,
            eval_file=args.eval_file,
            max_seq_length=args.max_seq_length,
            training_method=args.training_method,
            adaptation_method=args.adaptation_method,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules.split(","),
            kl_coef=args.kl_coef,
            clip_range=args.clip_range,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            num_ppo_epochs=args.num_ppo_epochs,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            seed=args.seed,
            fp16=args.fp16,
            bf16=args.bf16,
            local_rank=args.local_rank,
            use_deepspeed=args.use_deepspeed,
            use_fsdp=args.use_fsdp,
        )
        
        # Create the appropriate trainer based on the training method
        try:
            if config.training_method_enum == TrainingMethod.SFT:
                trainer = SupervisedTrainer(config)
            elif config.training_method_enum == TrainingMethod.PPO:
                trainer = PPOTrainer(config)
            elif config.training_method_enum == TrainingMethod.GRPO:
                trainer = GRPOTrainer(config)
            else:
                raise ValueError(f"Unsupported training method: {config.training_method}")
        except Exception as e:
            logger.error(f"Error creating trainer: {e}")
            raise
        
        # Train the model
        trainer.train()
    
    elif args.mode == "merge_lora":
        # Validate required arguments
        if args.model_name_or_path is None or args.lora_model_path is None or args.merged_model_path is None:
            raise ValueError("--model_name_or_path, --lora_model_path, and --merged_model_path are required for merge_lora mode")
        
        # Merge LoRA weights
        LoRAMerger.merge_lora_weights(
            base_model_path=args.model_name_or_path,
            lora_model_path=args.lora_model_path,
            output_path=args.merged_model_path,
            device="cuda" if args.fp16 or args.bf16 else "cpu",
        )
    
    elif args.mode == "inference":
        # Validate required arguments
        if args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required for inference mode")
        
        if args.prompt is None:
            raise ValueError("--prompt is required for inference mode")
        
        # Initialize optimized inference
        inference = OptimizedInference(
            model_path=args.model_name_or_path,
            tokenizer_path=args.tokenizer_name_or_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="float16" if args.fp16 else "bfloat16" if args.bf16 else "auto",
            quantization=args.quantization,
        )
        
        # Run inference
        output = inference.generate(
            prompts=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
        )
        
        # Print output
        print(f"\nPrompt: {args.prompt}")
        print(f"\nGenerated text: {output[0]}")
        
        # Clean up
        inference.cleanup()
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


class LoRAMerger:
    """Utility for merging LoRA weights into base model.
    
    This class provides functionality to merge LoRA adapter weights
    back into the base model for efficient inference without the
    PEFT library.
    """
    
    @staticmethod
    def merge_lora_weights(
        base_model_path: str,
        lora_model_path: str,
        output_path: str,
        device: str = "auto"
    ) -> PreTrainedModel:
        """Merge LoRA weights into base model for efficient inference.
        
        This method loads a base model and its LoRA adapter weights,
        merges them together, and saves the resulting model to disk.
        
        Args:
            base_model_path: Path to the base model
            lora_model_path: Path to the LoRA adapter weights
            output_path: Path to save the merged model
            device: Device to load models on ('cpu', 'cuda', 'auto')
            
        Returns:
            The merged model
            
        Raises:
            ValueError: If the paths are invalid or models can't be loaded
            RuntimeError: If merging fails
        """
        logger.info(f"Merging LoRA weights from {lora_model_path} into base model {base_model_path}")
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base model
        logger.info(f"Loading base model from {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA model
        logger.info(f"Loading LoRA model from {lora_model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
        )
        
        # Merge weights
        logger.info("Merging weights")
        model = model.merge_and_unload()
        
        # Save merged model
        logger.info(f"Saving merged model to {output_path}")
        model.save_pretrained(output_path)
        
        # Save tokenizer if available
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            tokenizer.save_pretrained(output_path)
            logger.info("Tokenizer saved")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")
        
        logger.info("LoRA weights merged successfully")
        return model


class OptimizedInference:
    """Helper class for efficient inference with language models.
    
    This class provides optimized inference capabilities for language models,
    including:
    - Automatic device placement
    - Mixed precision inference
    - Quantization support
    - Batched generation
    - Streaming generation
    - Memory optimization
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        quantization: Optional[str] = None,
        max_memory: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize optimized inference.
        
        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer (if None, use model_path)
            device: Device to load models on ('cpu', 'cuda', 'auto')
            dtype: Data type for inference ('float32', 'float16', 'bfloat16', 'auto')
            quantization: Quantization method (None, '4bit', '8bit')
            max_memory: Maximum memory allocation per device
            
        Raises:
            ValueError: If the model or tokenizer can't be loaded
            ImportError: If quantization is requested but bitsandbytes is not installed
        """
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine dtype
        if dtype == "auto":
            dtype = "float16" if device == "cuda" else "float32"
        
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype, torch.float16)
        
        # Prepare quantization arguments
        quantization_config = None
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading model from {model_path} (device={device}, dtype={dtype}, quantization={quantization})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else "cpu",
            quantization_config=quantization_config,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        
        # Optimize model for generation
        if hasattr(self.model, "config") and hasattr(self.model.config, "pretraining_tp"):
            self.model.config.use_cache = True
            if self.model.config.pretraining_tp > 1:
                self.model.config.pretraining_tp = 1
        
        # Enable flash attention if available
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            try:
                # Try to enable flash attention for faster inference
                if hasattr(self.model.config, "attn_implementation"):
                    self.model.config.attn_implementation = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled")
            except Exception:
                logger.warning("Failed to enable Flash Attention, continuing without it")
        
        self.device = device
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> Union[List[str], TextIteratorStreamer]:
        """
        Generate text using optimized inference.
        
        Args:
            prompts: A single prompt or list of prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter (0-1)
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition (1.0 = no penalty)
            do_sample: Whether to use sampling (True) or greedy generation (False)
            stream: Whether to stream the output (for UI integration)
            batch_size: Batch size for processing multiple prompts (None = auto)
        
        Returns:
            List of generated texts or a streamer if stream=True
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Set default batch size based on hardware
        if batch_size is None:
            if self.device == "cuda":
                batch_size = min(len(prompts), 4)  # Default batch size for GPU
            else:
                batch_size = 1  # Default batch size for CPU
        
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Setup streamer if needed
        streamer = None
        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Process prompts in batches
        all_outputs = []
        
        # Context for disabling generation optimization like kv-cache for some models
        no_kv_cache = False
        context = nullcontext()
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize inputs
            input_tokens = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            
            # Move to device
            if self.device != "cpu":
                input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
            
            # Generate text
            if stream and len(batch_prompts) == 1:
                # For streaming, we handle it differently
                generation_kwargs = dict(
                    **input_tokens,
                    streamer=streamer,
                    generation_config=generation_config,
                )
                
                thread = threading.Thread(
                    target=self.model.generate,
                    kwargs=generation_kwargs,
                )
                thread.start()
                
                return streamer
            else:
                # For non-streaming or batch generation
                with torch.no_grad(), context:
                    outputs = self.model.generate(
                        **input_tokens,
                        generation_config=generation_config,
                    )
                
                # Decode outputs
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Remove prompt from the outputs
                for j, (prompt, output) in enumerate(zip(batch_prompts, decoded_outputs)):
                    if output.startswith(prompt):
                        decoded_outputs[j] = output[len(prompt):].strip()
                
                all_outputs.extend(decoded_outputs)
        
        return all_outputs
    
    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments to pass to the tokenizer
        
        Returns:
            Dictionary of encoded inputs
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs,
        )
        
        if self.device != "cpu":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        return encoded
    
    def embed_texts(self, texts: List[str], pooling: str = "mean", **kwargs) -> torch.Tensor:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            pooling: Pooling method ('mean', 'max', 'cls')
            **kwargs: Additional arguments to pass to the tokenizer
        
        Returns:
            Tensor of embeddings (batch_size, hidden_size)
        """
        # Encode texts
        encoded = self.batch_encode(texts, **kwargs)
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Apply pooling
        if pooling == "mean":
            # Mean pooling
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        elif pooling == "max":
            # Max pooling
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            hidden_states = hidden_states.masked_fill(attention_mask == 0, -1e9)
            embeddings = torch.max(hidden_states, dim=1)[0]
        elif pooling == "cls":
            # CLS token (first token) pooling
            embeddings = hidden_states[:, 0]
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        return embeddings
    
    def cleanup(self):
        """
        Clean up resources.
        """
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


# Example usage of the LoRA merger and inference

def merge_lora_example():
    """Example for merging LoRA weights."""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_model", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Merge weights
    LoRAMerger.merge_lora_weights(
        args.base_model,
        args.lora_model,
        args.output_path,
        args.device,
    )


def inference_example():
    """Example for optimized inference."""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with an optimized LLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (default: same as model)")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type (float32, float16, bfloat16, auto)")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method (None, 4bit, 8bit)")
    
    args = parser.parse_args()
    
    # Initialize optimized inference
    inference = OptimizedInference(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    
    # Generate text
    output = inference.generate(
        prompts=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    print(f"Input: {args.prompt}")
    print(f"Output: {output[0]}")
    
    # Clean up
    inference.cleanup()


if __name__ == "__main__":
    main()