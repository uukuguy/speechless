# llm_pretraining.py
"""
Framework for pretraining large language models.
Supports efficient distributed training with model and data parallelism.
"""

import os
import math
import json
import time
import logging
import itertools
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    SchedulerType,
    get_scheduler,
)

import datasets
from datasets import load_dataset, concatenate_datasets

import wandb
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import deepspeed and accelerate if available
try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    logger.warning("DeepSpeed not available, some features will be disabled")
    DEEPSPEED_AVAILABLE = False

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedType
    from accelerate.state import AcceleratorState
    ACCELERATE_AVAILABLE = True
except ImportError:
    logger.warning("Accelerate not available, some features will be disabled")
    ACCELERATE_AVAILABLE = False


@dataclass
class PretrainingConfig:
    """Configuration for pretraining a language model."""
    
    # Model configuration
    model_name_or_path: Optional[str] = None  # For starting from a pretrained model
    model_type: str = "gpt2"  # Model architecture if starting from scratch
    config_name: Optional[str] = None  # Model config if starting from scratch
    tokenizer_name: Optional[str] = None  # Tokenizer to use
    vocab_size: int = 50257  # Vocabulary size if building a new tokenizer
    hidden_size: int = 768  # Hidden size if building a new model
    num_hidden_layers: int = 12  # Number of layers if building a new model
    num_attention_heads: int = 12  # Number of attention heads if building a new model
    intermediate_size: int = 3072  # FFN intermediate size if building a new model
    max_position_embeddings: int = 2048  # Max sequence length if building a new model
    
    # Dataset and processing
    dataset_name: Optional[str] = None  # HuggingFace dataset name
    dataset_config_name: Optional[str] = None  # HuggingFace dataset config
    train_file: Optional[str] = None  # Path to training file
    validation_file: Optional[str] = None  # Path to validation file
    dataset_paths: List[str] = field(default_factory=list)  # List of paths to datasets
    data_cache_dir: Optional[str] = None  # Directory to cache datasets
    max_seq_length: int = 1024  # Maximum sequence length to use for training
    preprocessing_num_workers: Optional[int] = None  # Number of workers for preprocessing
    
    # Training parameters
    per_device_train_batch_size: int = 8  # Batch size per device for training
    per_device_eval_batch_size: int = 8  # Batch size per device for evaluation
    learning_rate: float = 5e-5  # Learning rate
    weight_decay: float = 0.0  # Weight decay
    adam_beta1: float = 0.9  # Adam beta1
    adam_beta2: float = 0.999  # Adam beta2
    adam_epsilon: float = 1e-8  # Adam epsilon
    max_grad_norm: float = 1.0  # Maximum gradient norm for gradient clipping
    num_train_epochs: int = 3  # Number of training epochs
    max_steps: int = -1  # Maximum number of training steps
    lr_scheduler_type: Union[SchedulerType, str] = "linear"  # Learning rate scheduler type
    warmup_ratio: float = 0.1  # Ratio of steps for warmup
    warmup_steps: int = 0  # Number of steps for warmup
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    
    # Distributed training
    local_rank: int = -1  # Local rank for distributed training
    use_deepspeed: bool = False  # Whether to use DeepSpeed
    deepspeed_config: Optional[str] = None  # Path to DeepSpeed config file
    use_fsdp: bool = False  # Whether to use FSDP
    use_accelerate: bool = True  # Whether to use Accelerate
    mixed_precision: Optional[str] = None  # Mixed precision mode
    
    # System and optimization
    seed: int = 42  # Random seed
    fp16: bool = False  # Whether to use FP16 precision
    bf16: bool = False  # Whether to use BF16 precision
    tf32: bool = False  # Whether to use TF32 precision
    
    # Logging and checkpointing
    output_dir: str = "./output"  # Output directory
    logging_dir: Optional[str] = None  # Directory for logs
    logging_strategy: str = "steps"  # Logging strategy
    logging_steps: int = 500  # Logging steps
    save_strategy: str = "steps"  # Saving strategy
    save_steps: int = 500  # Saving steps
    save_total_limit: Optional[int] = None  # Maximum number of checkpoints to keep
    resume_from_checkpoint: Optional[str] = None  # Resume from checkpoint
    
    # Evaluation
    do_eval: bool = False  # Whether to evaluate during training
    eval_strategy: str = "steps"  # Evaluation strategy
    eval_steps: int = 500  # Evaluation steps
    
    # Monitoring
    use_wandb: bool = False  # Whether to use Weights & Biases
    wandb_project: Optional[str] = None  # Weights & Biases project name
    wandb_entity: Optional[str] = None  # Weights & Biases entity name
    
    # Data streaming
    streaming: bool = False  # Whether to stream data from disk
    
    def __post_init__(self):
        """Validate and set defaults for config."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set logging directory if not specified
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
            os.makedirs(self.logging_dir, exist_ok=True)
        
        # Validate mixed precision options
        if self.mixed_precision not in [None, "no", "fp16", "bf16"]:
            raise ValueError(f"Invalid mixed precision mode: {self.mixed_precision}")
        
        # Set fp16/bf16 if mixed precision is specified
        if self.mixed_precision == "fp16":
            self.fp16 = True
        elif self.mixed_precision == "bf16":
            self.bf16 = True
        
        # Validate gradient accumulation steps
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps: {self.gradient_accumulation_steps}, should be >= 1")
        
        # Validate DeepSpeed config
        if self.use_deepspeed:
            if not DEEPSPEED_AVAILABLE:
                raise ImportError("DeepSpeed is not available. Please install it first.")
            if self.deepspeed_config is None:
                raise ValueError("DeepSpeed config file must be specified when using DeepSpeed")
            if not os.path.exists(self.deepspeed_config):
                raise FileNotFoundError(f"DeepSpeed config file not found: {self.deepspeed_config}")
        
        # Validate FSDP
        if self.use_fsdp and self.use_deepspeed:
            raise ValueError("Cannot use both FSDP and DeepSpeed simultaneously")


class TokenizerBuilder:
    """Utility for building or loading tokenizers."""
    
    @staticmethod
    def build_tokenizer(config: PretrainingConfig):
        """Build or load a tokenizer based on the config."""
        # If tokenizer_name is specified, load it
        if config.tokenizer_name:
            logger.info(f"Loading tokenizer: {config.tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name,
                cache_dir=config.data_cache_dir,
                use_fast=True,
            )
        # If model_name_or_path is specified, load tokenizer from it
        elif config.model_name_or_path:
            logger.info(f"Loading tokenizer from model: {config.model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                cache_dir=config.data_cache_dir,
                use_fast=True,
            )
        # Otherwise, raise an error
        else:
            raise ValueError(
                "Either tokenizer_name or model_name_or_path must be specified"
            )
        
        # Ensure the tokenizer has padding and eos tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer


class ModelBuilder:
    """Utility for building or loading models."""
    
    @staticmethod
    def build_model(config: PretrainingConfig):
        """Build or load a model based on the config."""
        # If model_name_or_path is specified, load it
        if config.model_name_or_path:
            logger.info(f"Loading model: {config.model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                cache_dir=config.data_cache_dir,
                torch_dtype=torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else None,
            )
        # If config_name is specified, load it
        elif config.config_name:
            logger.info(f"Loading model config: {config.config_name}")
            model_config = AutoConfig.from_pretrained(
                config.config_name,
                cache_dir=config.data_cache_dir,
            )
            logger.info(f"Building new model from config")
            model = AutoModelForCausalLM.from_config(model_config)
        # Otherwise, create a new config and model
        else:
            logger.info(f"Building new model from scratch with type: {config.model_type}")
            model_config = AutoConfig.from_pretrained(
                config.model_type,
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                cache_dir=config.data_cache_dir,
            )
            model = AutoModelForCausalLM.from_config(model_config)
        
        # Set up Pytorch 2.0 compile if available
        if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
            logger.info("Using torch.compile")
            model = torch.compile(model)
        
        # Configure tensor precision
        if config.tf32 and torch.cuda.is_available():
            logger.info("Using TF32 precision")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        return model


class PretrainingDatasetProcessor:
    """Processor for pretraining datasets."""
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: PretrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def load_and_prepare_datasets(self):
        """Load and prepare datasets for pretraining."""
        # Load raw datasets
        raw_datasets = self._load_raw_datasets()
        
        # Preprocess datasets
        train_dataset = self._preprocess_dataset(raw_datasets["train"])
        
        eval_dataset = None
        if self.config.do_eval and "validation" in raw_datasets:
            eval_dataset = self._preprocess_dataset(raw_datasets["validation"])
        
        return train_dataset, eval_dataset
    
    def _load_raw_datasets(self):
        """Load raw datasets from HuggingFace or local files."""
        # If dataset_name is specified, load it from HuggingFace
        if self.config.dataset_name:
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            raw_datasets = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config_name,
                cache_dir=self.config.data_cache_dir,
                streaming=self.config.streaming,
            )
        # If train_file is specified, load it
        elif self.config.train_file:
            data_files = {}
            extension = self.config.train_file.split(".")[-1]
            if self.config.train_file:
                data_files["train"] = self.config.train_file
            if self.config.validation_file:
                data_files["validation"] = self.config.validation_file
            
            logger.info(f"Loading dataset from files: {data_files}")
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.config.data_cache_dir,
                streaming=self.config.streaming,
            )
        # If dataset_paths are specified, load them
        elif self.config.dataset_paths:
            datasets_list = []
            for path in self.config.dataset_paths:
                logger.info(f"Loading dataset from path: {path}")
                # Determine if it's a directory of files or a single file
                if os.path.isdir(path):
                    # Get all files in the directory
                    files = [os.path.join(path, f) for f in os.listdir(path)]
                else:
                    files = [path]
                
                # Load each file
                for file_path in files:
                    extension = file_path.split(".")[-1]
                    dataset = load_dataset(
                        extension,
                        data_files=file_path,
                        split="train",
                        cache_dir=self.config.data_cache_dir,
                        streaming=self.config.streaming,
                    )
                    datasets_list.append(dataset)
            
            # Combine datasets
            if self.config.streaming:
                raw_datasets = {"train": datasets.interleave_datasets(datasets_list)}
            else:
                combined_dataset = concatenate_datasets(datasets_list)
                # Split into train and validation
                if self.config.do_eval:
                    split = combined_dataset.train_test_split(test_size=0.05, seed=self.config.seed)
                    raw_datasets = {"train": split["train"], "validation": split["test"]}
                else:
                    raw_datasets = {"train": combined_dataset}
        else:
            raise ValueError(
                "Either dataset_name, train_file, or dataset_paths must be specified"
            )
        
        return raw_datasets
    
    def _preprocess_dataset(self, dataset):
        """Preprocess a dataset for pretraining."""
        # Define preprocessing function
        def preprocess_function(examples):
            # Tokenize texts
            text_column_name = "text" if "text" in examples else list(examples.keys())[0]
            texts = examples[text_column_name]
            
            # Tokenize with padding and truncation
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
                return_special_tokens_mask=True,
            )
            
            return tokenized
        
        # Apply preprocessing
        if self.config.streaming:
            # For streaming datasets, map on-the-fly
            processed_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
        else:
            # For non-streaming datasets, process all at once
            column_names = dataset.column_names
            processed_dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.config.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.config.streaming,
                desc="Tokenizing dataset",
            )
        
        return processed_dataset


class PretrainingDataCollator:
    """Data collator for pretraining language models."""
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm: bool = False, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack inputs
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        
        # Build causal language modeling inputs
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
        
        # If using masked language modeling, mask some tokens
        if self.mlm:
            batch["input_ids"], batch["labels"] = self._mask_tokens(
                batch["input_ids"], batch["attention_mask"]
            )
        
        return batch
    
    def _mask_tokens(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask tokens for masked language modeling."""
        labels = inputs.clone()
        
        # Sample a few tokens in each sequence for MLM training
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ],
            dtype=torch.bool,
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # Rest of the time, keep the original word
        
        # Set labels for non-masked tokens to -100 to ignore them in the loss
        labels[~masked_indices] = -100
        
        return inputs, labels


class DeepSpeedTrainer:
    """Trainer for language models using DeepSpeed."""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        
        # Set up tokenizer
        self.tokenizer = TokenizerBuilder.build_tokenizer(config)
        
        # Set up model
        self.model = ModelBuilder.build_model(config)
        
        # Set up data processor
        self.data_processor = PretrainingDatasetProcessor(self.tokenizer, config)
        
        # Set up data collator
        self.data_collator = PretrainingDataCollator(self.tokenizer)
        
        # Set up WandB if enabled
        if config.use_wandb:
            if config.wandb_project is None:
                raise ValueError("wandb_project must be specified when use_wandb is True")
            
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=vars(config),
            )
    
    def _prepare_deepspeed(self):
        """Prepare DeepSpeed for training."""
        # Initialize process group
        deepspeed.init_distributed()
        
        # Load DeepSpeed config
        with open(self.config.deepspeed_config, "r") as f:
            ds_config = json.load(f)
        
        # Update DeepSpeed config with training parameters
        ds_config.update({
            "train_batch_size": self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps * dist.get_world_size(),
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
        })
        
        # Convert model to DeepSpeed
        engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config,
        )
        
        return engine, optimizer
    
    def train(self):
        """Train the model using DeepSpeed."""
        # Prepare datasets
        train_dataset, eval_dataset = self.data_processor.load_and_prepare_datasets()
        
        # Prepare dataloaders
        if isinstance(train_dataset, IterableDataset):
            # For streaming datasets, no sampler
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.config.preprocessing_num_workers,
            )
        else:
            # For regular datasets, use DistributedSampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                seed=self.config.seed,
            )
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                num_workers=self.config.preprocessing_num_workers,
            )
        
        # Prepare eval dataloader if needed
        eval_dataloader = None
        if self.config.do_eval and eval_dataset is not None:
            if isinstance(eval_dataset, IterableDataset):
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.per_device_eval_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.config.preprocessing_num_workers,
                )
            else:
                eval_sampler = DistributedSampler(
                    eval_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    seed=self.config.seed,
                )
                
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.per_device_eval_batch_size,
                    sampler=eval_sampler,
                    collate_fn=self.data_collator,
                    num_workers=self.config.preprocessing_num_workers,
                )
        
        # Prepare DeepSpeed
        engine, _ = self._prepare_deepspeed()
        
        # Calculate number of training steps
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
        else:
            if isinstance(train_dataset, IterableDataset):
                # For streaming datasets, set a large number of steps
                max_steps = self.config.num_train_epochs * 10000
            else:
                max_steps = math.ceil(
                    len(train_dataloader) / self.config.gradient_accumulation_steps * self.config.num_train_epochs
                )
        
        # Calculate warmup steps
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = math.ceil(max_steps * self.config.warmup_ratio)
        
        # Get scheduler
        scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=engine.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        
        # Log training info
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset) if not isinstance(train_dataset, IterableDataset) else 'Unknown (streaming)'}")
        logger.info(f"  Num epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size per device = {self.config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Warmup steps = {warmup_steps}")
        
        # Training loop
        global_step = 0
        total_loss = 0.0
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            _, client_state = engine.load_checkpoint(self.config.resume_from_checkpoint)
            if client_state:
                global_step = client_state["global_step"]
                logger.info(f"Resuming from step {global_step}")
        
        # Progress bar
        progress_bar = tqdm(range(global_step, max_steps), disable=not dist.get_rank() == 0)
        
        # Training loop
        for epoch in range(self.config.num_train_epochs):
            engine.train()
            
            # Set epoch for sampler
            if isinstance(train_dataset, IterableDataset):
                # For streaming datasets, no need to set epoch
                pass
            else:
                train_dataloader.sampler.set_epoch(epoch)
            
            for step, batch in enumerate(train_dataloader):
                # Skip steps if resuming
                if global_step < step:
                    continue
                
                # Forward pass
                outputs = engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                loss = outputs.loss
                
                # Backward pass
                engine.backward(loss)
                
                # Update weights
                engine.step()
                
                # Update progress
                global_step += 1
                total_loss += loss.item()
                
                # Update progress bar
                if global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    progress_bar.set_description(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}")
                    
                    # Log to WandB
                    if self.config.use_wandb and dist.get_rank() == 0:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch + (step + 1) / len(train_dataloader),
                        })
                    
                    total_loss = 0.0
                
                progress_bar.update(1)
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    client_state = {"global_step": global_step}
                    checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
                    engine.save_checkpoint(checkpoint_path, client_state=client_state)
                    
                    # Save tokenizer
                    if dist.get_rank() == 0:
                        self.tokenizer.save_pretrained(checkpoint_path)
                    
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Evaluate
                if self.config.do_eval and eval_dataloader is not None and global_step % self.config.eval_steps == 0:
                    eval_results = self._evaluate(engine, eval_dataloader)
                    
                    # Log to WandB
                    if self.config.use_wandb and dist.get_rank() == 0:
                        wandb.log({"eval/loss": eval_results["loss"]})
                    
                    # Log to console
                    logger.info(f"Eval Loss: {eval_results['loss']:.4f}")
                    logger.info(f"Eval Perplexity: {math.exp(eval_results['loss']):.4f}")
                
                # Check if we've reached max steps
                if global_step >= max_steps:
                    break
            
            # Check if we've reached max steps after an epoch
            if global_step >= max_steps:
                break
        
        # Save final model
        client_state = {"global_step": global_step}
        final_checkpoint_path = os.path.join(self.config.output_dir, "final_checkpoint")
        engine.save_checkpoint(final_checkpoint_path, client_state=client_state)
        
        # Save tokenizer
        if dist.get_rank() == 0:
            self.tokenizer.save_pretrained(final_checkpoint_path)
        
        logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        
        return global_step
    
    def _evaluate(self, engine, eval_dataloader):
        """Evaluate the model."""
        engine.eval()
        
        eval_loss = 0.0
        num_eval_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                loss = outputs.loss
                eval_loss += loss.item()
                num_eval_steps += 1
        
        # Gather losses from all processes
        eval_loss_tensor = torch.tensor(eval_loss, device=engine.device)
        num_steps_tensor = torch.tensor(num_eval_steps, device=engine.device)
        
        dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_steps_tensor, op=dist.ReduceOp.SUM)
        
        eval_loss = eval_loss_tensor.item() / num_steps_tensor.item()
        
        results = {"loss": eval_loss}
        
        return results


class AccelerateTrainer:
    """Trainer for language models using Accelerate."""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        
        # Set up accelerator
        kwargs = {}
        if config.mixed_precision is not None:
            kwargs["mixed_precision"] = config.mixed_precision
        
        self.accelerator = Accelerator(**kwargs)
        
        # Set random seed
        self.accelerator.set_seed(config.seed)
        
        # Set up tokenizer
        self.tokenizer = TokenizerBuilder.build_tokenizer(config)
        
        # Set up model
        self.model = ModelBuilder.build_model(config)
        
        # Set up data processor
        self.data_processor = PretrainingDatasetProcessor(self.tokenizer, config)
        
        # Set up data collator
        self.data_collator = PretrainingDataCollator(self.tokenizer)
        
        # Set up WandB if enabled
        if config.use_wandb and self.accelerator.is_local_main_process:
            if config.wandb_project is None:
                raise ValueError("wandb_project must be specified when use_wandb is True")
            
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=vars(config),
            )
    
    def train(self):
        """Train the model using Accelerate."""
        # Prepare datasets
        train_dataset, eval_dataset = self.data_processor.load_and_prepare_datasets()
        
        # Prepare dataloaders
        if isinstance(train_dataset, IterableDataset):
            # For streaming datasets, no sampler
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.config.preprocessing_num_workers,
            )
        else:
            # For regular datasets, use default sampler (will be wrapped by Accelerate)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                num_workers=self.config.preprocessing_num_workers,
            )
        
        # Prepare eval dataloader if needed
        eval_dataloader = None
        if self.config.do_eval and eval_dataset is not None:
            if isinstance(eval_dataset, IterableDataset):
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.per_device_eval_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.config.preprocessing_num_workers,
                )
            else:
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.per_device_eval_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.config.preprocessing_num_workers,
                )
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        # Calculate number of training steps
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
        else:
            if isinstance(train_dataset, IterableDataset):
                # For streaming datasets, set a large number of steps
                max_steps = self.config.num_train_epochs * 10000
            else:
                max_steps = math.ceil(
                    len(train_dataloader) / self.config.gradient_accumulation_steps * self.config.num_train_epochs
                )
        
        # Calculate warmup steps
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = math.ceil(max_steps * self.config.warmup_ratio)
        
        # Get scheduler
        scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        
        # Prepare model, optimizer, dataloader, and scheduler with accelerate
        self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Log training info
        self.accelerator.print("***** Running training *****")
        self.accelerator.print(f"  Num examples = {len(train_dataset) if not isinstance(train_dataset, IterableDataset) else 'Unknown (streaming)'}")
        self.accelerator.print(f"  Num epochs = {self.config.num_train_epochs}")
        self.accelerator.print(f"  Batch size per device = {self.config.per_device_train_batch_size}")
        self.accelerator.print(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        self.accelerator.print(f"  Total optimization steps = {max_steps}")
        self.accelerator.print(f"  Warmup steps = {warmup_steps}")
        
        # Training loop
        global_step = 0
        total_loss = 0.0
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            if os.path.isdir(self.config.resume_from_checkpoint):
                accelerator_checkpoint = os.path.join(self.config.resume_from_checkpoint, "accelerate_checkpoint")
                if os.path.exists(accelerator_checkpoint):
                    self.accelerator.load_state(accelerator_checkpoint)
                    global_step = int(self.config.resume_from_checkpoint.split("-")[-1])
                    self.accelerator.print(f"Resuming from step {global_step}")
        
        # Progress bar
        progress_bar = tqdm(range(global_step, max_steps), disable=not self.accelerator.is_local_main_process)
        
        # Training loop
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            
            for step, batch in enumerate(train_dataloader):
                # Skip steps if resuming
                if global_step < step:
                    continue
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # Clip gradients
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update progress
                    global_step += 1
                    
                    # Update progress bar
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = self.accelerator.gather(loss).mean().item() * self.config.gradient_accumulation_steps
                        progress_bar.set_description(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}")
                        
                        # Log to WandB
                        if self.config.use_wandb and self.accelerator.is_local_main_process:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/epoch": epoch + (step + 1) / len(train_dataloader),
                                "train/step": global_step,
                            })
                    
                    progress_bar.update(1)
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
                        self.accelerator.save_state(os.path.join(checkpoint_path, "accelerate_checkpoint"))
                        
                        # Save model and tokenizer
                        if self.accelerator.is_local_main_process:
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            unwrapped_model.save_pretrained(
                                checkpoint_path,
                                is_main_process=self.accelerator.is_local_main_process,
                                save_function=self.accelerator.save,
                            )
                            self.tokenizer.save_pretrained(checkpoint_path)
                        
                        self.accelerator.print(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Evaluate
                    if self.config.do_eval and eval_dataloader is not None and global_step % self.config.eval_steps == 0:
                        eval_results = self._evaluate(eval_dataloader)
                        
                        # Log to WandB
                        if self.config.use_wandb and self.accelerator.is_local_main_process:
                            wandb.log({
                                "eval/loss": eval_results["loss"],
                                "eval/perplexity": math.exp(eval_results["loss"]),
                                "eval/step": global_step,
                            })
                        
                        # Log to console
                        self.accelerator.print(f"Eval Loss: {eval_results['loss']:.4f}")
                        self.accelerator.print(f"Eval Perplexity: {math.exp(eval_results['loss']):.4f}")
                    
                    # Check if we've reached max steps
                    if global_step >= max_steps:
                        break
                
                # Update total loss
                total_loss += loss.detach().float()
            
            # Check if we've reached max steps after an epoch
            if global_step >= max_steps:
                break
        
        # Save final model
        final_checkpoint_path = os.path.join(self.config.output_dir, "final_checkpoint")
        self.accelerator.save_state(os.path.join(final_checkpoint_path, "accelerate_checkpoint"))
        
        # Save model and tokenizer
        if self.accelerator.is_local_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                final_checkpoint_path,
                is_main_process=self.accelerator.is_local_main_process,
                save_function=self.accelerator.save,
            )
            self.tokenizer.save_pretrained(final_checkpoint_path)
        
        self.accelerator.print(f"Saved final checkpoint to {final_checkpoint_path}")
        
        return global_step
    
    def _evaluate(self, eval_dataloader):
        """Evaluate the model."""
        self.model.eval()
        
        losses = []
        
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            
            loss = outputs.loss
            losses.append(self.accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].shape[0])))
        
        losses = torch.cat(losses)
        mean_loss = torch.mean(losses)
        
        return {"loss": mean_loss.item()}


def main():
    """Run pretraining with the specified config."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pretrain a language model")
    
    # Config file argument
    parser.add_argument("--config_file", type=str, help="Path to config file")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, default="gpt2", help="Model type if starting from scratch")
    parser.add_argument("--config_name", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset to use (via the datasets library)")
    parser.add_argument("--dataset_config_name", type=str, help="The configuration name of the dataset to use")
    parser.add_argument("--train_file", type=str, help="The input training data file")
    parser.add_argument("--validation_file", type=str, help="An optional input evaluation data file")
    parser.add_argument("--dataset_paths", type=str, nargs="+", help="Paths to datasets")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="The maximum total input sequence length after tokenization")
    parser.add_argument("--preprocessing_num_workers", type=int, help="The number of processes to use for preprocessing")
    parser.add_argument("--data_cache_dir", type=str, help="Directory to cache datasets")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    
    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay amount")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of training steps (overrides num_train_epochs)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before performing a backward/update pass")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps for a linear warmup")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of steps for a linear warmup")
    
    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--use_deepspeed", action="store_true", help="Whether to use DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, help="Path to DeepSpeed config file")
    parser.add_argument("--use_fsdp", action="store_true", help="Whether to use FSDP")
    parser.add_argument("--mixed_precision", type=str, help="Mixed precision mode")
    
    # Logging and checkpointing arguments
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=500, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving checkpoints")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluations")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to a checkpoint to resume training from")
    
    # Evaluation arguments
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    
    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, help="Weights & Biases entity name")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16 precision")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bf16 precision")
    parser.add_argument("--tf32", action="store_true", help="Whether to enable tf32 precision")
    
    args = parser.parse_args()
    
    # If config file is specified, load it
    config = None
    if args.config_file:
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
        
        # Convert to PretrainingConfig
        config = PretrainingConfig(**config_dict)
    else:
        # Convert args to dictionary
        config_dict = {k: v for k, v in vars(args).items() if v is not None}
        
        # Convert to PretrainingConfig
        config = PretrainingConfig(**config_dict)
    
    # Set up trainer
    if config.use_deepspeed:
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not available. Please install it first.")
        
        trainer = DeepSpeedTrainer(config)
    else:
        if not ACCELERATE_AVAILABLE:
            raise ImportError("Accelerate is not available. Please install it first.")
        
        trainer = AccelerateTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()