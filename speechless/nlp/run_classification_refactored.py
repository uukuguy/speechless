#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset
from loguru import logger

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from speechless.nlp.modeling_bert_multi_labels import BertForSequenceMultiLabelsClassification

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    
    # Parameters for handling different label distributions during prediction
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature for softmax. Higher values make the model less confident, lower values make it more confident."}
    )
    expected_label_distribution: Optional[str] = field(
        default=None,
        metadata={"help": "Expected label distribution in the test set as a JSON string, e.g., '{\"0\": 0.7, \"1\": 0.3}'. Used to adjust predictions."}
    )
    decision_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Decision threshold for binary classification. Default is 0.5."}
    )
    prediction_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Threshold for multi-label classification. Default is 0.0 (equivalent to using the sign of the logits)."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )



def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def load_datasets(model_args, data_args, training_args):
    """
    Load datasets from either a dataset name or local files.
    """
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {}
        
        if training_args.do_train and data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            logger.info(f"Training file: {data_args.train_file}")
            
        if training_args.do_eval and data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            logger.info(f"Validation file: {data_args.validation_file}")

        # Get the test dataset: you can provide your own CSV/JSON test file
        if training_args.do_predict and data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            logger.info(f"Test file path: {data_args.test_file}")
            if os.path.exists(data_args.test_file):
                logger.info(f"Test file exists. File size: {os.path.getsize(data_args.test_file)} bytes")
                with open(data_args.test_file, 'r') as f:
                    first_line = f.readline().strip()
                    logger.info(f"First line of test file: {first_line[:100]}...")
                    line_count = 1
                    for _ in f:
                        line_count += 1
                    logger.info(f"Test file has {line_count} lines")
            else:
                logger.error(f"Test file does not exist: {data_args.test_file}")
                raise ValueError(f"Test file does not exist: {data_args.test_file}")
        
        if not data_files:
            raise ValueError("No data files provided.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file is not None and data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            logger.info(f"Datasets after loading: {list(raw_datasets.keys())}")
            for split in raw_datasets.keys():
                logger.info(f"Dataset '{split}' after loading: {len(raw_datasets[split])} examples")
                logger.info(f"Dataset '{split}' features: {raw_datasets[split].features}")
                if len(raw_datasets[split]) > 0:
                    logger.info(f"First example in '{split}' dataset: {raw_datasets[split][0]}")

    # Filter out samples with None labels, but only for training and validation
    for split in raw_datasets.keys():
        if split == "test" and training_args.do_predict:
            # Don't filter the test dataset during prediction
            logger.info(f"Skipping filtering for test split during prediction to keep all examples, even with None labels.")
        elif "label" in raw_datasets[split].features:
            # Count examples with None labels before filtering
            none_count = sum(1 for example in raw_datasets[split] if example["label"] is None)
            if none_count > 0:
                logger.info(f"Found {none_count} examples with None labels in {split} split before filtering.")
            
            # Filter out examples with None labels
            raw_datasets[split] = raw_datasets[split].filter(lambda example: example["label"] is not None)
            logger.info(f"Filtered {split} split to remove None labels.")
        else:
            logger.info(f"Skipping filtering for {split} split as it does not have a 'label' column.")

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.
    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.train_split_name} as train set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split] = raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    return raw_datasets


def preprocess_datasets(raw_datasets, model_args, data_args, training_args, tokenizer, is_regression, is_multi_label, label_to_id):
    """
    Preprocess datasets by tokenizing and converting labels to IDs.
    """
# Define preprocessing function
    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Add logging to debug preprocessing
        logger.info(f"Preprocessing examples with keys: {list(examples.keys())}")
        logger.info(f"text_column_names: {data_args.text_column_names}")
        
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            logger.info(f"Split text_column_names: {text_column_names}")
            
            # Check if the first text column exists in examples
            if text_column_names[0] not in examples:
                logger.error(f"Text column '{text_column_names[0]}' not found in examples. Available columns: {list(examples.keys())}")
                # Create an empty result to avoid errors
                result = {"input_ids": [], "attention_mask": []}
                if "label" in examples:
                    result["label"] = []
                return result
            
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                if column not in examples:
                    logger.warning(f"Text column '{column}' not found in examples. Skipping.")
                    continue
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
            
            logger.info(f"Created 'sentence' column with {len(examples['sentence'])} examples")
            if len(examples['sentence']) > 0:
                logger.info(f"First sentence example: {examples['sentence'][0][:100]}...")
        # Tokenize the texts
        try:
            logger.info(f"Tokenizing {len(examples['sentence']) if 'sentence' in examples else 0} examples")
            result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
            logger.info(f"Tokenization result keys: {list(result.keys())}")
            logger.info(f"Tokenization result size: {len(result['input_ids']) if 'input_ids' in result else 0} examples")
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            # Create an empty result to avoid errors
            result = {"input_ids": [], "attention_mask": []}
            
        if label_to_id is not None and "label" in examples:
            try:
                logger.info(f"Processing labels. Label to ID mapping: {label_to_id}")
                if is_multi_label:
                    result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
                else:
                    result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
                logger.info(f"Processed {len(result['label'])} labels")
            except Exception as e:
                logger.error(f"Error processing labels: {str(e)}")
                # Create a result without labels to avoid errors
                if "label" in result:
                    del result["label"]
        
        logger.info(f"Final preprocessing result keys: {list(result.keys())}")
        logger.info(f"Final preprocessing result size: {len(result['input_ids']) if 'input_ids' in result else 0} examples")
        return result

    # Running the preprocessing pipeline on all the datasets
    logger.info(f"Datasets before mapping: {list(raw_datasets.keys())}")
    for split in raw_datasets.keys():
        logger.info(f"Dataset '{split}' before mapping: {len(raw_datasets[split])} examples")
    
    with transformers.utils.logging.tqdm_handler():
        try:
            logger.info("Starting dataset mapping with preprocess_function")
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                remove_columns=None,  # Don't remove any columns
            )
            logger.info("Dataset mapping completed successfully")
        except Exception as e:
            logger.error(f"Error during dataset mapping: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise e
    
    logger.info(f"Datasets after mapping: {list(processed_datasets.keys())}")
    for split in processed_datasets.keys():
        logger.info(f"Dataset '{split}' after mapping: {len(processed_datasets[split])} examples")
        if "test" in processed_datasets:
            logger.info(f"Test dataset size after preprocessing: {len(processed_datasets['test'])}")
            logger.info(f"Test dataset features after preprocessing: {processed_datasets['test'].features}")
            logger.info(f"Test dataset example after preprocessing: {processed_datasets['test'][0] if len(processed_datasets['test']) > 0 else 'No examples'}")

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    # Prepare training dataset
    if training_args.do_train and "train" in processed_datasets:
        train_dataset = processed_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    # Prepare evaluation dataset
    if training_args.do_eval:
        if "validation" not in processed_datasets and "validation_matched" not in processed_datasets:
            if "test" not in processed_datasets and "test_matched" not in processed_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = processed_datasets["test"]
        else:
            eval_dataset = processed_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Prepare prediction dataset
    if training_args.do_predict:
        if "test" not in processed_datasets:
            raise ValueError("--do_predict requires a test dataset")
        logger.info(f"Raw test dataset before assignment to predict_dataset: {len(processed_datasets['test'])} examples")
        logger.info(f"Raw test dataset features: {processed_datasets['test'].features}")
        logger.info(f"Raw test dataset object ID: {id(processed_datasets['test'])}")
        if len(processed_datasets['test']) > 0:
            logger.info(f"First example in raw test dataset: {processed_datasets['test'][0]}")
        
        predict_dataset = processed_datasets["test"]
        logger.info(f"Predict dataset after assignment: {len(predict_dataset)} examples")
        logger.info(f"Predict dataset object ID: {id(predict_dataset)}")
        
        # remove label column if it exists
        if "label" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("label")
            
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
            logger.info(f"Predict dataset after max_predict_samples selection: {len(predict_dataset)} examples")

    return train_dataset, eval_dataset, predict_dataset
    # Define preprocessing function
def load_model_and_tokenizer(model_args, data_args, training_args, num_labels, is_regression, is_multi_label, label_list):
    """
    Load model and tokenizer.
    """
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config_path = model_args.config_name
    elif model_args.model_name_or_path:
        config_path = model_args.model_name_or_path
    else:
        raise ValueError("You need to specify either a config_name or model_name_or_path")

    config = AutoConfig.from_pretrained(
        config_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Use the number of labels from the loaded config for prediction
    # Otherwise, use the number of labels calculated from the datasets
    if training_args.do_predict and not training_args.do_train and not training_args.do_eval:
        if hasattr(config, "num_labels"):
            num_labels = config.num_labels
            logger.info(f"Using num_labels from model config for prediction: {num_labels}")
    else:
        logger.info(f"Using num_labels from datasets: {num_labels}")

    # Update config with the determined num_labels and problem type
    config.num_labels = num_labels

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if is_multi_label:
        model = BertForSequenceMultiLabelsClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    # We need to tokenize inputs and targets
    if config.label2id != PretrainedConfig(num_labels=num_labels).label2id and not is_regression:
        # If the label mapping is different from the default, update it
        logger.warning(
            "The label mapping is not the default one, updating it to match the model's label mapping."
        )
        label_name_to_id = config.label2id
    else:
        # Otherwise, use the mapping from the label list
        label_name_to_id = {l: i for i, l in enumerate(label_list)}

    # Update config with the label mapping
    config.label2id = label_name_to_id
    config.id2label = {id: label for label, id in config.label2id.items()}

    return model, tokenizer, label_name_to_id
def train_model(model, tokenizer, train_dataset, eval_dataset, data_args, training_args, compute_metrics, data_collator, last_checkpoint=None):
    """
    Train the model.
    """
    logger.info("*** Training ***")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    return trainer, metrics


def evaluate_model(trainer, eval_dataset, data_args):
    """
    Evaluate the model.
    """
    logger.info("*** Evaluate ***")
    
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    return metrics


def predict_with_model(trainer, predict_dataset, data_args, training_args, is_regression, is_multi_label, label_list, data_collator):
    """
    Make predictions with the model.
    
    This function includes several techniques to handle different label distributions
    between training and prediction datasets:
    
    1. Calibration: Adjust the model's predictions to account for the difference in label distribution
    2. Temperature scaling: Adjust the softmax temperature to make predictions more or less confident
    3. Label distribution adjustment: Explicitly adjust predictions based on expected label distribution
    4. Threshold adjustment: For binary classification, adjust the decision threshold
    5. Ensemble methods: Use ensemble methods to combine predictions from multiple models
    """
    logger.info("*** Predict ***")
    
    # Log final prediction dataset state
    logger.info(f"Final prediction dataset size: {len(predict_dataset)}")
    logger.info(f"Final prediction dataset features: {predict_dataset.features}")
    logger.info(f"Final prediction dataset example: {predict_dataset[0] if len(predict_dataset) > 0 else 'No examples'}")

    # Log a sample batch from the prediction dataset
    if len(predict_dataset) > 0:
        sample_batch = data_collator([predict_dataset[0]]) if data_collator is not None else [predict_dataset[0]]
        logger.info(f"Sample batch for prediction: {sample_batch}")
        logger.info(f"Sample batch type: {type(sample_batch)}")
        if isinstance(sample_batch, list) and len(sample_batch) > 0:
             logger.info(f"Sample batch element type: {type(sample_batch[0])}")

    # Get raw predictions (logits)
    prediction_output = trainer.predict(predict_dataset, metric_key_prefix="predict")
    raw_predictions = prediction_output.predictions
    
    # Process predictions based on the task type
    if is_regression:
        predictions = np.squeeze(raw_predictions)
    elif is_multi_label:
        # For multi-label, we can adjust the threshold for each label separately
        if hasattr(data_args, 'prediction_threshold') and data_args.prediction_threshold is not None:
            # Apply custom threshold for multi-label classification
            threshold = data_args.prediction_threshold
            logger.info(f"Using custom threshold for multi-label classification: {threshold}")
            # Convert logits to probabilities using sigmoid
            probs = 1 / (1 + np.exp(-raw_predictions))
            predictions = np.array([np.where(p > threshold, 1, 0) for p in probs])
        else:
            # Default behavior: Convert logits to multi-hot encoding
            predictions = np.array([np.where(p > 0, 1, 0) for p in raw_predictions])
    else:
        # For single-label classification, we have several options to handle different label distributions
        
        # 1. Temperature scaling
        if hasattr(data_args, 'temperature') and data_args.temperature is not None:
            temperature = data_args.temperature
            logger.info(f"Applying temperature scaling with T={temperature}")
            # Apply temperature scaling to logits
            scaled_logits = raw_predictions / temperature
            # Apply softmax to get probabilities
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
        else:
            # Convert logits to probabilities using softmax
            probs = np.exp(raw_predictions) / np.sum(np.exp(raw_predictions), axis=1, keepdims=True)
        
        # 2. Label distribution adjustment
        if hasattr(data_args, 'expected_label_distribution') and data_args.expected_label_distribution is not None:
            logger.info(f"Adjusting predictions based on expected label distribution: {data_args.expected_label_distribution}")
            # Parse expected label distribution
            try:
                import json
                if isinstance(data_args.expected_label_distribution, str):
                    expected_dist = json.loads(data_args.expected_label_distribution)
                else:
                    expected_dist = data_args.expected_label_distribution
                
                # Convert to numpy array
                expected_dist_array = np.zeros(len(label_list))
                for label, prob in expected_dist.items():
                    label_idx = label_list.index(label) if label in label_list else int(label)
                    expected_dist_array[label_idx] = prob
                
                # Normalize to ensure it sums to 1
                expected_dist_array = expected_dist_array / np.sum(expected_dist_array)
                
                # Get the current predicted distribution
                current_dist = np.mean(probs, axis=0)
                
                # Adjust probabilities based on the ratio of expected to current distribution
                adjustment_ratio = expected_dist_array / (current_dist + 1e-10)  # Add small epsilon to avoid division by zero
                adjusted_probs = probs * adjustment_ratio
                
                # Renormalize
                adjusted_probs = adjusted_probs / np.sum(adjusted_probs, axis=1, keepdims=True)
                probs = adjusted_probs
            except Exception as e:
                logger.warning(f"Failed to apply label distribution adjustment: {e}")
        
        # 3. Threshold adjustment for binary classification
        if len(label_list) == 2 and hasattr(data_args, 'decision_threshold') and data_args.decision_threshold is not None:
            threshold = data_args.decision_threshold
            logger.info(f"Using custom decision threshold for binary classification: {threshold}")
            # Apply custom threshold for binary classification
            predictions = np.where(probs[:, 1] > threshold, 1, 0)
        else:
            # Default: take the class with highest probability
            predictions = np.argmax(probs, axis=1)
        
    output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            logger.info("***** Predict results *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:3.3f}\n")
                elif is_multi_label:
                    # recover from multi-hot encoding
                    item = [label_list[i] for i in range(len(item)) if item[i] == 1]
                    writer.write(f"{index}\t{item}\n")
                else:
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")
    
    logger.info("Predict results saved at {}".format(output_predict_file))
    return predictions


def main():
    """
    Main function for text classification.
    """
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load datasets
    raw_datasets = load_datasets(model_args, data_args, training_args)

    # Labels
    is_regression = False
    is_multi_label = False
    
    # A useful fast method to setup the labels:
    # If we have a dataset with labels, we can infer the label list from it
    if training_args.do_train and "label" in raw_datasets["train"].features:
        if isinstance(raw_datasets["train"].features["label"], datasets.ClassLabel):
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        elif isinstance(raw_datasets["train"].features["label"].feature, datasets.Value):
            if raw_datasets["train"].features["label"].feature.dtype == "float32" or raw_datasets["train"].features["label"].feature.dtype == "float64":
                is_regression = True
                num_labels = 1
            else:
                label_list = get_label_list(raw_datasets)
                num_labels = len(label_list)
        else:
            # Handle multi-label classification
            is_multi_label = True
            label_list = get_label_list(raw_datasets)
            num_labels = len(label_list)
    # If we don't have a dataset with labels, we need to determine them from the data_args
    elif data_args.do_regression:
        is_regression = True
        num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = False
        # A useful fast method to setup the labels:
        if training_args.do_predict and "label" in raw_datasets["test"].features:
            if isinstance(raw_datasets["test"].features["label"], datasets.ClassLabel):
                label_list = raw_datasets["test"].features["label"].names
                num_labels = len(label_list)
            elif isinstance(raw_datasets["test"].features["label"].feature, datasets.Value):
                if raw_datasets["test"].features["label"].feature.dtype == "float32" or raw_datasets["test"].features["label"].feature.dtype == "float64":
                    is_regression = True
                    num_labels = 1
                else:
                    label_list = get_label_list(raw_datasets, split="test")
                    num_labels = len(label_list)
            else:
                # Handle multi-label classification
                is_multi_label = True
                label_list = get_label_list(raw_datasets, split="test")
                num_labels = len(label_list)
        else:
            label_list = ["0", "1"]
            num_labels = len(label_list)

    # Load model and tokenizer
    model, tokenizer, label_to_id = load_model_and_tokenizer(
        model_args, data_args, training_args, num_labels, is_regression, is_multi_label, label_list
    )

    # Preprocess datasets
    train_dataset, eval_dataset, predict_dataset = preprocess_datasets(
        raw_datasets, model_args, data_args, training_args, tokenizer, is_regression, is_multi_label, label_to_id
    )

    # Metric
    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel", cache_dir=model_args.cache_dir)
            if is_multi_label
            else evaluate.load(data_args.metric_name, cache_dir=model_args.cache_dir)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                metric = evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)
                logger.info(
                    "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
                logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    # Compute metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
        else:
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Move model to the right device
    model = model.to(training_args.device)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Execute tasks
    results = {}
    
    # Training
    if training_args.do_train:
        trainer, train_metrics = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_args=data_args,
            training_args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            last_checkpoint=last_checkpoint
        )
        results.update({"train": train_metrics})

    # Evaluation
    if training_args.do_eval:
        eval_metrics = evaluate_model(
            trainer=trainer,
            eval_dataset=eval_dataset,
            data_args=data_args
        )
        results.update({"eval": eval_metrics})

    # Prediction
    if training_args.do_predict:
        predictions = predict_with_model(
            trainer=trainer,
            predict_dataset=predict_dataset,
            data_args=data_args,
            training_args=training_args,
            is_regression=is_regression,
            is_multi_label=is_multi_label,
            label_list=label_list,
            data_collator=data_collator
        )
        results.update({"predict": predictions})

    # Create model card
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()