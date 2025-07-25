"""
Generic data preparation for GSM8K and similar math datasets to parquet format.
Maintains full backward compatibility while adding configurability.

Usage:
    # Original usage (GSM8K)
    python generic_data_prepare.py --local_dir data/verl/gsm8k
    
    # Custom datasets (future extension)
    python generic_data_prepare.py --local_dir data/verl/custom --config math_config
"""

import os
import datasets
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset preprocessing"""
    data_source: str = 'data/openai/gsm8k'
    dataset_name: str = 'main'
    input_key: str = 'question'
    output_key: str = 'answer'
    ability: str = 'math'
    reward_style: str = 'rule'
    extract_answer_fn: Optional[Callable[[str], str]] = None
    prompt_format_fn: Optional[Callable[[str], str]] = None


def preprocess_verl_dataset(config: DatasetConfig,
                            output_dir: str,
                            splits: List[str] = ['train', 'test']) -> Dict[str, Any]:
    """
    Generic dataset preprocessing for math problem datasets.
    
    Args:
        config: Dataset configuration with source and field mapping
        output_dir: Directory to save processed parquet files
        splits: Dataset splits to process (default: ['train', 'test'])
        
    Returns:
        Dict mapping split names to processed datasets
    """
    dataset = datasets.load_dataset(config.data_source, config.dataset_name)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    def process_fn(example, idx, split_name: str):
        """Process a single example"""
        # Use get to handle potential missing keys gracefully
        question_raw = example[config.input_key]

        # Use prompt format function if provided, otherwise use the raw question
        if config.prompt_format_fn is not None:
            question = config.prompt_format_fn(str(question_raw))
        else:
            question = str(question_raw)

        answer_raw = example[config.output_key]

        if config.extract_answer_fn is not None:
            extract_fn = config.extract_answer_fn
            ground_truth = extract_fn(str(answer_raw))
        else:
            ground_truth = str(answer_raw)

        data = {
            "data_source": f"{config.data_source}/{config.dataset_name}",
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": config.ability,
            "reward_model": {
                "style": config.reward_style,
                "ground_truth": ground_truth
            },
            "extra_info": {
                'split': split_name,
                'index': idx,
                'answer': str(answer_raw),
                "question": str(question_raw),
            }
        }
        return data

    processed_datasets = {}

    for split in splits:
        if split in dataset:
            processed_dataset = dataset[split].map(
                function=lambda ex, idx: process_fn(ex, idx, split), with_indices=True
            )
            processed_datasets[split] = processed_dataset

            # Save to parquet
            output_path = os.path.join(output_dir, f'{split}.parquet')
            processed_dataset.to_parquet(output_path)
            print(f"Saved {split} dataset to {output_path} ({len(processed_dataset)} examples)")

    return processed_datasets


def generic_main(config: DatasetConfig, custom_args: Optional[Dict[str, Any]] = None):
    """
    Generic main function for dataset preprocessing.
    
    Args:
        config: Dataset configuration
        custom_args: Optional custom command line arguments
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/processed', help='Output directory for processed datasets')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'], help='Dataset splits to process')

    if custom_args:
        for arg_name, arg_config in custom_args.items():
            if arg_name != 'local_dir':  # Don't duplicate local_dir
                parser.add_argument(f'--{arg_name}', **arg_config)
            else:
                # Override the default local_dir if provided in custom_args
                parser.add_argument('--local_dir', **arg_config)

    args = parser.parse_args()

    # Create a copy of config and override with any provided arguments
    import copy
    final_config = copy.deepcopy(config)

    # Override config fields if corresponding arguments are provided and not None
    if hasattr(args, 'ability') and args.ability is not None:
        final_config.ability = args.ability

    processed = preprocess_verl_dataset(config=final_config, output_dir=args.local_dir, splits=args.splits)

    print(f"Successfully processed datasets: {list(processed.keys())}")
    return processed
