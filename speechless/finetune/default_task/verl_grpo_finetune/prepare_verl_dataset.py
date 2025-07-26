#!/usr/bin/env python

import json, re, yaml
import os, time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datasets import Dataset, load_dataset
from pathlib import Path
from loguru import logger
import rich

@dataclass
class ProcessorConfig:
    """Base configuration for data processors"""
    name: str
    data_source: str
    dataset_name: str = "main"
    input_key: str = "question"
    output_key: str = "answer"
    ability: str = "math"
    reward_style: str = "rule"
    splits: List[str] = field(default_factory=lambda: ["train", "test"])
    output_format: str = "parquet"
    
    # Processing options
    extract_answer: bool = True
    format_prompt: bool = True
    
    # Custom processing parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir:str, filename:str = None):
        """Save configuration to YAML file"""
        if filename is None:
            filename = self.name
        config_file = Path(output_dir) / f"{filename}.yaml"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert dataclass to dict
        config_dict = asdict(self)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_name: str) -> ProcessorConfig:
        """Load configuration from YAML file"""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        return ProcessorConfig(**config_data)
    
    def save_config(self, config: ProcessorConfig, config_name: str):
        """Save configuration to YAML file"""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        # Convert dataclass to dict
        config_dict = asdict(config)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
    def list_configs(self) -> List[str]:
        """List available configuration files"""
        return [f.stem for f in self.config_dir.glob("*.yaml")]
    

class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
    
    @abstractmethod
    def extract_answer(self, raw_answer: str) -> str:
        """Extract the final answer from raw answer text"""
        pass
    
    @abstractmethod
    def format_prompt(self, question: str) -> str:
        """Format the question into a proper prompt"""
        pass
    
    def process_example(self, example: Dict[str, Any], idx: int, split_name: str) -> Dict[str, Any]:
        """Process a single example into the target format"""
        try:
            # Get raw data
            question_raw = example[self.config.input_key]
            answer_raw = example[self.config.output_key]
            
            # Format prompt if enabled
            if self.config.format_prompt:
                question = self.format_prompt(str(question_raw))
            else:
                question = str(question_raw)
            
            # Extract answer if enabled
            if self.config.extract_answer:
                ground_truth = self.extract_answer(str(answer_raw))
            else:
                ground_truth = str(answer_raw)
            
            # Create structured output
            processed_data = {
                "data_source": f"{self.config.data_source}/{self.config.dataset_name}",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": self.config.ability,
                "reward_model": {
                    "style": self.config.reward_style,
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    'split': split_name,
                    'index': idx,
                    'answer': str(answer_raw),
                    "question": str(question_raw),
                }
            }
            
            return processed_data
            
        except Exception as e:
            raise Exception(f"Failed to process example {idx}: {str(e)}")
    
    def load_dataset(self) -> Dataset:
        """Load the dataset from the configured source"""
        try:
            return load_dataset(self.config.data_source, self.config.dataset_name)
        except Exception as e:
            raise Exception(f"Failed to load dataset {self.config.data_source}: {str(e)}")


class GSM8KProcessor(DataProcessor):
    """Processor for GSM8K mathematics dataset"""
    
    def extract_answer(self, raw_answer: str) -> str:
        """Extract numerical answer from GSM8K solution"""
        pattern = self.config.custom_params.get("answer_pattern", r"#### ([\-]?[0-9\.\,]+)")
        
        match = re.search(pattern, raw_answer)
        if match is None:
            raise Exception(f"No numerical answer found in: {raw_answer[:100]}...")
            
        final_answer = match.group(1).replace(',', '')
        return final_answer
    
    def format_prompt(self, question: str) -> str:
        """Format GSM8K question with thinking instruction"""
        template = self.config.custom_params.get(
            "prompt_template", 
            "{question} Let's think step by step and output the final answer after \"\"\"."
        )
        return template.format(question=question)

class MathProcessor(DataProcessor):
    """Processor for MATH dataset"""
    
    def extract_answer(self, raw_answer: str) -> str:
        """Extract answer from LaTeX boxed format"""
        pattern = self.config.custom_params.get("answer_pattern", r"\\boxed{(.+?)}")
        
        match = re.search(pattern, raw_answer)
        if match is None:
            raise Exception(f"No boxed answer found in: {raw_answer[:100]}...")
            
        return match.group(1)
    
    def format_prompt(self, question: str) -> str:
        """Format MATH problem with solution prompt"""
        template = self.config.custom_params.get(
            "prompt_template",
            "Problem: {question}\nSolution:"
        )
        return template.format(question=question)


class GenericProcessor(DataProcessor):
    """Generic processor that uses regex patterns from config"""
    
    def extract_answer(self, raw_answer: str) -> str:
        """Extract answer using configured regex pattern"""
        pattern = self.config.custom_params.get("answer_pattern")
        
        if not pattern:
            return raw_answer  # No extraction pattern provided
            
        match = re.search(pattern, raw_answer)
        if match is None:
            raise Exception(f"Pattern '{pattern}' not found in: {raw_answer[:100]}...")
            
        return match.group(1) if match.groups() else match.group(0)
    
    def format_prompt(self, question: str) -> str:
        """Format prompt using configured template"""
        template = self.config.custom_params.get("prompt_template", "{question}")
        return template.format(question=question)


class ProcessorFactory:
    """Factory for creating appropriate processors"""
    
    _processors = {
        'gsm8k': GSM8KProcessor,
        'math': MathProcessor,
        'generic': GenericProcessor,
    }
    
    @classmethod
    def create_processor(cls, config: ProcessorConfig) -> DataProcessor:
        """Create a processor based on configuration"""
        processor_type = config.name.lower()
        
        if processor_type in cls._processors:
            return cls._processors[processor_type](config)
        else:
            # Fall back to generic processor for unknown types
            return GenericProcessor(config)
    
    @classmethod
    def register_processor(cls, name: str, processor_class: type):
        """Register a new processor type"""
        cls._processors[name.lower()] = processor_class
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """List available processor types"""
        return list(cls._processors.keys())

class ProcessingPipeline:
    """Main pipeline for data preprocessing with comprehensive error handling"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()

    def save_dataset(self, dataset: Dataset, split_name: str, output_dir: str) -> str:
        """Save dataset with error handling and verification"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{split_name}.parquet"
        
        # # Handle existing files
        # if output_path.exists():
        #     if not self.config.overwrite_existing:
        #         raise FileExistsError(
        #             f"Output file {output_path} exists. Use --overwrite to replace."
        #         )
        #     else:
        #         logger.info(f"Overwriting existing file: {output_path}")
        if output_path.exists():
            logger.info(f"Overwriting existing file: {output_path}")
        
        try:
            # Save with compression for efficiency
            dataset.to_parquet(str(output_path))
            
            # Verify the saved file
            file_size = output_path.stat().st_size
            logger.info(
                f"Saved {split_name}: {output_path} "
                f"({len(dataset)} examples, {file_size:,} bytes)"
            )
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save {split_name} dataset: {e}")
            raise
        
    def process_dataset(self, 
                       config_name: str, 
                       output_dir: str,
                       splits: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main method to process a dataset with comprehensive error handling.
        
        Args:
            config_name: Name of the configuration to use
            output_dir: Directory to save processed data
            splits: List of splits to process (default from config)
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        results = {
            'success': False,
            'config_name': config_name,
            'output_dir': output_dir,
            'processed_splits': {},
            'errors': [],
            'processing_time': 0,
            'total_examples': 0
        }
        
        try:
            # Load configuration
            logger.info(f"Loading configuration: {config_name}")
            config = self.config_manager.load_config(config_name)
            
            # Use splits from parameter or config
            target_splits = splits or config.splits
            logger.info(f"Processing splits: {target_splits}")
            
            # Create processor
            processor = ProcessorFactory.create_processor(config)
            
            # Load dataset
            logger.info(f"Loading dataset: {config.data_source}")
            raw_dataset = processor.load_dataset()

            # Validate splits exist in dataset
            available_splits = list(raw_dataset.keys())
            missing_splits = [s for s in target_splits if s not in available_splits]
            if missing_splits:
                raise ValueError(f"Splits {missing_splits} not found in dataset. "
                               f"Available splits: {available_splits}")
            
            # Process each split
            processed_datasets = {}
            for split in target_splits:
                try:
                    logger.info(f"Processing split: {split}")
                    split_result = self._process_split(
                        processor, raw_dataset[split], split, config_name
                    )
                    processed_datasets[split] = split_result['dataset']
                    results['processed_splits'][split] = split_result['stats']
                    results['total_examples'] += split_result['stats']['num_examples']
                    
                except Exception as e:
                    error_msg = f"Failed to process split '{split}': {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    continue
            
            # Save processed data if not dry run
            if processed_datasets:
                logger.info(f"Saving processed data to: {output_dir}")

                file_sizes = {}
                for split_name, dataset in processed_datasets.items():
                    output_path = self.save_dataset(dataset, split_name, output_dir)
                    file_size = Path(output_path).stat().st_size
                    file_sizes[split_name]=file_size
                
                # Create metadata
                metadata = {
                    'config_name': config_name,
                    'num_examples': {k:len(ds) for k, ds in processed_datasets.items()},
                    'processing_time': time.time() - start_time,
                    **asdict(config)
                }

                metadata_file = f"{output_dir}/dataset_metadata.json"
                json.dump(metadata, open(metadata_file, 'w', encoding='utf-8'), indent=2)

                config.save(output_dir)


            results['success'] = len(processed_datasets) > 0
            results['processing_time'] = time.time() - start_time
            
            if results['success']:
                logger.info(f"Successfully processed {len(processed_datasets)} splits "
                               f"with {results['total_examples']} total examples "
                               f"in {results['processing_time']:.2f} seconds")
            else:
                logger.warning("No splits were successfully processed")
                
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['processing_time'] = time.time() - start_time
            
        return results
    
    def _process_split(self, processor, dataset, split_name: str, config_name: str) -> Dict[str, Any]:
        """Process a single dataset split with error tracking"""
        processed_examples = []
        errors = []
        
        for idx, example in enumerate(dataset):
            try:
                processed_example = processor.process_example(example, idx, split_name)
                processed_examples.append(processed_example)
            except Exception as e:
                error_msg = f"Example {idx}: {str(e)}"
                errors.append(error_msg)
                if len(errors) <= 5:  # Log first 5 errors
                    logger.warning(f"Processing error in {split_name} {error_msg}")
        
        if errors:
            logger.warning(f"Split '{split_name}' had {len(errors)} processing errors")
        
        # Convert back to dataset format
        processed_dataset = Dataset.from_list(processed_examples)
        
        return {
            'dataset': processed_dataset,
            'stats': {
                'num_examples': len(processed_examples),
                'num_errors': len(errors),
                'error_rate': len(errors) / len(dataset) if len(dataset) > 0 else 0,
                'errors': errors[:10]  # Keep first 10 errors for debugging
            }
        }
    
    def list_available_configs(self) -> List[str]:
        """List all available configurations"""
        return self.config_manager.list_configs()
    
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_name", type=str, required=True, help="Name of the configuration to use")
    parser.add_argument("--configs_dir", type=str, default=None, help="Directory containing configuration files")
    parser.add_argument("--data_source", type=str, help="Dataset name or path")
    parser.add_argument("--dataset_name", type=str, default="main", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save processed data")
                    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config_name = args.config_name
    configs_dir = args.configs_dir
    output_dir = args.output_dir or f"verl_data_processed/{config_name}"


    # Initialize pipeline
    if configs_dir is None:
        configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        print(f"Configs directory: {configs_dir}")

    config_manager = ConfigManager(configs_dir)
    pipeline = ProcessingPipeline(config_manager)

    # List available configurations
    configs = pipeline.list_available_configs()
    print(f"Available configs: {configs}")

    result = pipeline.process_dataset(
        config_name=config_name,
        output_dir=output_dir
    )
    
    print(f"Processing result: {result}")

if __name__ == "__main__":
    main()