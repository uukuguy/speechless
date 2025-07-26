"""
Modular data preprocessing architecture with clear separation of concerns.
"""

import re
import os
import datasets
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from config import ProcessorConfig


class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.validate_config()
    
    def validate_config(self):
        """Validate processor-specific configuration"""
        required_keys = [self.config.input_key, self.config.output_key]
        # This would be expanded in real implementation
        pass
    
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
            raise ProcessingError(f"Failed to process example {idx}: {str(e)}")
    
    def load_dataset(self) -> datasets.Dataset:
        """Load the dataset from the configured source"""
        try:
            return datasets.load_dataset(self.config.data_source, self.config.dataset_name)
        except Exception as e:
            raise DataLoadError(f"Failed to load dataset {self.config.data_source}: {str(e)}")


class GSM8KProcessor(DataProcessor):
    """Processor for GSM8K mathematics dataset"""
    
    def extract_answer(self, raw_answer: str) -> str:
        """Extract numerical answer from GSM8K solution"""
        pattern = self.config.custom_params.get("answer_pattern", r"#### ([\-]?[0-9\.\,]+)")
        
        match = re.search(pattern, raw_answer)
        if match is None:
            raise AnswerExtractionError(f"No numerical answer found in: {raw_answer[:100]}...")
            
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
            raise AnswerExtractionError(f"No boxed answer found in: {raw_answer[:100]}...")
            
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
            raise AnswerExtractionError(f"Pattern '{pattern}' not found in: {raw_answer[:100]}...")
            
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


# Custom exceptions for better error handling
class ProcessingError(Exception):
    """Base exception for data processing errors"""
    pass


class DataLoadError(ProcessingError):
    """Exception for dataset loading errors"""
    pass


class AnswerExtractionError(ProcessingError):
    """Exception for answer extraction failures"""
    pass


class ConfigurationError(ProcessingError):
    """Exception for configuration validation errors"""
    pass