"""
Improved configuration management system for data preprocessing.
Uses YAML files for better maintainability and validation.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


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
        config_dict = {
            'name': config.name,
            'data_source': config.data_source,
            'dataset_name': config.dataset_name,
            'input_key': config.input_key,
            'output_key': config.output_key,
            'ability': config.ability,
            'reward_style': config.reward_style,
            'splits': config.splits,
            'output_format': config.output_format,
            'extract_answer': config.extract_answer,
            'format_prompt': config.format_prompt,
            'custom_params': config.custom_params
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
    def list_configs(self) -> List[str]:
        """List available configuration files"""
        return [f.stem for f in self.config_dir.glob("*.yaml")]
    
    def validate_config(self, config: ProcessorConfig) -> bool:
        """Validate configuration parameters"""
        required_fields = ['name', 'data_source', 'input_key', 'output_key']
        
        for field in required_fields:
            if not getattr(config, field):
                raise ValueError(f"Required field '{field}' is missing or empty")
                
        if config.output_format not in ['parquet', 'jsonl', 'json']:
            raise ValueError(f"Unsupported output format: {config.output_format}")
            
        return True


# Create default configurations
def create_default_configs():
    """Create default configuration files for common datasets"""
    manager = ConfigManager()
    
    # GSM8K configuration
    gsm8k_config = ProcessorConfig(
        name="gsm8k",
        data_source="data/openai/gsm8k",
        dataset_name="main",
        input_key="question",
        output_key="answer",
        ability="math",
        reward_style="rule",
        custom_params={
            "answer_pattern": r"#### ([\-]?[0-9\.\,]+)",
            "prompt_template": "{question} Let's think step by step and output the final answer after \"\"\"."
        }
    )
    
    manager.save_config(gsm8k_config, "gsm8k")
    
    # MATH dataset configuration
    math_config = ProcessorConfig(
        name="math",
        data_source="data/hendrycks/math",
        dataset_name="all",
        input_key="problem",
        output_key="solution",
        ability="math",
        reward_style="rule",
        custom_params={
            "answer_pattern": r"\\boxed{(.+?)}",
            "prompt_template": "Problem: {problem}\nSolution:"
        }
    )
    
    manager.save_config(math_config, "math")
    
    return manager


if __name__ == "__main__":
    # Create example configurations
    create_default_configs()
    print("Default configurations created successfully!")