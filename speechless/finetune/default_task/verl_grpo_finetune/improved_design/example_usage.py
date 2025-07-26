#!/usr/bin/env python3
"""
Example usage of the improved data processing pipeline.
"""

from config import ConfigManager, ProcessorConfig
from pipeline import ProcessingPipeline, BatchProcessor
from processors import ProcessorFactory


def example_single_processing():
    """Example of processing a single dataset"""
    print("=== Single Dataset Processing Example ===")
    
    # Initialize pipeline
    config_manager = ConfigManager("configs")
    pipeline = ProcessingPipeline(config_manager)
    
    # List available configurations
    configs = pipeline.list_available_configs()
    print(f"Available configs: {configs}")
    
    if "gsm8k" in configs:
        # Validate configuration
        validation = pipeline.validate_config("gsm8k")
        print(f"GSM8K config validation: {validation['valid']}")
        
        # Get config information
        info = pipeline.get_config_info("gsm8k")
        print(f"Config info: {info}")
        
        # Process dataset (dry run)
        print("Processing GSM8K dataset (dry run)...")
        result = pipeline.process_dataset(
            config_name="gsm8k",
            output_dir="data/processed/gsm8k",
            dry_run=True
        )
        
        print(f"Processing result: {result}")
    else:
        print("GSM8K config not found. Run: python config.py to create default configs.")


def example_batch_processing():
    """Example of batch processing multiple datasets"""
    print("\n=== Batch Processing Example ===")
    
    # Initialize batch processor
    pipeline = ProcessingPipeline()
    batch_processor = BatchProcessor(pipeline)
    
    # Available configs to process
    config_names = ["gsm8k", "math"]
    
    # Process multiple configs
    print(f"Batch processing configs: {config_names}")
    results = batch_processor.process_multiple_configs(
        config_names=config_names,
        base_output_dir="data/processed",
        fail_fast=False
    )
    
    print(f"Batch results: {results}")


def example_custom_processor():
    """Example of creating a custom processor"""
    print("\n=== Custom Processor Example ===")
    
    # Create custom configuration
    custom_config = ProcessorConfig(
        name="custom",
        data_source="my/custom/dataset",
        input_key="text",
        output_key="label",
        ability="classification",
        custom_params={
            "prompt_template": "Classify this text: {text}"
        }
    )
    
    # Create processor
    processor = ProcessorFactory.create_processor(custom_config)
    print(f"Created processor: {type(processor).__name__}")
    
    # Test example processing (mock data)
    example = {
        "text": "This is a sample text",
        "label": "positive"
    }
    
    try:
        # This would normally process the example
        print(f"Would process example: {example}")
        print("Note: This is just a demonstration - actual processing requires a real dataset")
    except Exception as e:
        print(f"Expected error (no real dataset): {e}")


def main():
    """Run all examples"""
    try:
        example_single_processing()
        example_batch_processing()
        example_custom_processor()
        
        print("\n=== Examples completed successfully! ===")
        print("\nTo run the actual CLI:")
        print("1. python cli.py init-configs  # Create default configurations")
        print("2. python cli.py list-configs  # List available configurations")
        print("3. python cli.py validate --config gsm8k  # Validate a configuration")
        print("4. python cli.py process --config gsm8k --output data/processed --dry-run")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all required files are in place and dependencies are installed")


if __name__ == "__main__":
    main()