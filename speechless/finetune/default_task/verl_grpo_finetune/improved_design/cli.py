"""
Command line interface for the improved data processing system.
"""

import argparse
import sys
import json
from typing import List, Optional
from pathlib import Path

from config import ConfigManager, create_default_configs
from pipeline import ProcessingPipeline, BatchProcessor


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced data preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process GSM8K dataset
  python cli.py process --config gsm8k --output data/processed/gsm8k
  
  # Create default configurations
  python cli.py init-configs
  
  # Validate a configuration
  python cli.py validate --config gsm8k
  
  # List available configurations
  python cli.py list-configs
  
  # Batch process multiple datasets
  python cli.py batch --configs gsm8k math --output data/processed
  
  # Dry run (validate without writing)
  python cli.py process --config gsm8k --output data/processed/gsm8k --dry-run
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a dataset')
    process_parser.add_argument('--config', required=True, help='Configuration name')
    process_parser.add_argument('--output', required=True, help='Output directory')
    process_parser.add_argument('--splits', nargs='+', help='Specific splits to process')
    process_parser.add_argument('--dry-run', action='store_true', help='Validate without writing output')
    process_parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    
    # Batch process command
    batch_parser = subparsers.add_parser('batch', help='Process multiple datasets')
    batch_parser.add_argument('--configs', nargs='+', required=True, help='Configuration names')
    batch_parser.add_argument('--output', required=True, help='Base output directory')
    batch_parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    batch_parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration')
    validate_parser.add_argument('--config', required=True, help='Configuration name')
    validate_parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    
    # List configurations command
    list_parser = subparsers.add_parser('list-configs', help='List available configurations')
    list_parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show configuration details')
    info_parser.add_argument('--config', required=True, help='Configuration name')
    info_parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    
    # Init configurations command
    init_parser = subparsers.add_parser('init-configs', help='Create default configurations')
    init_parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    init_parser.add_argument('--force', action='store_true', help='Overwrite existing configurations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'process':
            handle_process_command(args)
        elif args.command == 'batch':
            handle_batch_command(args)
        elif args.command == 'validate':
            handle_validate_command(args)
        elif args.command == 'list-configs':
            handle_list_configs_command(args)
        elif args.command == 'info':
            handle_info_command(args)
        elif args.command == 'init-configs':
            handle_init_configs_command(args)
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def handle_process_command(args):
    """Handle the process command"""
    config_manager = ConfigManager(args.config_dir)
    pipeline = ProcessingPipeline(config_manager)
    
    print(f"Processing dataset with config: {args.config}")
    result = pipeline.process_dataset(
        config_name=args.config,
        output_dir=args.output,
        splits=args.splits,
        dry_run=args.dry_run
    )
    
    if result['success']:
        print(f"✅ Successfully processed {result['total_examples']} examples")
        print(f"   Processing time: {result['processing_time']:.2f} seconds")
        if not args.dry_run:
            print(f"   Output saved to: {args.output}")
        else:
            print("   (Dry run - no output written)")
    else:
        print("❌ Processing failed")
        for error in result['errors']:
            print(f"   Error: {error}")
        sys.exit(1)


def handle_batch_command(args):
    """Handle the batch command"""
    config_manager = ConfigManager(args.config_dir)
    pipeline = ProcessingPipeline(config_manager)
    batch_processor = BatchProcessor(pipeline)
    
    print(f"Batch processing {len(args.configs)} configurations...")
    results = batch_processor.process_multiple_configs(
        config_names=args.configs,
        base_output_dir=args.output,
        fail_fast=args.fail_fast
    )
    
    print(f"\n📊 Batch Results:")
    print(f"   Total configs: {results['total_configs']}")
    print(f"   Successful: {results['successful_configs']}")
    print(f"   Failed: {results['failed_configs']}")
    
    for config_name, result in results['batch_results'].items():
        status = "✅" if result.get('success', False) else "❌"
        print(f"   {status} {config_name}")
        if not result.get('success', False) and 'error' in result:
            print(f"      Error: {result['error']}")


def handle_validate_command(args):
    """Handle the validate command"""
    config_manager = ConfigManager(args.config_dir)
    pipeline = ProcessingPipeline(config_manager)
    
    print(f"Validating configuration: {args.config}")
    result = pipeline.validate_config(args.config)
    
    if result['valid']:
        print("✅ Configuration is valid")
        print(f"   Processor type: {result['processor_type']}")
    else:
        print("❌ Configuration validation failed")
        print(f"   Error: {result['message']}")
        sys.exit(1)


def handle_list_configs_command(args):
    """Handle the list-configs command"""
    config_manager = ConfigManager(args.config_dir)
    pipeline = ProcessingPipeline(config_manager)
    
    configs = pipeline.list_available_configs()
    
    if not configs:
        print("No configurations found.")
        print(f"Run 'python cli.py init-configs --config-dir {args.config_dir}' to create default configurations.")
        return
    
    print(f"📋 Available configurations ({len(configs)}):")
    
    if args.detailed:
        for config_name in configs:
            info = pipeline.get_config_info(config_name)
            if 'error' in info:
                print(f"   ❌ {config_name}: {info['error']}")
            else:
                status = "✅" if info['valid'] else "❌"
                print(f"   {status} {config_name}")
                print(f"      Data source: {info['data_source']}")
                print(f"      Processor: {info['processor_type']}")
                print(f"      Splits: {info['splits']}")
                print(f"      Format: {info['output_format']}")
    else:
        for config_name in configs:
            print(f"   • {config_name}")


def handle_info_command(args):
    """Handle the info command"""
    config_manager = ConfigManager(args.config_dir)
    pipeline = ProcessingPipeline(config_manager)
    
    info = pipeline.get_config_info(args.config)
    
    if 'error' in info:
        print(f"❌ Error getting config info: {info['error']}")
        sys.exit(1)
    
    print(f"📋 Configuration: {args.config}")
    print(f"   Name: {info['name']}")
    print(f"   Data source: {info['data_source']}")
    print(f"   Dataset name: {info['dataset_name']}")
    print(f"   Processor type: {info['processor_type']}")
    print(f"   Splits: {info['splits']}")
    print(f"   Output format: {info['output_format']}")
    print(f"   Valid: {'✅' if info['valid'] else '❌'}")
    
    if info['custom_params']:
        print(f"   Custom parameters:")
        for key, value in info['custom_params'].items():
            print(f"      {key}: {value}")


def handle_init_configs_command(args):
    """Handle the init-configs command"""
    config_dir = Path(args.config_dir)
    
    if config_dir.exists() and any(config_dir.glob("*.yaml")) and not args.force:
        print(f"Configuration directory {config_dir} already contains YAML files.")
        print("Use --force to overwrite existing configurations.")
        sys.exit(1)
    
    # Temporarily change the ConfigManager directory
    import config
    original_dir = config.ConfigManager.__init__.__defaults__[0] if config.ConfigManager.__init__.__defaults__ else "configs"
    
    # Create manager with specified directory
    manager = config.ConfigManager(args.config_dir)
    
    # Create default configs
    create_default_configs()
    
    print(f"✅ Default configurations created in: {config_dir}")
    configs = manager.list_configs()
    for config_name in configs:
        print(f"   • {config_name}.yaml")


if __name__ == "__main__":
    main()