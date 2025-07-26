"""
Main data processing pipeline with error handling and validation.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from config import ConfigManager, ProcessorConfig
from processors import ProcessorFactory, ProcessingError, DataLoadError
from output_manager import OutputManager


class ProcessingPipeline:
    """Main pipeline for data preprocessing with comprehensive error handling"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        logger = logging.getLogger('data_processing')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def process_dataset(self, 
                       config_name: str, 
                       output_dir: str,
                       splits: Optional[List[str]] = None,
                       dry_run: bool = False) -> Dict[str, Any]:
        """
        Main method to process a dataset with comprehensive error handling.
        
        Args:
            config_name: Name of the configuration to use
            output_dir: Directory to save processed data
            splits: List of splits to process (default from config)
            dry_run: If True, validate without writing outputs
            
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
            # Load and validate configuration
            self.logger.info(f"Loading configuration: {config_name}")
            config = self.config_manager.load_config(config_name)
            self.config_manager.validate_config(config)
            
            # Use splits from parameter or config
            target_splits = splits or config.splits
            self.logger.info(f"Processing splits: {target_splits}")
            
            # Create processor
            processor = ProcessorFactory.create_processor(config)
            
            # Load dataset
            self.logger.info(f"Loading dataset: {config.data_source}")
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
                    self.logger.info(f"Processing split: {split}")
                    split_result = self._process_split(
                        processor, raw_dataset[split], split, config_name
                    )
                    processed_datasets[split] = split_result['dataset']
                    results['processed_splits'][split] = split_result['stats']
                    results['total_examples'] += split_result['stats']['num_examples']
                    
                except Exception as e:
                    error_msg = f"Failed to process split '{split}': {str(e)}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    continue
            
            # Save processed data if not dry run
            if not dry_run and processed_datasets:
                self.logger.info(f"Saving processed data to: {output_dir}")
                output_manager = OutputManager(output_dir, config.output_format)
                
                # Create metadata
                config_metadata = {
                    'config_name': config_name,
                    'data_source': config.data_source,
                    'dataset_name': config.dataset_name,
                    'processor_type': config.name,
                    'processing_time': time.time() - start_time
                }
                
                output_manager.write_splits(processed_datasets, config_metadata)
            
            results['success'] = len(processed_datasets) > 0
            results['processing_time'] = time.time() - start_time
            
            if results['success']:
                self.logger.info(f"Successfully processed {len(processed_datasets)} splits "
                               f"with {results['total_examples']} total examples "
                               f"in {results['processing_time']:.2f} seconds")
            else:
                self.logger.warning("No splits were successfully processed")
                
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
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
            except ProcessingError as e:
                error_msg = f"Example {idx}: {str(e)}"
                errors.append(error_msg)
                if len(errors) <= 5:  # Log first 5 errors
                    self.logger.warning(f"Processing error in {split_name} {error_msg}")
        
        if errors:
            self.logger.warning(f"Split '{split_name}' had {len(errors)} processing errors")
        
        # Convert back to dataset format
        import datasets
        processed_dataset = datasets.Dataset.from_list(processed_examples)
        
        return {
            'dataset': processed_dataset,
            'stats': {
                'num_examples': len(processed_examples),
                'num_errors': len(errors),
                'error_rate': len(errors) / len(dataset) if len(dataset) > 0 else 0,
                'errors': errors[:10]  # Keep first 10 errors for debugging
            }
        }
    
    def validate_config(self, config_name: str) -> Dict[str, Any]:
        """Validate a configuration without processing data"""
        try:
            config = self.config_manager.load_config(config_name)
            self.config_manager.validate_config(config)
            
            # Try to create processor
            processor = ProcessorFactory.create_processor(config)
            
            return {
                'valid': True,
                'config': config,
                'processor_type': type(processor).__name__,
                'message': 'Configuration is valid'
            }
        except Exception as e:
            return {
                'valid': False,
                'config': None,
                'processor_type': None,
                'message': f'Configuration validation failed: {str(e)}'
            }
    
    def list_available_configs(self) -> List[str]:
        """List all available configurations"""
        return self.config_manager.list_configs()
    
    def get_config_info(self, config_name: str) -> Dict[str, Any]:
        """Get detailed information about a configuration"""
        try:
            config = self.config_manager.load_config(config_name)
            validation_result = self.validate_config(config_name)
            
            return {
                'name': config.name,
                'data_source': config.data_source,
                'dataset_name': config.dataset_name,
                'splits': config.splits,
                'output_format': config.output_format,
                'processor_type': validation_result['processor_type'],
                'valid': validation_result['valid'],
                'custom_params': config.custom_params
            }
        except Exception as e:
            return {
                'error': f"Failed to load config info: {str(e)}"
            }


class BatchProcessor:
    """Process multiple datasets in batch"""
    
    def __init__(self, pipeline: Optional[ProcessingPipeline] = None):
        self.pipeline = pipeline or ProcessingPipeline()
    
    def process_multiple_configs(self, 
                                config_names: List[str],
                                base_output_dir: str,
                                fail_fast: bool = False) -> Dict[str, Any]:
        """Process multiple configurations"""
        results = {}
        base_path = Path(base_output_dir)
        
        for config_name in config_names:
            try:
                output_dir = base_path / config_name
                result = self.pipeline.process_dataset(config_name, str(output_dir))
                results[config_name] = result
                
                if not result['success'] and fail_fast:
                    break
                    
            except Exception as e:
                results[config_name] = {
                    'success': False,
                    'error': f"Batch processing failed: {str(e)}"
                }
                
                if fail_fast:
                    break
        
        return {
            'batch_results': results,
            'total_configs': len(config_names),
            'successful_configs': sum(1 for r in results.values() if r.get('success', False)),
            'failed_configs': sum(1 for r in results.values() if not r.get('success', False))
        }

