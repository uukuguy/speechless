"""
Output management for different data formats with consistent interface.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import datasets


class OutputWriter(ABC):
    """Abstract base class for output writers"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    @abstractmethod
    def write_dataset(self, dataset: datasets.Dataset, split_name: str, metadata: Dict[str, Any] = None):
        """Write dataset to output format"""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for this output format"""
        pass


class ParquetWriter(OutputWriter):
    """Writer for Parquet format"""
    
    def write_dataset(self, dataset: datasets.Dataset, split_name: str, metadata: Dict[str, Any] = None):
        output_path = self.output_dir / f"{split_name}.parquet"
        
        # Add metadata if provided
        if metadata:
            # Create a metadata file alongside the parquet
            metadata_path = self.output_dir / f"{split_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        dataset.to_parquet(str(output_path))
        print(f"Saved {split_name} dataset to {output_path} ({len(dataset)} examples)")
    
    def get_file_extension(self) -> str:
        return ".parquet"


class JSONLWriter(OutputWriter):
    """Writer for JSONL format"""
    
    def write_dataset(self, dataset: datasets.Dataset, split_name: str, metadata: Dict[str, Any] = None):
        output_path = self.output_dir / f"{split_name}.jsonl"
        
        # Write dataset
        dataset.to_json(str(output_path), lines=True, orient='records')
        
        # Write metadata if provided
        if metadata:
            metadata_path = self.output_dir / f"{split_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {split_name} dataset to {output_path} ({len(dataset)} examples)")
    
    def get_file_extension(self) -> str:
        return ".jsonl"


class JSONWriter(OutputWriter):
    """Writer for JSON format"""
    
    def write_dataset(self, dataset: datasets.Dataset, split_name: str, metadata: Dict[str, Any] = None):
        output_path = self.output_dir / f"{split_name}.json"
        
        # Convert dataset to list of dicts
        data_list = [example for example in dataset]
        
        # Create output structure
        output_data = {
            "data": data_list,
            "metadata": metadata or {}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {split_name} dataset to {output_path} ({len(dataset)} examples)")
    
    def get_file_extension(self) -> str:
        return ".json"


class OutputManager:
    """Manages output writing with format selection"""
    
    _writers = {
        'parquet': ParquetWriter,
        'jsonl': JSONLWriter,
        'json': JSONWriter,
    }
    
    def __init__(self, output_dir: str, output_format: str = 'parquet'):
        self.output_format = output_format.lower()
        
        if self.output_format not in self._writers:
            raise ValueError(f"Unsupported output format: {output_format}. "
                           f"Available formats: {list(self._writers.keys())}")
        
        self.writer = self._writers[self.output_format](output_dir)
    
    def write_splits(self, datasets: Dict[str, datasets.Dataset], config_metadata: Dict[str, Any] = None):
        """Write all dataset splits to output directory"""
        for split_name, dataset in datasets.items():
            # Create metadata for this split
            split_metadata = {
                "split": split_name,
                "num_examples": len(dataset),
                "output_format": self.output_format,
                **(config_metadata or {})
            }
            
            self.writer.write_dataset(dataset, split_name, split_metadata)
        
        # Create summary metadata
        summary_metadata = {
            "splits": list(datasets.keys()),
            "total_examples": sum(len(ds) for ds in datasets.values()),
            "output_format": self.output_format,
            **(config_metadata or {})
        }
        
        summary_path = self.writer.output_dir / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset summary saved to {summary_path}")
    
    @classmethod
    def supported_formats(cls) -> List[str]:
        """List supported output formats"""
        return list(cls._writers.keys())
    
    @classmethod
    def register_writer(cls, format_name: str, writer_class: type):
        """Register a new output writer"""
        cls._writers[format_name.lower()] = writer_class