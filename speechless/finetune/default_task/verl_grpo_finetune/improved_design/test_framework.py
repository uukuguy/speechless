"""
Comprehensive testing framework for the data processing pipeline.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

from config import ConfigManager, ProcessorConfig
from processors import ProcessorFactory, GSM8KProcessor, MathProcessor, GenericProcessor
from output_manager import OutputManager
from pipeline import ProcessingPipeline


class MockDataset:
    """Mock dataset for testing"""
    
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def map(self, function, with_indices=False):
        if with_indices:
            processed = [function(item, idx) for idx, item in enumerate(self.data)]
        else:
            processed = [function(item) for item in self.data]
        return MockDataset(processed)
    
    def to_parquet(self, path):
        # Mock saving to parquet
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(f"Mock parquet data: {len(self.data)} examples")
    
    def to_json(self, path, lines=True, orient='records'):
        # Mock saving to JSON
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(f"Mock JSON data: {len(self.data)} examples")


class TestConfigManager(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_and_load_config(self):
        """Test creating and loading configurations"""
        config = ProcessorConfig(
            name="test",
            data_source="test/data",
            input_key="question",
            output_key="answer"
        )
        
        # Save config
        self.config_manager.save_config(config, "test")
        
        # Load config
        loaded_config = self.config_manager.load_config("test")
        
        self.assertEqual(loaded_config.name, "test")
        self.assertEqual(loaded_config.data_source, "test/data")
        self.assertEqual(loaded_config.input_key, "question")
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = ProcessorConfig(
            name="test",
            data_source="test/data",
            input_key="question",
            output_key="answer"
        )
        
        self.assertTrue(self.config_manager.validate_config(valid_config))
        
        # Invalid config (missing required field)
        invalid_config = ProcessorConfig(
            name="",
            data_source="test/data",
            input_key="question",
            output_key="answer"
        )
        
        with self.assertRaises(ValueError):
            self.config_manager.validate_config(invalid_config)
    
    def test_list_configs(self):
        """Test listing configurations"""
        # Initially empty
        self.assertEqual(len(self.config_manager.list_configs()), 0)
        
        # Add configs
        config1 = ProcessorConfig(name="test1", data_source="test/data1")
        config2 = ProcessorConfig(name="test2", data_source="test/data2")
        
        self.config_manager.save_config(config1, "test1")
        self.config_manager.save_config(config2, "test2")
        
        configs = self.config_manager.list_configs()
        self.assertEqual(len(configs), 2)
        self.assertIn("test1", configs)
        self.assertIn("test2", configs)


class TestProcessors(unittest.TestCase):
    """Test data processors"""
    
    def setUp(self):
        self.gsm8k_config = ProcessorConfig(
            name="gsm8k",
            data_source="test/gsm8k",
            input_key="question",
            output_key="answer",
            custom_params={
                "answer_pattern": r"#### ([\-]?[0-9\.\,]+)",
                "prompt_template": "{question} Let's think step by step."
            }
        )
    
    def test_gsm8k_processor(self):
        """Test GSM8K processor"""
        processor = GSM8KProcessor(self.gsm8k_config)
        
        # Test answer extraction
        answer = "The calculation is 42 + 8 = 50\n#### 50"
        extracted = processor.extract_answer(answer)
        self.assertEqual(extracted, "50")
        
        # Test prompt formatting
        question = "What is 2 + 2?"
        formatted = processor.format_prompt(question)
        self.assertEqual(formatted, "What is 2 + 2? Let's think step by step.")
    
    def test_processor_factory(self):
        """Test processor factory"""
        # Test GSM8K processor creation
        gsm8k_processor = ProcessorFactory.create_processor(self.gsm8k_config)
        self.assertIsInstance(gsm8k_processor, GSM8KProcessor)
        
        # Test generic processor for unknown type
        unknown_config = ProcessorConfig(name="unknown", data_source="test")
        generic_processor = ProcessorFactory.create_processor(unknown_config)
        self.assertIsInstance(generic_processor, GenericProcessor)
    
    def test_process_example(self):
        """Test example processing"""
        processor = GSM8KProcessor(self.gsm8k_config)
        
        example = {
            "question": "What is 2 + 3?",
            "answer": "2 + 3 = 5\n#### 5"
        }
        
        with patch.object(processor, 'load_dataset'):
            processed = processor.process_example(example, 0, "test")
            
            self.assertEqual(processed["ability"], "math")
            self.assertEqual(processed["reward_model"]["ground_truth"], "5")
            self.assertIn("role", processed["prompt"][0])
            self.assertEqual(processed["prompt"][0]["role"], "user")


class TestOutputManager(unittest.TestCase):
    """Test output management"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_parquet_writer(self):
        """Test parquet output writing"""
        output_manager = OutputManager(self.temp_dir, "parquet")
        
        # Create mock dataset
        mock_data = [
            {"text": "example 1", "label": "A"},
            {"text": "example 2", "label": "B"}
        ]
        
        datasets = {
            "train": MockDataset(mock_data),
            "test": MockDataset(mock_data[:1])
        }
        
        # Write datasets
        output_manager.write_splits(datasets, {"test": True})
        
        # Check files were created
        self.assertTrue((Path(self.temp_dir) / "train.parquet").exists())
        self.assertTrue((Path(self.temp_dir) / "test.parquet").exists())
        self.assertTrue((Path(self.temp_dir) / "dataset_summary.json").exists())
    
    def test_supported_formats(self):
        """Test supported output formats"""
        formats = OutputManager.supported_formats()
        self.assertIn("parquet", formats)
        self.assertIn("jsonl", formats)
        self.assertIn("json", formats)


class TestPipeline(unittest.TestCase):
    """Test the main processing pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        
        # Create test config
        self.test_config = ProcessorConfig(
            name="test",
            data_source="test/data",
            input_key="question",
            output_key="answer",
            splits=["train", "test"]
        )
        self.config_manager.save_config(self.test_config, "test")
        
        self.pipeline = ProcessingPipeline(self.config_manager)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_config_validation(self):
        """Test configuration validation in pipeline"""
        result = self.pipeline.validate_config("test")
        self.assertTrue(result["valid"])
        
        # Test non-existent config
        result = self.pipeline.validate_config("nonexistent")
        self.assertFalse(result["valid"])
    
    def test_list_configs(self):
        """Test listing configurations"""
        configs = self.pipeline.list_available_configs()
        self.assertIn("test", configs)
    
    def test_get_config_info(self):
        """Test getting config information"""
        info = self.pipeline.get_config_info("test")
        self.assertEqual(info["name"], "test")
        self.assertEqual(info["data_source"], "test/data")
        self.assertIn("valid", info)
    
    @patch('processors.datasets.load_dataset')
    def test_dry_run_processing(self, mock_load):
        """Test dry run processing"""
        # Mock dataset
        mock_data = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"}
        ]
        
        mock_dataset = {
            "train": MockDataset(mock_data),
            "test": MockDataset(mock_data[:1])
        }
        mock_load.return_value = mock_dataset
        
        # Process with dry run
        result = self.pipeline.process_dataset(
            "test", 
            self.temp_dir + "/output", 
            dry_run=True
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["total_examples"], 3)  # 2 train + 1 test


class IntegrationTest(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('processors.datasets.load_dataset')
    def test_end_to_end_processing(self, mock_load):
        """Test complete end-to-end processing"""
        # Setup
        config_manager = ConfigManager(self.temp_dir + "/configs")
        
        # Create GSM8K-like config
        config = ProcessorConfig(
            name="gsm8k",
            data_source="test/gsm8k",
            input_key="question",
            output_key="answer",
            custom_params={
                "answer_pattern": r"#### ([\-]?[0-9\.\,]+)",
                "prompt_template": "{question} Let's think step by step."
            }
        )
        config_manager.save_config(config, "gsm8k")
        
        # Mock dataset
        mock_data = [
            {
                "question": "What is 2+2?", 
                "answer": "2 + 2 = 4\n#### 4"
            },
            {
                "question": "What is 5*3?", 
                "answer": "5 * 3 = 15\n#### 15"
            }
        ]
        
        mock_dataset = {
            "train": MockDataset(mock_data),
            "test": MockDataset(mock_data[:1])
        }
        mock_load.return_value = mock_dataset
        
        # Process
        pipeline = ProcessingPipeline(config_manager)
        result = pipeline.process_dataset(
            "gsm8k",
            self.temp_dir + "/output"
        )
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["total_examples"], 3)
        self.assertEqual(len(result["processed_splits"]), 2)
        self.assertIn("train", result["processed_splits"])
        self.assertIn("test", result["processed_splits"])


def run_tests():
    """Run all tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfigManager,
        TestProcessors,
        TestOutputManager,
        TestPipeline,
        IntegrationTest
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)