#!/usr/bin/env python
"""
Unit tests for the LightLLM framework.

This module contains tests for the core functionality of the LightLLM framework,
including configuration validation, dataset handling, model adaptation, and training.
"""

import os
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from speechless.finetune.lightllm import (
    TrainingConfig,
    TrainingMethod,
    AdaptationMethod,
    DataFormat,
    TextDataset,
    LoRAAdapter,
    FullAdapter,
    SupervisedTrainer,
    PPOTrainer,
    GRPOTrainer,
    LoRAMerger,
    OptimizedInference,
)


class TestTrainingConfig(unittest.TestCase):
    """Tests for the TrainingConfig class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainingConfig()
        self.assertEqual(config.model_name_or_path, "facebook/opt-350m")
        self.assertEqual(config.training_method, "sft")
        self.assertEqual(config.adaptation_method, "lora")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.num_train_epochs, 3)
    
    def test_tokenizer_path_default(self):
        """Test that tokenizer_path defaults to model_path if not provided."""
        config = TrainingConfig(model_name_or_path="test/model")
        self.assertEqual(config.tokenizer_name_or_path, "test/model")
    
    def test_training_method_validation(self):
        """Test that invalid training methods are rejected."""
        with self.assertRaises(ValueError):
            TrainingConfig(training_method="invalid_method")
    
    def test_adaptation_method_validation(self):
        """Test that invalid adaptation methods are rejected."""
        with self.assertRaises(ValueError):
            TrainingConfig(adaptation_method="invalid_method")


class TestTextDataset(unittest.TestCase):
    """Tests for the TextDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        self.tokenizer.pad_token = "[PAD]"
        
        # Create temporary SFT data file
        self.sft_data_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(self.sft_data_file.name, "w") as f:
            json.dump([{"input": "Hello", "output": " world"}], f)
        
        # Create temporary RL data file
        self.rl_data_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(self.rl_data_file.name, "w") as f:
            json.dump([{"prompt": "Hello", "response": "world", "reward": 1.0}], f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.sft_data_file.name)
        os.unlink(self.rl_data_file.name)
    
    def test_sft_data_format_detection(self):
        """Test that SFT data format is correctly detected."""
        dataset = TextDataset(self.sft_data_file.name, self.tokenizer, 512)
        self.assertEqual(dataset.data_format, DataFormat.SFT)
    
    def test_rl_data_format_detection(self):
        """Test that RL data format is correctly detected."""
        dataset = TextDataset(self.rl_data_file.name, self.tokenizer, 512)
        self.assertEqual(dataset.data_format, DataFormat.RL)
    
    def test_invalid_data_file(self):
        """Test that invalid data files are rejected."""
        with self.assertRaises(FileNotFoundError):
            TextDataset("nonexistent_file.json", self.tokenizer, 512)


class TestModelAdapters(unittest.TestCase):
    """Tests for the model adapter classes."""
    
    @patch("speechless.finetune.lightllm.get_peft_model")
    @patch("speechless.finetune.lightllm.prepare_model_for_kbit_training")
    def test_lora_adapter(self, mock_prepare, mock_get_peft):
        """Test LoRA adapter functionality."""
        # Setup mocks
        model = MagicMock()
        model.is_quantized = True
        config = TrainingConfig(lora_r=16, lora_alpha=32)
        
        # Call the adapter
        LoRAAdapter.prepare_model(model, config)
        
        # Verify mocks were called correctly
        mock_prepare.assert_called_once_with(model)
        mock_get_peft.assert_called_once()
    
    def test_full_adapter(self):
        """Test full fine-tuning adapter functionality."""
        model = MagicMock()
        config = TrainingConfig()
        
        # Full adapter should return the model unchanged
        result = FullAdapter.prepare_model(model, config)
        self.assertEqual(result, model)


class TestLoRAMerger(unittest.TestCase):
    """Tests for the LoRA merger functionality."""
    
    @patch("speechless.finetune.lightllm.PeftModel.from_pretrained")
    @patch("speechless.finetune.lightllm.AutoModelForCausalLM.from_pretrained")
    @patch("speechless.finetune.lightllm.AutoTokenizer.from_pretrained")
    def test_merge_lora_weights(self, mock_tokenizer, mock_model, mock_peft):
        """Test merging LoRA weights into a base model."""
        # Setup mocks
        base_model = MagicMock()
        mock_model.return_value = base_model
        
        peft_model = MagicMock()
        mock_peft.return_value = peft_model
        merged_model = MagicMock()
        peft_model.merge_and_unload.return_value = merged_model
        
        # Call the merger
        result = LoRAMerger.merge_lora_weights(
            "base/model/path",
            "lora/model/path",
            "output/path",
            "cpu"
        )
        
        # Verify the result
        self.assertEqual(result, merged_model)
        mock_model.assert_called_once()
        mock_peft.assert_called_once_with(base_model, "lora/model/path")
        peft_model.merge_and_unload.assert_called_once()
        merged_model.save_pretrained.assert_called_once_with("output/path")


if __name__ == "__main__":
    unittest.main()