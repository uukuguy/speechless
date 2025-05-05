"""
Tests for Reward Function Utilities

This module contains unit tests for the utility functions:
- create_reward_function
- example_usage
"""

import unittest
from unittest.mock import patch, MagicMock

from ..utils import create_reward_function
from ..text_rewards import LengthReward, FormatReward, CoherenceReward
from ..math_rewards import MathReward, MathVerifyReward
from ..code_rewards import CodeReward
from ..factuality_rewards import FactualityReward
from ..task_rewards import TaskSpecificReward
from ..tag_rewards import TagReward
from ..combined_rewards import CombinedReward


class TestCreateRewardFunction(unittest.TestCase):
    """Test cases for the create_reward_function utility."""
    
    def test_create_length_reward(self):
        """Test creating a LengthReward from configuration."""
        config = {
            'type': 'length',
            'min_length': 100,
            'max_length': 500,
            'optimal_length': 300,
            'weight': 2.0
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, LengthReward)
        self.assertEqual(reward.min_length, 100)
        self.assertEqual(reward.max_length, 500)
        self.assertEqual(reward.optimal_length, 300)
        self.assertEqual(reward.weight, 2.0)
    
    def test_create_format_reward(self):
        """Test creating a FormatReward from configuration."""
        config = {
            'type': 'format',
            'format_type': 'json',
            'weight': 1.5
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, FormatReward)
        self.assertEqual(reward.format_type, 'json')
        self.assertEqual(reward.weight, 1.5)
    
    def test_create_math_reward(self):
        """Test creating a MathReward from configuration."""
        config = {
            'type': 'math',
            'check_final_answer': True,
            'check_reasoning_steps': False,
            'weight': 3.0
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, MathReward)
        self.assertTrue(reward.check_final_answer)
        self.assertFalse(reward.check_reasoning_steps)
        self.assertEqual(reward.weight, 3.0)
    
    def test_create_code_reward(self):
        """Test creating a CodeReward from configuration."""
        config = {
            'type': 'code',
            'check_syntax': True,
            'check_execution': False,
            'check_style': True,
            'language': 'python',
            'weight': 2.5
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, CodeReward)
        self.assertTrue(reward.check_syntax)
        self.assertFalse(reward.check_execution)
        self.assertTrue(reward.check_style)
        self.assertEqual(reward.language, 'python')
        self.assertEqual(reward.weight, 2.5)
    
    def test_create_factuality_reward(self):
        """Test creating a FactualityReward from configuration."""
        config = {
            'type': 'factuality',
            'reference_texts': ['Text 1', 'Text 2'],
            'check_contradictions': True,
            'weight': 1.8
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, FactualityReward)
        self.assertEqual(reward.reference_texts, ['Text 1', 'Text 2'])
        self.assertTrue(reward.check_contradictions)
        self.assertEqual(reward.weight, 1.8)
    
    def test_create_coherence_reward(self):
        """Test creating a CoherenceReward from configuration."""
        config = {
            'type': 'coherence',
            'check_logical_flow': True,
            'check_consistency': False,
            'check_clarity': True,
            'weight': 1.2
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, CoherenceReward)
        self.assertTrue(reward.check_logical_flow)
        self.assertFalse(reward.check_consistency)
        self.assertTrue(reward.check_clarity)
        self.assertEqual(reward.weight, 1.2)
    
    def test_create_task_specific_reward(self):
        """Test creating a TaskSpecificReward from configuration."""
        config = {
            'type': 'task_specific',
            'task_type': 'summarization',
            'task_params': {'target_ratio': 0.1},
            'weight': 2.2
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, TaskSpecificReward)
        self.assertEqual(reward.task_type, 'summarization')
        self.assertEqual(reward.task_params, {'target_ratio': 0.1})
        self.assertEqual(reward.weight, 2.2)
    
    def test_create_tag_reward(self):
        """Test creating a TagReward from configuration."""
        tag_specs = {
            'thinking': {'required': True},
            'answer': {'required': True}
        }
        
        config = {
            'type': 'tag',
            'tag_specs': tag_specs,
            'strict_nesting': True,
            'weight': 1.7
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, TagReward)
        self.assertEqual(reward.tag_specs['thinking']['required'], True)
        self.assertEqual(reward.tag_specs['answer']['required'], True)
        self.assertTrue(reward.strict_nesting)
        self.assertEqual(reward.weight, 1.7)
    
    def test_create_math_verify_reward(self):
        """Test creating a MathVerifyReward from configuration."""
        config = {
            'type': 'math_verify',
            'boxed_format': False,
            'weight': 2.3
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, MathVerifyReward)
        self.assertFalse(reward.boxed_format)
        self.assertEqual(reward.weight, 2.3)
    
    def test_create_combined_reward(self):
        """Test creating a CombinedReward from configuration."""
        config = {
            'type': 'combined',
            'name': 'custom_combined',
            'reward_functions': [
                {'type': 'length', 'min_length': 100, 'weight': 1.0},
                {'type': 'format', 'format_type': 'json', 'weight': 2.0}
            ]
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, CombinedReward)
        self.assertEqual(reward.name, 'custom_combined')
        self.assertEqual(len(reward.reward_functions), 2)
        self.assertIsInstance(reward.reward_functions[0], LengthReward)
        self.assertIsInstance(reward.reward_functions[1], FormatReward)
        self.assertEqual(reward.weights, [1/3, 2/3])  # Normalized weights
    
    def test_missing_type(self):
        """Test that create_reward_function raises an error for missing type."""
        config = {
            'min_length': 100,
            'max_length': 500
        }
        
        with self.assertRaises(ValueError):
            create_reward_function(config)
    
    def test_unknown_type(self):
        """Test that create_reward_function raises an error for unknown type."""
        config = {
            'type': 'unknown',
            'param1': 'value1'
        }
        
        with self.assertRaises(ValueError):
            create_reward_function(config)
    
    def test_combined_reward_no_functions(self):
        """Test that create_reward_function raises an error for combined reward with no functions."""
        config = {
            'type': 'combined',
            'name': 'empty_combined',
            'reward_functions': []
        }
        
        with self.assertRaises(ValueError):
            create_reward_function(config)


class TestExampleUsage(unittest.TestCase):
    """Test cases for the example_usage utility."""
    
    @patch('builtins.print')
    def test_example_usage(self, mock_print):
        """Test that example_usage runs without errors."""
        from ..utils import example_usage
        
        # Just verify that it runs without errors
        example_usage()
        
        # Verify that print was called multiple times
        self.assertGreater(mock_print.call_count, 5)


if __name__ == "__main__":
    unittest.main()