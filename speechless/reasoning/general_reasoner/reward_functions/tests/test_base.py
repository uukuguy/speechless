"""
Tests for the BaseReward class

This module contains unit tests for the BaseReward class and its methods.
"""

import unittest
from unittest.mock import MagicMock

from ..base import BaseReward


class TestBaseReward(unittest.TestCase):
    """Test cases for the BaseReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation of BaseReward for testing
        class ConcreteReward(BaseReward):
            def compute_reward(self, response, prompt=None, reference=None, **kwargs):
                # Simple implementation that returns 0.5 for testing
                if isinstance(response, list):
                    return [0.5] * len(response)
                return 0.5
        
        self.reward = ConcreteReward(name="test_reward", weight=2.0)
    
    def test_initialization(self):
        """Test that the BaseReward initializes correctly."""
        self.assertEqual(self.reward.name, "test_reward")
        self.assertEqual(self.reward.weight, 2.0)
    
    def test_callable(self):
        """Test that the BaseReward is callable and delegates to compute_reward."""
        result = self.reward("test_response")
        self.assertEqual(result, 0.5)
    
    def test_ensure_list_with_string(self):
        """Test that _ensure_list converts a string to a list."""
        result = self.reward._ensure_list("test")
        self.assertEqual(result, ["test"])
    
    def test_ensure_list_with_list(self):
        """Test that _ensure_list leaves a list unchanged."""
        test_list = ["item1", "item2"]
        result = self.reward._ensure_list(test_list)
        self.assertEqual(result, test_list)
    
    def test_normalize_score_within_range(self):
        """Test that _normalize_score returns the score if it's within range."""
        self.assertEqual(self.reward._normalize_score(0.5), 0.5)
    
    def test_normalize_score_below_min(self):
        """Test that _normalize_score clamps to min_val if score is below range."""
        self.assertEqual(self.reward._normalize_score(-0.5), 0.0)
    
    def test_normalize_score_above_max(self):
        """Test that _normalize_score clamps to max_val if score is above range."""
        self.assertEqual(self.reward._normalize_score(1.5), 1.0)
    
    def test_normalize_score_custom_range(self):
        """Test that _normalize_score works with custom min/max values."""
        self.assertEqual(self.reward._normalize_score(15, min_val=10, max_val=20), 15)
        self.assertEqual(self.reward._normalize_score(5, min_val=10, max_val=20), 10)
        self.assertEqual(self.reward._normalize_score(25, min_val=10, max_val=20), 20)
    
    def test_batch_processing(self):
        """Test that the reward function handles batch inputs correctly."""
        responses = ["response1", "response2", "response3"]
        results = self.reward(responses)
        self.assertEqual(results, [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()