"""
Tests for Combined Reward Functions

This module contains unit tests for the combined reward functions:
- CombinedReward
"""

import unittest
from unittest.mock import MagicMock

from ..base import BaseReward
from ..combined_rewards import CombinedReward


class TestCombinedReward(unittest.TestCase):
    """Test cases for the CombinedReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock reward functions
        self.reward1 = MagicMock(spec=BaseReward)
        self.reward1.name = "reward1"
        self.reward1.weight = 1.0
        self.reward1.return_value = 0.8
        
        self.reward2 = MagicMock(spec=BaseReward)
        self.reward2.name = "reward2"
        self.reward2.weight = 2.0
        self.reward2.return_value = 0.6
        
        self.reward3 = MagicMock(spec=BaseReward)
        self.reward3.name = "reward3"
        self.reward3.weight = 3.0
        self.reward3.return_value = 0.4
        
        # Create combined reward
        self.combined_reward = CombinedReward(
            reward_functions=[self.reward1, self.reward2, self.reward3]
        )
    
    def test_initialization(self):
        """Test that CombinedReward initializes correctly."""
        # Test with provided weights
        combined = CombinedReward(
            reward_functions=[self.reward1, self.reward2, self.reward3],
            weights=[1.0, 2.0, 3.0],
            name="test_combined"
        )
        
        self.assertEqual(combined.name, "test_combined")
        self.assertEqual(len(combined.reward_functions), 3)
        self.assertEqual(combined.weights, [1/6, 2/6, 3/6])  # Normalized weights
        
        # Test with default weights from reward functions
        combined = CombinedReward(
            reward_functions=[self.reward1, self.reward2, self.reward3]
        )
        
        self.assertEqual(combined.name, "combined")
        self.assertEqual(len(combined.reward_functions), 3)
        self.assertEqual(combined.weights, [1/6, 2/6, 3/6])  # Normalized weights
    
    def test_initialization_validation(self):
        """Test that CombinedReward validates inputs correctly."""
        # Test with empty reward functions
        with self.assertRaises(ValueError):
            CombinedReward(reward_functions=[])
        
        # Test with mismatched weights
        with self.assertRaises(ValueError):
            CombinedReward(
                reward_functions=[self.reward1, self.reward2, self.reward3],
                weights=[1.0, 2.0]
            )
    
    def test_compute_reward_single(self):
        """Test that compute_reward correctly combines rewards for a single response."""
        # Setup mock return values
        self.reward1.side_effect = lambda *args, **kwargs: 0.8
        self.reward2.side_effect = lambda *args, **kwargs: 0.6
        self.reward3.side_effect = lambda *args, **kwargs: 0.4
        
        # Test with a single response
        response = "test response"
        prompt = "test prompt"
        reference = "test reference"
        
        score = self.combined_reward.compute_reward(response, prompt, reference)
        
        # Expected score: (0.8 * 1/6) + (0.6 * 2/6) + (0.4 * 3/6) = 0.5333...
        self.assertAlmostEqual(score, 0.5333, places=3)
        
        # Verify each reward function was called with the right arguments
        self.reward1.assert_called_once_with(response, prompt, reference)
        self.reward2.assert_called_once_with(response, prompt, reference)
        self.reward3.assert_called_once_with(response, prompt, reference)
    
    def test_compute_reward_batch(self):
        """Test that compute_reward correctly combines rewards for batch responses."""
        # Setup mock return values for batch processing
        self.reward1.side_effect = lambda *args, **kwargs: [0.8, 0.7, 0.9]
        self.reward2.side_effect = lambda *args, **kwargs: [0.6, 0.5, 0.4]
        self.reward3.side_effect = lambda *args, **kwargs: [0.4, 0.3, 0.2]
        
        # Test with multiple responses
        responses = ["response1", "response2", "response3"]
        
        scores = self.combined_reward.compute_reward(responses)
        
        # Expected scores:
        # (0.8 * 1/6) + (0.6 * 2/6) + (0.4 * 3/6) = 0.5333...
        # (0.7 * 1/6) + (0.5 * 2/6) + (0.3 * 3/6) = 0.4333...
        # (0.9 * 1/6) + (0.4 * 2/6) + (0.2 * 3/6) = 0.3833...
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 0.5333, places=3)
        self.assertAlmostEqual(scores[1], 0.4333, places=3)
        self.assertAlmostEqual(scores[2], 0.3833, places=3)
    
    def test_compute_reward_mismatched_batch_sizes(self):
        """Test that compute_reward handles mismatched batch sizes correctly."""
        # Setup mock return values with mismatched batch sizes
        self.reward1.side_effect = lambda *args, **kwargs: [0.8, 0.7, 0.9]
        self.reward2.side_effect = lambda *args, **kwargs: 0.6  # Single value
        self.reward3.side_effect = lambda *args, **kwargs: [0.4, 0.3]  # Shorter batch
        
        # Test with multiple responses
        responses = ["response1", "response2", "response3"]
        
        scores = self.combined_reward.compute_reward(responses)
        
        # Should handle the mismatches by extending single values and using the first value for shorter batches
        self.assertEqual(len(scores), 3)
    
    def test_error_handling(self):
        """Test that compute_reward handles errors in reward functions gracefully."""
        # Setup one reward function to raise an exception
        self.reward1.side_effect = lambda *args, **kwargs: 0.8
        self.reward2.side_effect = Exception("Test error")
        self.reward3.side_effect = lambda *args, **kwargs: 0.4
        
        # Test with a single response
        response = "test response"
        
        score = self.combined_reward.compute_reward(response)
        
        # Should skip the failing reward function and normalize the remaining weights
        # Expected score: (0.8 * 1/4) + (0.4 * 3/4) = 0.5
        self.assertAlmostEqual(score, 0.5, places=3)
    
    def test_normalize_score(self):
        """Test that _normalize_score correctly clamps scores to [0, 1]."""
        # Create a combined reward with mock functions that return out-of-range values
        reward1 = MagicMock(spec=BaseReward)
        reward1.name = "reward1"
        reward1.weight = 1.0
        reward1.side_effect = lambda *args, **kwargs: 1.5  # Above range
        
        reward2 = MagicMock(spec=BaseReward)
        reward2.name = "reward2"
        reward2.weight = 1.0
        reward2.side_effect = lambda *args, **kwargs: -0.5  # Below range
        
        combined = CombinedReward(
            reward_functions=[reward1, reward2],
            weights=[0.5, 0.5]
        )
        
        # Test with a single response
        response = "test response"
        
        score = combined.compute_reward(response)
        
        # Expected score: (1.0 * 0.5) + (0.0 * 0.5) = 0.5
        # (out-of-range values should be clamped to [0, 1])
        self.assertEqual(score, 0.5)
    
    def test_custom_weights(self):
        """Test that custom weights are used correctly."""
        # Setup mock return values
        self.reward1.side_effect = lambda *args, **kwargs: 0.8
        self.reward2.side_effect = lambda *args, **kwargs: 0.6
        self.reward3.side_effect = lambda *args, **kwargs: 0.4
        
        # Create combined reward with custom weights
        combined = CombinedReward(
            reward_functions=[self.reward1, self.reward2, self.reward3],
            weights=[0.2, 0.3, 0.5]  # Custom weights (already normalized)
        )
        
        # Test with a single response
        response = "test response"
        
        score = combined.compute_reward(response)
        
        # Expected score: (0.8 * 0.2) + (0.6 * 0.3) + (0.4 * 0.5) = 0.56
        self.assertAlmostEqual(score, 0.56, places=3)


if __name__ == "__main__":
    unittest.main()