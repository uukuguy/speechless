"""
Tests for Math-Based Reward Functions

This module contains unit tests for the math-based reward functions:
- MathReward
- MathVerifyReward
"""

import unittest
from unittest.mock import MagicMock, patch

from ..math_rewards import MathReward, MathVerifyReward, MATH_VERIFY_AVAILABLE


class TestMathReward(unittest.TestCase):
    """Test cases for the MathReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward = MathReward()
    
    def test_extract_answer(self):
        """Test that _extract_answer correctly extracts numerical answers."""
        # Test with explicit answer format
        text = "The answer is 42."
        self.assertEqual(self.reward._extract_answer(text), 42)
        
        # Test with different format
        text = "The result = 3.14"
        self.assertEqual(self.reward._extract_answer(text), 3.14)
        
        # Test with no explicit answer marker but number present
        text = "After calculating, we get 123."
        self.assertEqual(self.reward._extract_answer(text), 123)
        
        # Test with no numbers
        text = "The answer is unknown."
        self.assertIsNone(self.reward._extract_answer(text))
    
    def test_evaluate_reasoning_steps(self):
        """Test that _evaluate_reasoning_steps correctly scores reasoning quality."""
        # Good reasoning with steps, operations, and explanations
        good_reasoning = """
        Step 1: First, we identify that 6 * 7 = 42.
        Step 2: Next, we add 10 to get 42 + 10 = 52.
        Therefore, the answer is 52.
        """
        
        # Poor reasoning with minimal explanation
        poor_reasoning = "The answer is 52."
        
        good_score = self.reward._evaluate_reasoning_steps(good_reasoning)
        poor_score = self.reward._evaluate_reasoning_steps(poor_reasoning)
        
        self.assertGreater(good_score, 0.7)  # Good reasoning should get high score
        self.assertLess(poor_score, 0.5)     # Poor reasoning should get low score
    
    def test_check_answer_correctness(self):
        """Test that _check_answer_correctness correctly scores answer correctness."""
        # Exact match
        self.assertEqual(self.reward._check_answer_correctness(42, 42), 1.0)
        
        # Very close match
        self.assertGreater(self.reward._check_answer_correctness(42.0000001, 42), 0.8)
        
        # Within 1%
        self.assertGreater(self.reward._check_answer_correctness(42.3, 42), 0.7)
        
        # Within 5%
        self.assertGreater(self.reward._check_answer_correctness(44, 42), 0.5)
        
        # Within 10%
        self.assertGreater(self.reward._check_answer_correctness(46, 42), 0.3)
        
        # Far off
        self.assertLess(self.reward._check_answer_correctness(100, 42), 0.3)
    
    def test_compute_reward(self):
        """Test that compute_reward correctly combines reasoning and answer scores."""
        # Good reasoning with correct answer
        good_response = """
        Step 1: We calculate 6 * 7 = 42.
        Therefore, the answer is 42.
        """
        
        # Good reasoning with wrong answer
        mixed_response = """
        Step 1: We calculate 6 * 7 = 42.
        Step 2: We add 10 to get 52.
        Therefore, the answer is 52.
        """
        
        # Poor reasoning with correct answer
        poor_reasoning = "The answer is 42."
        
        # Test with reference answer
        good_score = self.reward.compute_reward(good_response, reference_answer=42)
        mixed_score = self.reward.compute_reward(mixed_response, reference_answer=42)
        poor_score = self.reward.compute_reward(poor_reasoning, reference_answer=42)
        
        # Good reasoning + correct answer should score highest
        self.assertGreater(good_score, mixed_score)
        self.assertGreater(good_score, poor_score)
        
        # Test with reference text
        ref_text = "The answer is 42."
        good_score_ref = self.reward.compute_reward(good_response, reference=ref_text)
        self.assertAlmostEqual(good_score, good_score_ref, places=1)
        
        # Test with no reference (should only evaluate reasoning)
        reasoning_only = self.reward.compute_reward(good_response)
        self.assertGreater(reasoning_only, 0.5)  # Should get decent score for good reasoning
    
    def test_batch_processing(self):
        """Test that the reward function handles batch inputs correctly."""
        responses = [
            "The answer is 42.",
            "Step 1: Calculate 6 * 7 = 42. Therefore, the answer is 42.",
            "The result is 52."  # Wrong answer
        ]
        
        scores = self.reward.compute_reward(responses, reference_answer=[42, 42, 42])
        
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[1], scores[0])  # Response with reasoning should score higher
        self.assertGreater(scores[0], scores[2])  # Correct answer should score higher than wrong answer


@unittest.skipIf(not MATH_VERIFY_AVAILABLE, "math-verify not installed")
class TestMathVerifyReward(unittest.TestCase):
    """Test cases for the MathVerifyReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward = MathVerifyReward()
    
    @patch('math_verify.metric.math_metric')
    def test_compute_reward_with_mock(self, mock_math_metric):
        """Test compute_reward using a mock for math_metric."""
        # Setup mock
        mock_verify_func = MagicMock()
        mock_verify_func.return_value = (0.8, None)
        mock_math_metric.return_value = mock_verify_func
        
        # Test with a single response
        score = self.reward.compute_reward("The answer is 42.", reference="42")
        self.assertEqual(score, 0.8)
        
        # Verify mock was called correctly
        mock_math_metric.assert_called_once()
        mock_verify_func.assert_called_once()
        
        # Check that boxed format was applied
        args = mock_verify_func.call_args[0]
        self.assertTrue(args[0][0].startswith("\\boxed{"))
    
    @patch('math_verify.metric.math_metric')
    def test_boxed_format_option(self, mock_math_metric):
        """Test that boxed_format option works correctly."""
        # Setup mock
        mock_verify_func = MagicMock()
        mock_verify_func.return_value = (0.8, None)
        mock_math_metric.return_value = mock_verify_func
        
        # Create reward with boxed_format=False
        reward = MathVerifyReward(boxed_format=False)
        
        # Test with a single response
        reward.compute_reward("The answer is 42.", reference="42")
        
        # Check that boxed format was not applied
        args = mock_verify_func.call_args[0]
        self.assertEqual(args[0][0], "42")  # Should not be boxed
    
    @patch('math_verify.metric.math_metric')
    def test_batch_processing(self, mock_math_metric):
        """Test that the reward function handles batch inputs correctly."""
        # Setup mock
        mock_verify_func = MagicMock()
        mock_verify_func.return_value = (0.8, None)
        mock_math_metric.return_value = mock_verify_func
        
        # Test with multiple responses
        responses = ["Answer 1", "Answer 2", "Answer 3"]
        references = ["Ref 1", "Ref 2", "Ref 3"]
        
        scores = self.reward.compute_reward(responses, reference=references)
        
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores, [0.8, 0.8, 0.8])  # All should get the same mock score
        self.assertEqual(mock_verify_func.call_count, 3)  # Should be called once per response
    
    @patch('math_verify.metric.math_metric')
    def test_error_handling(self, mock_math_metric):
        """Test that errors are handled gracefully."""
        # Setup mock to raise exception
        mock_verify_func = MagicMock()
        mock_verify_func.side_effect = Exception("Test error")
        mock_math_metric.return_value = mock_verify_func
        
        # Test with a single response
        score = self.reward.compute_reward("The answer is 42.", reference="42")
        
        # Should return 0.0 on error
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()