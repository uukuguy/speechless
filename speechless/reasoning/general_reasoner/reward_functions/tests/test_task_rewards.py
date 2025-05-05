"""
Tests for Task-Specific Reward Functions

This module contains unit tests for the task-specific reward functions:
- TaskSpecificReward
"""

import unittest
from unittest.mock import MagicMock

from ..task_rewards import TaskSpecificReward


class TestTaskSpecificReward(unittest.TestCase):
    """Test cases for the TaskSpecificReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.summarization_reward = TaskSpecificReward(task_type='summarization')
        self.translation_reward = TaskSpecificReward(task_type='translation')
        self.qa_reward = TaskSpecificReward(task_type='qa')
    
    def test_summarization_reward(self):
        """Test that _summarization_reward correctly evaluates summaries."""
        # Original text
        original = """
        Machine learning is a field of study in artificial intelligence concerned with the development
        of algorithms and statistical models that computer systems use to perform tasks without explicit
        instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence.
        Machine learning algorithms build a mathematical model based on sample data, known as "training data",
        in order to make predictions or decisions without being explicitly programmed to perform the task.
        Machine learning algorithms are used in a wide variety of applications, such as email filtering and
        computer vision, where it is difficult or infeasible to develop conventional algorithms to perform
        the needed tasks.
        """
        
        # Good summary (concise, covers key points)
        good_summary = """
        Machine learning is an AI field focused on developing algorithms that allow computers to perform tasks
        using patterns and inference rather than explicit instructions. It uses training data to build models
        for making predictions and is applied in areas like email filtering and computer vision.
        """
        
        # Bad summary (too short, misses key points)
        bad_summary = "Machine learning is a type of AI."
        
        # Too verbose (almost as long as original)
        verbose_summary = """
        Machine learning is a field of study that falls under artificial intelligence. It focuses on developing
        algorithms and statistical models that enable computer systems to perform tasks without explicit
        instructions. Instead, these systems rely on patterns and inference. Machine learning is considered
        a subset of artificial intelligence. The algorithms in machine learning build mathematical models
        using sample data, which is referred to as "training data". These models then make predictions or
        decisions without being explicitly programmed for the specific task. Machine learning algorithms
        have wide applications.
        """
        
        good_score = self.summarization_reward._summarization_reward(good_summary, original)
        bad_score = self.summarization_reward._summarization_reward(bad_summary, original)
        verbose_score = self.summarization_reward._summarization_reward(verbose_summary, original)
        
        self.assertGreater(good_score, bad_score)
        self.assertGreater(good_score, verbose_score)
    
    def test_translation_reward(self):
        """Test that _translation_reward correctly evaluates translations."""
        # Reference translation
        reference = "The quick brown fox jumps over the lazy dog."
        
        # Good translation (very close to reference)
        good_translation = "The quick brown fox jumps over the lazy dog."
        
        # Decent translation (same meaning, different words)
        decent_translation = "A fast brown fox leaps above the lazy dog."
        
        # Bad translation (different meaning)
        bad_translation = "The slow gray wolf runs under the active cat."
        
        good_score = self.translation_reward._translation_reward(good_translation, reference)
        decent_score = self.translation_reward._translation_reward(decent_translation, reference)
        bad_score = self.translation_reward._translation_reward(bad_translation, reference)
        
        self.assertGreater(good_score, decent_score)
        self.assertGreater(decent_score, bad_score)
    
    def test_qa_reward(self):
        """Test that _qa_reward correctly evaluates question answering."""
        # Reference answer
        reference = "Paris is the capital of France."
        
        # Question
        question = "What is the capital of France?"
        
        # Exact match
        exact_match = "Paris is the capital of France."
        
        # Correct but different wording
        correct_different = "The capital of France is Paris."
        
        # Substring match
        substring_match = "Paris."
        
        # Wrong answer
        wrong_answer = "Lyon is the capital of France."
        
        exact_score = self.qa_reward._qa_reward(exact_match, reference, question)
        different_score = self.qa_reward._qa_reward(correct_different, reference, question)
        substring_score = self.qa_reward._qa_reward(substring_match, reference, question)
        wrong_score = self.qa_reward._qa_reward(wrong_answer, reference, question)
        
        self.assertEqual(exact_score, 1.0)  # Exact match should get perfect score
        self.assertGreater(different_score, 0.7)  # Correct but different should get high score
        self.assertGreater(substring_score, 0.7)  # Substring match should get high score
        self.assertLess(wrong_score, 0.5)  # Wrong answer should get low score
    
    def test_custom_reward_function(self):
        """Test that custom_reward_fn is used when provided."""
        # Create a mock custom reward function
        mock_reward_fn = MagicMock(return_value=0.75)
        
        # Create a reward with the mock function
        reward = TaskSpecificReward(
            task_type='custom',
            custom_reward_fn=mock_reward_fn
        )
        
        # Test the reward
        score = reward.compute_reward("test response", "test prompt", "test reference")
        
        # Verify the mock was called with the right arguments
        mock_reward_fn.assert_called_once_with("test response", "test prompt", "test reference")
        self.assertEqual(score, 0.75)
    
    def test_custom_reward_function_error(self):
        """Test that errors in custom_reward_fn are handled gracefully."""
        # Create a mock custom reward function that raises an exception
        mock_reward_fn = MagicMock(side_effect=Exception("Test error"))
        
        # Create a reward with the mock function
        reward = TaskSpecificReward(
            task_type='custom',
            custom_reward_fn=mock_reward_fn
        )
        
        # Test the reward
        score = reward.compute_reward("test response")
        
        # Should return neutral score on error
        self.assertEqual(score, 0.5)
    
    def test_task_params(self):
        """Test that task_params are used correctly."""
        # Create a summarization reward with custom target ratio
        reward = TaskSpecificReward(
            task_type='summarization',
            task_params={'target_ratio': 0.1}  # Very concise summary
        )
        
        # Original text
        original = "This is a long text that should be summarized very concisely."
        
        # Very concise summary (good for target_ratio=0.1)
        concise_summary = "Text needs concise summary."
        
        # Less concise summary (good for default target_ratio=0.2)
        less_concise = "This text should be summarized concisely."
        
        concise_score = reward._summarization_reward(concise_summary, original)
        less_concise_score = reward._summarization_reward(less_concise, original)
        
        # With target_ratio=0.1, the more concise summary should score higher
        self.assertGreater(concise_score, less_concise_score)
        
        # With default target_ratio=0.2, the less concise summary should score higher
        default_reward = TaskSpecificReward(task_type='summarization')
        default_concise_score = default_reward._summarization_reward(concise_summary, original)
        default_less_concise_score = default_reward._summarization_reward(less_concise, original)
        
        self.assertGreater(default_less_concise_score, default_concise_score)
    
    def test_compute_reward_no_reference(self):
        """Test that compute_reward handles missing references gracefully."""
        # Test with no reference
        score = self.summarization_reward.compute_reward("test response")
        
        # Should return neutral score when no reference is provided
        self.assertEqual(score, 0.5)
    
    def test_compute_reward_mismatched_lengths(self):
        """Test that compute_reward handles mismatched response and reference lengths."""
        # Multiple responses but single reference
        responses = ["response1", "response2", "response3"]
        reference = "reference"
        
        scores = self.summarization_reward.compute_reward(responses, reference=reference)
        
        # Should return a score for each response
        self.assertEqual(len(scores), 3)
        
        # Multiple references but single response
        response = "response"
        references = ["reference1", "reference2", "reference3"]
        
        score = self.summarization_reward.compute_reward(response, reference=references)
        
        # Should return a single score
        self.assertIsInstance(score, float)
    
    def test_unknown_task_type(self):
        """Test that compute_reward handles unknown task types gracefully."""
        # Create a reward with an unknown task type
        reward = TaskSpecificReward(task_type='unknown')
        
        # Test the reward
        score = reward.compute_reward("test response", reference="test reference")
        
        # Should return neutral score for unknown task type
        self.assertEqual(score, 0.5)
    
    def test_batch_processing(self):
        """Test that the reward function handles batch inputs correctly."""
        # Original texts
        originals = [
            "Text 1 that needs to be summarized.",
            "Text 2 that needs to be summarized.",
            "Text 3 that needs to be summarized."
        ]
        
        # Summaries
        summaries = [
            "Summary 1.",
            "Summary 2.",
            "Summary 3."
        ]
        
        scores = self.summarization_reward.compute_reward(summaries, reference=originals)
        
        # Should return a score for each summary
        self.assertEqual(len(scores), 3)


if __name__ == "__main__":
    unittest.main()