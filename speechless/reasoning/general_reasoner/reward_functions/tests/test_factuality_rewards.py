"""
Tests for Factuality Reward Functions

This module contains unit tests for the factuality reward functions:
- FactualityReward
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from ..factuality_rewards import FactualityReward


class TestFactualityReward(unittest.TestCase):
    """Test cases for the FactualityReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reference_texts = [
            "The Earth is the third planet from the Sun.",
            "Water boils at 100 degrees Celsius at standard pressure.",
            "Paris is the capital of France."
        ]
        self.reward = FactualityReward(reference_texts=self.reference_texts)
    
    def test_extract_facts(self):
        """Test that _extract_facts correctly extracts factual statements."""
        # Text with multiple facts
        text = """
        The Earth is the third planet from the Sun. It has one natural satellite called the Moon.
        Water freezes at 0 degrees Celsius. Is that interesting?
        Paris has a population of over 2 million people in 2023.
        """
        
        facts = self.reward._extract_facts(text)
        
        self.assertEqual(len(facts), 3)  # Should extract 3 facts
        self.assertIn("The Earth is the third planet from the Sun", facts[0])
        self.assertIn("Water freezes at 0 degrees Celsius", facts[1])
        self.assertIn("Paris has a population of over 2 million people in 2023", facts[2])
        
        # Text with questions and short statements (should be filtered out)
        text = "Is the Earth flat? No. Yes."
        facts = self.reward._extract_facts(text)
        self.assertEqual(len(facts), 0)  # Should extract 0 facts
    
    def test_compute_similarity_text_based(self):
        """Test that _compute_similarity correctly computes text-based similarity."""
        # Identical texts
        similarity = self.reward._compute_similarity(
            "The Earth is the third planet from the Sun.",
            "The Earth is the third planet from the Sun."
        )
        self.assertEqual(similarity, 1.0)
        
        # Similar texts
        similarity = self.reward._compute_similarity(
            "The Earth is the third planet from the Sun.",
            "Earth is the third planet from our Sun."
        )
        self.assertGreater(similarity, 0.5)
        
        # Different texts
        similarity = self.reward._compute_similarity(
            "The Earth is the third planet from the Sun.",
            "Jupiter is the fifth planet from the Sun."
        )
        self.assertLess(similarity, 0.5)
    
    @patch('numpy.dot')
    @patch('numpy.linalg.norm')
    def test_compute_similarity_embedding_based(self, mock_norm, mock_dot):
        """Test that _compute_similarity correctly computes embedding-based similarity."""
        # Create a mock embedding model
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts: np.array([[1.0, 0.0]] * len(texts))
        
        # Setup mocks for numpy functions
        mock_dot.return_value = 0.8
        mock_norm.return_value = 1.0
        
        # Create a reward with embeddings
        reward = FactualityReward(
            reference_texts=self.reference_texts,
            use_embeddings=True,
            embedding_model=mock_model
        )
        
        # Test similarity computation
        similarity = reward._compute_similarity("text1", "text2")
        
        self.assertEqual(similarity, 0.8)
        mock_model.encode.assert_called()
        mock_dot.assert_called_once()
        self.assertEqual(mock_norm.call_count, 2)
    
    def test_check_fact_support(self):
        """Test that _check_fact_support correctly evaluates fact support."""
        # Fact that matches a reference exactly
        support = self.reward._check_fact_support(
            "The Earth is the third planet from the Sun.",
            self.reference_texts
        )
        self.assertEqual(support, 1.0)
        
        # Fact that partially matches a reference
        support = self.reward._check_fact_support(
            "Earth is the third planet from the Sun.",
            self.reference_texts
        )
        self.assertGreater(support, 0.5)
        
        # Fact that doesn't match any reference
        support = self.reward._check_fact_support(
            "Jupiter has 79 known moons.",
            self.reference_texts
        )
        self.assertLess(support, 0.5)
    
    def test_detect_contradictions(self):
        """Test that _detect_contradictions correctly identifies contradictions."""
        # Facts with no contradictions
        facts = [
            "The Earth is the third planet from the Sun.",
            "Water boils at 100 degrees Celsius at standard pressure."
        ]
        
        penalty = self.reward._detect_contradictions(facts, self.reference_texts)
        self.assertEqual(penalty, 1.0)  # No contradictions should give full score
        
        # Facts with contradictions
        facts = [
            "The Earth is not the third planet from the Sun.",
            "Water boils at 100 degrees Celsius at standard pressure."
        ]
        
        # Add a contradictory reference
        references = self.reference_texts + ["The Earth is not the third planet from the Sun."]
        
        penalty = self.reward._detect_contradictions(facts, references)
        self.assertLess(penalty, 1.0)  # Contradictions should reduce the score
    
    def test_compute_reward(self):
        """Test that compute_reward correctly evaluates factual accuracy."""
        # Factual response
        factual_response = """
        The Earth is the third planet from the Sun.
        Water boils at 100 degrees Celsius at standard pressure.
        Paris is the capital of France.
        """
        
        # Non-factual response
        non_factual_response = """
        The Earth is the fourth planet from the Sun.
        Water boils at 90 degrees Celsius at standard pressure.
        Berlin is the capital of France.
        """
        
        # Mixed response
        mixed_response = """
        The Earth is the third planet from the Sun.
        Water boils at 90 degrees Celsius at standard pressure.
        Paris is the capital of France.
        """
        
        factual_score = self.reward.compute_reward(factual_response)
        non_factual_score = self.reward.compute_reward(non_factual_response)
        mixed_score = self.reward.compute_reward(mixed_response)
        
        self.assertGreater(factual_score, non_factual_score)
        self.assertGreater(factual_score, mixed_score)
        self.assertGreater(mixed_score, non_factual_score)
    
    def test_compute_reward_with_additional_references(self):
        """Test that compute_reward correctly uses additional references."""
        # Create a reward with no pre-configured references
        reward = FactualityReward()
        
        # Response with a fact
        response = "The Earth is the third planet from the Sun."
        
        # Test with no references
        score_no_ref = reward.compute_reward(response)
        self.assertEqual(score_no_ref, 0.5)  # Should return neutral score
        
        # Test with provided reference
        score_with_ref = reward.compute_reward(
            response, 
            reference="The Earth is the third planet from the Sun."
        )
        self.assertGreater(score_with_ref, 0.5)  # Should return higher score
    
    def test_batch_processing(self):
        """Test that the reward function handles batch inputs correctly."""
        responses = [
            "The Earth is the third planet from the Sun.",
            "The Earth is the fourth planet from the Sun.",
            "Paris is the capital of France."
        ]
        
        scores = self.reward.compute_reward(responses)
        
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], scores[1])  # Factual should score higher than non-factual
        self.assertGreater(scores[2], scores[1])  # Factual should score higher than non-factual


if __name__ == "__main__":
    unittest.main()