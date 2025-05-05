"""
Tests for Tag-Based Reward Functions

This module contains unit tests for the tag-based reward functions:
- TagReward
"""

import unittest

from ..tag_rewards import TagReward


class TestTagReward(unittest.TestCase):
    """Test cases for the TagReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tag_specs = {
            'thinking': {'required': True, 'min_count': 1, 'max_count': 1},
            'answer': {'required': True, 'min_count': 1, 'max_count': 1},
            'code': {'required': False, 'max_count': 3}
        }
        self.reward = TagReward(tag_specs=self.tag_specs)
    
    def test_find_tag_pairs(self):
        """Test that _find_tag_pairs correctly identifies tag pairs."""
        # Text with all required tags
        text = """
        <thinking>This is a math problem. 6 * 7 = 42</thinking>
        <answer>42</answer>
        """
        
        tag_pairs = self.reward._find_tag_pairs(text)
        
        self.assertIn('thinking', tag_pairs)
        self.assertIn('answer', tag_pairs)
        self.assertEqual(len(tag_pairs['thinking']), 1)
        self.assertEqual(len(tag_pairs['answer']), 1)
        self.assertEqual(tag_pairs['thinking'][0][2], "This is a math problem. 6 * 7 = 42")
        self.assertEqual(tag_pairs['answer'][0][2], "42")
        
        # Text with multiple instances of a tag
        text = """
        <code>function hello() { return 'world'; }</code>
        <code>print("Hello, world!")</code>
        """
        
        tag_pairs = self.reward._find_tag_pairs(text)
        
        self.assertIn('code', tag_pairs)
        self.assertEqual(len(tag_pairs['code']), 2)
        
        # Text with no tags
        text = "This text has no tags."
        
        tag_pairs = self.reward._find_tag_pairs(text)
        
        self.assertEqual(len(tag_pairs['thinking']), 0)
        self.assertEqual(len(tag_pairs['answer']), 0)
        self.assertEqual(len(tag_pairs['code']), 0)
        
        # Text with unclosed tags (should not be found)
        text = "<thinking>This tag is not closed"
        
        tag_pairs = self.reward._find_tag_pairs(text)
        
        self.assertEqual(len(tag_pairs['thinking']), 0)
    
    def test_check_nesting(self):
        """Test that _check_nesting correctly validates tag nesting."""
        # Properly nested tags
        text = """
        <thinking>
            This is a math problem.
            <code>6 * 7 = 42</code>
        </thinking>
        <answer>42</answer>
        """
        
        tag_pairs = self.reward._find_tag_pairs(text)
        self.assertTrue(self.reward._check_nesting(text, tag_pairs))
        
        # Improperly nested tags
        text = """
        <thinking>
            This is a math problem.
            <code>6 * 7 = 42
        </thinking>
        </code>
        <answer>42</answer>
        """
        
        # For this test, we need to manually create tag pairs since _find_tag_pairs
        # won't find improperly nested tags
        thinking_tag = (text.find("<thinking>"), text.find("</thinking>") + len("</thinking>"), 
                        text[text.find("<thinking>") + len("<thinking>"):text.find("</thinking>")])
        code_tag = (text.find("<code>"), text.find("</code>") + len("</code>"), 
                    text[text.find("<code>") + len("<code>"):text.find("</code>")])
        answer_tag = (text.find("<answer>"), text.find("</answer>") + len("</answer>"), 
                      text[text.find("<answer>") + len("<answer>"):text.find("</answer>")])
        
        tag_pairs = {
            'thinking': [thinking_tag],
            'code': [code_tag],
            'answer': [answer_tag]
        }
        
        # This should fail because the tags are not properly nested
        self.assertFalse(self.reward._check_nesting(text, tag_pairs))
    
    def test_evaluate_tag_compliance(self):
        """Test that _evaluate_tag_compliance correctly scores tag usage."""
        # Perfect compliance
        text = """
        <thinking>This is a math problem. 6 * 7 = 42</thinking>
        <answer>42</answer>
        """
        
        score, details = self.reward._evaluate_tag_compliance(text)
        self.assertGreaterEqual(score, 0.9)  # Should get high score
        
        # Missing required tag
        text = """
        <thinking>This is a math problem. 6 * 7 = 42</thinking>
        """
        
        score, details = self.reward._evaluate_tag_compliance(text)
        self.assertLess(score, 0.5)  # Should get low score
        self.assertIn('answer', details['tag_scores'])
        self.assertEqual(details['tag_scores']['answer']['score'], 0.0)
        
        # Too many instances of a tag
        text = """
        <thinking>This is a math problem.</thinking>
        <thinking>6 * 7 = 42</thinking>
        <answer>42</answer>
        """
        
        score, details = self.reward._evaluate_tag_compliance(text)
        self.assertLess(score, 0.9)  # Should get reduced score
        self.assertIn('thinking', details['tag_scores'])
        self.assertLess(details['tag_scores']['thinking']['score'], 1.0)
        
        # Empty content in tag
        text = """
        <thinking></thinking>
        <answer>42</answer>
        """
        
        score, details = self.reward._evaluate_tag_compliance(text)
        self.assertLess(score, 0.9)  # Should get reduced score
        
        # Optional tag present
        text = """
        <thinking>This is a math problem. 6 * 7 = 42</thinking>
        <answer>42</answer>
        <code>print(6 * 7)</code>
        """
        
        score, details = self.reward._evaluate_tag_compliance(text)
        self.assertGreaterEqual(score, 0.9)  # Should still get high score
        self.assertIn('code', details['tag_scores'])
        self.assertGreaterEqual(details['tag_scores']['code']['score'], 0.9)
    
    def test_strict_nesting_option(self):
        """Test that strict_nesting option works correctly."""
        # Create a reward with strict_nesting=False
        reward = TagReward(tag_specs=self.tag_specs, strict_nesting=False)
        
        # Improperly nested tags
        text = """
        <thinking>
            This is a math problem.
            <code>6 * 7 = 42
        </thinking>
        </code>
        <answer>42</answer>
        """
        
        # This should still give a score even with improper nesting
        score, _ = reward._evaluate_tag_compliance(text)
        self.assertGreater(score, 0.0)
        
        # With strict_nesting=True (default), this should fail
        score, _ = self.reward._evaluate_tag_compliance(text)
        self.assertEqual(score, 0.0)
    
    def test_content_regex(self):
        """Test that content_regex specification works correctly."""
        # Create a reward with content_regex for the answer tag
        tag_specs = {
            'thinking': {'required': True},
            'answer': {'required': True, 'content_regex': r'^\d+$'}  # Only digits
        }
        reward = TagReward(tag_specs=tag_specs)
        
        # Valid content
        text = """
        <thinking>This is a math problem.</thinking>
        <answer>42</answer>
        """
        
        score, details = reward._evaluate_tag_compliance(text)
        self.assertGreaterEqual(score, 0.9)
        self.assertGreaterEqual(details['tag_scores']['answer']['score'], 0.9)
        
        # Invalid content
        text = """
        <thinking>This is a math problem.</thinking>
        <answer>forty-two</answer>
        """
        
        score, details = reward._evaluate_tag_compliance(text)
        self.assertLess(score, 0.9)
        self.assertLess(details['tag_scores']['answer']['score'], 0.9)
    
    def test_compute_reward(self):
        """Test that compute_reward correctly evaluates tag compliance."""
        # Perfect compliance
        text = """
        <thinking>This is a math problem. 6 * 7 = 42</thinking>
        <answer>42</answer>
        """
        
        score = self.reward.compute_reward(text)
        self.assertGreaterEqual(score, 0.9)
        
        # Missing required tag
        text = """
        <thinking>This is a math problem. 6 * 7 = 42</thinking>
        """
        
        score = self.reward.compute_reward(text)
        self.assertLess(score, 0.5)
    
    def test_batch_processing(self):
        """Test that the reward function handles batch inputs correctly."""
        responses = [
            "<thinking>Problem 1</thinking><answer>42</answer>",
            "<thinking>Problem 2</thinking>",  # Missing answer tag
            "<thinking>Problem 3</thinking><answer>100</answer><code>print(100)</code>"
        ]
        
        scores = self.reward.compute_reward(responses)
        
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], scores[1])  # Complete should score higher than incomplete
        self.assertGreaterEqual(scores[2], scores[0])  # Optional tags should not reduce score


if __name__ == "__main__":
    unittest.main()