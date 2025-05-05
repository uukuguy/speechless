"""
Tests for Text-Based Reward Functions

This module contains unit tests for the text-based reward functions:
- LengthReward
- FormatReward
- CoherenceReward
"""

import unittest
from unittest.mock import MagicMock

from ..text_rewards import LengthReward, FormatReward, CoherenceReward


class TestLengthReward(unittest.TestCase):
    """Test cases for the LengthReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward = LengthReward(min_length=10, max_length=50)
        
        # Create a mock tokenizer for token-based tests
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode = lambda text: [0] * len(text.split())  # Simple word count
        self.token_reward = LengthReward(
            min_length=5, 
            max_length=20, 
            token_based=True, 
            tokenizer=self.mock_tokenizer
        )
    
    def test_too_short(self):
        """Test that responses shorter than min_length are penalized."""
        short_response = "Too short"  # 2 words, 9 chars
        score = self.reward(short_response)
        self.assertLess(score, 0.5)  # Should be penalized
    
    def test_too_long(self):
        """Test that responses longer than max_length are penalized."""
        long_response = "This is a very long response that exceeds the maximum length limit set for this test case."
        score = self.reward(long_response)
        self.assertLess(score, 0.5)  # Should be penalized
    
    def test_optimal_length(self):
        """Test that responses within range get good scores."""
        good_response = "This is a good length response."
        score = self.reward(good_response)
        self.assertGreaterEqual(score, 0.5)  # Should be rewarded
    
    def test_optimal_length_specified(self):
        """Test that responses close to optimal_length get the best scores."""
        reward_with_optimal = LengthReward(min_length=10, max_length=50, optimal_length=30)
        
        # Test responses at different distances from optimal
        at_optimal = "x" * 30
        near_optimal = "x" * 35
        far_from_optimal = "x" * 49
        
        score_optimal = reward_with_optimal(at_optimal)
        score_near = reward_with_optimal(near_optimal)
        score_far = reward_with_optimal(far_from_optimal)
        
        self.assertGreater(score_optimal, score_far)
        self.assertGreaterEqual(score_optimal, score_near)
    
    def test_token_based(self):
        """Test token-based length calculation."""
        # This has 7 words/tokens
        response = "This is a test of token based reward"
        score = self.token_reward(response)
        
        # Should be in the good range (5-20 tokens)
        self.assertGreaterEqual(score, 0.5)
        
        # Test with too many tokens
        long_response = "This is a very long response with many tokens that should exceed our maximum token limit for this test case and therefore receive a lower score"
        score = self.token_reward(long_response)
        self.assertLess(score, 0.5)


class TestFormatReward(unittest.TestCase):
    """Test cases for the FormatReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.json_reward = FormatReward(format_type='json')
        self.bullet_reward = FormatReward(format_type='bullet')
        self.custom_reward = FormatReward(format_regex=r'^Title:\s+.*\nContent:\s+.*$')
    
    def test_json_format(self):
        """Test JSON format detection and scoring."""
        valid_json = '{"key": "value", "number": 42}'
        invalid_json = '{key: value}'
        
        valid_score = self.json_reward(valid_json)
        invalid_score = self.json_reward(invalid_json)
        
        self.assertGreater(valid_score, 0.8)  # Valid JSON should get high score
        self.assertLess(invalid_score, 0.5)   # Invalid JSON should get low score
    
    def test_bullet_format(self):
        """Test bullet list format detection and scoring."""
        valid_bullets = "- Item 1\n- Item 2\n- Item 3"
        invalid_bullets = "Item 1\nItem 2\nItem 3"
        
        valid_score = self.bullet_reward(valid_bullets)
        invalid_score = self.bullet_reward(invalid_bullets)
        
        self.assertGreater(valid_score, 0.8)  # Valid bullets should get high score
        self.assertLess(invalid_score, 0.5)   # Invalid bullets should get low score
    
    def test_custom_format(self):
        """Test custom regex format detection and scoring."""
        valid_format = "Title: My Document\nContent: This is the content."
        invalid_format = "My Document\nThis is the content."
        
        valid_score = self.custom_reward(valid_format)
        invalid_score = self.custom_reward(invalid_format)
        
        self.assertGreater(valid_score, 0.8)  # Valid format should get high score
        self.assertLess(invalid_score, 0.5)   # Invalid format should get low score
    
    def test_format_detection_from_prompt(self):
        """Test format detection from prompt."""
        format_reward = FormatReward()  # No format specified
        
        # Test with JSON prompt
        json_prompt = "Please respond in JSON format."
        json_response = '{"result": "success"}'
        
        # Test with bullet prompt
        bullet_prompt = "List the items using bullet points."
        bullet_response = "- Item 1\n- Item 2"
        
        json_score = format_reward(json_response, prompt=json_prompt)
        bullet_score = format_reward(bullet_response, prompt=bullet_prompt)
        
        self.assertGreater(json_score, 0.5)
        self.assertGreater(bullet_score, 0.5)


class TestCoherenceReward(unittest.TestCase):
    """Test cases for the CoherenceReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward = CoherenceReward()
    
    def test_logical_flow(self):
        """Test logical flow detection and scoring."""
        good_flow = """
        First, we need to understand the problem. 
        Second, we should analyze the available data.
        Finally, we can draw conclusions based on our analysis.
        Therefore, the answer is 42.
        """
        
        bad_flow = """
        The answer is 42.
        Data shows various patterns.
        Problem statement is unclear.
        """
        
        # Create a reward that only checks logical flow
        flow_reward = CoherenceReward(
            check_logical_flow=True,
            check_consistency=False,
            check_clarity=False
        )
        
        good_score = flow_reward(good_flow)
        bad_score = flow_reward(bad_flow)
        
        self.assertGreater(good_score, bad_score)
    
    def test_consistency(self):
        """Test consistency detection and scoring."""
        consistent = """
        The Earth orbits the Sun. This journey takes 365.25 days.
        The Moon orbits the Earth in approximately 27.3 days.
        Both orbits are elliptical rather than perfectly circular.
        """
        
        inconsistent = """
        The Earth orbits the Sun. This journey takes 365.25 days.
        The Earth does not orbit the Sun. It is stationary.
        The Moon orbits the Earth in approximately 27.3 days.
        """
        
        # Create a reward that only checks consistency
        consistency_reward = CoherenceReward(
            check_logical_flow=False,
            check_consistency=True,
            check_clarity=False
        )
        
        consistent_score = consistency_reward(consistent)
        inconsistent_score = consistency_reward(inconsistent)
        
        self.assertGreater(consistent_score, inconsistent_score)
    
    def test_clarity(self):
        """Test clarity detection and scoring."""
        clear = """
        The water cycle has four main stages. First, water evaporates from oceans and lakes.
        Then, it condenses to form clouds. Next, precipitation occurs as rain or snow.
        Finally, the water returns to oceans and lakes through rivers.
        """
        
        unclear = """
        The hydrological cycle's initial phase involves the transformation of liquid dihydrogen monoxide 
        to its gaseous state via solar radiation absorption, subsequently followed by the atmospheric 
        condensation of said gaseous molecules into visible, suspended aqueous formations, ultimately 
        culminating in the gravitational descent of the aforementioned condensed matter.
        """
        
        # Create a reward that only checks clarity
        clarity_reward = CoherenceReward(
            check_logical_flow=False,
            check_consistency=False,
            check_clarity=True
        )
        
        clear_score = clarity_reward(clear)
        unclear_score = clarity_reward(unclear)
        
        self.assertGreater(clear_score, unclear_score)
    
    def test_combined_coherence(self):
        """Test combined coherence scoring."""
        good_text = """
        Introduction: This essay discusses climate change.
        
        First, we'll examine the causes of climate change. The primary cause is the emission
        of greenhouse gases from human activities. These gases trap heat in the atmosphere.
        
        Second, we'll look at the effects. Climate change leads to rising sea levels,
        extreme weather events, and disruptions to ecosystems.
        
        Finally, we'll consider solutions. Reducing emissions, developing renewable energy,
        and adapting to changes are all important strategies.
        
        In conclusion, climate change is a serious issue that requires global cooperation.
        """
        
        bad_text = """
        Climate change is bad. It's caused by stuff we do. The climate is not changing
        according to some people. The weather gets weird. The climate is definitely changing
        based on data. Solutions are hard but necessary. Some solutions might work but
        others won't. The end.
        """
        
        good_score = self.reward(good_text)
        bad_score = self.reward(bad_text)
        
        self.assertGreater(good_score, bad_score)


if __name__ == "__main__":
    unittest.main()