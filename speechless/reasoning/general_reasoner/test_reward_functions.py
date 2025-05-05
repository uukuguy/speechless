# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for reward functions.

This module contains unit tests for the reward functions defined in the reward_functions package.
"""

import unittest
import json
import numpy as np
from typing import List, Dict, Any

# Updated imports from the new package structure
from speechless.reasoning.general_reasoner.reward_functions.base import BaseReward
from speechless.reasoning.general_reasoner.reward_functions.text_rewards import LengthReward, FormatReward, CoherenceReward
from speechless.reasoning.general_reasoner.reward_functions.math_rewards import MathReward
from speechless.reasoning.general_reasoner.reward_functions.code_rewards import CodeReward
from speechless.reasoning.general_reasoner.reward_functions.factuality_rewards import FactualityReward
from speechless.reasoning.general_reasoner.reward_functions.task_rewards import TaskSpecificReward
from speechless.reasoning.general_reasoner.reward_functions.tag_rewards import TagReward
from speechless.reasoning.general_reasoner.reward_functions.combined_rewards import CombinedReward
from speechless.reasoning.general_reasoner.reward_functions.utils import create_reward_function


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def encode(self, text):
        """Mock encode method that returns a token count based on word count."""
        return text.split()


class TestBaseReward(unittest.TestCase):
    """Test the BaseReward class."""
    
    def test_ensure_list(self):
        """Test the _ensure_list method."""
        class TestReward(BaseReward):
            def compute_reward(self, response, prompt=None, reference=None, **kwargs):
                return 0.5
        
        reward = TestReward(name="test")
        self.assertEqual(reward._ensure_list("test"), ["test"])
        self.assertEqual(reward._ensure_list(["test"]), ["test"])
    
    def test_normalize_score(self):
        """Test the _normalize_score method."""
        class TestReward(BaseReward):
            def compute_reward(self, response, prompt=None, reference=None, **kwargs):
                return 0.5
        
        reward = TestReward(name="test")
        self.assertEqual(reward._normalize_score(0.5), 0.5)
        self.assertEqual(reward._normalize_score(1.5), 1.0)
        self.assertEqual(reward._normalize_score(-0.5), 0.0)


class TestLengthReward(unittest.TestCase):
    """Test the LengthReward class."""
    
    def test_character_based(self):
        """Test character-based length reward."""
        reward = LengthReward(min_length=10, max_length=50)
        
        # Too short
        self.assertLess(reward.compute_reward("short"), 0.5)
        
        # Too long
        long_text = "a" * 100
        self.assertLess(reward.compute_reward(long_text), 0.5)
        
        # Optimal length
        optimal_text = "a" * 30
        self.assertGreater(reward.compute_reward(optimal_text), 0.5)
    
    def test_token_based(self):
        """Test token-based length reward."""
        tokenizer = MockTokenizer()
        reward = LengthReward(min_length=5, max_length=20, token_based=True, tokenizer=tokenizer)
        
        # Too short
        self.assertLess(reward.compute_reward("short text"), 0.5)
        
        # Too long
        long_text = "this is a very long text that exceeds the maximum token limit set for this test"
        self.assertLess(reward.compute_reward(long_text), 0.5)
        
        # Optimal length
        optimal_text = "this is a text with optimal length for the test"
        self.assertGreater(reward.compute_reward(optimal_text), 0.5)
    
    def test_optimal_length(self):
        """Test length reward with optimal length specified."""
        reward = LengthReward(min_length=10, max_length=100, optimal_length=50)
        
        # Near optimal
        near_optimal = "a" * 55
        self.assertGreater(reward.compute_reward(near_optimal), 0.8)
        
        # Far from optimal
        far_from_optimal = "a" * 90
        self.assertLess(reward.compute_reward(far_from_optimal), 0.8)
    
    def test_batch_processing(self):
        """Test batch processing of responses."""
        reward = LengthReward(min_length=10, max_length=50)
        
        responses = ["short", "a" * 30, "a" * 100]
        rewards = reward.compute_reward(responses)
        
        self.assertEqual(len(rewards), 3)
        self.assertLess(rewards[0], 0.5)  # Too short
        self.assertGreater(rewards[1], 0.5)  # Optimal
        self.assertLess(rewards[2], 0.5)  # Too long


class TestFormatReward(unittest.TestCase):
    """Test the FormatReward class."""
    
    def test_json_format(self):
        """Test JSON format reward."""
        reward = FormatReward(format_type='json')
        
        # Valid JSON
        valid_json = '{"name": "test", "value": 42}'
        self.assertGreater(reward.compute_reward(valid_json), 0.7)
        
        # Invalid JSON
        invalid_json = '{"name": "test", value: 42}'
        self.assertLess(reward.compute_reward(invalid_json), 0.5)
    
    def test_markdown_format(self):
        """Test markdown format reward."""
        reward = FormatReward(format_type='markdown_heading')
        
        # Valid markdown
        valid_markdown = "# Heading\n\nThis is a paragraph."
        self.assertGreater(reward.compute_reward(valid_markdown), 0.7)
        
        # Not markdown
        not_markdown = "This is just plain text."
        self.assertLess(reward.compute_reward(not_markdown), 0.5)
    
    def test_bullet_format(self):
        """Test bullet list format reward."""
        reward = FormatReward(format_type='bullet')
        
        # Valid bullet list
        valid_bullets = "- Item 1\n- Item 2\n- Item 3"
        self.assertGreater(reward.compute_reward(valid_bullets), 0.7)
        
        # Not a bullet list
        not_bullets = "Item 1\nItem 2\nItem 3"
        self.assertLess(reward.compute_reward(not_bullets), 0.5)
    
    def test_custom_regex(self):
        """Test custom regex format reward."""
        # Custom regex for email format
        reward = FormatReward(format_regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
        
        # Valid email
        valid_email = "test@example.com"
        self.assertGreater(reward.compute_reward(valid_email), 0.7)
        
        # Invalid email
        invalid_email = "test@example"
        self.assertLess(reward.compute_reward(invalid_email), 0.5)
    
    def test_format_detection_from_prompt(self):
        """Test format detection from prompt."""
        reward = FormatReward()
        
        # JSON format in prompt
        json_prompt = "Please provide the answer in JSON format."
        json_response = '{"answer": 42}'
        self.assertGreater(reward.compute_reward(json_response, prompt=json_prompt), 0.7)
        
        # Bullet format in prompt
        bullet_prompt = "List the items using bullet points."
        bullet_response = "- Item 1\n- Item 2\n- Item 3"
        self.assertGreater(reward.compute_reward(bullet_response, prompt=bullet_prompt), 0.7)


class TestMathReward(unittest.TestCase):
    """Test the MathReward class."""
    
    def test_answer_extraction(self):
        """Test answer extraction."""
        reward = MathReward()
        
        # Clear answer format
        text = "The answer is 42."
        self.assertEqual(reward._extract_answer(text), 42)
        
        # Different format
        text = "Result: 3.14"
        self.assertEqual(reward._extract_answer(text), 3.14)
        
        # No clear answer
        text = "The solution involves calculus."
        self.assertIsNone(reward._extract_answer(text))
    
    def test_reasoning_evaluation(self):
        """Test reasoning evaluation."""
        reward = MathReward()
        
        # Good reasoning
        good_reasoning = """
        To solve this problem, we need to follow these steps:
        Step 1: Calculate 6 * 7 = 42
        Step 2: Add 10 to get 42 + 10 = 52
        Therefore, the answer is 52.
        """
        self.assertGreater(reward._evaluate_reasoning_steps(good_reasoning), 0.7)
        
        # Poor reasoning
        poor_reasoning = "The answer is 52."
        self.assertLess(reward._evaluate_reasoning_steps(poor_reasoning), 0.5)
    
    def test_answer_correctness(self):
        """Test answer correctness checking."""
        reward = MathReward()
        
        # Exact match
        self.assertEqual(reward._check_answer_correctness(42, 42), 1.0)
        
        # Close match
        self.assertGreater(reward._check_answer_correctness(42.001, 42), 0.8)
        
        # Wrong answer
        self.assertLess(reward._check_answer_correctness(43, 42), 0.5)
    
    def test_complete_reward(self):
        """Test complete math reward."""
        reward = MathReward()
        
        # Good response with correct answer
        good_response = """
        To find the product of 6 and 7, we multiply them:
        6 * 7 = 42
        Therefore, the answer is 42.
        """
        reference_answer = 42
        
        self.assertGreater(reward.compute_reward(good_response, reference_answer=reference_answer), 0.7)
        
        # Good reasoning but wrong answer
        wrong_answer = """
        To find the product of 6 and 7, we multiply them:
        6 * 7 = 43
        Therefore, the answer is 43.
        """
        self.assertLess(reward.compute_reward(wrong_answer, reference_answer=reference_answer), 0.7)


class TestCodeReward(unittest.TestCase):
    """Test the CodeReward class."""
    
    def test_language_detection(self):
        """Test programming language detection."""
        reward = CodeReward()
        
        # Python code
        python_code = """
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n-1)
        """
        self.assertEqual(reward._detect_language(python_code), 'python')
        
        # JavaScript code
        js_code = """
        function factorial(n) {
            if (n <= 1) return 1;
            return n * factorial(n-1);
        }
        """
        self.assertEqual(reward._detect_language(js_code), 'javascript')
    
    def test_syntax_checking(self):
        """Test syntax checking."""
        reward = CodeReward()
        
        # Valid Python syntax
        valid_python = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
        is_valid, score = reward._check_syntax(valid_python, 'python')
        self.assertTrue(is_valid)
        self.assertGreater(score, 0.7)
        
        # Invalid Python syntax
        invalid_python = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1
"""
        is_valid, score = reward._check_syntax(invalid_python, 'python')
        self.assertFalse(is_valid)
        self.assertLess(score, 0.7)
    
    def test_style_checking(self):
        """Test code style checking."""
        reward = CodeReward()
        
        # Good style Python
        good_style = """
def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
        self.assertGreater(reward._check_style(good_style, 'python'), 0.6)
        
        # Poor style Python
        poor_style = """
def factorial(n):
    if n<=1:
     return 1
    return n*factorial(n-1)
"""
        self.assertLess(reward._check_style(poor_style, 'python'), 0.6)
    
    def test_code_extraction(self):
        """Test code block extraction."""
        reward = CodeReward()
        
        # Markdown code block
        markdown = """
Here's a factorial function:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```

This function is recursive.
"""
        code_blocks = reward._extract_code_blocks(markdown, 'python')
        self.assertEqual(len(code_blocks), 1)
        self.assertIn("def factorial", code_blocks[0])
    
    def test_complete_reward(self):
        """Test complete code reward."""
        reward = CodeReward(check_execution=False)
        
        # Good code
        good_code = """
```python
def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)
```
"""
        self.assertGreater(reward.compute_reward(good_code), 0.7)
        
        # Bad code
        bad_code = """
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1
```
"""
        self.assertLess(reward.compute_reward(bad_code), 0.7)


class TestFactualityReward(unittest.TestCase):
    """Test the FactualityReward class."""
    
    def test_fact_extraction(self):
        """Test fact extraction."""
        reward = FactualityReward()
        
        # Text with facts
        text = """
        The Earth is the third planet from the Sun. It has a diameter of approximately 12,742 kilometers.
        Water covers about 71% of the Earth's surface. The Earth's atmosphere is composed primarily of nitrogen and oxygen.
        """
        facts = reward._extract_facts(text)
        self.assertGreaterEqual(len(facts), 3)
        
        # Text without clear facts
        text = "What is the Earth like? How big is it? Where is it located?"
        facts = reward._extract_facts(text)
        self.assertEqual(len(facts), 0)
    
    def test_fact_support(self):
        """Test fact support checking."""
        reward = FactualityReward()
        
        fact = "The Earth is the third planet from the Sun."
        references = [
            "Our solar system has eight planets. Mercury is first, Venus is second, and Earth is third from the Sun.",
            "Mars is the fourth planet from the Sun, after Earth."
        ]
        
        support = reward._check_fact_support(fact, references)
        self.assertGreater(support, 0.5)
        
        # Unsupported fact
        fact = "The Earth is flat."
        support = reward._check_fact_support(fact, references)
        self.assertLess(support, 0.5)
    
    def test_contradiction_detection(self):
        """Test contradiction detection."""
        reward = FactualityReward()
        
        facts = ["The Earth is the third planet from the Sun.", "The Earth has one natural satellite."]
        references = [
            "The Earth is the third planet from the Sun.",
            "The Moon is Earth's only natural satellite."
        ]
        
        penalty = reward._detect_contradictions(facts, references)
        self.assertGreater(penalty, 0.7)  # No contradictions
        
        # With contradictions
        facts = ["The Earth is the third planet from the Sun.", "The Earth has no natural satellites."]
        penalty = reward._detect_contradictions(facts, references)
        self.assertLess(penalty, 0.7)  # Contradiction detected
    
    def test_complete_reward(self):
        """Test complete factuality reward."""
        references = [
            "The Earth is the third planet from the Sun. It has a diameter of approximately 12,742 kilometers.",
            "The Moon is Earth's only natural satellite. It has a diameter of about 3,474 kilometers."
        ]
        
        reward = FactualityReward(reference_texts=references)
        
        # Factual response
        factual = "The Earth is the third planet from the Sun and has one moon."
        self.assertGreater(reward.compute_reward(factual), 0.6)
        
        # Non-factual response
        non_factual = "The Earth is the fifth planet from the Sun and has three moons."
        self.assertLess(reward.compute_reward(non_factual), 0.6)


class TestCoherenceReward(unittest.TestCase):
    """Test the CoherenceReward class."""
    
    def test_logical_flow(self):
        """Test logical flow checking."""
        reward = CoherenceReward()
        
        # Good logical flow
        good_flow = """
        First, we need to understand the problem. The issue is related to climate change.
        
        Second, we should analyze the causes. Human activities are the primary driver of climate change.
        
        Finally, we can discuss solutions. Reducing carbon emissions is essential to mitigate climate change.
        """
        self.assertGreater(reward._check_logical_flow(good_flow), 0.7)
        
        # Poor logical flow
        poor_flow = """
        Climate change is a problem. Carbon emissions. We need solutions now. The temperature is rising.
        """
        self.assertLess(reward._check_logical_flow(poor_flow), 0.7)
    
    def test_consistency(self):
        """Test consistency checking."""
        reward = CoherenceReward()
        
        # Consistent text
        consistent = """
        The Earth orbits the Sun. It takes approximately 365 days to complete one orbit.
        The Moon orbits the Earth and takes about 27 days to complete one orbit.
        """
        self.assertGreater(reward._check_consistency(consistent), 0.7)
        
        # Inconsistent text
        inconsistent = """
        The Earth orbits the Sun. It takes approximately 365 days to complete one orbit.
        The Earth takes 30 days to orbit the Sun. The Moon orbits the Earth.
        """
        self.assertLess(reward._check_consistency(inconsistent), 0.7)
    
    def test_clarity(self):
        """Test clarity checking."""
        reward = CoherenceReward()
        
        # Clear text
        clear = """
        Climate change is a global problem. It is caused by greenhouse gas emissions.
        These emissions trap heat in the atmosphere. This leads to rising temperatures.
        """
        self.assertGreater(reward._check_clarity(clear), 0.7)
        
        # Unclear text
        unclear = """
        Climate change global issue greenhouse gases atmospheric heat retention temperature elevation
        problematic consequences require mitigation strategies implementation urgency.
        """
        self.assertLess(reward._check_clarity(unclear), 0.7)
    
    def test_complete_reward(self):
        """Test complete coherence reward."""
        reward = CoherenceReward()
        
        # Coherent response
        coherent = """
        Climate change is a global challenge. It is primarily caused by human activities that release greenhouse gases.
        
        First, burning fossil fuels releases carbon dioxide. Second, deforestation reduces carbon absorption.
        
        To address this issue, we need to reduce emissions and increase renewable energy use.
        """
        self.assertGreater(reward.compute_reward(coherent), 0.7)
        
        # Incoherent response
        incoherent = """
        Climate change problem. Greenhouse gases. Temperature rising. Ice melting. Weather extreme.
        Carbon dioxide. Fossil fuels. Renewable energy. Trees important. Oceans warming.
        """
        self.assertLess(reward.compute_reward(incoherent), 0.7)


class TestTaskSpecificReward(unittest.TestCase):
    """Test the TaskSpecificReward class."""
    
    def test_summarization_reward(self):
        """Test summarization reward."""
        reward = TaskSpecificReward(task_type='summarization')
        
        # Original text
        original = """
        Machine learning is a field of study in artificial intelligence concerned with the development
        of algorithms and statistical models that computer systems use to perform tasks without explicit
        instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence.
        Machine learning algorithms build a mathematical model based on sample data, known as "training data",
        in order to make predictions or decisions without being explicitly programmed to perform the task.
        """
        
        # Good summary
        good_summary = "Machine learning is an AI field that develops algorithms allowing computers to perform tasks using patterns and inference rather than explicit instructions."
        self.assertGreater(reward._summarization_reward(good_summary, original), 0.7)
        
        # Bad summary (too short)
        bad_summary = "Machine learning uses algorithms."
        self.assertLess(reward._summarization_reward(bad_summary, original), 0.7)
        
        # Bad summary (too verbose)
        verbose_summary = original  # Just repeating the original
        self.assertLess(reward._summarization_reward(verbose_summary, original), 0.7)
    
    def test_qa_reward(self):
        """Test QA reward."""
        reward = TaskSpecificReward(task_type='qa')
        
        # Reference answer
        reference = "Paris is the capital of France."
        
        # Correct answer
        correct = "The capital of France is Paris."
        self.assertGreater(reward._qa_reward(correct, reference), 0.7)
        
        # Wrong answer
        wrong = "The capital of France is Lyon."
        self.assertLess(reward._qa_reward(wrong, reference), 0.7)
    
    def test_custom_reward_function(self):
        """Test custom reward function."""
        def custom_fn(response, prompt=None, reference=None, **kwargs):
            """Simple custom reward function that rewards longer responses."""
            return min(1.0, len(response) / 100)
        
        reward = TaskSpecificReward(task_type='custom', custom_reward_fn=custom_fn)
        
        # Short response
        short = "Short response."
        self.assertLess(reward.compute_reward(short), 0.5)
        
        # Long response
        long = "This is a much longer response that should receive a higher score from our custom reward function."
        self.assertGreater(reward.compute_reward(long), 0.5)


class TestCombinedReward(unittest.TestCase):
    """Test the CombinedReward class."""
    
    def test_combined_reward(self):
        """Test combined reward with equal weights."""
        length_reward = LengthReward(min_length=10, max_length=50)
        format_reward = FormatReward(format_type='json')
        
        combined = CombinedReward(
            reward_functions=[length_reward, format_reward],
            weights=[0.5, 0.5]
        )
        
        # Good length, bad format
        response1 = "This is a good length but not JSON format."
        
        # Bad length, good format
        response2 = '{"short":true}'
        
        # Good length, good format
        response3 = '{"content":"This is a good length and proper JSON format.","value":42}'
        
        scores = combined.compute_reward([response1, response2, response3])
        
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[2], scores[0])  # Good on both should be better than good on one
        self.assertGreater(scores[2], scores[1])  # Good on both should be better than good on one
    
    def test_weighted_combined_reward(self):
        """Test combined reward with custom weights."""
        length_reward = LengthReward(min_length=10, max_length=50)
        format_reward = FormatReward(format_type='json')
        
        # Length-focused combined reward
        length_focused = CombinedReward(
            reward_functions=[length_reward, format_reward],
            weights=[0.8, 0.2]
        )
        
        # Format-focused combined reward
        format_focused = CombinedReward(
            reward_functions=[length_reward, format_reward],
            weights=[0.2, 0.8]
        )
        
        # Good length, bad format
        response1 = "This is a good length but not JSON format."
        
        # Bad length, good format
        response2 = '{"short":true}'
        
        length_score1 = length_focused.compute_reward(response1)
        length_score2 = length_focused.compute_reward(response2)
        
        format_score1 = format_focused.compute_reward(response1)
        format_score2 = format_focused.compute_reward(response2)
        
        # Length-focused should prefer response1
        self.assertGreater(length_score1, length_score2)
        
        # Format-focused should prefer response2
        self.assertGreater(format_score2, format_score1)


class TestCreateRewardFunction(unittest.TestCase):
    """Test the create_reward_function utility."""
    
    def test_create_length_reward(self):
        """Test creating a LengthReward from configuration."""
        config = {
            'type': 'length',
            'min_length': 10,
            'max_length': 50,
            'weight': 2.0
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, LengthReward)
        self.assertEqual(reward.min_length, 10)
        self.assertEqual(reward.max_length, 50)
        self.assertEqual(reward.weight, 2.0)
    
    def test_create_combined_reward(self):
        """Test creating a CombinedReward from configuration."""
        config = {
            'type': 'combined',
            'reward_functions': [
                {
                    'type': 'length',
                    'min_length': 10,
                    'max_length': 50,
                    'weight': 1.0
                },
                {
                    'type': 'format',
                    'format_type': 'json',
                    'weight': 2.0
                }
            ]
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, CombinedReward)
        self.assertEqual(len(reward.reward_functions), 2)
        self.assertIsInstance(reward.reward_functions[0], LengthReward)
        self.assertIsInstance(reward.reward_functions[1], FormatReward)
    
    def test_invalid_reward_type(self):
        """Test error handling for invalid reward type."""
        config = {
            'type': 'invalid_type'
        }
        
        with self.assertRaises(ValueError):
            create_reward_function(config)


class TestTagReward(unittest.TestCase):
    """Test the TagReward class."""
    
    def test_basic_tag_compliance(self):
        """Test basic tag compliance."""
        tag_specs = {
            'thinking': {'required': True},
            'answer': {'required': True}
        }
        
        reward = TagReward(tag_specs=tag_specs)
        
        # Compliant response
        compliant = "<thinking>This is a math problem. 6 * 7 = 42</thinking><answer>42</answer>"
        self.assertGreater(reward.compute_reward(compliant), 0.8)
        
        # Missing required tag
        missing_tag = "<thinking>This is a math problem. 6 * 7 = 42</thinking>"
        self.assertLess(reward.compute_reward(missing_tag), 0.5)
        
        # Empty tag content
        empty_content = "<thinking></thinking><answer>42</answer>"
        self.assertLess(reward.compute_reward(empty_content), 0.8)
    
    def test_nested_tags(self):
        """Test nested tags."""
        tag_specs = {
            'thinking': {'required': True},
            'code': {'required': False},
            'answer': {'required': True}
        }
        
        reward = TagReward(tag_specs=tag_specs)
        
        # Properly nested tags
        nested = """
        <thinking>
            This is a math problem.
            <code>6 * 7 = 42</code>
        </thinking>
        <answer>42</answer>
        """
        self.assertGreater(reward.compute_reward(nested), 0.8)
        
        # Improperly nested tags with strict_nesting=True (default)
        improper = """
        <thinking>
            This is a math problem.
            <code>6 * 7 = 42
        </thinking>
        </code>
        <answer>42</answer>
        """
        self.assertLess(reward.compute_reward(improper), 0.5)
        
        # Improperly nested tags with strict_nesting=False
        lenient_reward = TagReward(tag_specs=tag_specs, strict_nesting=False)
        self.assertGreater(lenient_reward.compute_reward(improper), 0.5)
    
    def test_content_requirements(self):
        """Test content requirements."""
        tag_specs = {
            'thinking': {'required': True, 'content_required': True},
            'answer': {'required': True, 'content_regex': r'^\d+$'}
        }
        
        reward = TagReward(tag_specs=tag_specs)
        
        # Valid content
        valid = "<thinking>This is a math problem. 6 * 7 = 42</thinking><answer>42</answer>"
        self.assertGreater(reward.compute_reward(valid), 0.8)
        
        # Invalid content (non-numeric answer)
        invalid = "<thinking>This is a math problem. 6 * 7 = 42</thinking><answer>forty-two</answer>"
        self.assertLess(reward.compute_reward(invalid), 0.8)