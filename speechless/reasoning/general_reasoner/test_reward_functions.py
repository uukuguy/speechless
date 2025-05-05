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

This module contains unit tests for the reward functions defined in reward_functions.py.
"""

import unittest
import json
import numpy as np
from typing import List, Dict, Any

from reward_functions import (
    BaseReward, LengthReward, FormatReward, MathReward, CodeReward,
    FactualityReward, CoherenceReward, TaskSpecificReward, TagReward, CombinedReward,
    create_reward_function
)


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
        Climate change is a serious issue. It is caused by human activities that release greenhouse gases.
        These gases trap heat in the atmosphere, leading to global warming. We need to reduce emissions.
        """
        self.assertGreater(reward._check_clarity(clear), 0.7)
        
        # Unclear text
        unclear = """
        The aforementioned anthropogenic climatological perturbations necessitate immediate multilateral
        intergovernmental interventions to mitigate the deleterious ramifications on global ecosystems
        and socioeconomic infrastructures.
        """
        self.assertLess(reward._check_clarity(unclear), 0.7)
    
    def test_complete_reward(self):
        """Test complete coherence reward."""
        reward = CoherenceReward()
        
        # Coherent response
        coherent = """
        First, let's understand what climate change is. Climate change refers to long-term shifts in temperatures and weather patterns.
        
        Second, we need to recognize the causes. Human activities, particularly the burning of fossil fuels, are the main drivers of climate change.
        
        Finally, we should consider solutions. Reducing carbon emissions, transitioning to renewable energy, and implementing sustainable practices are essential steps.
        """
        self.assertGreater(reward.compute_reward(coherent), 0.7)
        
        # Incoherent response
        incoherent = """
        Climate change is bad. We need to stop it now. The temperature is rising. Ice caps are melting.
        Fossil fuels are a problem. The Earth is getting warmer. We need solutions. Renewable energy is good.
        """
        self.assertLess(reward.compute_reward(incoherent), 0.7)


class TestTaskSpecificReward(unittest.TestCase):
    """Test the TaskSpecificReward class."""
    
    def test_summarization_reward(self):
        """Test summarization reward."""
        reward = TaskSpecificReward(task_type='summarization')
        
        original_text = """
        Climate change is a long-term shift in global or regional climate patterns. Often climate change refers 
        specifically to the rise in global temperatures from the mid-20th century to present.
        
        Climate change is caused by factors such as biotic processes, variations in solar radiation received by Earth, 
        plate tectonics, and volcanic eruptions. Certain human activities have been identified as primary causes of 
        ongoing climate change, often referred to as global warming.
        
        Scientists use observations from the ground, air and space, along with theoretical models, to monitor and study 
        past, present and future climate change. Climate data records provide evidence of climate change key indicators, 
        such as global land and ocean temperature increases; rising sea levels; ice loss at Earth's poles and in mountain 
        glaciers; frequency and severity changes in extreme weather such as hurricanes, heatwaves, wildfires, droughts, 
        floods and precipitation; and cloud and vegetation cover changes, to name but a few.
        """
        
        # Good summary
        good_summary = """
        Climate change refers to long-term shifts in climate patterns, particularly global warming since the mid-20th century.
        It's caused by natural factors and human activities. Scientists study it using observations and models, tracking
        indicators like temperature increases, sea level rise, and extreme weather events.
        """
        self.assertGreater(reward.compute_reward(good_summary, reference=original_text), 0.7)
        
        # Too short summary
        too_short = "Climate change is a shift in climate patterns caused by natural and human factors."
        self.assertLess(reward.compute_reward(too_short, reference=original_text), 0.7)
        
        # Off-topic summary
        off_topic = """
        Global warming is a serious issue that needs immediate attention. We must reduce carbon emissions
        and transition to renewable energy sources to mitigate its effects.
        """
        self.assertLess(reward.compute_reward(off_topic, reference=original_text), 0.7)
    
    def test_qa_reward(self):
        """Test question-answering reward."""
        reward = TaskSpecificReward(task_type='qa')
        
        question = "What is the capital of France?"
        reference = "The capital of France is Paris."
        
        # Correct answer
        correct = "Paris is the capital of France."
        self.assertGreater(reward.compute_reward(correct, reference=reference, question=question), 0.7)
        
        # Incorrect answer
        incorrect = "The capital of France is Lyon."
        self.assertLess(reward.compute_reward(incorrect, reference=reference, question=question), 0.7)
        
        # Partially correct
        partial = "France has many cities, including Paris which is its capital."
        self.assertGreater(reward.compute_reward(partial, reference=reference, question=question), 0.5)
    
    def test_custom_reward_function(self):
        """Test custom reward function."""
        def custom_fn(response, prompt=None, reference=None, **kwargs):
            # Simple custom function that rewards responses containing a specific keyword
            if isinstance(response, list):
                return [1.0 if "keyword" in r else 0.0 for r in response]
            return 1.0 if "keyword" in response else 0.0
        
        reward = TaskSpecificReward(task_type='custom', custom_reward_fn=custom_fn)
        
        # Response with keyword
        with_keyword = "This response contains the keyword we're looking for."
        self.assertEqual(reward.compute_reward(with_keyword), 1.0)
        
        # Response without keyword
        without_keyword = "This response does not contain what we're looking for."
        self.assertEqual(reward.compute_reward(without_keyword), 0.0)


class TestCombinedReward(unittest.TestCase):
    """Test the CombinedReward class."""
    
    def test_combined_reward(self):
        """Test combined reward function."""
        length_reward = LengthReward(min_length=10, max_length=100)
        format_reward = FormatReward(format_type='json')
        
        # Equal weights
        combined = CombinedReward(reward_functions=[length_reward, format_reward])
        
        # JSON response with good length
        good_json = '{"result": "This is a good length JSON response with enough characters."}'
        self.assertGreater(combined.compute_reward(good_json), 0.7)
        
        # JSON response that's too short
        short_json = '{"result": "Too short"}'
        self.assertLess(combined.compute_reward(short_json), 0.7)
        
        # Non-JSON response with good length
        non_json = "This is a good length response but it's not in JSON format as required."
        self.assertLess(combined.compute_reward(non_json), 0.7)
    
    def test_weighted_combined_reward(self):
        """Test combined reward with custom weights."""
        length_reward = LengthReward(min_length=10, max_length=100)
        format_reward = FormatReward(format_type='json')
        
        # Length weighted more heavily
        combined = CombinedReward(
            reward_functions=[length_reward, format_reward],
            weights=[0.8, 0.2]
        )
        
        # Good length but not JSON
        good_length_not_json = "This is a response with good length but not in JSON format."
        
        # Short but valid JSON
        short_json = '{"result": "Short"}'
        
        # With length weighted more, the non-JSON should score higher
        self.assertGreater(
            combined.compute_reward(good_length_not_json),
            combined.compute_reward(short_json)
        )
        
        # Format weighted more heavily
        combined = CombinedReward(
            reward_functions=[length_reward, format_reward],
            weights=[0.2, 0.8]
        )
        
        # With format weighted more, the JSON should score higher
        self.assertGreater(
            combined.compute_reward(short_json),
            combined.compute_reward(good_length_not_json)
        )


class TestCreateRewardFunction(unittest.TestCase):
    """Test the create_reward_function utility."""
    
    def test_create_length_reward(self):
        """Test creating a length reward from config."""
        config = {
            'type': 'length',
            'min_length': 20,
            'max_length': 200,
            'weight': 1.5
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, LengthReward)
        self.assertEqual(reward.min_length, 20)
        self.assertEqual(reward.max_length, 200)
        self.assertEqual(reward.weight, 1.5)
    
    def test_create_combined_reward(self):
        """Test creating a combined reward from config."""
        config = {
            'type': 'combined',
            'name': 'test_combined',
            'reward_functions': [
                {
                    'type': 'length',
                    'min_length': 20,
                    'weight': 0.7
                },
                {
                    'type': 'format',
                    'format_type': 'json',
                    'weight': 0.3
                }
            ]
        }
        
        reward = create_reward_function(config)
        
        self.assertIsInstance(reward, CombinedReward)
        self.assertEqual(reward.name, 'test_combined')
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


if __name__ == '__main__':
    unittest.main()
class TestTagReward(unittest.TestCase):
    """Test the TagReward class."""
    
    def test_basic_tag_compliance(self):
        """Test basic tag compliance checking."""
        # Define tag specifications
        tag_specs = {
            'think': {'required': True, 'min_count': 1, 'max_count': 1},
            'answer': {'required': True, 'min_count': 1, 'max_count': 1},
            'code': {'required': False, 'max_count': 2}
        }
        
        reward = TagReward(tag_specs=tag_specs)
        
        # Perfect compliance
        perfect = """
        <think>This is my thinking process</think>
        <answer>This is my answer</answer>
        """
        self.assertGreater(reward.compute_reward(perfect), 0.9)
        
        # Missing required tag
        missing_required = """
        <think>This is my thinking process</think>
        """
        self.assertLess(reward.compute_reward(missing_required), 0.5)
        
        # Too many occurrences
        too_many = """
        <think>First thought</think>
        <think>Second thought</think>
        <answer>This is my answer</answer>
        """
        self.assertLess(reward.compute_reward(too_many), 0.8)
        
        # With optional tag
        with_optional = """
        <think>This is my thinking process</think>
        <answer>This is my answer</answer>
        <code>function example() { return true; }</code>
        """
        self.assertGreater(reward.compute_reward(with_optional), 0.9)
    
    def test_nested_tags(self):
        """Test nested tag compliance checking."""
        # Define tag specifications
        tag_specs = {
            'outer': {'required': True, 'min_count': 1, 'max_count': 1},
            'inner': {'required': True, 'min_count': 1, 'max_count': 2}
        }
        
        # With strict nesting
        strict_reward = TagReward(tag_specs=tag_specs, strict_nesting=True)
        
        # Properly nested
        properly_nested = """
        <outer>
            This is outer content
            <inner>This is inner content</inner>
        </outer>
        """
        self.assertGreater(strict_reward.compute_reward(properly_nested), 0.9)
        
        # Improperly nested
        improperly_nested = """
        <outer>
            This is outer content
            <inner>This is inner content
        </outer>
        </inner>
        """
        self.assertLess(strict_reward.compute_reward(improperly_nested), 0.5)
        
        # Without strict nesting
        non_strict_reward = TagReward(tag_specs=tag_specs, strict_nesting=False)
        self.assertGreater(non_strict_reward.compute_reward(improperly_nested), 0.5)
    
    def test_content_requirements(self):
        """Test content requirements for tags."""
        # Define tag specifications with content requirements
        tag_specs = {
            'think': {
                'required': True, 
                'content_required': True,
                'content_regex': r'.*\bstep\b.*'  # Must contain the word "step"
            },
            'answer': {'required': True}
        }
        
        reward = TagReward(tag_specs=tag_specs)
        
        # Valid content
        valid_content = """
        <think>First step: analyze the problem. Second step: solve it.</think>
        <answer>The solution is 42.</answer>
        """
        self.assertGreater(reward.compute_reward(valid_content), 0.9)
        
        # Invalid content (missing required word)
        invalid_content = """
        <think>I need to analyze the problem and then solve it.</think>
        <answer>The solution is 42.</answer>
        """
        self.assertLess(reward.compute_reward(invalid_content), 0.9)
        
        # Empty content
        empty_content = """
        <think></think>
        <answer>The solution is 42.</answer>
        """
        self.assertLess(reward.compute_reward(empty_content), 0.7)