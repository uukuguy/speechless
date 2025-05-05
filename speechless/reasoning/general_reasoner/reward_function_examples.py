#!/usr/bin/env python
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
Example usage of reward functions for RL fine-tuning.

This script demonstrates how to use the reward functions defined in reward_functions.py
for evaluating language model outputs.
"""

import json
import argparse
from typing import List, Dict, Any

from reward_functions import (
    LengthReward, FormatReward, MathReward, CodeReward,
    FactualityReward, CoherenceReward, TaskSpecificReward, TagReward, CombinedReward,
    create_reward_function
)


def demonstrate_length_reward():
    """Demonstrate the LengthReward function."""
    print("\n=== Length Reward ===")
    
    # Create a length reward
    reward = LengthReward(min_length=20, max_length=100, optimal_length=50)
    
    # Example responses of different lengths
    responses = [
        "Too short.",
        "This is a response with a good length that should score well with the length reward function.",
        "This is a very long response that exceeds the maximum length limit and should receive a lower score. " * 5
    ]
    
    # Evaluate each response
    for i, response in enumerate(responses):
        score = reward.compute_reward(response)
        print(f"Response {i+1} ({len(response)} chars): {score:.2f}")


def demonstrate_format_reward():
    """Demonstrate the FormatReward function."""
    print("\n=== Format Reward ===")
    
    # Create a format reward for JSON
    json_reward = FormatReward(format_type='json')
    
    # Example responses
    responses = [
        '{"name": "John", "age": 30}',  # Valid JSON
        '{"name": "John", age: 30}',     # Invalid JSON
        'Name: John, Age: 30'            # Not JSON at all
    ]
    
    # Evaluate each response
    print("JSON Format:")
    for i, response in enumerate(responses):
        score = json_reward.compute_reward(response)
        print(f"Response {i+1}: {score:.2f}")
    
    # Create a format reward for bullet points
    bullet_reward = FormatReward(format_type='bullet')
    
    # Example responses
    responses = [
        "- Item 1\n- Item 2\n- Item 3",  # Valid bullet list
        "• Item 1\n• Item 2\n• Item 3",  # Alternative bullet style
        "Item 1\nItem 2\nItem 3"         # Not a bullet list
    ]
    
    # Evaluate each response
    print("\nBullet Format:")
    for i, response in enumerate(responses):
        score = bullet_reward.compute_reward(response)
        print(f"Response {i+1}: {score:.2f}")


def demonstrate_math_reward():
    """Demonstrate the MathReward function."""
    print("\n=== Math Reward ===")
    
    # Create a math reward
    reward = MathReward()
    
    # Example responses with reference answer
    reference_answer = 42
    responses = [
        "The answer is 42.",  # Correct answer, no reasoning
        """
        To solve this problem, I need to multiply 6 by 7.
        6 * 7 = 42
        Therefore, the answer is 42.
        """,  # Correct answer with reasoning
        """
        To solve this problem, I need to multiply 6 by 7.
        6 * 7 = 43
        Therefore, the answer is 43.
        """  # Incorrect answer with reasoning
    ]
    
    # Evaluate each response
    for i, response in enumerate(responses):
        score = reward.compute_reward(response, reference_answer=reference_answer)
        print(f"Response {i+1}: {score:.2f}")


def demonstrate_code_reward():
    """Demonstrate the CodeReward function."""
    print("\n=== Code Reward ===")
    
    # Create a code reward
    reward = CodeReward(check_execution=False)
    
    # Example responses
    responses = [
        # Good code with docstring and good style
        """
```python
def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)
```
        """,
        
        # Code with syntax error
        """
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1
```
        """,
        
        # Code with poor style
        """
```python
def factorial(n):
    if n<=1:
     return 1
    return n*factorial(n-1)
```
        """
    ]
    
    # Evaluate each response
    for i, response in enumerate(responses):
        score = reward.compute_reward(response)
        print(f"Response {i+1}: {score:.2f}")


def demonstrate_factuality_reward():
    """Demonstrate the FactualityReward function."""
    print("\n=== Factuality Reward ===")
    
    # Create a factuality reward with reference texts
    reference_texts = [
        "The Earth is the third planet from the Sun.",
        "The Moon is Earth's only natural satellite.",
        "The Earth has a diameter of approximately 12,742 kilometers."
    ]
    
    reward = FactualityReward(reference_texts=reference_texts)
    
    # Example responses
    responses = [
        "The Earth is the third planet from the Sun and has one moon.",  # Factual
        "The Earth is the fifth planet from the Sun and has three moons.",  # Non-factual
        "The Earth orbits the Sun and has a diameter of about 12,700 km."  # Partially factual
    ]
    
    # Evaluate each response
    for i, response in enumerate(responses):
        score = reward.compute_reward(response)
        print(f"Response {i+1}: {score:.2f}")


def demonstrate_coherence_reward():
    """Demonstrate the CoherenceReward function."""
    print("\n=== Coherence Reward ===")
    
    # Create a coherence reward
    reward = CoherenceReward()
    
    # Example responses
    responses = [
        # Well-structured, coherent response
        """
First, let's understand what climate change is. Climate change refers to long-term shifts in temperatures and weather patterns.

Second, we need to recognize the causes. Human activities, particularly the burning of fossil fuels, are the main drivers of climate change.

Finally, we should consider solutions. Reducing carbon emissions, transitioning to renewable energy, and implementing sustainable practices are essential steps.
        """,
        
        # Poorly structured, incoherent response
        """
Climate change is bad. We need to stop it now. The temperature is rising. Ice caps are melting.
Fossil fuels are a problem. The Earth is getting warmer. We need solutions. Renewable energy is good.
        """
    ]
    
    # Evaluate each response
    for i, response in enumerate(responses):
        score = reward.compute_reward(response)
        print(f"Response {i+1}: {score:.2f}")


def demonstrate_task_specific_reward():
    """Demonstrate the TaskSpecificReward function."""
    print("\n=== Task-Specific Reward ===")
    
    # Original text for summarization
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
    
    # Create a summarization reward
    summarization_reward = TaskSpecificReward(task_type='summarization')
    
    # Example summaries
    summaries = [
        # Good summary
        """
Climate change refers to long-term shifts in climate patterns, particularly global warming since the mid-20th century.
It's caused by natural factors and human activities. Scientists study it using observations and models, tracking
indicators like temperature increases, sea level rise, and extreme weather events.
        """,
        
        # Too short summary
        "Climate change is a shift in climate patterns caused by natural and human factors."
    ]
    
    # Evaluate each summary
    print("Summarization:")
    for i, summary in enumerate(summaries):
        score = summarization_reward.compute_reward(summary, reference=original_text)
        print(f"Summary {i+1}: {score:.2f}")
    
    # Create a QA reward
    qa_reward = TaskSpecificReward(task_type='qa')
    
    # Example QA
    question = "What is the capital of France?"
    reference = "The capital of France is Paris."
    
    answers = [
        "Paris is the capital of France.",  # Correct
        "The capital of France is Lyon."    # Incorrect
    ]
    
    # Evaluate each answer
    print("\nQuestion Answering:")
    for i, answer in enumerate(answers):
        score = qa_reward.compute_reward(answer, reference=reference, question=question)
        print(f"Answer {i+1}: {score:.2f}")


def demonstrate_combined_reward():
    """Demonstrate the CombinedReward function."""
    print("\n=== Combined Reward ===")
    
    # Create individual reward functions
    length_reward = LengthReward(min_length=20, max_length=100)
    format_reward = FormatReward(format_type='json')
    
    # Combine them with weights
    combined_reward = CombinedReward(
        reward_functions=[length_reward, format_reward],
        weights=[0.7, 0.3]
    )
    
    # Example responses
    responses = [
        '{"result": "This is a good length JSON response that should score well with both reward functions."}',
        '{"result": "Short"}',
        'This is a good length response but not in JSON format, so it should score well with length but poorly with format.'
    ]
    
    # Evaluate each response
    for i, response in enumerate(responses):
        length_score = length_reward.compute_reward(response)
        format_score = format_reward.compute_reward(response)
        combined_score = combined_reward.compute_reward(response)
        
        print(f"Response {i+1}:")
        print(f"  Length: {length_score:.2f}")
        print(f"  Format: {format_score:.2f}")
        print(f"  Combined: {combined_score:.2f}")


def demonstrate_config_based_creation():
    """Demonstrate creating reward functions from configuration."""
    print("\n=== Config-Based Creation ===")
    
    # Configuration for a combined reward
    config = {
        'type': 'combined',
        'name': 'my_combined_reward',
        'reward_functions': [
            {
                'type': 'length',
                'min_length': 20,
                'max_length': 100,
                'weight': 0.7
            },
            {
                'type': 'format',
                'format_type': 'json',
                'weight': 0.3
            }
        ]
    }
    
    # Create the reward function from config
    reward = create_reward_function(config)
    
    # Example response
    response = '{"result": "This is a good length JSON response."}'
    
    # Evaluate the response
    score = reward.compute_reward(response)
    print(f"Config-based reward score: {score:.2f}")


def demonstrate_tag_reward():
    """Demonstrate the TagReward function."""
    print("\n=== Tag Reward ===")
    
    # Define tag specifications
    tag_specs = {
        'think': {'required': True, 'min_count': 1, 'max_count': 1},
        'answer': {'required': True, 'min_count': 1, 'max_count': 1},
        'code': {'required': False, 'max_count': 2}
    }
    
    # Create a tag reward
    reward = TagReward(tag_specs=tag_specs)
    
    # Example responses
    responses = [
        # Perfect compliance with required tags
        """
        <think>I need to analyze this problem carefully. First, I'll consider the requirements and then develop a solution.</think>
        <answer>The solution is to implement a recursive algorithm that handles all edge cases.</answer>
        """,
        
        # Missing a required tag
        """
        <think>I need to analyze this problem carefully. First, I'll consider the requirements and then develop a solution.</think>
        """,
        
        # With optional tag
        """
        <think>I need to analyze this problem carefully.</think>
        <answer>The solution is to implement a recursive algorithm.</answer>
        <code>
        function solve(input) {
            if (input.length === 0) return [];
            return [input[0]].concat(solve(input.slice(1)));
        }
        </code>
        """,
        
        # Too many occurrences of a tag
        """
        <think>First thought.</think>
        <think>Second thought.</think>
        <answer>The solution is to implement a recursive algorithm.</answer>
        """
    ]
    
    # Evaluate each response
    for i, response in enumerate(responses):
        score = reward.compute_reward(response)
        print(f"Response {i+1}: {score:.2f}")
    
    # Example with content requirements
    print("\nTag Reward with Content Requirements:")
    
    # Define tag specifications with content requirements
    tag_specs_with_content = {
        'think': {
            'required': True,
            'content_required': True,
            'content_regex': r'.*\bstep\b.*'  # Must contain the word "step"
        },
        'answer': {'required': True}
    }
    
    content_reward = TagReward(tag_specs=tag_specs_with_content)
    
    # Example responses
    content_responses = [
        # Valid content
        """
        <think>Step 1: Analyze the problem. Step 2: Develop a solution.</think>
        <answer>The solution is 42.</answer>
        """,
        
        # Invalid content (missing required word)
        """
        <think>I need to analyze the problem and then solve it.</think>
        <answer>The solution is 42.</answer>
        """
    ]
    
    # Evaluate each response
    for i, response in enumerate(content_responses):
        score = content_reward.compute_reward(response)
        print(f"Response {i+1}: {score:.2f}")


def main():
    """Main function to demonstrate reward functions."""
    parser = argparse.ArgumentParser(description="Demonstrate reward functions")
    parser.add_argument("--all", action="store_true", help="Run all demonstrations")
    parser.add_argument("--length", action="store_true", help="Demonstrate length reward")
    parser.add_argument("--format", action="store_true", help="Demonstrate format reward")
    parser.add_argument("--math", action="store_true", help="Demonstrate math reward")
    parser.add_argument("--code", action="store_true", help="Demonstrate code reward")
    parser.add_argument("--factuality", action="store_true", help="Demonstrate factuality reward")
    parser.add_argument("--coherence", action="store_true", help="Demonstrate coherence reward")
    parser.add_argument("--task", action="store_true", help="Demonstrate task-specific reward")
    parser.add_argument("--tag", action="store_true", help="Demonstrate tag reward")
    parser.add_argument("--combined", action="store_true", help="Demonstrate combined reward")
    parser.add_argument("--config", action="store_true", help="Demonstrate config-based creation")
    
    args = parser.parse_args()
    
    # If no specific demonstrations are requested, run all
    run_all = args.all or not any([
        args.length, args.format, args.math, args.code, args.factuality,
        args.coherence, args.task, args.tag, args.combined, args.config
    ])
    
    print("Reward Functions Demonstration")
    print("=============================")
    
    if run_all or args.length:
        demonstrate_length_reward()
    
    if run_all or args.format:
        demonstrate_format_reward()
    
    if run_all or args.math:
        demonstrate_math_reward()
    
    if run_all or args.code:
        demonstrate_code_reward()
    
    if run_all or args.factuality:
        demonstrate_factuality_reward()
    
    if run_all or args.coherence:
        demonstrate_coherence_reward()
    
    if run_all or args.task:
        demonstrate_task_specific_reward()
    
    if run_all or args.tag:
        demonstrate_tag_reward()
    
    if run_all or args.combined:
        demonstrate_combined_reward()
    
    if run_all or args.config:
        demonstrate_config_based_creation()


if __name__ == "__main__":
    main()