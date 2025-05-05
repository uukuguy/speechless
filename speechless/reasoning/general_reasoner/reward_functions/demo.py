#!/usr/bin/env python
"""
Demonstration of Reward Functions

This script demonstrates how to use the refactored reward functions.
"""

from base import BaseReward
from text_rewards import LengthReward, FormatReward, CoherenceReward
from math_rewards import MathReward
from code_rewards import CodeReward
from factuality_rewards import FactualityReward
from task_rewards import TaskSpecificReward
from tag_rewards import TagReward
from combined_rewards import CombinedReward


def demonstrate_reward_functions():
    """
    Demonstrate how to use the reward functions.
    """
    print("\n=== Reward Functions Demonstration ===\n")
    
    # Create individual reward functions
    length_reward = LengthReward(min_length=50, max_length=200)
    format_reward = FormatReward(format_type='json')
    math_reward = MathReward()
    code_reward = CodeReward(check_execution=False)
    
    # Example responses
    responses = [
        "The answer is 42.",
        '{"result": "success", "value": 42}',
        "To solve this problem, we need to calculate 6 * 7. The answer is 42."
    ]
    
    # Evaluate with individual reward functions
    print("Individual Reward Function Scores:")
    print("---------------------------------")
    
    print("Length Reward:")
    length_scores = length_reward(responses)
    for i, (resp, score) in enumerate(zip(responses, length_scores)):
        print(f"  Response {i+1}: {resp[:30]}{'...' if len(resp) > 30 else ''} - Score: {score:.2f}")
    
    print("\nFormat Reward (JSON):")
    format_scores = format_reward(responses)
    for i, (resp, score) in enumerate(zip(responses, format_scores)):
        print(f"  Response {i+1}: {resp[:30]}{'...' if len(resp) > 30 else ''} - Score: {score:.2f}")
    
    print("\nMath Reward:")
    math_scores = math_reward(responses, reference_answer=[42, 42, 42])
    for i, (resp, score) in enumerate(zip(responses, math_scores)):
        print(f"  Response {i+1}: {resp[:30]}{'...' if len(resp) > 30 else ''} - Score: {score:.2f}")
    
    # Combine reward functions with weights
    print("\nCombined Reward Function:")
    print("-----------------------")
    combined_reward = CombinedReward(
        reward_functions=[length_reward, format_reward, math_reward, code_reward],
        weights=[0.2, 0.2, 0.3, 0.3]
    )
    
    # Compute combined rewards
    combined_scores = combined_reward(responses, reference_answer=[42, 42, 42])
    
    for i, (resp, score) in enumerate(zip(responses, combined_scores)):
        print(f"  Response {i+1}: {resp[:30]}{'...' if len(resp) > 30 else ''} - Score: {score:.2f}")
    
    # Demonstrate task-specific rewards
    print("\nTask-Specific Rewards:")
    print("--------------------")
    
    # Summarization
    original_text = """
    Machine learning is a field of study in artificial intelligence concerned with the development
    of algorithms and statistical models that computer systems use to perform tasks without explicit
    instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence.
    """
    
    summary = "Machine learning is an AI field focused on developing algorithms that allow computers to perform tasks using patterns and inference rather than explicit instructions."
    
    summarization_reward = TaskSpecificReward(task_type='summarization')
    summary_score = summarization_reward(summary, reference=original_text)
    
    print(f"Summarization Score: {summary_score:.2f}")
    
    # Tag reward
    print("\nTag Reward:")
    print("---------")
    
    tag_specs = {
        'thinking': {'required': True},
        'answer': {'required': True}
    }
    
    tag_reward = TagReward(tag_specs=tag_specs)
    
    tag_response = "<thinking>This is a math problem. 6 * 7 = 42</thinking><answer>42</answer>"
    tag_score = tag_reward(tag_response)
    
    print(f"Tag Score: {tag_score:.2f}")
    
    print("\nThis demonstration shows how to use the refactored reward functions.")
    print("The modular structure makes it easy to use individual reward functions")
    print("or combine them for comprehensive evaluation.")


if __name__ == "__main__":
    demonstrate_reward_functions()