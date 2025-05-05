"""
Reward Function Utilities

This module provides utility functions for creating and using reward functions:
- create_reward_function: Creates a reward function from a configuration dictionary
- example_usage: Demonstrates how to use the reward functions
"""

import logging
from typing import Dict, Any, List

from .base import BaseReward
from .text_rewards import LengthReward, FormatReward, CoherenceReward
from .math_rewards import MathReward, MathVerifyReward
from .code_rewards import CodeReward
from .factuality_rewards import FactualityReward
from .task_rewards import TaskSpecificReward
from .tag_rewards import TagReward
from .combined_rewards import CombinedReward

# Configure logging
logger = logging.getLogger(__name__)


def create_reward_function(config: Dict[str, Any]) -> BaseReward:
    """
    Create a reward function from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with reward function details
        
    Returns:
        Instantiated reward function
        
    Raises:
        ValueError: If the reward type is unknown or required parameters are missing
    """
    reward_type = config.get('type')
    if not reward_type:
        raise ValueError("Reward type must be specified")
    
    weight = config.get('weight', 1.0)
    
    if reward_type == 'length':
        return LengthReward(
            min_length=config.get('min_length', 50),
            max_length=config.get('max_length', 500),
            optimal_length=config.get('optimal_length'),
            token_based=config.get('token_based', False),
            tokenizer=config.get('tokenizer'),
            weight=weight
        )
    
    elif reward_type == 'format':
        return FormatReward(
            format_type=config.get('format_type'),
            format_regex=config.get('format_regex'),
            json_schema=config.get('json_schema'),
            weight=weight
        )
    
    elif reward_type == 'math':
        return MathReward(
            check_final_answer=config.get('check_final_answer', True),
            check_reasoning_steps=config.get('check_reasoning_steps', True),
            answer_regex=config.get('answer_regex', r'(?:answer|result)(?:\s+is)?(?:\s*[:=])?\s*(-?\d+\.?\d*|\d*\.\d+)'),
            weight=weight
        )
    
    elif reward_type == 'code':
        return CodeReward(
            check_syntax=config.get('check_syntax', True),
            check_execution=config.get('check_execution', False),
            check_style=config.get('check_style', True),
            language=config.get('language'),
            test_cases=config.get('test_cases'),
            timeout=config.get('timeout', 5),
            weight=weight
        )
    
    elif reward_type == 'factuality':
        return FactualityReward(
            reference_texts=config.get('reference_texts'),
            use_embeddings=config.get('use_embeddings', False),
            embedding_model=config.get('embedding_model'),
            check_contradictions=config.get('check_contradictions', True),
            weight=weight
        )
    
    elif reward_type == 'coherence':
        return CoherenceReward(
            check_logical_flow=config.get('check_logical_flow', True),
            check_consistency=config.get('check_consistency', True),
            check_clarity=config.get('check_clarity', True),
            weight=weight
        )
    
    elif reward_type == 'task_specific':
        return TaskSpecificReward(
            task_type=config.get('task_type', 'custom'),
            custom_reward_fn=config.get('custom_reward_fn'),
            task_params=config.get('task_params', {}),
            weight=weight
        )
    
    elif reward_type == 'tag':
        return TagReward(
            tag_specs=config.get('tag_specs', {}),
            strict_nesting=config.get('strict_nesting', True),
            weight=weight
        )
    
    elif reward_type == 'math_verify':
        return MathVerifyReward(
            boxed_format=config.get('boxed_format', True),
            weight=weight
        )
    
    elif reward_type == 'combined':
        reward_configs = config.get('reward_functions', [])
        if not reward_configs:
            raise ValueError("Combined reward requires at least one reward function")
        
        reward_functions = [create_reward_function(cfg) for cfg in reward_configs]
        weights = [cfg.get('weight', 1.0) for cfg in reward_configs]
        
        return CombinedReward(
            reward_functions=reward_functions,
            weights=weights,
            name=config.get('name', 'combined')
        )
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def example_usage():
    """
    Example usage of the reward functions.
    
    This function demonstrates how to create and use various reward functions
    individually and in combination. It shows how to evaluate different types
    of responses and interpret the results.
    """
    print("\n=== Reward Functions Example Usage ===\n")
    
    # Create individual reward functions
    length_reward = LengthReward(min_length=50, max_length=200)
    format_reward = FormatReward(format_type='json')
    math_reward = MathReward()
    code_reward = CodeReward(check_execution=False)
    coherence_reward = CoherenceReward()
    
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
        reward_functions=[length_reward, format_reward, math_reward, code_reward, coherence_reward],
        weights=[0.2, 0.2, 0.3, 0.2, 0.1]
    )
    
    # Compute combined rewards
    combined_scores = combined_reward(responses, reference_answer=[42, 42, 42])
    
    for i, (resp, score) in enumerate(zip(responses, combined_scores)):
        print(f"  Response {i+1}: {resp[:30]}{'...' if len(resp) > 30 else ''} - Score: {score:.2f}")
    
    print("\nThis example demonstrates how reward functions can be used to evaluate")
    print("different aspects of model responses and combined for comprehensive assessment.")
    
    # Example of creating a reward function from a configuration
    print("\nCreating Reward Functions from Configuration:")
    print("------------------------------------------")
    
    config = {
        'type': 'combined',
        'name': 'custom_combined',
        'reward_functions': [
            {'type': 'length', 'min_length': 100, 'max_length': 300, 'weight': 0.3},
            {'type': 'coherence', 'check_logical_flow': True, 'weight': 0.7}
        ]
    }
    
    custom_reward = create_reward_function(config)
    print(f"Created {custom_reward.name} reward with {len(custom_reward.reward_functions)} sub-rewards")
    
    # Example with tag reward
    print("\nTag Reward Example:")
    print("-----------------")
    
    tag_specs = {
        'thinking': {'required': True, 'min_count': 1, 'max_count': 1},
        'answer': {'required': True, 'min_count': 1, 'max_count': 1}
    }
    
    tag_reward = TagReward(tag_specs=tag_specs)
    
    tag_responses = [
        "<thinking>This is a math problem. 6 * 7 = 42</thinking><answer>42</answer>",
        "<answer>42</answer>",
        "<thinking>Let me think...</thinking><answer>42</answer><extra>Additional info</extra>"
    ]
    
    tag_scores = tag_reward(tag_responses)
    for i, (resp, score) in enumerate(zip(tag_responses, tag_scores)):
        print(f"  Response {i+1}: {resp[:50]}{'...' if len(resp) > 50 else ''} - Score: {score:.2f}")


if __name__ == "__main__":
    example_usage()