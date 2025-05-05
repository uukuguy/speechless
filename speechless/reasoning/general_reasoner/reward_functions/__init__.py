"""
Reward Functions for General Reasoner RL Fine-tuning

This package provides a comprehensive suite of reward functions for reinforcement learning
fine-tuning of language models. Each reward function evaluates a specific aspect of model
outputs and returns a normalized score. These functions are designed to be used with the
veRL framework for training language models with reinforcement learning.

The reward functions can be used individually or combined using the CombinedReward class.
Each function supports both single-instance and batched processing for computational efficiency.

Modules:
    base: Contains the BaseReward abstract base class
    text_rewards: Text-based rewards (length, format, coherence)
    math_rewards: Math-related rewards (math problem solving, verification)
    code_rewards: Code quality and correctness rewards
    factuality_rewards: Factual accuracy rewards
    task_rewards: Domain-specific task rewards
    tag_rewards: XML-style tag usage rewards
    combined_rewards: Combining multiple rewards
    utils: Utility functions for creating and managing rewards

For backward compatibility, all reward classes are imported directly into the
reward_functions namespace.
"""

# Import all reward classes for backward compatibility
from .base import BaseReward
from .text_rewards import LengthReward, FormatReward, CoherenceReward
from .math_rewards import MathReward, MathVerifyReward
from .code_rewards import CodeReward
from .factuality_rewards import FactualityReward
from .task_rewards import TaskSpecificReward
from .tag_rewards import TagReward
from .combined_rewards import CombinedReward
from .utils import create_reward_function, example_usage

# For backward compatibility, expose all classes at the package level
__all__ = [
    'BaseReward',
    'LengthReward',
    'FormatReward',
    'MathReward',
    'CodeReward',
    'FactualityReward',
    'CoherenceReward',
    'TaskSpecificReward',
    'TagReward',
    'CombinedReward',
    'MathVerifyReward',
    'create_reward_function',
    'example_usage',
]