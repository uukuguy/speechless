"""
Combined Reward Functions

This module provides functionality for combining multiple reward functions:
- CombinedReward: Combines multiple reward functions with weighted averaging
"""

import logging
from typing import List, Dict, Any, Union, Optional
from loguru import logger
import numpy as np

from .base import BaseReward

class CombinedReward(BaseReward):
    """
    Combines multiple reward functions with weighted averaging.
    
    This reward function allows for combining multiple reward functions with
    different weights to create a comprehensive evaluation metric.
    """
    
    def __init__(self, 
                 reward_functions: List[BaseReward],
                 weights: Optional[List[float]] = None,
                 name: str = "combined",
                 max_cached_scores: int=10):
        """
        Initialize the combined reward function.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: Optional list of weights for each reward function
                    If not provided, the weights from each reward function will be used
            name: Name of the combined reward function (default: "combined")
        """
        if not reward_functions:
            raise ValueError("At least one reward function must be provided")
        
        if weights is None:
            weights = [fn.weight for fn in reward_functions]
        
        if len(weights) != len(reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
        
        super().__init__(name=name, weight=1.0)
        self.reward_functions = reward_functions
        self.weights = weights
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]

        self.cached_rewards = []
        self.max_cached_scores = max_cached_scores
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the combined reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Optional prompt(s) that generated the response
            reference: Optional reference response(s) for comparison
            **kwargs: Additional arguments passed to all reward functions
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        
        # Initialize rewards array
        combined_rewards = [0.0] * len(responses)
        
        score_list = []
        # Compute rewards for each function and combine with weights
        for i, (reward_fn, weight) in enumerate(zip(self.reward_functions, self.weights)):
            try:
                rewards = reward_fn(response, prompt, reference, **kwargs)
                rewards = self._ensure_list(rewards)
                
                # Ensure rewards has the same length as responses
                if len(rewards) != len(responses):
                    if len(rewards) == 1:
                        rewards = rewards * len(responses)
                    else:
                        logger.warning(f"Reward function {reward_fn.name} returned {len(rewards)} rewards for {len(responses)} responses. Using first reward for all.")
                        rewards = [rewards[0]] * len(responses)
                
                mean_score = np.mean(rewards)
                score_list.append((reward_fn.name, weight, mean_score))

                # Add weighted rewards
                for j in range(len(combined_rewards)):
                    combined_rewards[j] += weight * rewards[j]
            except Exception as e:
                logger.error(f"Error in reward function {reward_fn.name}: {e}")
                # Skip this reward function on error
        mean_combined_score = np.mean(combined_rewards) 
        score_list.append(("combined", 1.0, mean_combined_score))

        self.cached_rewards.append(score_list)

        if len(self.cached_rewards) >= self.max_cached_scores:
            reward_metas = [None] * (len(self.reward_functions) + 1)
            reward_scores = [0.0] * (len(self.reward_functions) + 1)
            for score_list in self.cached_rewards:
                for j, (n, w, s) in enumerate(score_list):
                    reward_scores[j] += s
                    reward_metas[j] = (n, w)
            for i in range(len(reward_scores)):
                reward_scores[i] /= len(self.cached_rewards)

            # score_list_str = "|".join([f"{n}({w:.2f}): {s:.3f}" for (n, w), s in zip(reward_metas, reward_scores)])
            # score_list_str = " | ".join([f"{n}: {round(s,3) if s != 0 else 0}" for (n, w), s in zip(reward_metas, reward_scores)])
            # logger.info(score_list_str)
            score_list_str = " | ".join([f"{n}: {s:.3f)}" for (n, w), s in zip(reward_metas, reward_scores)])
            print(score_list_str)
            self.cached_rewards.clear()

        # Normalize final rewards
        # combined_rewards = [self._normalize_score(r) for r in combined_rewards]

        return combined_rewards[0] if len(combined_rewards) == 1 else combined_rewards