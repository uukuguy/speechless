"""
Base Reward Class

This module provides the abstract base class for all reward functions.
All reward functions should inherit from BaseReward and implement the compute_reward method.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseReward(ABC):
    """
    Abstract base class for all reward functions.
    
    All reward functions should inherit from this class and implement the compute_reward method.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize the reward function.
        
        Args:
            name: Name of the reward function
            weight: Weight of this reward when combined with others (default: 1.0)
        """
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the reward for a given response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Optional prompt(s) that generated the response
            reference: Optional reference response(s) for comparison
            **kwargs: Additional arguments specific to the reward function
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Union[float, List[float]]:
        """
        Make the reward function callable.
        
        Args:
            *args: Positional arguments to pass to compute_reward
            **kwargs: Keyword arguments to pass to compute_reward
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        return self.compute_reward(*args, **kwargs)
    
    def _ensure_list(self, item: Union[Any, List[Any]]) -> List[Any]:
        """
        Ensure the input is a list.
        
        Args:
            item: Input item or list of items
            
        Returns:
            List of items
        """
        if not isinstance(item, list):
            return [item]
        return item
    
    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize a score to be between min_val and max_val.
        
        Args:
            score: Score to normalize
            min_val: Minimum value for normalization (default: 0.0)
            max_val: Maximum value for normalization (default: 1.0)
            
        Returns:
            Normalized score
        """
        return max(min_val, min(max_val, score))