"""
Tag-Based Reward Functions

This module provides reward functions for evaluating XML-style tag usage:
- TagReward: Evaluates the correct usage of XML-style tags in responses
"""

import re
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

from .base import BaseReward

# Configure logging
logger = logging.getLogger(__name__)


class TagReward(BaseReward):
    """
    Reward function that evaluates the correct usage of XML-style tags in responses.
    
    This reward checks for the presence and proper formatting of specified tag pairs,
    with support for mandatory and optional tags, and constraints on the number of
    occurrences for each tag type.
    """
    
    def __init__(self,
                 tag_specs: Dict[str, Dict[str, Any]],
                 strict_nesting: bool = True,
                 weight: float = 1.0):
        """
        Initialize the tag reward function.
        
        Args:
            tag_specs: Dictionary mapping tag names to specifications.
                Each specification is a dictionary with the following keys:
                - 'required' (bool): Whether the tag is mandatory
                - 'min_count' (int): Minimum number of occurrences (default: 0 for optional, 1 for required)
                - 'max_count' (int): Maximum number of occurrences (default: 1)
                - 'content_required' (bool): Whether content is required between tags (default: True)
                - 'content_regex' (str): Optional regex pattern that content must match
            strict_nesting: Whether to enforce proper nesting of tags (default: True)
            weight: Weight of this reward when combined with others (default: 1.0)
            
        Example:
            tag_specs = {
                'think': {'required': True, 'min_count': 1, 'max_count': 1},
                'answer': {'required': True, 'min_count': 1, 'max_count': 1},
                'code': {'required': False, 'max_count': 3}
            }
        """
        super().__init__(name="tag", weight=weight)
        self.tag_specs = {}
        
        # Process and validate tag specifications
        for tag_name, spec in tag_specs.items():
            # Create a copy to avoid modifying the original
            processed_spec = dict(spec)
            
            # Set defaults based on 'required' flag
            required = processed_spec.get('required', False)
            processed_spec['required'] = required
            
            if 'min_count' not in processed_spec:
                processed_spec['min_count'] = 1 if required else 0
                
            if 'max_count' not in processed_spec:
                processed_spec['max_count'] = 1
                
            if 'content_required' not in processed_spec:
                processed_spec['content_required'] = True
                
            self.tag_specs[tag_name] = processed_spec
            
        self.strict_nesting = strict_nesting
    
    def _find_tag_pairs(self, text: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Find all tag pairs in the text.
        
        Args:
            text: Text to search for tags
            
        Returns:
            Dictionary mapping tag names to lists of (start_pos, end_pos, content) tuples
        """
        tag_pairs = {tag: [] for tag in self.tag_specs}
        
        # Find all opening and closing tags
        for tag_name in self.tag_specs:
            # Create regex patterns for this tag
            open_pattern = f"<{tag_name}>"
            close_pattern = f"</{tag_name}>"
            
            # Find all instances of this tag pair
            start_pos = 0
            while True:
                # Find opening tag
                open_match = re.search(open_pattern, text[start_pos:])
                if not open_match:
                    break
                    
                open_pos = start_pos + open_match.start()
                content_start = start_pos + open_match.end()
                
                # Find corresponding closing tag
                close_match = re.search(close_pattern, text[content_start:])
                if not close_match:
                    break
                    
                content_end = content_start + close_match.start()
                close_end = content_start + close_match.end()
                
                # Extract content between tags
                content = text[content_start:content_end]
                
                # Add to results
                tag_pairs[tag_name].append((open_pos, close_end, content))
                
                # Move start position for next search
                start_pos = close_end
        
        return tag_pairs
    
    def _check_nesting(self, text: str, tag_pairs: Dict[str, List[Tuple[int, int, str]]]) -> bool:
        """
        Check if tags are properly nested.
        
        Args:
            text: Original text
            tag_pairs: Dictionary of tag pairs as returned by _find_tag_pairs
            
        Returns:
            True if tags are properly nested, False otherwise
        """
        # Flatten all tag positions
        all_tags = []
        for tag_name, pairs in tag_pairs.items():
            for start_pos, end_pos, _ in pairs:
                # Add opening tag
                all_tags.append((start_pos, f"<{tag_name}>", True))
                # Add closing tag
                all_tags.append((end_pos - len(f"</{tag_name}>"), f"</{tag_name}>", False))
        
        # Sort by position
        all_tags.sort()
        
        # Check nesting using a stack
        stack = []
        for _, tag, is_opening in all_tags:
            if is_opening:
                stack.append(tag)
            else:
                # Closing tag should match the last opening tag
                if not stack:
                    return False
                
                opening_tag = stack.pop()
                closing_tag = tag
                
                # Check if tags match
                if opening_tag[1:-1] != closing_tag[2:-1]:
                    return False
        
        # Stack should be empty if all tags are properly nested
        return len(stack) == 0
    
    def _evaluate_tag_compliance(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate how well the text complies with tag specifications.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Tuple of (score, details) where details is a dictionary with information
            about tag compliance
        """
        # Find all tag pairs
        tag_pairs = self._find_tag_pairs(text)
        
        # Check nesting if required
        if self.strict_nesting and not self._check_nesting(text, tag_pairs):
            return 0.0, {"error": "Tags are not properly nested"}
        
        # Check each tag specification
        tag_scores = {}
        total_score = 0.0
        total_weight = 0.0
        
        for tag_name, spec in self.tag_specs.items():
            pairs = tag_pairs.get(tag_name, [])
            count = len(pairs)
            
            # Check count constraints
            min_count = spec['min_count']
            max_count = spec['max_count']
            
            if count < min_count:
                tag_scores[tag_name] = {
                    "score": 0.0,
                    "reason": f"Too few occurrences: {count} < {min_count}"
                }
                
                # Required tags that are missing have higher weight
                weight = 2.0 if spec['required'] else 1.0
                total_score += 0.0 * weight
                total_weight += weight
                continue
                
            if count > max_count:
                tag_scores[tag_name] = {
                    "score": 0.2,
                    "reason": f"Too many occurrences: {count} > {max_count}"
                }
                
                weight = 1.0
                total_score += 0.2 * weight
                total_weight += weight
                continue
            
            # Check content requirements
            content_required = spec['content_required']
            content_regex = spec.get('content_regex')
            
            content_scores = []
            for _, _, content in pairs:
                # Check if content is required but missing
                if content_required and not content.strip():
                    content_scores.append(0.5)
                    continue
                
                # Check if content matches regex pattern
                if content_regex and not re.search(content_regex, content, re.DOTALL):
                    content_scores.append(0.7)
                    continue
                
                # Content is valid
                content_scores.append(1.0)
            
            # Calculate average content score
            avg_content_score = sum(content_scores) / len(content_scores) if content_scores else 1.0
            
            # Calculate overall tag score
            if min_count <= count <= max_count:
                count_score = 1.0
            else:
                # This shouldn't happen due to earlier checks, but just in case
                count_score = 0.5
            
            tag_score = 0.6 * count_score + 0.4 * avg_content_score
            
            tag_scores[tag_name] = {
                "score": tag_score,
                "count": count,
                "content_score": avg_content_score
            }
            
            # Add to total score with appropriate weight
            weight = 2.0 if spec['required'] else 1.0
            total_score += tag_score * weight
            total_weight += weight
        
        # Calculate final score
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        
        return final_score, {
            "tag_scores": tag_scores,
            "overall_score": final_score
        }
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the tag compliance reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Not used in this reward function
            reference: Not used in this reward function
            **kwargs: Additional arguments
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        
        rewards = []
        for resp in responses:
            score, _ = self._evaluate_tag_compliance(resp)
            rewards.append(self._normalize_score(score))
        
        return rewards[0] if len(rewards) == 1 else rewards