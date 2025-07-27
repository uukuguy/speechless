"""
Math-Based Reward Functions

This module provides reward functions for evaluating mathematical problem-solving:
- MathReward: Evaluates mathematical reasoning and correctness of solutions
- MathVerifyReward: Uses the math-verify library for symbolic verification of solutions
"""

import re
import logging
import math
from typing import List, Optional, Union, Dict, Any

from .base import BaseReward

# Configure logging
logger = logging.getLogger(__name__)

# Try to import math-verify
try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    logger.warning("To use MathVerifyReward, please install math-verify by running `pip install math-verify`.")


class MathReward(BaseReward):
    """
    Reward function that evaluates mathematical problem-solving.
    
    This reward assesses the correctness of mathematical solutions and the quality
    of the reasoning steps provided.
    """
    
    def __init__(self, 
                 check_final_answer: bool = True,
                 check_reasoning_steps: bool = True,
                 answer_regex: str = r'(?:answer|result)(?:\s+is)?(?:\s*[:=])?\s*(-?\d+\.?\d*|\d*\.\d+)',
                 weight: float = 1.0):
        """
        Initialize the math reward function.
        
        Args:
            check_final_answer: Whether to check the final numerical answer (default: True)
            check_reasoning_steps: Whether to check the quality of reasoning steps (default: True)
            answer_regex: Regex pattern to extract the final answer (default: matches common formats)
            weight: Weight of this reward when combined with others (default: 1.0)
        """
        super().__init__(name="math", weight=weight)
        self.check_final_answer = check_final_answer
        self.check_reasoning_steps = check_reasoning_steps
        self.answer_regex = answer_regex
    
    def _extract_answer(self, text: str) -> Optional[float]:
        """
        Extract the final numerical answer from text.
        
        Args:
            text: Text to extract answer from
            
        Returns:
            Extracted numerical answer or None if not found
        """
        match = re.search(self.answer_regex, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        
        # Try to find the last number in the text
        numbers = re.findall(r'-?\d+\.?\d*|\d*\.\d+', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def _evaluate_reasoning_steps(self, text: str) -> float:
        """
        Evaluate the quality of mathematical reasoning steps.
        
        Args:
            text: Text containing reasoning steps
            
        Returns:
            Score between 0 and 1 for reasoning quality
        """
        # Check for step-by-step working
        has_steps = bool(re.search(r'step\s+\d|first|second|third|next|then|finally', text, re.IGNORECASE))
        
        # Check for mathematical operations
        has_operations = bool(re.search(r'[\+\-\*\/\=\(\)]|plus|minus|times|divided by|multiply|subtract|add', text))
        
        # Check for explanations
        has_explanations = bool(re.search(r'because|since|therefore|thus|so|as a result', text, re.IGNORECASE))
        
        # Count mathematical symbols and numbers
        math_symbols = len(re.findall(r'[\+\-\*\/\=\(\)\^\{\}\[\]]', text))
        numbers = len(re.findall(r'\d+\.?\d*|\d*\.\d+', text))
        
        # Calculate reasoning score
        score = 0.0
        if has_steps:
            score += 0.3
        if has_operations:
            score += 0.3
        if has_explanations:
            score += 0.2
        
        # Bonus for richness of mathematical content
        math_content_score = min(0.2, (math_symbols + numbers) / 50)
        score += math_content_score
        
        return self._normalize_score(score)
    
    def _check_answer_correctness(self, response_answer: float, reference_answer: float) -> float:
        """
        Check the correctness of the numerical answer.
        
        Args:
            response_answer: Answer extracted from the response
            reference_answer: Correct answer
            
        Returns:
            Score between 0 and 1 for answer correctness
        """
        # Exact match
        if response_answer == reference_answer:
            return 1.0
        
        # Check for approximate match with relative tolerance
        rel_tol = 1e-9
        abs_tol = 1e-9
        
        if abs(response_answer - reference_answer) <= abs_tol + rel_tol * abs(reference_answer):
            return 0.9
        
        # Calculate relative error
        if reference_answer != 0:
            rel_error = abs((response_answer - reference_answer) / reference_answer)
            
            # Score based on relative error
            if rel_error < 0.01:  # Within 1%
                return 0.8
            elif rel_error < 0.05:  # Within 5%
                return 0.6
            elif rel_error < 0.1:  # Within 10%
                return 0.4
            elif rel_error < 0.5:  # Within 50%
                return 0.2
        
        return 0.0
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       reference_answer: Optional[Union[float, List[float]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the math reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Not used in this reward function
            reference: Reference response(s) containing the correct answer
            reference_answer: Optional explicit reference answer(s)
            **kwargs: Additional arguments
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        
        if reference_answer is not None:
            ref_answers = self._ensure_list(reference_answer)
        elif reference is not None:
            references = self._ensure_list(reference)
            ref_answers = [self._extract_answer(ref) for ref in references]
        else:
            # No reference provided
            logger.warning("No reference answer provided for math reward. Evaluating reasoning only.")
            ref_answers = [None] * len(responses)
        
        rewards = []
        for resp, ref_ans in zip(responses, ref_answers):
            score = 0.0
            components = 0
            
            # Evaluate reasoning steps
            if self.check_reasoning_steps:
                reasoning_score = self._evaluate_reasoning_steps(resp)
                score += reasoning_score
                components += 1
            
            # Check final answer if reference is available
            if self.check_final_answer and ref_ans is not None:
                resp_ans = self._extract_answer(resp)
                if resp_ans is not None:
                    answer_score = self._check_answer_correctness(resp_ans, ref_ans)
                    score += answer_score
                    components += 1
                else:
                    # Penalize for not providing a clear numerical answer
                    score += 0.0
                    components += 1
            
            # Calculate final score
            final_score = score / components if components > 0 else 0.5
            rewards.append(self._normalize_score(final_score))
        
        return rewards

class MathVerifyReward(BaseReward):
    """
    Reward function that evaluates mathematical solutions using the math-verify library.
    
    This reward assesses the correctness of mathematical solutions by comparing them
    to reference answers using symbolic verification techniques.
    """
    
    def __init__(self,
                 boxed_format: bool = True,
                 weight: float = 1.0):
        """
        Initialize the math verification reward function.
        
        Args:
            boxed_format: Whether to wrap the ground truth in \\boxed{} format (default: True)
            weight: Weight of this reward when combined with others (default: 1.0)
        """
        super().__init__(name="math_verify", weight=weight)
        self.boxed_format = boxed_format
        
        if not MATH_VERIFY_AVAILABLE:
            logger.warning("MathVerifyReward initialized but math-verify is not installed.")
    
    def compute_reward(self,
                       response: Union[str, List[str]],
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the math verification reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Not used in this reward function
            reference: Reference answer(s) containing the correct solution
            **kwargs: Additional arguments including:
                - data_source: Optional source of the problem
                - extra_info: Optional additional information
        
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        if not MATH_VERIFY_AVAILABLE:
            logger.warning("math-verify is not installed. Returning neutral score.")
            return 0.5
        
        responses = self._ensure_list(response)
        
        if reference is None:
            logger.warning("No reference answer provided for math verification. Returning neutral score.")
            return [0.5] * len(responses)
        
        references = self._ensure_list(reference)
        
        # Ensure equal length of responses and references
        if len(responses) != len(references):
            if len(references) == 1:
                references = references * len(responses)
            else:
                logger.warning("Mismatched number of responses and references. Using first reference for all.")
                references = [references[0]] * len(responses)
        
        data_source = kwargs.get('data_source', None)
        extra_info = kwargs.get('extra_info', None)
        
        rewards = []
        for resp, ref in zip(responses, references):
            # Wrap the ground truth in \boxed{} format for verification if needed
            if self.boxed_format and not ref.startswith("\\boxed{"):
                ref_boxed = "\\boxed{" + ref + "}"
            else:
                ref_boxed = ref
            
            # Initialize the verification function
            verify_func = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
            
            # Compute the score
            try:
                score, _ = verify_func([ref_boxed], [resp])
            except TimeoutException:
                logger.warning("Timeout during math verification. Returning zero score.")
                score = 0.0
            except Exception as e:
                logger.error(f"Error during math verification: {e}")
                score = 0.0
            
            rewards.append(self._normalize_score(score))
        
        return rewards