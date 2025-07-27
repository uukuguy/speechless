"""
Task-Specific Reward Functions

This module provides reward functions for domain-specific objectives:
- TaskSpecificReward: Customizable reward for specific tasks like summarization, translation, or QA
"""

import re
import math
import logging
from typing import List, Dict, Any, Union, Optional, Callable

from .base import BaseReward

# Configure logging
logger = logging.getLogger(__name__)


class TaskSpecificReward(BaseReward):
    """
    Reward function for domain-specific objectives.
    
    This reward allows for customization based on specific tasks such as
    summarization, translation, or other domain-specific requirements.
    """
    
    def __init__(self, 
                 task_type: str,
                 custom_reward_fn: Optional[Callable] = None,
                 task_params: Optional[Dict[str, Any]] = None,
                 weight: float = 1.0):
        """
        Initialize the task-specific reward function.
        
        Args:
            task_type: Type of task ('summarization', 'translation', 'qa', etc.)
            custom_reward_fn: Optional custom reward function
            task_params: Optional parameters specific to the task
            weight: Weight of this reward when combined with others (default: 1.0)
        """
        super().__init__(name=f"task_{task_type}", weight=weight)
        self.task_type = task_type
        self.custom_reward_fn = custom_reward_fn
        self.task_params = task_params or {}
    
    def _summarization_reward(self, response: str, reference: str) -> float:
        """
        Compute reward for summarization tasks.
        
        Args:
            response: Model-generated summary
            reference: Reference summary or original text
            
        Returns:
            Score between 0 and 1
        """
        # Check length ratio
        target_ratio = self.task_params.get('target_ratio', 0.2)
        response_words = len(response.split())
        reference_words = len(reference.split())
        
        length_ratio = response_words / reference_words if reference_words > 0 else 0
        length_score = 1.0 - min(1.0, abs(length_ratio - target_ratio) / target_ratio)
        
        # Check content coverage using simple lexical overlap
        response_tokens = set(re.findall(r'\b\w+\b', response.lower()))
        reference_tokens = set(re.findall(r'\b\w+\b', reference.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
        response_tokens = response_tokens - stop_words
        reference_tokens = reference_tokens - stop_words
        
        # Calculate coverage
        if not reference_tokens:
            coverage_score = 0.5
        else:
            # Focus on important words (longer words tend to be more important)
            important_ref_tokens = {token for token in reference_tokens if len(token) > 4}
            important_resp_tokens = {token for token in response_tokens if len(token) > 4}
            
            if not important_ref_tokens:
                coverage_score = 0.5
            else:
                coverage = len(important_resp_tokens.intersection(important_ref_tokens)) / len(important_ref_tokens)
                coverage_score = min(1.0, coverage * 1.5)  # Boost coverage score
        
        # Check for novel n-grams (paraphrasing)
        ref_bigrams = set()
        for i in range(len(reference.split()) - 1):
            ref_bigrams.add(' '.join(reference.split()[i:i+2]).lower())
        
        resp_bigrams = set()
        for i in range(len(response.split()) - 1):
            resp_bigrams.add(' '.join(response.split()[i:i+2]).lower())
        
        if not ref_bigrams:
            novelty_score = 0.5
        else:
            # Some novel bigrams are good (paraphrasing), but too many might indicate hallucination
            novel_ratio = 1 - (len(resp_bigrams.intersection(ref_bigrams)) / len(resp_bigrams) if resp_bigrams else 0)
            
            # Ideal novelty is around 30-70%
            if 0.3 <= novel_ratio <= 0.7:
                novelty_score = 0.9
            elif novel_ratio < 0.3:
                novelty_score = 0.7  # Too similar to original
            else:
                novelty_score = 0.5  # Too different from original
        
        # Combine scores with weights
        final_score = 0.3 * length_score + 0.5 * coverage_score + 0.2 * novelty_score
        
        return self._normalize_score(final_score)
    
    def _translation_reward(self, response: str, reference: str) -> float:
        """
        Compute reward for translation tasks.
        
        Args:
            response: Model-generated translation
            reference: Reference translation
            
        Returns:
            Score between 0 and 1
        """
        # This is a simplified implementation
        # A more sophisticated approach would use BLEU, METEOR, or other MT metrics
        
        # Check length ratio
        response_words = len(response.split())
        reference_words = len(reference.split())
        
        length_ratio = response_words / reference_words if reference_words > 0 else 0
        length_score = 1.0 - min(1.0, abs(length_ratio - 1.0) / 0.5)
        
        # Check for n-gram overlap (simplified BLEU-like)
        def get_ngrams(text, n):
            words = text.lower().split()
            ngrams = []
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i+n]))
            return ngrams
        
        overlap_scores = []
        for n in range(1, 5):  # 1-gram to 4-gram
            ref_ngrams = get_ngrams(reference, n)
            resp_ngrams = get_ngrams(response, n)
            
            if not resp_ngrams:
                overlap_scores.append(0.0)
                continue
            
            matches = sum(1 for ng in resp_ngrams if ng in ref_ngrams)
            precision = matches / len(resp_ngrams)
            overlap_scores.append(precision)
        
        # Apply brevity penalty
        bp = 1.0
        if response_words < reference_words:
            bp = math.exp(1 - reference_words / response_words) if response_words > 0 else 0.0
        
        # Calculate final score (geometric mean of n-gram precisions with brevity penalty)
        if all(score == 0 for score in overlap_scores):
            ngram_score = 0.0
        else:
            # Replace zeros with a small value to avoid zero product
            adjusted_scores = [max(0.01, score) for score in overlap_scores]
            ngram_score = bp * (adjusted_scores[0] ** 0.4 * 
                               adjusted_scores[1] ** 0.3 * 
                               adjusted_scores[2] ** 0.2 * 
                               adjusted_scores[3] ** 0.1)
        
        return self._normalize_score(ngram_score)
    
    def _qa_reward(self, response: str, reference: str, question: Optional[str] = None) -> float:
        """
        Compute reward for question-answering tasks.
        
        Args:
            response: Model-generated answer
            reference: Reference answer
            question: Optional question text
            
        Returns:
            Score between 0 and 1
        """
        # Check for exact match or substring match
        if response.lower() == reference.lower():
            return 1.0
        
        if response.lower() in reference.lower() or reference.lower() in response.lower():
            return 0.8
        
        # Check for semantic similarity using token overlap
        response_tokens = set(re.findall(r'\b\w+\b', response.lower()))
        reference_tokens = set(re.findall(r'\b\w+\b', reference.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
        response_tokens = response_tokens - stop_words
        reference_tokens = reference_tokens - stop_words
        
        if not reference_tokens:
            return 0.5
        
        # Calculate F1 score
        if not response_tokens:
            return 0.0
        
        precision = len(response_tokens.intersection(reference_tokens)) / len(response_tokens)
        recall = len(response_tokens.intersection(reference_tokens)) / len(reference_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        # Check if the answer addresses the question
        question_relevance = 0.8  # Default if no question provided
        if question:
            question_tokens = set(re.findall(r'\b\w+\b', question.lower())) - stop_words
            if question_tokens:
                question_overlap = len(response_tokens.intersection(question_tokens)) / len(question_tokens)
                question_relevance = min(1.0, question_overlap * 2)  # Scale up to reward relevance
        
        # Combine scores
        final_score = 0.7 * f1 + 0.3 * question_relevance
        
        return self._normalize_score(final_score)
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the task-specific reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Optional prompt(s) that generated the response
            reference: Optional reference response(s) for comparison
            **kwargs: Additional arguments specific to the task
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        
        # Use custom reward function if provided
        if self.custom_reward_fn:
            try:
                return self.custom_reward_fn(response, prompt, reference, **kwargs)
            except Exception as e:
                logger.error(f"Error in custom reward function: {e}")
                return [0.5] * len(responses)  # Neutral score on error
        
        # Ensure reference is provided for task-specific rewards
        if reference is None:
            logger.warning(f"No reference provided for {self.task_type} reward. Returning neutral score.")
            return [0.5] * len(responses)
        
        references = self._ensure_list(reference)
        
        # Ensure equal length of responses and references
        if len(responses) != len(references):
            if len(references) == 1:
                references = references * len(responses)
            else:
                logger.warning("Mismatched number of responses and references. Using first reference for all.")
                references = [references[0]] * len(responses)
        
        rewards = []
        for resp, ref in zip(responses, references):
            # Select the appropriate reward function based on task type
            if self.task_type == 'summarization':
                score = self._summarization_reward(resp, ref)
            elif self.task_type == 'translation':
                score = self._translation_reward(resp, ref)
            elif self.task_type == 'qa':
                question = kwargs.get('question', prompt[0] if prompt else None)
                score = self._qa_reward(resp, ref, question)
            else:
                logger.warning(f"Unknown task type: {self.task_type}. Returning neutral score.")
                score = 0.5
            
            rewards.append(score)
        
        return rewards