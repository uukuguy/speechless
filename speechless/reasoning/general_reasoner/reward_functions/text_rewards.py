"""
Text-Based Reward Functions

This module provides reward functions that evaluate text-based aspects of model outputs:
- LengthReward: Evaluates response length, penalizing responses that are too short or too long
- FormatReward: Evaluates adherence to specified output formats (JSON, markdown, etc.)
- CoherenceReward: Evaluates logical flow, clarity, and consistency of responses
"""

import re
import json
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from .base import BaseReward

# Configure logging
logger = logging.getLogger(__name__)


class LengthReward(BaseReward):
    """
    Reward function that penalizes responses that are too short or too long.
    
    This reward encourages optimal conciseness in responses by penalizing those
    that deviate from a target length range.
    """
    
    def __init__(self, 
                 min_length: int = 50, 
                 max_length: int = 500, 
                 optimal_length: Optional[int] = None,
                 token_based: bool = False,
                 tokenizer = None,
                 weight: float = 1.0):
        """
        Initialize the length reward function.
        
        Args:
            min_length: Minimum acceptable length (default: 50)
            max_length: Maximum acceptable length (default: 500)
            optimal_length: Optional optimal length; if provided, scores will be highest
                           for responses closest to this length
            token_based: Whether to count tokens instead of characters (default: False)
            tokenizer: Tokenizer to use if token_based is True
            weight: Weight of this reward when combined with others (default: 1.0)
        
        Raises:
            ValueError: If token_based is True but no tokenizer is provided
        """
        super().__init__(name="length", weight=weight)
        self.min_length = min_length
        self.max_length = max_length
        self.optimal_length = optimal_length
        self.token_based = token_based
        
        if token_based and tokenizer is None:
            raise ValueError("Tokenizer must be provided when token_based is True")
        
        self.tokenizer = tokenizer
    
    def compute_reward(self,
                       response: Union[str, List[str]],
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the length-based reward for a response.
        
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
            if self.token_based and self.tokenizer:
                length = len(self.tokenizer.encode(resp))
            else:
                length = len(resp)
            
            # If response is too short or too long, penalize it
            if length < self.min_length:
                # Linear scaling from 0 at length=0 to 0.5 at length=min_length
                reward = 0.5 * (length / self.min_length)
            elif length > self.max_length:
                # Exponential decay for very long responses
                excess = length - self.max_length
                decay_factor = 0.01  # Controls how quickly the score decays
                reward = 0.5 * math.exp(-decay_factor * excess)
            else:
                # Response is within acceptable range
                if self.optimal_length:
                    # Score based on proximity to optimal length
                    distance = abs(length - self.optimal_length)
                    max_distance = max(self.optimal_length - self.min_length,
                                      self.max_length - self.optimal_length)
                    proximity = 1 - (distance / max_distance if max_distance > 0 else 0)
                    reward = 0.5 + 0.5 * proximity
                else:
                    # Linear scaling from 0.5 at boundaries to 1.0 at midpoint
                    position = (length - self.min_length) / (self.max_length - self.min_length)
                    # Parabolic curve peaking at position=0.5
                    reward = 0.5 + 0.5 * (1 - 4 * (position - 0.5) ** 2)
            
            rewards.append(self._normalize_score(reward))
        
        return rewards[0] if len(rewards) == 1 else rewards


class FormatReward(BaseReward):
    """
    Reward function that evaluates adherence to specified output formats.
    
    This reward encourages responses to follow structured formats like JSON,
    markdown, bullet points, etc. when specified in the prompt.
    """
    
    def __init__(self, 
                 format_type: Optional[str] = None,
                 format_regex: Optional[str] = None,
                 json_schema: Optional[Dict] = None,
                 weight: float = 1.0):
        """
        Initialize the format reward function.
        
        Args:
            format_type: Type of format to check ('json', 'markdown', 'bullet', 'table', 'code')
            format_regex: Custom regex pattern to match the desired format
            json_schema: JSON schema to validate against (if format_type is 'json')
            weight: Weight of this reward when combined with others (default: 1.0)
            
        Note:
            Either format_type or format_regex must be provided.
        """
        super().__init__(name="format", weight=weight)
        self.format_type = format_type
        self.format_regex = format_regex
        self.json_schema = json_schema
        
        # Predefined regex patterns for common formats
        self.format_patterns = {
            'json': r'^\s*\{\s*".*}\s*$',
            'markdown_heading': r'^\s*(#{1,6}\s+.+|.+\n[=-]+\n)',
            'bullet': r'^\s*[-*+]\s+.+(\n\s*[-*+]\s+.+)*\s*$',
            'numbered_list': r'^\s*\d+\.\s+.+(\n\s*\d+\.\s+.+)*\s*$',
            'table': r'^\s*\|.+\|\s*\n\s*\|[-:| ]+\|\s*\n(\s*\|.+\|\s*\n)+\s*$',
            'code_block': r'```[\w]*\n[\s\S]*?\n```',
        }
    
    def _validate_json(self, text: str) -> Tuple[bool, float]:
        """
        Validate if the text is valid JSON and optionally check against a schema.
        
        Args:
            text: Text to validate as JSON
            
        Returns:
            Tuple of (is_valid, score)
        """
        try:
            # Try to parse as JSON
            data = json.loads(text)
            
            # Basic validation passed
            score = 0.7
            
            # Check against schema if provided
            if self.json_schema:
                try:
                    from jsonschema import validate, ValidationError
                    validate(instance=data, schema=self.json_schema)
                    score = 1.0
                except ValidationError:
                    score = 0.5
                except ImportError:
                    logger.warning("jsonschema package not installed. Schema validation skipped.")
            else:
                # No schema, but valid JSON
                score = 0.9
                
            return True, score
        except json.JSONDecodeError:
            # Check if it's close to valid JSON (e.g., missing quotes, commas)
            if re.search(r'^\s*\{.*\}\s*$', text, re.DOTALL):
                return False, 0.3
            return False, 0.0
    
    def _check_format_adherence(self, text: str) -> float:
        """
        Check how well the text adheres to the specified format.
        
        Args:
            text: Text to check
            
        Returns:
            Score between 0 and 1 indicating format adherence
        """
        if self.format_type == 'json':
            _, score = self._validate_json(text)
            return score
        
        # Use the appropriate regex pattern
        pattern = self.format_regex
        if not pattern and self.format_type in self.format_patterns:
            pattern = self.format_patterns[self.format_type]
        
        if not pattern:
            logger.warning("No format pattern specified. Returning neutral score.")
            return 0.5
        
        # Check if the text matches the pattern
        if re.search(pattern, text, re.DOTALL | re.MULTILINE):
            return 1.0
        
        # Partial matching for some formats
        if self.format_type == 'bullet' and re.search(r'[-*+]\s+', text):
            return 0.5
        if self.format_type == 'numbered_list' and re.search(r'\d+\.\s+', text):
            return 0.5
        if self.format_type == 'table' and '|' in text and '-' in text:
            return 0.3
        if self.format_type == 'code_block' and '```' in text:
            return 0.4
        
        return 0.0
    
    def _detect_format_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Detect the required format from the prompt.
        
        Args:
            prompt: Prompt to analyze
            
        Returns:
            Detected format type or None
        """
        prompt_lower = prompt.lower()
        
        # Check for explicit format requests
        if re.search(r'(respond|answer|reply|format).+(json|in json|as json)', prompt_lower):
            return 'json'
        if re.search(r'(use|respond with|format as|in) markdown', prompt_lower):
            return 'markdown_heading'
        if re.search(r'(use|respond with|format as|in) bullet points', prompt_lower):
            return 'bullet'
        if re.search(r'(use|respond with|format as|in) numbered list', prompt_lower):
            return 'numbered_list'
        if re.search(r'(use|respond with|format as|in) table', prompt_lower):
            return 'table'
        if re.search(r'(use|respond with|format as|in) code( block)?', prompt_lower):
            return 'code_block'
        
        return None
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the format adherence reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Optional prompt(s) that generated the response
            reference: Not used in this reward function
            **kwargs: Additional arguments
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        prompts = self._ensure_list(prompt) if prompt is not None else [None] * len(responses)
        
        rewards = []
        for resp, pmt in zip(responses, prompts):
            # If format_type not explicitly set, try to detect from prompt
            format_type = self.format_type
            if not format_type and pmt:
                format_type = self._detect_format_from_prompt(pmt)
            
            if not format_type and not self.format_regex:
                # No format specified or detected
                rewards.append(0.5)  # Neutral score
                continue
            
            # Store original format_type
            original_format_type = self.format_type
            if format_type and format_type != self.format_type:
                self.format_type = format_type
            
            # Check format adherence
            score = self._check_format_adherence(resp)
            rewards.append(score)
            
            # Restore original format_type
            self.format_type = original_format_type
        
        return rewards[0] if len(rewards) == 1 else rewards


class CoherenceReward(BaseReward):
    """
    Reward function that evaluates logical flow, clarity, and consistency.
    
    This reward measures the coherence of responses, particularly in multi-turn
    conversations or complex reasoning tasks.
    """
    
    def __init__(self, 
                 check_logical_flow: bool = True,
                 check_consistency: bool = True,
                 check_clarity: bool = True,
                 weight: float = 1.0):
        """
        Initialize the coherence reward function.
        
        Args:
            check_logical_flow: Whether to check for logical flow (default: True)
            check_consistency: Whether to check for internal consistency (default: True)
            check_clarity: Whether to check for clarity (default: True)
            weight: Weight of this reward when combined with others (default: 1.0)
        """
        super().__init__(name="coherence", weight=weight)
        self.check_logical_flow = check_logical_flow
        self.check_consistency = check_consistency
        self.check_clarity = check_clarity
    
    def _check_logical_flow(self, text: str) -> float:
        """
        Check the logical flow of the text.
        
        Args:
            text: Text to check
            
        Returns:
            Score between 0 and 1 for logical flow
        """
        # Check for discourse markers that indicate logical structure
        discourse_markers = [
            r'\bfirst\b|\bsecond\b|\bthird\b|\bfinally\b',
            r'\btherefore\b|\bthus\b|\bhence\b|\bconsequently\b',
            r'\bbecause\b|\bsince\b|\bas a result\b|\bdue to\b',
            r'\bhowever\b|\bnevertheless\b|\bin contrast\b|\bon the other hand\b',
            r'\bfor example\b|\bfor instance\b|\bsuch as\b',
            r'\bin conclusion\b|\bto summarize\b|\bin summary\b'
        ]
        
        # Count discourse markers
        marker_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in discourse_markers)
        
        # Normalize by text length
        text_length = len(text.split())
        normalized_count = min(1.0, marker_count / (text_length / 50))
        
        # Check for paragraph structure
        paragraphs = text.split('\n\n')
        has_paragraphs = len(paragraphs) > 1
        
        # Check for topic sentences
        has_topic_sentences = True
        for para in paragraphs:
            if len(para.split()) > 20:  # Only check substantial paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', para)
                if sentences and len(sentences[0].split()) < 5:
                    has_topic_sentences = False
        
        # Calculate score
        score = 0.3 * normalized_count
        if has_paragraphs:
            score += 0.3
        if has_topic_sentences:
            score += 0.2
        
        # Check for overall structure (beginning, middle, end)
        if len(text.split()) > 100:
            has_intro = bool(re.search(r'\b(introduction|begin|start|first|initially)\b', text[:len(text)//3], re.IGNORECASE))
            has_conclusion = bool(re.search(r'\b(conclusion|summary|finally|in summary|to conclude)\b', text[-len(text)//3:], re.IGNORECASE))
            
            if has_intro:
                score += 0.1
            if has_conclusion:
                score += 0.1
        
        return self._normalize_score(score)
    
    def _check_consistency(self, text: str) -> float:
        """
        Check the internal consistency of the text.
        
        Args:
            text: Text to check
            
        Returns:
            Score between 0 and 1 for consistency
        """
        # Extract statements that might contradict each other
        statements = re.split(r'(?<=[.!?])\s+', text)
        
        # Look for numerical inconsistencies
        numbers = {}
        for statement in statements:
            # Extract numbers with their context
            matches = re.findall(r'(\w+(?:\s+\w+){0,3})\s+(?:is|was|are|were)\s+(\d+)', statement)
            for context, number in matches:
                context = context.lower()
                if context in numbers and numbers[context] != number:
                    # Found inconsistency
                    return 0.5
                numbers[context] = number
        
        # Look for logical contradictions using simple patterns
        contradictions = 0
        for i, stmt1 in enumerate(statements):
            stmt1_lower = stmt1.lower()
            
            # Check for negation patterns
            is_positive = 'is ' in stmt1_lower or 'are ' in stmt1_lower or 'was ' in stmt1_lower or 'were ' in stmt1_lower
            is_negative = 'is not ' in stmt1_lower or 'are not ' in stmt1_lower or 'was not ' in stmt1_lower or 'were not ' in stmt1_lower
            
            if is_positive or is_negative:
                for stmt2 in statements[i+1:]:
                    stmt2_lower = stmt2.lower()
                    
                    # Check if the statements might be about the same subject
                    # This is a simplification; a more sophisticated approach would use coreference resolution
                    words1 = set(re.findall(r'\b\w+\b', stmt1_lower))
                    words2 = set(re.findall(r'\b\w+\b', stmt2_lower))
                    common_words = words1.intersection(words2)
                    
                    if len(common_words) >= 3:  # Arbitrary threshold for "same topic"
                        if (is_positive and 'not ' in stmt2_lower) or (is_negative and 'not ' not in stmt2_lower):
                            contradictions += 1
        
        # Penalize based on number of contradictions
        if contradictions > 0:
            return max(0.2, 1.0 - (contradictions * 0.2))
        
        return 0.9  # High score if no obvious contradictions found
    
    def _check_clarity(self, text: str) -> float:
        """
        Check the clarity of the text.
        
        Args:
            text: Text to check
            
        Returns:
            Score between 0 and 1 for clarity
        """
        # Check for readability using a simplified Flesch-Kincaid formula
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not words or not sentences:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Penalize very long sentences
        if avg_sentence_length > 25:
            sentence_score = 0.5
        elif avg_sentence_length > 20:
            sentence_score = 0.7
        else:
            sentence_score = 0.9
        
        # Check for use of jargon and complex words
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_ratio = complex_words / len(words) if words else 0
        
        if complex_ratio > 0.2:
            complexity_score = 0.6
        else:
            complexity_score = 0.9
        
        # Check for passive voice (simplified)
        passive_count = len(re.findall(r'\b(?:is|are|was|were|be|been|being)\s+\w+ed\b', text, re.IGNORECASE))
        passive_ratio = passive_count / len(sentences) if sentences else 0
        
        if passive_ratio > 0.5:
            passive_score = 0.7
        else:
            passive_score = 0.9
        
        # Calculate overall clarity score
        clarity_score = 0.4 * sentence_score + 0.4 * complexity_score + 0.2 * passive_score
        
        return self._normalize_score(clarity_score)
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the coherence reward for a response.
        
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
            score = 0.0
            components = 0
            
            # Check logical flow
            if self.check_logical_flow:
                flow_score = self._check_logical_flow(resp)
                score += flow_score
                components += 1
            
            # Check consistency
            if self.check_consistency:
                consistency_score = self._check_consistency(resp)
                score += consistency_score
                components += 1
            
            # Check clarity
            if self.check_clarity:
                clarity_score = self._check_clarity(resp)
                score += clarity_score
                components += 1
            
            # Calculate final score
            final_score = score / components if components > 0 else 0.5
            rewards.append(self._normalize_score(final_score))
        
        return rewards[0] if len(rewards) == 1 else rewards