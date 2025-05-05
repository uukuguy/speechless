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
Reward Functions for General Reasoner RL Fine-tuning

This module provides a comprehensive suite of reward functions for reinforcement learning
fine-tuning of language models. Each reward function evaluates a specific aspect of model
outputs and returns a normalized score. These functions are designed to be used with the
veRL framework for training language models with reinforcement learning.

The reward functions can be used individually or combined using the CombinedReward class.
Each function supports both single-instance and batched processing for computational efficiency.
"""

import re
import json
import math
import numpy as np
from typing import List, Dict, Any, Union, Callable, Optional, Tuple
import logging
from abc import ABC, abstractmethod
import ast
import subprocess
import tempfile
import os
import difflib
import time
from collections import Counter

# Try to import math-verify
try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not MATH_VERIFY_AVAILABLE:
    logger.warning("To use MathVerifyReward, please install math-verify by running `pip install math-verify`.")


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
        
        return rewards[0] if len(rewards) == 1 else rewards
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
class CodeReward(BaseReward):
    """
    Reward function that evaluates code quality and correctness.
    
    This reward assesses syntactical accuracy, efficiency, and functional correctness
    of generated code.
    """
    
    def __init__(self, 
                 check_syntax: bool = True,
                 check_execution: bool = False,
                 check_style: bool = True,
                 language: Optional[str] = None,
                 test_cases: Optional[List[Dict[str, Any]]] = None,
                 timeout: int = 5,
                 weight: float = 1.0):
        """
        Initialize the code reward function.
        
        Args:
            check_syntax: Whether to check code syntax (default: True)
            check_execution: Whether to execute code to check functionality (default: False)
            check_style: Whether to check code style (default: True)
            language: Programming language of the code (auto-detected if None)
            test_cases: List of test cases to run if check_execution is True
            timeout: Maximum execution time in seconds (default: 5)
            weight: Weight of this reward when combined with others (default: 1.0)
            
        Note:
            Setting check_execution to True will execute code in a sandbox environment.
            Use with caution and only with trusted inputs.
        """
        super().__init__(name="code", weight=weight)
        self.check_syntax = check_syntax
        self.check_execution = check_execution
        self.check_style = check_style
        self.language = language
        self.test_cases = test_cases or []
        self.timeout = timeout
        
        # Language detection patterns
        self.language_patterns = {
            'python': r'```python|def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import',
            'javascript': r'```js|```javascript|function\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=',
            'java': r'```java|public\s+(static\s+)?(class|void|int|String)|import\s+java\.',
            'c++': r'```cpp|#include\s*<\w+>|using\s+namespace\s+std|void\s+\w+\s*\(',
            'c#': r'```csharp|using\s+System|namespace\s+\w+|public\s+(static\s+)?(class|void|int|string)',
            'ruby': r'```ruby|def\s+\w+\s*(\|.*\|)?|require\s+[\'\"]',
            'go': r'```go|func\s+\w+\s*\(|package\s+\w+|import\s+\(',
            'rust': r'```rust|fn\s+\w+\s*\(|use\s+\w+::|let\s+mut\s+\w+',
            'php': r'```php|<\?php|\$\w+\s*=|function\s+\w+\s*\(',
            'swift': r'```swift|func\s+\w+\s*\(|import\s+\w+|var\s+\w+\s*:|let\s+\w+\s*:',
        }
    
    def _detect_language(self, code: str) -> str:
        """
        Detect the programming language of the code.
        
        Args:
            code: Code to analyze
            
        Returns:
            Detected language or 'unknown'
        """
        if self.language:
            return self.language
        
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                return lang
        
        # Default to Python if no language is detected
        return 'python'
    
    def _extract_code_blocks(self, text: str, language: str) -> List[str]:
        """
        Extract code blocks from text.
        
        Args:
            text: Text containing code blocks
            language: Programming language
            
        Returns:
            List of extracted code blocks
        """
        # Try to extract code from markdown code blocks
        code_blocks = re.findall(r'```(?:' + language + r')?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        
        if code_blocks:
            return code_blocks
        
        # If no code blocks found, treat the entire text as code
        # This is a heuristic and may not always be correct
        return [text]
    
    def _check_syntax(self, code: str, language: str) -> Tuple[bool, float]:
        """
        Check the syntax of the code.
        
        Args:
            code: Code to check
            language: Programming language
            
        Returns:
            Tuple of (is_valid, score)
        """
        if language == 'python':
            try:
                ast.parse(code)
                return True, 1.0
            except SyntaxError as e:
                # Calculate how far into the code the error occurred
                try:
                    error_position = e.lineno / len(code.split('\n'))
                    # If error is near the end, code is mostly correct
                    if error_position > 0.8:
                        return False, 0.7
                    elif error_position > 0.5:
                        return False, 0.5
                    else:
                        return False, 0.2
                except:
                    return False, 0.0
        
        # For other languages, use a simple heuristic based on balanced delimiters
        delimiters = {
            '(': ')', '[': ']', '{': '}', '"': '"', "'": "'"
        }
        
        stack = []
        for char in code:
            if char in delimiters.keys():
                stack.append(char)
            elif char in delimiters.values():
                if not stack or delimiters[stack.pop()] != char:
                    return False, 0.3
        
        if stack:
            # Unbalanced delimiters
            return False, 0.5
        
        # No obvious syntax errors detected
        return True, 0.8
    
    def _check_style(self, code: str, language: str) -> float:
        """
        Check the style of the code.
        
        Args:
            code: Code to check
            language: Programming language
            
        Returns:
            Style score between 0 and 1
        """
        # Initialize score
        score = 0.5
        
        # Check indentation consistency
        lines = code.split('\n')
        indentation_types = set()
        for line in lines:
            if line.strip():
                indent = re.match(r'^(\s*)', line).group(1)
                if indent:
                    indentation_types.add(indent[0])
        
        # Consistent indentation
        if len(indentation_types) <= 1:
            score += 0.1
        
        # Check for comments
        comment_patterns = {
            'python': r'#.*$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
            'javascript': r'//.*$|/\*[\s\S]*?\*/',
            'java': r'//.*$|/\*[\s\S]*?\*/',
            'c++': r'//.*$|/\*[\s\S]*?\*/',
            'c#': r'//.*$|/\*[\s\S]*?\*/',
            'ruby': r'#.*$|=begin[\s\S]*?=end',
            'go': r'//.*$|/\*[\s\S]*?\*/',
            'rust': r'//.*$|/\*[\s\S]*?\*/',
            'php': r'//.*$|#.*$|/\*[\s\S]*?\*/',
            'swift': r'//.*$|/\*[\s\S]*?\*/',
        }
        
        pattern = comment_patterns.get(language, r'//.*$|#.*$|/\*[\s\S]*?\*/')
        comments = re.findall(pattern, code)
        
        # Has comments
        if comments:
            score += 0.1
            
            # Check comment quality (length and frequency)
            comment_ratio = len(''.join(comments)) / len(code) if code else 0
            if 0.05 <= comment_ratio <= 0.3:  # Reasonable comment ratio
                score += 0.1
        
        # Check for function/method documentation
        if language == 'python':
            has_docstrings = bool(re.search(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code))
            if has_docstrings:
                score += 0.1
        
        # Check for consistent naming conventions
        if language in ['python', 'javascript', 'java', 'c#']:
            # Check for snake_case or camelCase consistency
            variable_names = re.findall(r'\b[a-zA-Z_]\w*\b', code)
            snake_case = sum(1 for name in variable_names if re.match(r'^[a-z_][a-z0-9_]*$', name))
            camel_case = sum(1 for name in variable_names if re.match(r'^[a-z][a-zA-Z0-9]*$', name))
            pascal_case = sum(1 for name in variable_names if re.match(r'^[A-Z][a-zA-Z0-9]*$', name))
            
            if variable_names:
                consistency = max(snake_case, camel_case, pascal_case) / len(variable_names)
                if consistency > 0.8:  # High naming consistency
                    score += 0.1
        
        return self._normalize_score(score)
    
    def _execute_code(self, code: str, language: str, test_cases: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """
        Execute code with test cases in a sandbox environment.
        
        Args:
            code: Code to execute
            language: Programming language
            test_cases: List of test cases to run
            
        Returns:
            Tuple of (success, score, output)
        """
        if not test_cases:
            return False, 0.0, "No test cases provided"
        
        if language != 'python':
            return False, 0.0, f"Execution not supported for {language}"
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            f.write(code.encode('utf-8'))
            temp_file = f.name
        
        try:
            # Prepare test harness
            test_results = []
            total_tests = len(test_cases)
            passed_tests = 0
            
            for i, test_case in enumerate(test_cases):
                # Extract test case details
                inputs = test_case.get('inputs', [])
                expected_output = test_case.get('expected_output')
                
                # Create a command to run the test
                cmd = [sys.executable, temp_file]
                
                # Run the test
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Provide inputs if any
                    stdin_data = '\n'.join(str(inp) for inp in inputs) if inputs else ''
                    
                    # Set timeout
                    stdout, stderr = process.communicate(stdin_data, timeout=self.timeout)
                    
                    # Check if the output matches the expected output
                    if expected_output is not None:
                        # Normalize outputs for comparison
                        actual = stdout.strip()
                        expected = str(expected_output).strip()
                        
                        if actual == expected:
                            passed_tests += 1
                            test_results.append(f"Test {i+1}: PASSED")
                        else:
                            test_results.append(f"Test {i+1}: FAILED - Expected '{expected}', got '{actual}'")
                    else:
                        # No expected output specified, just check for errors
                        if not stderr:
                            passed_tests += 1
                            test_results.append(f"Test {i+1}: PASSED (no errors)")
                        else:
                            test_results.append(f"Test {i+1}: FAILED - Error: {stderr}")
                
                except subprocess.TimeoutExpired:
                    test_results.append(f"Test {i+1}: FAILED - Timeout after {self.timeout} seconds")
                except Exception as e:
                    test_results.append(f"Test {i+1}: FAILED - Error: {str(e)}")
            
            # Calculate score based on passed tests
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            success = passed_tests == total_tests
            
            return success, score, '\n'.join(test_results)
        
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the code quality reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Not used in this reward function
            reference: Optional reference code for comparison
            **kwargs: Additional arguments including test_cases
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        test_cases = kwargs.get('test_cases', self.test_cases)
        
        rewards = []
        for resp in responses:
            # Detect language
            language = self._detect_language(resp)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(resp, language)
            
            if not code_blocks:
                rewards.append(0.0)
                continue
            
            # Use the largest code block (heuristic)
            code = max(code_blocks, key=len)
            
            # Initialize component scores
            syntax_score = 0.0
            style_score = 0.0
            execution_score = 0.0
            components = 0
            
            # Check syntax
            if self.check_syntax:
                _, syntax_score = self._check_syntax(code, language)
                components += 1
            
            # Check style
            if self.check_style:
                style_score = self._check_style(code, language)
                components += 1
            
            # Execute code if requested
            if self.check_execution and test_cases:
                _, execution_score, _ = self._execute_code(code, language, test_cases)
                components += 1
            
            # Calculate final score
            if components > 0:
                # Weighted average: syntax is most important, then execution, then style
                if self.check_syntax and self.check_execution and self.check_style:
                    final_score = 0.4 * syntax_score + 0.4 * execution_score + 0.2 * style_score
                elif self.check_syntax and self.check_execution:
                    final_score = 0.6 * syntax_score + 0.4 * execution_score
                elif self.check_syntax and self.check_style:
                    final_score = 0.7 * syntax_score + 0.3 * style_score
                else:
                    final_score = (syntax_score + style_score + execution_score) / components
            else:
                final_score = 0.5  # Neutral score if no components checked
            
            rewards.append(self._normalize_score(final_score))
        
        return rewards[0] if len(rewards) == 1 else rewards


class FactualityReward(BaseReward):
    """
    Reward function that evaluates factual accuracy of responses.
    
    This reward cross-references outputs with trusted sources to minimize hallucinations.
    """
    
    def __init__(self, 
                 reference_texts: Optional[List[str]] = None,
                 use_embeddings: bool = False,
                 embedding_model = None,
                 check_contradictions: bool = True,
                 weight: float = 1.0):
        """
        Initialize the factuality reward function.
        
        Args:
            reference_texts: List of reference texts containing factual information
            use_embeddings: Whether to use embeddings for semantic similarity (default: False)
            embedding_model: Model to use for embeddings if use_embeddings is True
            check_contradictions: Whether to check for contradictions (default: True)
            weight: Weight of this reward when combined with others (default: 1.0)
            
        Note:
            If use_embeddings is True, embedding_model must be provided.
        """
        super().__init__(name="factuality", weight=weight)
        self.reference_texts = reference_texts or []
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.check_contradictions = check_contradictions
        
        if use_embeddings and embedding_model is None:
            raise ValueError("Embedding model must be provided when use_embeddings is True")
        
        # Precompute reference embeddings if using embeddings
        self.reference_embeddings = None
        if self.use_embeddings and self.embedding_model and self.reference_texts:
            self.reference_embeddings = self._compute_embeddings(self.reference_texts)
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of texts to compute embeddings for
            
        Returns:
            Array of embeddings
        """
        try:
            return self.embedding_model.encode(texts)
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return np.zeros((len(texts), 768))  # Default embedding size
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.use_embeddings and self.embedding_model:
            # Compute embeddings
            emb1 = self.embedding_model.encode([text1])[0]
            emb2 = self.embedding_model.encode([text2])[0]
            
            # Compute cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 > 0 and norm2 > 0:
                return np.dot(emb1, emb2) / (norm1 * norm2)
            return 0.0
        else:
            # Use simple text-based similarity
            # Tokenize and compute Jaccard similarity
            tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
            tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
    
    def _extract_facts(self, text: str) -> List[str]:
        """
        Extract factual statements from text.
        
        Args:
            text: Text to extract facts from
            
        Returns:
            List of extracted factual statements
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter for likely factual statements
        facts = []
        for sentence in sentences:
            # Skip questions, commands, and very short sentences
            if re.search(r'\?$|!$', sentence) or len(sentence.split()) < 5:
                continue
            
            # Look for factual indicators
            if (re.search(r'\bis\b|\bwas\b|\bwere\b|\bare\b|\bhave\b|\bhas\b', sentence) or
                re.search(r'\bin\b|\bat\b|\bon\b|\bduring\b|\bfrom\b', sentence) or
                re.search(r'\d{4}|\d+%|\d+\s+\w+', sentence)):  # Years, percentages, quantities
                facts.append(sentence)
        
        return facts
    
    def _check_fact_support(self, fact: str, references: List[str]) -> float:
        """
        Check how well a fact is supported by reference texts.
        
        Args:
            fact: Factual statement to check
            references: List of reference texts
            
        Returns:
            Support score between 0 and 1
        """
        max_similarity = 0.0
        
        for ref in references:
            # Split reference into sentences
            ref_sentences = re.split(r'(?<=[.!?])\s+', ref)
            
            for ref_sentence in ref_sentences:
                similarity = self._compute_similarity(fact, ref_sentence)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _detect_contradictions(self, facts: List[str], references: List[str]) -> float:
        """
        Detect contradictions between facts and references.
        
        Args:
            facts: List of factual statements
            references: List of reference texts
            
        Returns:
            Contradiction penalty between 0 and 1 (0 = many contradictions, 1 = no contradictions)
        """
        # This is a simplified implementation that looks for negation patterns
        # A more sophisticated approach would use natural language inference
        
        contradiction_count = 0
        total_comparisons = 0
        
        for fact in facts:
            fact_lower = fact.lower()
            
            # Extract key entities and predicates from the fact
            entities = set(re.findall(r'\b[A-Z][a-z]+\b', fact))  # Proper nouns
            predicates = set(re.findall(r'\b(is|was|were|are|have|has)\b\s+\w+', fact_lower))
            
            # Check for contradictions in references
            for ref in references:
                ref_lower = ref.lower()
                
                # Check if the reference mentions the same entities
                if any(entity.lower() in ref_lower for entity in entities):
                    # Look for negated versions of the predicates
                    for predicate in predicates:
                        negated = re.sub(r'\b(is|was|were|are|have|has)\b', r'is not|was not|were not|are not|have not|has not', predicate)
                        if re.search(negated, ref_lower):
                            contradiction_count += 1
                    
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 1.0  # No contradictions found
        
        # Calculate contradiction penalty
        contradiction_ratio = contradiction_count / total_comparisons
        return 1.0 - contradiction_ratio
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the factuality reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Not used in this reward function
            reference: Optional reference text(s) containing factual information
            **kwargs: Additional arguments
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        
        # Combine provided references with pre-configured reference texts
        all_references = list(self.reference_texts)
        if reference is not None:
            all_references.extend(self._ensure_list(reference))
        
        if not all_references:
            logger.warning("No reference texts provided for factuality reward. Returning neutral score.")
            return [0.5] * len(responses)
        
        rewards = []
        for resp in responses:
            # Extract factual statements from the response
            facts = self._extract_facts(resp)
            
            if not facts:
                rewards.append(0.5)  # Neutral score if no facts found
                continue
            
            # Check fact support
            support_scores = [self._check_fact_support(fact, all_references) for fact in facts]
            avg_support = sum(support_scores) / len(support_scores) if support_scores else 0.5
            
            # Check for contradictions if enabled
            contradiction_penalty = 1.0
            if self.check_contradictions:
                contradiction_penalty = self._detect_contradictions(facts, all_references)
            
            # Calculate final score
            # Higher weight on contradiction penalty to discourage hallucinations
            final_score = 0.7 * avg_support + 0.3 * contradiction_penalty
            
            rewards.append(self._normalize_score(final_score))
        
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
        
        return rewards[0] if len(rewards) == 1 else rewards


class CombinedReward(BaseReward):
    """
    Combines multiple reward functions with weighted averaging.
    
    This reward function allows for combining multiple reward functions with
    different weights to create a comprehensive evaluation metric.
    """
    
    def __init__(self, 
                 reward_functions: List[BaseReward],
                 weights: Optional[List[float]] = None,
                 name: str = "combined"):
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
                
                # Add weighted rewards
                for j in range(len(combined_rewards)):
                    combined_rewards[j] += weight * rewards[j]
            except Exception as e:
                logger.error(f"Error in reward function {reward_fn.name}: {e}")
                # Skip this reward function on error
        
        # Normalize final rewards
        combined_rewards = [self._normalize_score(r) for r in combined_rewards]
        
        return combined_rewards[0] if len(combined_rewards) == 1 else combined_rewards


# Utility function to create a reward function from a configuration
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
    
    elif reward_type == 'math_verify':
        return MathVerifyReward(
            boxed_format=config.get('boxed_format', True),
            weight=weight
        )
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


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
        
        return rewards[0] if len(rewards) == 1 else rewards


# Utility function to create a reward function from a configuration
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
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# Example usage
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


if __name__ == "__main__":
    example_usage()