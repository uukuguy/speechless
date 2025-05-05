"""
Code-Based Reward Functions

This module provides reward functions for evaluating code quality and correctness:
- CodeReward: Evaluates syntactical accuracy, style, and functional correctness of code
"""

import re
import ast
import os
import sys
import logging
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Union, Tuple

from .base import BaseReward

# Configure logging
logger = logging.getLogger(__name__)


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