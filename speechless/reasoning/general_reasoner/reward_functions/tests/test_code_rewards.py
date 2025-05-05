"""
Tests for Code-Based Reward Functions

This module contains unit tests for the code-based reward functions:
- CodeReward
"""

import unittest
from unittest.mock import MagicMock, patch

from ..code_rewards import CodeReward


class TestCodeReward(unittest.TestCase):
    """Test cases for the CodeReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward = CodeReward()
    
    def test_detect_language(self):
        """Test that _detect_language correctly identifies programming languages."""
        # Python
        python_code = """
        def hello_world():
            print("Hello, world!")
            
        if __name__ == "__main__":
            hello_world()
        """
        self.assertEqual(self.reward._detect_language(python_code), "python")
        
        # JavaScript
        js_code = """
        function helloWorld() {
            console.log("Hello, world!");
        }
        
        const greeting = "Hello";
        """
        self.assertEqual(self.reward._detect_language(js_code), "javascript")
        
        # With explicit language marker
        marked_code = "```python\ndef hello():\n    pass\n```"
        self.assertEqual(self.reward._detect_language(marked_code), "python")
        
        # Test with explicit language setting
        reward = CodeReward(language="java")
        self.assertEqual(reward._detect_language("any code here"), "java")
    
    def test_extract_code_blocks(self):
        """Test that _extract_code_blocks correctly extracts code from markdown."""
        # Code with markdown code block
        markdown_code = """
        Here's a Python function:
        
        ```python
        def hello_world():
            print("Hello, world!")
        ```
        
        That's all!
        """
        
        blocks = self.reward._extract_code_blocks(markdown_code, "python")
        self.assertEqual(len(blocks), 1)
        self.assertIn("def hello_world():", blocks[0])
        
        # Multiple code blocks
        multiple_blocks = """
        ```python
        def func1():
            pass
        ```
        
        And another:
        
        ```python
        def func2():
            pass
        ```
        """
        
        blocks = self.reward._extract_code_blocks(multiple_blocks, "python")
        self.assertEqual(len(blocks), 2)
        
        # Plain code without markdown
        plain_code = "def hello(): pass"
        blocks = self.reward._extract_code_blocks(plain_code, "python")
        self.assertEqual(blocks, [plain_code])
    
    def test_check_syntax(self):
        """Test that _check_syntax correctly evaluates code syntax."""
        # Valid Python
        valid_python = "def hello(): return 'world'"
        is_valid, score = self.reward._check_syntax(valid_python, "python")
        self.assertTrue(is_valid)
        self.assertEqual(score, 1.0)
        
        # Invalid Python
        invalid_python = "def hello() return 'world'"  # Missing colon
        is_valid, score = self.reward._check_syntax(invalid_python, "python")
        self.assertFalse(is_valid)
        self.assertLess(score, 1.0)
        
        # Other language with balanced delimiters
        js_code = "function hello() { return 'world'; }"
        is_valid, score = self.reward._check_syntax(js_code, "javascript")
        self.assertTrue(is_valid)
        self.assertGreaterEqual(score, 0.8)
        
        # Other language with unbalanced delimiters
        invalid_js = "function hello() { return 'world';"  # Missing closing brace
        is_valid, score = self.reward._check_syntax(invalid_js, "javascript")
        self.assertFalse(is_valid)
        self.assertLess(score, 0.8)
    
    def test_check_style(self):
        """Test that _check_style correctly evaluates code style."""
        # Good style Python with comments and consistent indentation
        good_style = """
        # This function says hello
        def hello_world():
            '''
            A simple function that prints a greeting.
            '''
            # Print the greeting
            print("Hello, world!")
            return True
        """
        
        # Poor style with inconsistent indentation and no comments
        poor_style = """
        def hello_world():
          print("Hello, world!")
         return True
        """
        
        good_score = self.reward._check_style(good_style, "python")
        poor_score = self.reward._check_style(poor_style, "python")
        
        self.assertGreater(good_score, poor_score)
        self.assertGreater(good_score, 0.7)
        self.assertLess(poor_score, 0.7)
    
    @patch('subprocess.Popen')
    def test_execute_code(self, mock_popen):
        """Test that _execute_code correctly runs code with test cases."""
        # Mock the subprocess.Popen
        process_mock = MagicMock()
        process_mock.communicate.return_value = ("42", "")
        mock_popen.return_value = process_mock
        
        # Create test cases
        test_cases = [
            {"inputs": ["10"], "expected_output": "42"}
        ]
        
        # Test execution
        success, score, output = self.reward._execute_code(
            "print(int(input()) * 4 + 2)", "python", test_cases
        )
        
        self.assertTrue(success)
        self.assertEqual(score, 1.0)
        self.assertIn("PASSED", output)
        
        # Test with failing test case
        process_mock.communicate.return_value = ("43", "")
        success, score, output = self.reward._execute_code(
            "print(int(input()) * 4 + 3)", "python", test_cases
        )
        
        self.assertFalse(success)
        self.assertEqual(score, 0.0)
        self.assertIn("FAILED", output)
        
        # Test with error
        process_mock.communicate.return_value = ("", "Error message")
        success, score, output = self.reward._execute_code(
            "print(undefined_variable)", "python", test_cases
        )
        
        self.assertFalse(success)
        self.assertEqual(score, 0.0)
        self.assertIn("FAILED", output)
    
    def test_compute_reward(self):
        """Test that compute_reward correctly combines component scores."""
        # Create a reward that only checks syntax and style
        reward = CodeReward(check_syntax=True, check_style=True, check_execution=False)
        
        # Good code
        good_code = """
        # A function that adds two numbers
        def add(a, b):
            '''Add two numbers and return the result.'''
            return a + b
        """
        
        # Bad code with syntax error
        bad_code = """
        # A function with a syntax error
        def add(a, b)
            return a + b
        """
        
        good_score = reward.compute_reward(good_code)
        bad_score = reward.compute_reward(bad_code)
        
        self.assertGreater(good_score, bad_score)
        self.assertGreater(good_score, 0.7)
        self.assertLess(bad_score, 0.5)
    
    def test_batch_processing(self):
        """Test that the reward function handles batch inputs correctly."""
        # Create a reward that only checks syntax
        reward = CodeReward(check_syntax=True, check_style=False, check_execution=False)
        
        responses = [
            "def add(a, b): return a + b",  # Valid
            "def add(a, b) return a + b",   # Invalid
            "function add(a, b) { return a + b; }"  # Valid JS
        ]
        
        scores = reward.compute_reward(responses)
        
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], scores[1])  # Valid code should score higher than invalid


if __name__ == "__main__":
    unittest.main()