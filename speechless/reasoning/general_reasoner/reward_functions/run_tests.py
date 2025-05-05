#!/usr/bin/env python
"""
Test Runner for Reward Functions

This script runs all the unit tests for the reward functions package.
"""

import unittest
import sys
import os

if __name__ == "__main__":
    # Add the parent directory to the path so we can import the package
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())