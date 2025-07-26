#!/usr/bin/env python3
"""
Simplified testing without external dependencies.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic module imports"""
    print("Testing basic imports...")
    try:
        # These should work without external dependencies
        from processors import ProcessorFactory, GSM8KProcessor
        print("✅ Processors import successful")
        
        from output_manager import OutputManager, ParquetWriter
        print("✅ Output manager import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_processor_creation():
    """Test processor creation without external config"""
    print("\nTesting processor creation...")
    try:
        # Mock config class for testing
        class MockConfig:
            def __init__(self):
                self.name = "gsm8k"
                self.data_source = "test/data"
                self.input_key = "question"
                self.output_key = "answer"
                self.ability = "math"
                self.reward_style = "rule"
                self.format_prompt = True
                self.extract_answer = True
                self.custom_params = {
                    "answer_pattern": r"#### ([\-]?[0-9\.\,]+)",
                    "prompt_template": "{question} Let's think step by step."
                }
        
        mock_config = MockConfig()
        
        # Test GSM8K processor creation
        from processors import GSM8KProcessor
        processor = GSM8KProcessor(mock_config)
        print("✅ GSM8K processor created successfully")
        
        # Test answer extraction
        test_answer = "The calculation shows 2 + 2 = 4\n#### 4"
        extracted = processor.extract_answer(test_answer)
        assert extracted == "4", f"Expected '4', got '{extracted}'"
        print("✅ Answer extraction works correctly")
        
        # Test prompt formatting
        test_question = "What is 2 + 2?"
        formatted = processor.format_prompt(test_question)
        expected = "What is 2 + 2? Let's think step by step."
        assert formatted == expected, f"Expected '{expected}', got '{formatted}'"
        print("✅ Prompt formatting works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Processor test failed: {e}")
        return False

def test_output_manager():
    """Test output manager functionality"""
    print("\nTesting output manager...")
    try:
        temp_dir = tempfile.mkdtemp()
        
        from output_manager import OutputManager
        
        # Test supported formats
        formats = OutputManager.supported_formats()
        expected_formats = ['parquet', 'jsonl', 'json']
        for fmt in expected_formats:
            assert fmt in formats, f"Format {fmt} not in supported formats"
        print("✅ Supported formats check passed")
        
        # Test manager creation
        manager = OutputManager(temp_dir, 'json')  # Use JSON to avoid parquet dependency
        print("✅ Output manager created successfully")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Output manager test failed: {e}")
        return False

def test_factory_pattern():
    """Test processor factory pattern"""
    print("\nTesting factory pattern...")
    try:
        from processors import ProcessorFactory
        
        # Mock config
        class MockConfig:
            def __init__(self, name):
                self.name = name
                self.data_source = "test"
                self.input_key = "question"
                self.output_key = "answer" 
                self.ability = "math"
                self.reward_style = "rule"
                self.format_prompt = True
                self.extract_answer = True
                self.custom_params = {}
        
        # Test known processor type
        gsm8k_config = MockConfig("gsm8k")
        processor = ProcessorFactory.create_processor(gsm8k_config)
        from processors import GSM8KProcessor
        assert isinstance(processor, GSM8KProcessor), "Should create GSM8K processor"
        print("✅ Known processor type creation works")
        
        # Test unknown processor type (should fallback to generic)
        unknown_config = MockConfig("unknown") 
        processor = ProcessorFactory.create_processor(unknown_config)
        from processors import GenericProcessor
        assert isinstance(processor, GenericProcessor), "Should create Generic processor"
        print("✅ Unknown processor type fallback works")
        
        # Test available processors
        available = ProcessorFactory.list_processors()
        expected_processors = ['gsm8k', 'math', 'generic']
        for proc in expected_processors:
            assert proc in available, f"Processor {proc} not in available list"
        print("✅ Available processors list is correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        return False

def test_error_handling():
    """Test custom error classes"""
    print("\nTesting error handling...")
    try:
        from processors import ProcessingError, AnswerExtractionError
        
        # Test custom exception creation
        try:
            raise ProcessingError("Test processing error")
        except ProcessingError as e:
            assert str(e) == "Test processing error"
            print("✅ ProcessingError works correctly")
        
        try:  
            raise AnswerExtractionError("Test extraction error")
        except AnswerExtractionError as e:
            assert str(e) == "Test extraction error"
            print("✅ AnswerExtractionError works correctly")
            
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_simplified_tests():
    """Run all simplified tests"""
    print("🧪 Running Simplified Tests for Improved Data Processing Design")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_processor_creation,
        test_output_manager,
        test_factory_pattern,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved design is working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_simplified_tests()
    
    if success:
        print("\n🚀 Next Steps:")
        print("1. Install missing dependencies: pip install pyyaml datasets")
        print("2. Run full test suite: python test_framework.py") 
        print("3. Try the CLI: python cli.py init-configs")
    
    sys.exit(0 if success else 1)