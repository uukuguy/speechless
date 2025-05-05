# Reward Functions for RL Fine-tuning

This module provides a comprehensive suite of reward functions for reinforcement learning fine-tuning of language models. Each reward function evaluates a specific aspect of model outputs and returns a normalized score. These functions are designed to be used with the veRL framework for training language models with reinforcement learning.

## Overview

The reward functions can be used individually or combined using the `CombinedReward` class. Each function supports both single-instance and batched processing for computational efficiency.

### Available Reward Functions

1. **LengthReward**: Penalizes overly verbose or overly terse responses, encouraging optimal conciseness.
2. **FormatReward**: Ensures adherence to structured output formats (e.g., JSON, markdown, bullet points) when specified.
3. **MathReward**: Evaluates correctness and reasoning steps in mathematical problem-solving.
4. **CodeReward**: Assesses syntactical accuracy, efficiency, and functional correctness of generated code.
5. **FactualityReward**: Cross-references outputs with trusted sources to minimize hallucinations.
6. **CoherenceReward**: Measures logical flow, clarity, and consistency in multi-turn responses.
7. **TaskSpecificReward**: Customizable rewards for domain-specific objectives (e.g., summarization, translation).
8. **TagReward**: Evaluates the correct usage of XML-style tags in responses, with support for mandatory and optional tags.
9. **CombinedReward**: Combines multiple reward functions with weighted averaging.

## Installation

The reward functions are part of the `speechless` package. No additional installation is required if you have already installed the package.

## Usage

### Basic Usage

```python
from speechless.reasoning.general_reasoner.reward_functions import LengthReward, FormatReward, CombinedReward

# Create individual reward functions
length_reward = LengthReward(min_length=50, max_length=200)
format_reward = FormatReward(format_type='json')

# Combine them with weights
combined_reward = CombinedReward(
    reward_functions=[length_reward, format_reward],
    weights=[0.7, 0.3]
)

# Evaluate a response
response = "This is a sample response that should be evaluated."
reward_score = combined_reward.compute_reward(response)
print(f"Reward score: {reward_score}")

# Evaluate multiple responses at once (batch processing)
responses = [
    "This is the first response.",
    "This is the second, longer response with more content.",
    "This is the third response, which is also quite lengthy to demonstrate the batch processing capability."
]
reward_scores = combined_reward.compute_reward(responses)
print(f"Reward scores: {reward_scores}")
```

### Creating Reward Functions from Configuration

You can also create reward functions from configuration dictionaries using the `create_reward_function` utility:

```python
from speechless.reasoning.general_reasoner.reward_functions import create_reward_function

# Configuration for a combined reward
config = {
    'type': 'combined',
    'name': 'my_combined_reward',
    'reward_functions': [
        {
            'type': 'length',
            'min_length': 50,
            'max_length': 200,
            'weight': 0.7
        },
        {
            'type': 'format',
            'format_type': 'json',
            'weight': 0.3
        }
    ]
}

# Create the reward function from config
reward = create_reward_function(config)

# Use the reward function
response = "This is a sample response."
score = reward.compute_reward(response)
print(f"Score: {score}")
```

## Detailed Documentation

### LengthReward

Penalizes responses that are too short or too long, encouraging optimal conciseness.

```python
from speechless.reasoning.general_reasoner.reward_functions import LengthReward

# Character-based length reward
length_reward = LengthReward(
    min_length=50,        # Minimum acceptable length
    max_length=200,       # Maximum acceptable length
    optimal_length=100,   # Optional optimal length
    weight=1.0            # Weight when combined with other rewards
)

# Token-based length reward (requires a tokenizer)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

token_length_reward = LengthReward(
    min_length=10,        # Minimum acceptable token count
    max_length=50,        # Maximum acceptable token count
    token_based=True,     # Use token count instead of character count
    tokenizer=tokenizer   # Tokenizer to use for counting tokens
)
```

### FormatReward

Ensures adherence to structured output formats like JSON, markdown, bullet points, etc.

```python
from speechless.reasoning.general_reasoner.reward_functions import FormatReward

# JSON format reward
json_reward = FormatReward(
    format_type='json',   # Type of format to check
    weight=1.0            # Weight when combined with other rewards
)

# Custom format reward using regex
custom_format_reward = FormatReward(
    format_regex=r'^[\w\.-]+@[\w\.-]+\.\w+$',  # Email format regex
    weight=1.0
)

# JSON format with schema validation
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    },
    "required": ["name", "age"]
}

schema_reward = FormatReward(
    format_type='json',
    json_schema=json_schema
)
```

### MathReward

Evaluates mathematical problem-solving, including correctness of the final answer and quality of reasoning steps.

```python
from speechless.reasoning.general_reasoner.reward_functions import MathReward

# Basic math reward
math_reward = MathReward(
    check_final_answer=True,      # Check if the final answer is correct
    check_reasoning_steps=True,   # Check the quality of reasoning steps
    weight=1.0
)

# Custom answer extraction pattern
custom_math_reward = MathReward(
    answer_regex=r'x\s*=\s*(-?\d+\.?\d*)'  # Extract answers of the form "x = 42"
)

# Evaluate a response
response = "To solve 6 * 7, I multiply the numbers: 6 * 7 = 42. Therefore, the answer is 42."
reference_answer = 42
score = math_reward.compute_reward(response, reference_answer=reference_answer)
```

### CodeReward

Assesses code quality, including syntax correctness, style, and optionally execution results.

```python
from speechless.reasoning.general_reasoner.reward_functions import CodeReward

# Basic code reward (syntax and style only)
code_reward = CodeReward(
    check_syntax=True,        # Check code syntax
    check_style=True,         # Check code style
    check_execution=False,    # Don't execute the code
    language=None,            # Auto-detect language
    weight=1.0
)

# Code reward with execution (use with caution)
test_cases = [
    {
        "inputs": [5],
        "expected_output": "120"
    },
    {
        "inputs": [0],
        "expected_output": "1"
    }
]

execution_reward = CodeReward(
    check_execution=True,     # Execute the code
    test_cases=test_cases,    # Test cases to run
    timeout=5,                # Maximum execution time in seconds
    language="python"         # Explicitly specify language
)
```

### FactualityReward

Cross-references outputs with trusted sources to minimize hallucinations.

```python
from speechless.reasoning.general_reasoner.reward_functions import FactualityReward

# Basic factuality reward
reference_texts = [
    "The Earth is the third planet from the Sun.",
    "The Moon is Earth's only natural satellite."
]

factuality_reward = FactualityReward(
    reference_texts=reference_texts,    # Reference texts containing factual information
    check_contradictions=True,          # Check for contradictions
    weight=1.0
)

# Factuality reward with embeddings (requires a sentence transformer model)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_reward = FactualityReward(
    reference_texts=reference_texts,
    use_embeddings=True,               # Use embeddings for semantic similarity
    embedding_model=model,             # Model to use for embeddings
    check_contradictions=True
)
```

### CoherenceReward

Measures logical flow, clarity, and consistency in responses.

```python
from speechless.reasoning.general_reasoner.reward_functions import CoherenceReward

# Basic coherence reward
coherence_reward = CoherenceReward(
    check_logical_flow=True,     # Check for logical flow
    check_consistency=True,      # Check for internal consistency
    check_clarity=True,          # Check for clarity
    weight=1.0
)

# Focused coherence reward (only check logical flow)
flow_reward = CoherenceReward(
    check_logical_flow=True,
    check_consistency=False,
    check_clarity=False
)
```

### TaskSpecificReward

Customizable rewards for domain-specific objectives like summarization, translation, or QA.

```python
from speechless.reasoning.general_reasoner.reward_functions import TaskSpecificReward

# Summarization reward
summarization_reward = TaskSpecificReward(
    task_type='summarization',
    task_params={
        'target_ratio': 0.2     # Target length ratio (summary / original)
    },
    weight=1.0
)

# Translation reward
translation_reward = TaskSpecificReward(
    task_type='translation',
    weight=1.0
)

# QA reward
qa_reward = TaskSpecificReward(
    task_type='qa',
    weight=1.0
)

# Custom reward function
def custom_reward_fn(response, prompt=None, reference=None, **kwargs):
    # Custom reward logic
    return 0.5  # Return a score between 0 and 1

custom_reward = TaskSpecificReward(
    task_type='custom',
    custom_reward_fn=custom_reward_fn
)
```

### TagReward

Evaluates the correct usage of XML-style tags in responses, with support for mandatory and optional tags.

```python
from speechless.reasoning.general_reasoner.reward_functions import TagReward

# Basic tag reward with required and optional tags
tag_specs = {
    'think': {'required': True, 'min_count': 1, 'max_count': 1},
    'answer': {'required': True, 'min_count': 1, 'max_count': 1},
    'code': {'required': False, 'max_count': 2}
}

tag_reward = TagReward(
    tag_specs=tag_specs,    # Specifications for each tag
    strict_nesting=True,    # Enforce proper nesting of tags
    weight=1.0
)

# Tag reward with content requirements
content_tag_specs = {
    'think': {
        'required': True,
        'content_required': True,
        'content_regex': r'.*\bstep\b.*'  # Content must contain the word "step"
    },
    'answer': {'required': True}
}

content_tag_reward = TagReward(
    tag_specs=content_tag_specs,
    strict_nesting=True
)

# Example usage
response = """
<think>
First step: analyze the problem.
Second step: develop a solution.
</think>
<answer>The solution is to implement a recursive algorithm.</answer>
"""

score = tag_reward.compute_reward(response)
```

### CombinedReward

Combines multiple reward functions with weighted averaging.

```python
from speechless.reasoning.general_reasoner.reward_functions import (
    LengthReward, FormatReward, CoherenceReward, CombinedReward
)

# Create individual reward functions
length_reward = LengthReward(min_length=50, max_length=200)
format_reward = FormatReward(format_type='json')
coherence_reward = CoherenceReward()

# Combine them with equal weights
equal_weights_reward = CombinedReward(
    reward_functions=[length_reward, format_reward, coherence_reward]
)

# Combine them with custom weights
weighted_reward = CombinedReward(
    reward_functions=[length_reward, format_reward, coherence_reward],
    weights=[0.2, 0.5, 0.3]
)
```

## Integration with veRL Framework

These reward functions are designed to be used with the veRL framework for reinforcement learning fine-tuning of language models. Here's an example of how to integrate them:

```python
from speechless.reasoning.general_reasoner.reward_functions import (
    LengthReward, FormatReward, CombinedReward
)
from verl.workers.reward_manager import BatchRewardManager

# Create a combined reward function
length_reward = LengthReward(min_length=50, max_length=200)
format_reward = FormatReward(format_type='json')
combined_reward = CombinedReward(
    reward_functions=[length_reward, format_reward],
    weights=[0.7, 0.3]
)

# Create a reward manager
reward_manager = BatchRewardManager(
    tokenizer=tokenizer,
    compute_score=combined_reward,
    reward_fn_key="reward"
)

# Use the reward manager in your training configuration
# ...
```

## Running Tests

The module includes a comprehensive test suite. To run the tests:

```bash
cd speechless
# Run all tests
python -m unittest discover -s speechless/reasoning/general_reasoner/reward_functions/tests

# Run specific test files
python -m unittest speechless/reasoning/general_reasoner/reward_functions/tests/test_base.py
python -m unittest speechless/reasoning/general_reasoner/reward_functions/tests/test_text_rewards.py
```

## Contributing

Contributions to improve the reward functions or add new ones are welcome. Please ensure that any new reward functions:

1. Inherit from the `BaseReward` class
2. Implement the `compute_reward` method
3. Include comprehensive documentation
4. Add appropriate tests

## License

This module is licensed under the Apache License, Version 2.0. See the LICENSE file for details.
