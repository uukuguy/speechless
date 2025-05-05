# Reward Functions Package

This package provides a comprehensive suite of reward functions for reinforcement learning fine-tuning of language models. Each reward function evaluates a specific aspect of model outputs and returns a normalized score.

## Package Structure

```
reward_functions/
├── __init__.py                # Package exports
├── base.py                    # BaseReward abstract base class
├── text_rewards.py            # Text-based rewards (length, format, coherence)
├── math_rewards.py            # Math-related rewards
├── code_rewards.py            # Code quality and correctness rewards
├── factuality_rewards.py      # Factual accuracy rewards
├── task_rewards.py            # Domain-specific task rewards
├── tag_rewards.py             # XML-style tag usage rewards
├── combined_rewards.py        # Combining multiple rewards
├── utils.py                   # Utility functions
└── tests/                     # Unit tests
    ├── __init__.py
    ├── test_base.py
    ├── test_text_rewards.py
    └── ...
```

## Available Reward Functions

- **LengthReward**: Evaluates response length, penalizing responses that are too short or too long
- **FormatReward**: Evaluates adherence to specified output formats (JSON, markdown, etc.)
- **MathReward**: Evaluates mathematical reasoning and correctness of solutions
- **MathVerifyReward**: Uses the math-verify library for symbolic verification of solutions
- **CodeReward**: Evaluates code quality, syntax, and correctness
- **FactualityReward**: Evaluates factual accuracy by cross-referencing with trusted sources
- **CoherenceReward**: Evaluates logical flow, clarity, and consistency
- **TaskSpecificReward**: Customizable reward for specific tasks like summarization, translation, or QA
- **TagReward**: Evaluates the correct usage of XML-style tags in responses
- **CombinedReward**: Combines multiple reward functions with weighted averaging

## Usage Examples

### Basic Usage

```python
from reward_functions import LengthReward, FormatReward, MathReward

# Create reward functions
length_reward = LengthReward(min_length=50, max_length=200)
format_reward = FormatReward(format_type='json')
math_reward = MathReward()

# Evaluate responses
response = "The answer is 42."
length_score = length_reward(response)
format_score = format_reward(response)
math_score = math_reward(response, reference_answer=42)

print(f"Length score: {length_score:.2f}")
print(f"Format score: {format_score:.2f}")
print(f"Math score: {math_score:.2f}")
```

### Combining Rewards

```python
from reward_functions import LengthReward, FormatReward, CombinedReward

# Create individual reward functions
length_reward = LengthReward(min_length=50, max_length=200)
format_reward = FormatReward(format_type='json')

# Combine them with weights
combined_reward = CombinedReward(
    reward_functions=[length_reward, format_reward],
    weights=[0.3, 0.7]
)

# Evaluate responses
responses = [
    "This is a short response.",
    '{"result": "This is a valid JSON response with sufficient length to meet the requirements."}'
]

scores = combined_reward(responses)
for i, score in enumerate(scores):
    print(f"Response {i+1}: {score:.2f}")
```

### Creating Rewards from Configuration

```python
from reward_functions import create_reward_function

# Define a configuration
config = {
    'type': 'combined',
    'name': 'custom_combined',
    'reward_functions': [
        {'type': 'length', 'min_length': 100, 'max_length': 300, 'weight': 0.3},
        {'type': 'coherence', 'check_logical_flow': True, 'weight': 0.7}
    ]
}

# Create the reward function
reward = create_reward_function(config)

# Use it to evaluate responses
response = "This is a test response."
score = reward(response)
print(f"Score: {score:.2f}")
```

## Extending with Custom Rewards

You can create custom reward functions by inheriting from the `BaseReward` class:

```python
from reward_functions import BaseReward

class CustomReward(BaseReward):
    def __init__(self, custom_param=None, weight=1.0):
        super().__init__(name="custom", weight=weight)
        self.custom_param = custom_param
    
    def compute_reward(self, response, prompt=None, reference=None, **kwargs):
        # Implement your custom reward logic here
        # Return a score between 0 and 1
        return 0.5  # Example placeholder
```

## Running Tests

To run the unit tests:

```bash
cd speechless/reasoning/general_reasoner
python -m unittest discover -s reward_functions/tests
```

## Backward Compatibility

For backward compatibility with the original `compute_score` function, use:

```python
from speechless.reasoning.general_reasoner.compute_score import compute_score

score = compute_score(None, "The answer is 42.", "42")
```

This function now uses the `MathVerifyReward` class internally.