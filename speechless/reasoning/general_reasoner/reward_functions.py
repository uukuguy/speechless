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

This module is maintained for backward compatibility.
It is recommended to import directly from the reward_functions package instead.
"""

import warnings

warnings.warn(
    "Importing from reward_functions.py is deprecated. "
    "Please import from speechless.reasoning.general_reasoner.reward_functions instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new package structure
from speechless.reasoning.general_reasoner.reward_functions.base import BaseReward
from speechless.reasoning.general_reasoner.reward_functions.text_rewards import LengthReward, FormatReward, CoherenceReward
from speechless.reasoning.general_reasoner.reward_functions.math_rewards import MathReward, MathVerifyReward
from speechless.reasoning.general_reasoner.reward_functions.code_rewards import CodeReward
from speechless.reasoning.general_reasoner.reward_functions.factuality_rewards import FactualityReward
from speechless.reasoning.general_reasoner.reward_functions.task_rewards import TaskSpecificReward
from speechless.reasoning.general_reasoner.reward_functions.tag_rewards import TagReward
from speechless.reasoning.general_reasoner.reward_functions.combined_rewards import CombinedReward
from speechless.reasoning.general_reasoner.reward_functions.utils import create_reward_function, example_usage

# Re-export all classes
__all__ = [
    'BaseReward', 'LengthReward', 'FormatReward', 'CoherenceReward',
    'MathReward', 'MathVerifyReward', 'CodeReward', 'FactualityReward',
    'TaskSpecificReward', 'TagReward', 'CombinedReward',
    'create_reward_function', 'example_usage'
]

# For backward compatibility
if __name__ == "__main__":
    example_usage()