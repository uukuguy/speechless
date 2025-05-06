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
Math Verification Reward Function

This module provides backward compatibility for the compute_score function
by importing the MathVerifyReward class from the reward_functions package.
"""

from speechless.reasoning.general_reasoner.reward_functions import CombineReward, LengthReward, MathVerifyReward
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Legacy function to compute math verification score.
    
    This function is maintained for backward compatibility.
    It's recommended to use the MathVerifyReward class instead.
    
    Args:
        data_source: Source of the problem
        solution_str: Model's solution string
        ground_truth: Ground truth answer
        extra_info: Additional information
        
    Returns:
        Verification score between 0 and 1
    """
    length_reward = LengthReward(min_length=256, max_length=2048)
    math_reward = MathVerifyReward()

    combine_reward = CombineReward(rewards_functions=[length_reward, math_reward])

    return combine_reward.compute_reward(solution_str, reference=ground_truth)


# Example usage
if __name__ == "__main__":
    print("This module provides backward compatibility for the compute_score function.")
    print("For examples, see the example_usage function in reward_functions.utils module")
    
    # Simple example
    answer = "The answer is 42."
    ground_truth = "42"
    score = compute_score(None, answer, ground_truth)
    print(f"Score: {score}")