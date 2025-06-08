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
import re
import jieba
from rouge_chinese import Rouge
from loguru import logger
from typing import Union, List, Optional
from speechless.reasoning.general_reasoner.reward_functions import BaseReward, CombinedReward, LengthReward, MathVerifyReward

# -------------------- Common Reward Functions for DIFC2025 --------------------
def correctness_reward(solution_str, ground_truth, category):
    score = 0.0

    def get_boxed_value(text):
        boxed_value = None
        boxed_text = re.findall(
            r"boxed{(.*?)}", text, re.DOTALL | re.MULTILINE)
        if len(boxed_text) > 0:
            boxed_value = boxed_text[0]
        return boxed_value

    true_value = get_boxed_value(ground_truth)
    llm_value = get_boxed_value(solution_str)

    if category == "选择题":
        if llm_value is not None:
            if true_value == llm_value:
                score = 2.0
        if score == 0.0:
            logger.warning(f"{solution_str=}, {llm_value=}")
            logger.warning(f"{ground_truth=}, {true_value=}")
            logger.warning(f"{score=}")
        else:
            logger.debug(f"{score=}")
    elif category == "问答题":
        s1 = " ".join(jieba.cut(ground_truth, cut_all=True))
        s2 = " ".join(jieba.cut(solution_str, cut_all=True))
        r = Rouge()
        rouge_scores = r.get_scores(s1, s2)[0]
        score = (rouge_scores['rouge-l']['f'] + rouge_scores['rouge-1']
                 ['f'] + rouge_scores['rouge-2']['f']) / 3
        score = round(score, 4)  # Round to 4 decimal places
        logger.debug(f"{score=}")
    else:
        logger.warning(f"Unknown category: {category}")
        score = 0.0

    return score


def strict_format_reward(solution_str):
    pattern = r"\\boxed{[.*?]}$"
    match = re.match(pattern, solution_str, re.DOTALL | re.MULTILINE)

    score = 0.5 if match else 0.0
    return score


def soft_format_reward(solution_str):
    pattern = r"\\boxed{[.*?]}"
    found = re.findall(pattern, solution_str, re.DOTALL | re.MULTILINE)

    score = 0.0
    if len(found) >= 1:
        score += 0.25
        if len(found) == 1:
            score += 0.25
    return score


# -------------------- Reward Functions for Unsloth --------------------
def correctness_reward_func(prompts, completions, targets, categories, **kwargs) -> list[float]:
    # responses = [completion[0]['content'] for completion in completions]
    responses = [completion.strip() for completion in completions]

    scores = []
    for prompt, generated_text, true_target, category in zip(prompts, responses, targets, categories):
        score = correctness_reward(generated_text, true_target, category)

        scores.append(score)

    logger.info(f"{scores=}")
    return scores


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]

    scores = [strict_format_reward(r) for r in responses]
    return scores


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]

    scores = [soft_format_reward(r) for r in responses]
    return scores


# -------------------- Reward Functions for speechless.reasoning.general_reasoner --------------------
class DIFC2025CorrectnessReward(BaseReward):
    def __init__(self, **kwargs):
        super().__init__(name="DIFC2025-CorrectnessReward", weight=1.0)

    def compute_reward(self,
                       response: Union[str, List[str]],
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        responses = self._ensure_list(response)
        rewards = []

        for resp in responses:
            score = correctness_reward(resp, reference, category=kwargs['category'])
            rewards.append(self._normalize_score(score))

        return rewards[0] if len(rewards) == 1 else rewards


class DIFC2025StrictFormatReward(BaseReward):

    def __init__(self, **kwargs):
        super().__init__(name="DIFC2025-StrictFormatReward", weight=1.0)

    def compute_reward(self,
                       response: Union[str, List[str]],
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        responses = self._ensure_list(response)
        rewards = []

        for resp in responses:
            score = strict_format_reward(resp)
            rewards.append(self._normalize_score(score))

        return rewards[0] if len(rewards) == 1 else rewards


class DIFC2025SoftFormatReward(BaseReward):
    def __init__(self, **kwargs):
        super().__init__(name="DIFC2025-StrictFormatReward", weight=1.0)

    def compute_reward(self,
                       response: Union[str, List[str]],
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        responses = self._ensure_list(response)
        rewards = []

        for resp in responses:
            score = soft_format_reward(resp)
            rewards.append(self._normalize_score(score))

        return rewards[0] if len(rewards) == 1 else rewards


# -------------------- compute score --------------------
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
    correctness_reward = DIFC2025CorrectnessReward()
    strict_format_reward = DIFC2025StrictFormatReward()
    soft_format_reward = DIFC2025SoftFormatReward()

    combined_reward = CombinedReward(
        reward_functions=[correctness_reward, strict_format_reward, soft_format_reward])

    return combined_reward.compute_reward(response=solution_str, reference=ground_truth, category=extra_info['category'])


# -------------------- demo compute score --------------------
def demo_compute_score(data_source, solution_str, ground_truth, extra_info=None):
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

    combined_reward = CombinedReward(
        reward_functions=[length_reward, math_reward])

    return combined_reward.compute_reward(solution_str, reference=ground_truth)


# Example usage
if __name__ == "__main__":
    print("This module provides backward compatibility for the compute_score function.")
    print("For examples, see the example_usage function in reward_functions.utils module")

    # Simple example
    answer = "The answer is 42."
    ground_truth = "42"
    score = demo_compute_score(None, answer, ground_truth)
    print(f"Score: {score}")
