"""
Math Verification Reward Function

This module provides backward compatibility for the compute_score function
by importing the MathVerifyReward class from the reward_functions package.
"""

from typing import List, Union, Optional, Tuple, Dict, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from speechless.reasoning.general_reasoner.reward_functions import BaseReward, CombinedReward, LengthReward, TagReward


class BoxedTagReward(BaseReward):
    def __init__(self, weight: float = 1.0):
        super().__init__(name="boxed_tag", weight=weight)
    
    def _evaluate_tag_compliance(self, text: str) -> float:
        full_pattern = r"\$\\boxed{.*?}\$\.$"
        if len(re.findall(full_pattern, text)) == 1:
            return 1.0
        less_pattern = r"\$\\boxed{.*?}\$"
        if len(re.findall(less_pattern, text)) == 1:
            return 0.8
        
        score = 0.0
        one_pattern =  r"\\boxed{.*?}"
        num_found = len(re.findall(one_pattern, text))
        if num_found > 0:
            score += 0.2


        score = 0.0

        
        return score


    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        responses = self._ensure_list(response)
        
        rewards = []
        for resp in responses:
            score = self._evaluate_tag_compliance(resp)
            rewards.append(self._normalize_score(score))
        
        return rewards[0] if len(rewards) == 1 else rewards

def check_answer(ans, ref):
    score = 0.0
    remain_ans = re.sub(r"[0-9]", "", ans).replace('-', '').replace('+', '').replace('.', '').replace('%', '').replace(' ', '').replace('$', '')
    if len(remain_ans) > 0:
        # 计算文本语义相似度
        documents = [ans, ref]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    else:
        if remain_ans == ref:
            score = 1.0
    return score


class CorrectionReward(BaseReward):
    def __init__(self, weight: float = 1.0):
        super().__init__(name="correction", weight=weight)

    
    def _evaluate_tag_compliance(self, text: str, ref: str) -> float:
        score = 0.0
        one_pattern =  r"\\boxed{(.*?)}"
        found = re.findall(one_pattern, text)
        if len(found) > 0:
            ans = found[-1]
            score = check_answer(ans, ref)
        
        return score


    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        responses = self._ensure_list(response)
        references = self._ensure_list(reference)
        
        rewards = []
        for resp, ref in zip(responses, references):
            score = self._evaluate_tag_compliance(resp, ref)
            rewards.append(self._normalize_score(score))
        
        return rewards[0] if len(rewards) == 1 else rewards




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
    length_reward = LengthReward(min_length=64, max_length=1024)
    # math_reward = MathVerifyReward()


    tag_specs = {
        'think': {'required': True, 'min_count': 1, 'max_count': 1}
    }
    think_tag_reward = TagReward(tag_specs)
    boxed_tag_reward = BoxedTagReward()
    correction_reward = CorrectionReward()

    combined_reward = CombinedReward(reward_functions=[length_reward, think_tag_reward, boxed_tag_reward, correction_reward])

    return combined_reward.compute_reward(solution_str, reference=ground_truth)


# Example usage
if __name__ == "__main__":
    print("This module provides backward compatibility for the compute_score function.")
    print("For examples, see the example_usage function in reward_functions.utils module")
    
    # Simple example
    answer = "The answer is 42."
    ground_truth = "42"
    score = compute_score(None, answer, ground_truth)
    print(f"Score: {score}")
