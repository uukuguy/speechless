import os, json, re
from typing import Any, List, Union, Optional
from loguru import logger

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
            score += 0.5

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
        
        return rewards

def check_answer(ans, ref):
    ans = ans.strip()
    ref = ref.strip()
    if len(ans) == 0 or len(ref) == 0:
        return 0.0

    if ans == ref:
        return 1.0

    score = 0.0
    # remain_ans = re.sub(r"[0-9]", "", ans).replace('-', '').replace('+', '').replace('.', '').replace('%', '').replace(' ', '').replace('$', '')
    # if len(remain_ans) > 0:
    if True:
        # 计算 Levenshtein 相似度 (通常是 1 - (距离 / 最大长度))
        import Levenshtein
        score = Levenshtein.ratio(ans, ref)
        # distance = Levenshtein.distance(ans, ref)
        # score = 1 - (distance / max(len(ans), len(ref)))

        # # 计算文本语义TF-IDF相似度
        # from sklearn.feature_extraction.text import TfidfVectorizer
        # from sklearn.metrics.pairwise import cosine_similarity
        # documents = [ans, ref]
        # vectorizer = TfidfVectorizer()
        # tfidf_matrix = vectorizer.fit_transform(documents)
        # score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    else:
        if remain_ans == ref:
            score = 1.0
    return score


class CorrectnessReward(BaseReward):
    def __init__(self, weight: float = 1.0):
        super().__init__(name="correction", weight=weight)

    
    def _evaluate_tag_compliance(self, text: str, ref: str) -> float:
        # text = "60\\% \\text{ (Stock)}, 500\\%  \\text{ (Call Option)}}$"
        # ref = "the percentage gain for buying the stock is 60%, and the percentage gain for buying the call option is 300%}$."

        def get_boxed_value(text):
            boxed_value=None
            boxed_text = re.findall(r"boxed{(.*?)}", text, re.DOTALL | re.MULTILINE)
            if len(boxed_text) > 0:
                boxed_value = boxed_text[-1]
            return boxed_value

        score = 0.0
        ans = get_boxed_value(text)
        ref = get_boxed_value(ref)
        if ans is not None:
            score = check_answer(ans, ref)
        # logger.debug(f"check_answer: {ans=}")
        # logger.debug(f"check_answer: {ref=}")
        # logger.debug(f"check_answer: {score=}")
        
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
        
        return rewards


# -------------------- Reward Functions --------------------
reward_funcs = []

def json_loads(json_str: str, ensure_ascii: bool = False, use_json_repair: bool = True) -> Any:
    if use_json_repair:
        from json_repair import repair_json
        return repair_json(json_str, return_objects=True, ensure_ascii=ensure_ascii)
    else:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error: {e}")
            return None

# def correctness_reward_func(prompts, completions, targets, **kwargs) -> list[float]:
#     # responses = [completion[0]['content'] for completion in completions]
#     responses = [completion.strip() for completion in completions]

#     scores = []
#     for prompt, generated_text, true_target in zip(prompts, responses, targets):
#         # logger.debug(f"{prompt[:30]=}")
#         # logger.debug(f"{generated_text=}")
#         score = 0.0
#         # logger.info(f"{true_target=}")

#         def get_boxed_value(text):
#             boxed_value=None
#             boxed_text = re.findall(r"boxed{(.*?)}", text, re.DOTALL | re.MULTILINE)
#             if len(boxed_text) > 0:
#                 boxed_text = boxed_text[-1]
#                 if boxed_text in ["1", "0"]:
#                     boxed_value = int(boxed_text)
#             return boxed_value

#         true_value = get_boxed_value(true_target)
#         llm_value = get_boxed_value(generated_text)

#         if llm_value is not None: 
#             if true_value == llm_value:
#                 score = 2.0
#         if score == 0.0:
#             logger.warning(f"{generated_text=}")
#             logger.warning(f"{true_target=}")
            
#         logger.debug(f"{score=}")
#         scores.append(score)

#     # logger.debug(f"{responses=}")
#     logger.info(f"{scores=}")
#     return scores

def correctness_reward_func(prompts, completions, targets, **kwargs) -> list[float]:
    # responses = [completion[0]['content'] for completion in completions]
    responses = [completion.strip() for completion in completions]

    scores = []
    correctnewss_reward = CorrectnessReward()
    scores = correctnewss_reward(response=responses, reference=targets)

    for i, (p, r, s, t) in enumerate(zip(prompts, responses, scores, targets)):
        if s > 0.0:
            logger.info(f"{s}, {p=}")
            logger.info(f"{r=}")
            logger.info(f"{t=}")
        else:
            logger.warning(f"{s}, {p=}")
            logger.warning(f"{r=}")
            logger.warning(f"{t=}")
        if i >= 0:
            break
    return scores

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]
    pattern = r"<think>.*?</think>[\n\s]*\\boxed{[01]}"
    matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    scores = [0.5 if match else 0.0 for match in matches]

    pattern = r"^<think>\n.*?\n</think>\n\\boxed{[01]}"
    matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    for i, match in enumerate(matches):
        if match:
            scores[i] += 0.5
    return scores

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>[\n\s]*\\boxed{[01]}"
    responses = [completion for completion in completions]
    # matches = [re.match(pattern, r) for r in responses]
    # return [0.5 if match else 0.0 for match in matches]
    matches = [re.findall(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    scores = [] 
    for m in matches:
        s = 0.0
        if len(m) >= 1:
            s += 0.5
            if len(m) == 1:
                s += 0.5
        scores.append(s)
    return scores

def think_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]

    # pattern = r"<think>.*?</think>"
    # found = [0.5 if len(re.findall(pattern, r, re.DOTALL | re.MULTILINE)) == 1 else 0.0 for r in responses]
    # return found

    tag_specs = {
        'think': {'required': True, 'min_count': 1, 'max_count': 1}
    }
    think_count = []
    for r in responses:
        num_think_s = len(re.findall(r"<think>", r))
        num_think_e = len(re.findall(r"</think>", r))
        think_count.append((num_think_s, num_think_e))
    logger.debug(f"Think Format count: {think_count}")
    think_tag_reward = TagReward(tag_specs)
    scores = think_tag_reward(responses)
    logger.debug(f"Think Format scores: {scores}")
    return scores

    # boxed_tag_reward = BoxedTagReward()

def boxed_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]

    # pattern = r"\\boxed{[01]}"
    # found = [0.5 if len(re.findall(pattern, r, re.DOTALL | re.MULTILINE)) == 1 else 0.0 for r in responses]
    # return found

    answer_texts = [ r.split("</think>")[-1] for r in responses]
    boxed_tag_reward = BoxedTagReward()
    return boxed_tag_reward(answer_texts)

def think_length_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    # thinkings = [ len(r.split("<think>")[-1].split("</think>")[0]) for r in responses]
    # def get_score(length):
    #     min_length = 128
    #     max_length = 768
    #     if length < min_length or length > max_length:
    #         return 0.0
    #     else:
    #         return (length - min_length) / (max_length - min_length)
            
    # return [get_score(t) for t in thinkings]

    thinkings = [ r.split("<think>")[-1].split("</think>")[0] for r in responses]
    length_reward = LengthReward(min_length=64, max_length=4096, optimal_length=512)
    logger.debug(f"Think Length lens: {[ len(t) for t in thinkings]}")
    scores = length_reward(thinkings)
    logger.debug(f"Think Length scores: {scores}")
    return scores

reward_funcs = [
    # strict_format_reward_func,
    # soft_format_reward_func,
    think_format_reward_func,
    boxed_format_reward_func,
    think_length_reward_func,
    correctness_reward_func,
]

# # Reward functions
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     logger.debug('-'*20 + 
#         f"\nQuestion:\n{q}" + 
#         f"\nAnswer:\n{answer[0]}" + 
#         f"\nResponse:\n{responses[0]}" + 
#         f"\nExtracted:\n{extracted_responses[0]}")
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]

# def soft_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]

# def count_xml(text) -> float:
#     count = 0.0
#     if text.count("<reasoning>\n") == 1:
#         count += 0.125
#     if text.count("\n</reasoning>\n") == 1:
#         count += 0.125
#     if text.count("\n<answer>\n") == 1:
#         count += 0.125
#         count -= len(text.split("\n</answer>\n")[-1])*0.001
#     if text.count("\n</answer>") == 1:
#         count += 0.125
#         count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
#     return count

# def xmlcount_reward_func(completions, **kwargs) -> list[float]:
#     contents = [completion[0]["content"] for completion in completions]
#     return [count_xml(c) for c in contents]

# reward_funcs = [
#     xmlcount_reward_func,
#     soft_format_reward_func,
#     strict_format_reward_func,
#     int_reward_func,
#     correctness_reward_func,
# ]

