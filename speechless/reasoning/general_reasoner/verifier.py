import logging
import os
import re

import torch
from vllm import LLM, SamplingParams
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl import DataProto
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer from the input text.
    
    Returns:
        The content inside the last \boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.
    
    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()
    
    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)


class RewardModelWorker(Worker):
    def __init__(self, config):
        """
        Initializes the reward model worker with its configuration and sampling parameters.
        """
        super().__init__()
        self.config = config
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialize the language model and tokenizer.
        """
        self.llm = LLM(model=self.config.model.path, gpu_memory_utilization=0.5)
        self.tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False)
        )
        self.llm.sleep(2)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute the reward model score for each data item.
        
        For every data instance, the function decodes the sequence of prompt and response
        tokens, extracts the solution, and then uses a language model to verify the answer.
        A reward score is then computed based on whether the verified answer is correct and the
        token length difference from the ground truth.
        
        Returns:
            A DataProto object containing the computed reward scores.
        """
        torch.cuda.empty_cache()
        self.llm.wake_up()
        sequence_strs = []
        ground_truths = []
        questions = []
        valid_response_lengths = []

        # Process each data item to create a sequence string and extract necessary fields.
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_lengths.append(valid_response_length)

            # Concatenate valid prompt and response tokens.
            sequence = torch.cat((valid_prompt_ids, response_ids[:valid_response_length]))
            sequence_str = self.tokenizer.decode(sequence[-1024:]) # avoid risk of getting too long answer extracted
            sequence_strs.append(sequence_str)

            # Extract question and ground truth from non-tensor batch.
            question = data_item.non_tensor_batch["extra_info"]["question"]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            questions.append(question)
            ground_truths.append(ground_truth)

        # Extract solutions from the decoded sequences.
        solutions = [extract_solution(seq) for seq in sequence_strs]


        # Prepare messages for the verification prompt.
        messages = [
            VERIFIER_PROMPT_TEMPLATE.format(question=q, ground_truth=gt, student_answer=sol)
            for q, gt, sol in zip(questions, ground_truths, solutions)
        ]

        # Generate verification responses using the language model.
        outputs = self.llm.generate(messages, self.sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]

        # Initialize reward tensor with the same shape as responses.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Compute a reward score for each data item.
        for i, (ground_truth, solution, verification, valid_response_length) in enumerate(
            zip(ground_truths, solutions, responses, valid_response_lengths)
        ):
            score = 0.0
            # Penalize if solution extraction failed.
            if solution is None:
                score -= 0.5
            # If solution is empty, assign a default value.
            if not solution:
                solution = "No Answer"
            # Award a score and adjust based on token length difference if verification passes.
            if VERIFIER_PASS_TAG in verification:
                score += 1.0
                tokenized_solution = self.tokenizer.encode(solution)
                tokenized_ground_truth = self.tokenizer.encode(ground_truth)
                # Penalize based on the absolute difference in token count (capped to 10 tokens).
                difference = abs(len(tokenized_solution) - len(tokenized_ground_truth))
                difference = min(difference, 10)
                score -= difference * 0.05
            # Record the score at the final valid response token index.
            reward_tensor[i, valid_response_length - 1] = score

        batch = TensorDict({"rm_scores": reward_tensor}, batch_size=reward_tensor.shape[0])
        self.llm.sleep(2)
        torch.cuda.empty_cache()
        return DataProto(batch=batch)
