import sys
import os
import regex as re
from tqdm import tqdm, trange
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .math_util import last_boxed_only_string, _strip_string
from ..base import Task


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(torch.sum(input_mask_expanded, dim=1), min=1e-9)

@torch.no_grad()
def get_text_representation(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        output = model(**inputs)
    text_embeddings = mean_pooling(output, inputs['attention_mask'])
    
    # Normalize embeddings
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    return text_embeddings


class MATH(Task):
    _template = "<s> [INST] You are expert at solving math problems that require multi-step reasoning, and here is your task:\n"
    _template_split = "You are expert at solving math problems that require multi-step reasoning, and here is your task:\n"
    _response_split = "Let's think step by step"
    
    _initial_instructions = [
        "A play has two different male roles, two different female roles and two different roles that can be either gender. Only a man can be assigned to a male role, and only a woman can be assigned to a female role. If five men and six women audition, in how many ways can the six roles be assigned?",
        "Solve for $x$:\n\n$$x^2 + 4x + 3 = -(x + 3)(x + 5).$$",
        "Jenny places a total of 18 red Easter eggs in several green baskets and a total of 24 orange Easter eggs in some blue baskets.  Each basket contains the same number of eggs and there are at least 4 eggs in each basket.  How many eggs did Jenny put in each basket?",
        "What multiple of 15 is closest to 2009?",
        "What is the value of $3 \\times (7 - 5) - 5$?",
        "Compute $(1 + i)^4.$",
        "How many integer side lengths are possible to complete a triangle in which the other sides measure 6 units and 3 units?",
        "The number 210 is the product of two consecutive positive integers and is also the product of three consecutive integers. What is the sum of those five integers?",
        "Find the value of $k$ so that the line $3x + 5y + k = 0$ is tangent to the parabola $y^2 = 24x.$",
        "Find the number of positive integers $n \\le 1000$ that can be expressed in the form\n\\[\\lfloor x \\rfloor + \\lfloor 2x \\rfloor + \\lfloor 3x \\rfloor = n\\]for some real number $x.$"
    ]
    batch_size_compute_text_representations = 1
    _similarity_threshold = 0.8
    
    ######################################## Part for synthesizing instructions ########################################    
    def postprocess_synthesized_instructions(self, outputs: List):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').cuda()
        
        all_instructions = self._initial_instructions.copy()
        all_instruction_representations = get_text_representation(all_instructions, tokenizer, model)
        
        retained = []
        discarded = []
        for i in trange(0, len(outputs), self.batch_size_compute_text_representations, desc='postprocessing synthesized instructions...'):
            instructions = []
            texts = []
            for j in range(i, min(i + self.batch_size_compute_text_representations, len(outputs))):
                instruction = outputs[j]['instruction']
                # maybe "[/INST]" or other similar tokens still in instruction
                instruction = re.sub("\[\s*/[\sA-Z]*?\]", '', instruction).strip()
                
                instruction = re.sub("( ){2,}", r'\1', instruction).strip()
                instruction_text = instruction
                instructions.append(instruction)
                texts.append(instruction_text)
            
            text_representations = get_text_representation(texts, tokenizer, model)
            all_sims = torch.matmul(text_representations, all_instruction_representations.t())
            sims = all_sims.max(dim=1).values
            index = []
            for j in range(sims.shape[0]):
                if sims[j] < self._similarity_threshold:
                    all_instructions.append(instructions[j])
                    retained.append({
                        'id': outputs[i + j]['id'],
                        'instruction': instructions[j]
                    })
                    index.append(j)
                else:
                    most_similar_instructions = {
                        all_instructions[k]: all_sims[j][k].item() for k in torch.argsort(all_sims[j])[-5:].flip(dims=(0,))
                    }
                    discarded.append({
                        'discarded_instruction': instructions[j],
                        'most_similar_instructions': most_similar_instructions
                    })
                    
            all_instruction_representations = torch.cat([all_instruction_representations, text_representations[index]], dim=0)
                
        return retained, discarded
    
    ######################################## Part for completing synthesized instructions ########################################
    def fill_request_for_completion(self, synthesized_instruction: Dict):
        system_message = 'You are expert at solving math problems that require multi-step reasoning. I will provide you with several challenging math problems described in LaTex that require knowledge such as calculus, algebra, number theory, counting and probability, etc. For each problem, you must answer with multiple steps of reasoning and put your final answer in "$\\boxed{}$". All your reasoning and answer should be in the form of LaTex and wrapped in a pair of backquotes.'
        instruction = synthesized_instruction['instruction']
        return {
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': 'Answer the following math problem wrapped in backquotes with multiple steps of reasoning. Your final answer should be placed in "$\\boxed{}$". All your reasoning and answer should be in the form of LaTex. A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem. ```Solve for $x$:\n\n$$x^2 + 4x + 3 = -(x + 3)(x + 5).$$```'},
                {'role': 'assistant', 'content': "```Expanding the product on the right, we have $x^2 + 4x + 3 = -(x^2 + 8x + 15),$ so $x^2 + 4x + 3 + (x^2 + 8x + 15) = 0$.  Simplifying the left side gives $2x^2 + 12x + 18 = 0.$ Dividing by 2, we have $x^2 + 6x + 9 = 0$, so $(x + 3)(x + 3) = 0.$ The only solution for $x$ is $\\boxed{-3}.$```"},
                {'role': 'user', 'content': 'Answer the following math problem wrapped in backquotes with multiple steps of reasoning. Your final answer should be placed in "$\\boxed{}$". All your reasoning and answer should be in the form of LaTex. A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem.' + f' ```{instruction}```'}
            ]
        }
    
    def postprocess_completion(self, synthesized_instruction: Dict, content: str):
        
        def check(content):
            if not (content.startswith('```') and content.endswith('```')):
                return None
            if not(content := content[3:-3].strip()):
                return None
            
            try:
                start, end = last_boxed_only_string(content)
            except:
                return None
            
            boxed_answer = content[start: end]
            answer = boxed_answer[len("\\boxed{"): -1].strip()
            if ',' in answer or any(kw in answer.lower() for kw in ["\\text", "emptyset", "infty", "approx"]):
                return None
            
            answer = _strip_string(answer)
            return content, answer
        
        result = check(content)
        if result is None:
            return None
        
        reasoning, answer = result
        return {
            'instruction': self._template_split + synthesized_instruction['instruction'],
            'output': "Let's think step by step.\n" + reasoning + '\nThe answer is ' + "$\\boxed{" + answer + "}$",
            'reasoning': reasoning,
            'answer': answer
        }
    
    ######################################## Part for verifying completions ########################################
    def prepare_for_verification(self, completion: Dict):
        pass
    