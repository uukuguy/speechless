import sys
import os
import regex as re
from tqdm import tqdm, trange
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

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


class MBPP(Task):
    _template = "<s> [INST] You are an expert Python programmer, and here is your task: Write"
    _template_split = "You are an expert Python programmer, and here is your task: "
    _response_split = re.compile("(?:You should|Your code should)", re.I)
    
    _initial_instructions = [
        "Write a function to extract a specified column from a given nested list.\nYour code should pass these tests:\n\nassert extract_column([[1, 2, 3], [2, 4, 5], [1, 1, 1]],0)==[1, 2, 1]\nassert extract_column([[1, 2, 3], [-2, 4, -5], [1, -1, 1]],2)==[3, -5, 1]\nassert extract_column([[1, 3], [5, 7], [1, 3], [13, 15, 17], [5, 7], [9, 11]],0)==[1, 5, 1, 13, 5, 9]",
        "Write a function to remove particular data type elements from the given tuple.\nYour code should pass these tests:\n\nassert remove_datatype((4, 5, 4, 7.7, 1.2), int) == [7.7, 1.2]\nassert remove_datatype((7, 8, 9, \"SR\"), str) == [7, 8, 9]\nassert remove_datatype((7, 1.1, 2, 2.2), float) == [7, 2]",
        "Write a function to get the sum of a non-negative integer.\nYour code should pass these tests:\n\nassert sum_digits(345)==12\nassert sum_digits(12)==3\nassert sum_digits(97)==16",
        "Write a python function to find the surface area of the square pyramid.\nYour code should pass these tests:\n\nassert surface_Area(3,4) == 33\nassert surface_Area(4,5) == 56\nassert surface_Area(1,2) == 5",
        "Write a python function to find minimum sum of factors of a given number.\nYour code should pass these tests:\n\nassert find_Min_Sum(12) == 7\nassert find_Min_Sum(105) == 15\nassert find_Min_Sum(2) == 2",
        "Write a python function to find the first repeated word in a given string.\nYour code should pass these tests:\n\nassert first_repeated_word(\"ab ca bc ab\") == \"ab\"\nassert first_repeated_word(\"ab ca bc\") == 'None'\nassert first_repeated_word(\"ab ca bc ca ab bc\") == \"ca\"",
        "Write a python function to find the maximum element in a sorted and rotated array.\nYour code should pass these tests:\n\nassert find_Max([2,3,5,6,9],0,4) == 9\nassert find_Max([3,4,5,2,1],0,4) == 5\nassert find_Max([1,2,3],0,2) == 3",
        "Write a python function to check whether an array is subarray of another or not.\nYour code should pass these tests:\n\nassert is_Sub_Array([1,4,3,5],[1,2],4,2) == False\nassert is_Sub_Array([1,2,1],[1,2,1],3,3) == True\nassert is_Sub_Array([1,0,2,2],[2,2,0],4,3) ==False",
        "Write a function to print check if the triangle is scalene or not.\nYour code should pass these tests:\n\nassert check_isosceles(6,8,12)==True\nassert check_isosceles(6,6,12)==False\nassert check_isosceles(6,15,20)==True",
        "Write a python function to find the difference between sum of cubes of first n natural numbers and the sum of first n natural numbers.\nYour code should pass these tests:\n\nassert difference(3) == 30\nassert difference(5) == 210\nassert difference(2) == 6"
    ]
    batch_size_compute_text_representations = 1
    _similarity_threshold = 0.8
    
    ######################################## Part for synthesizing instructions ########################################
    def get_response(self, generation: str):
        response = generation.split(self._template_split)[1].strip()
        return self._response_split.split(response)[0].strip()
        
    def postprocess_synthesized_instructions(self, outputs: List):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').cuda()
        
        all_instructions = self._initial_instructions.copy()
        all_instruction_texts = [inst.split('Your code should pass these tests:')[0].strip() for inst in all_instructions]
        all_instruction_representations = get_text_representation(all_instruction_texts, tokenizer, model)
        
        retained = []
        discarded = []
        for i in trange(0, len(outputs), self.batch_size_compute_text_representations, desc='postprocessing synthesized instructions...'):
            instructions = []
            texts = []
            for j in range(i, min(i + self.batch_size_compute_text_representations, len(outputs))):
                instruction = outputs[j]['instruction']
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
        system_message = "You are provided with an instruction for code completion. The instruction describes a Python problem but lacks test cases. You are asked to generate a canonical and correct solution to the instruction (import proper packages if necessary). Three test cases to help better understand the instruction are also required. Test cases must accord with your solution. You should put your solution and test cases into a pair of backquotes."
        instruction = synthesized_instruction['instruction']
        return {
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': "Provide a solution to the following instruction wrapped in backquotes and also three test cases designed for it: ```Write a function to get the sum of a non-negative integer.```"},
                {'role': 'assistant', 'content': "```Solution:\n\ndef sum_digits(n):\n    if n == 0:\n        return 0\n    else:\n        return n % 10 + sum_digits(int(n / 10))\n\nTest cases:\n\nassert sum_digits(345)==12\nassert sum_digits(12)==3\nassert sum_digits(97)==16```"},
                {'role': 'user', 'content': f"Provide a solution to the following instruction wrapped in backquotes and also three test cases designed for it: ```{instruction}```"}
            ]
        }
    
    def postprocess_completion(self, synthesized_instruction: Dict, content: str):
        
        def check(content):
            if not (content.startswith('```') and content.endswith('```')):
                return None
            if not(content := content[3:-3].strip()):
                return None
            
            try:
                solution, tests = re.split('Test cases:', content, flags=re.I)
            except:
                return None
            
            solution = solution.strip()
            if not solution.lower().startswith('solution:') or ('return' not in solution and 'lambda' not in solution):
                return None
            solution = solution[len('Solution:'):].strip()
            
            tests = tests.strip()
            return solution, tests
        
        result = check(content)
        if result is None:
            return None
        
        solution, tests = result
        return {
            'instruction': synthesized_instruction['instruction'],
            'output': '[PYTHON]\n' + solution + '\n[/PYTHON]',
            'test': tests
        }
    
    ######################################## Part for verifying completions ########################################
    def prepare_for_verification(self, completion: Dict):
        instruction = completion['instruction']
        output = completion['output'].split('[PYTHON]')[1].split('[/PYTHON]')[0].strip()
        
        if 'test' in completion:
            test_list = re.split("\n{1,2}", completion['test'].strip())
            return {'test_list': test_list}, output
        
        try:
            test_list = re.split("Your code should pass these tests:", instruction, flags=re.I)[1].strip()
            test_list = re.split("Your code should start with", test_list, flags=re.I)[0].strip()
            test_list = re.split("\n{1,2}", test_list)
            return {'test_list': test_list}, output
        except:
            return None
        