import sys
import os
import random
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


class HumanEval(Task):
    _template = "<s> [INST] You are an expert Python programmer, complete the function below based on its docstring and the given test cases:\n"
    _template_split = "You are an expert Python programmer, complete the function below based on its docstring and the given test cases:\n"
    _response_split = ">>>"
    
    _initial_instructions = [
        "def is_not_prime(n: int):\n    \"\"\" Check if the given number is non-prime.\n\n    >>> is_not_prime(2)\n    False\n    >>> is_not_prime(10)\n    True\n    \"\"\"\n",
        "def differ_At_One_Bit_Pos(a, b):\n    \"\"\" Return whether the two numbers differ at one bit position only or not.\n\n    >>> differ_At_One_Bit_Pos(13,9)\n    True\n    >>> differ_At_One_Bit_Pos(15,8)\n    False\n    \"\"\"\n",
        "def find_char_long(text):\n    \"\"\" From a given string, generate all words which are at least 4 characters long.\n\n    >>> find_char_long('Please move back to stream')\n    ['Please', 'move', 'back', 'stream']\n    >>> find_char_long('Jing Eco and Tech')\n    ['Jing', 'Tech']\n    \"\"\"\n",
        "def remove_Occ(s: str, ch: str) -> str:\n    \"\"\" removes first and last occurrence of a given character from the string.\n\n    >>> remove_Occ('hello', 'l')\n    'heo'\n    >>> remove_Occ('hellollol', 'l')\n    'helollol'\n    \"\"\"\n",
        "def is_acronym(words, s):\n    \"\"\" Given an array of strings words and a string s, determine if s is an acronym of words.\n    The string s is considered an acronym of words if it can be formed by concatenating the \n    first character of each string in words in order.\n\n    >>> is_acronym([\"alice\",\"bob\",\"charlie\"], \"abc\")\n    True\n    >>> is_acronym([\"an\",\"apple\"], \"a\")\n    False\n    >>> is_acronym([\"never\",\"gonna\",\"give\",\"up\",\"on\",\"you\"], \"ngguoy\")\n    True\n    \"\"\"\n",
        "def k_avoiding_sum(n, k):\n    \"\"\" Given n and k, teturn the minimum possible sum of a k-avoiding array of length n.\n    An array of distinct positive integers is called a k-avoiding array if there does \n    not exist any pair of distinct elements that sum to k.\n\n    >>> k_avoiding_sum(5, 4)\n    18\n    >>> k_avoiding_sum(2, 6)\n    3\n    \"\"\"\n",
        "def sum_of_sepcial_squares(nums: List[int]) -> int:\n    \"\"\" Given an integer array nums of length n. \n    The element nums[i] is called special if (i+1) divides n, i.e. n % (i+1) == 0.\n    Return the sum of the squares of all special elements of nums.\n\n    >>> sum_of_sepcial_squares([1,2,3,4])\n    21\n    >>> sum_of_sepcial_squares([2,7,1,19,18,3])\n    63\n    \"\"\"\n",
        "def pair_string_count(words):\n    \"\"\" You are given an array words consisting of distinct strings.\n    The string words[i] can be paired with the string words[j] if i < j\n    and words[i] is equal to the reversed string of words[j].\n    Count the number of pairs that can be formed from the array words.\n\n    >>> pair_string_count([\"cd\",\"ac\",\"dc\",\"ca\",\"zz\"])\n    2\n    >>> pair_string_count([\"ab\",\"ba\",\"cc\"])\n    1\n    >>> pair_string_count([\"aa\",\"ab\"])\n    0\n    \"\"\"\n",
        "def find_non_min_nor_max(numbers):\n    \"\"\" Given an integer array nums containing distinct positive integers, \n    find and return any number from the array that is neither the minimum nor \n    the maximum value in the array, or -1 if there is no such number.\n\n    >>> find_non_min_nor_max([3,2,1,4])\n    2\n    >>> find_non_min_nor_max([1,2])\n    -1\n    >>> find_non_min_nor_max([2,1,3])\n    2\n    \"\"\"\n",
        "def minimum_operations(num_str):\n    \"\"\" You are given a string num_str representing a non-negative integer.\n    In one operation, you can pick any digit of num and delete it. \n    If you delete all the digits of num, num becomes 0.\n    Return the minimum number of operations required to make the number divisible by 25.\n\n    >>> minimum_operations(\"2245047\")\n    2\n    Explanation: Delete the last two digits. The resulting number is 22450 which is divisible by 25.\n    >>> minimum_operations(\"2908305\")\n    3\n    >>> minimum_operations(\"10\")\n    1\n    \"\"\"\n"
    ]
    batch_size_compute_text_representations = 1
    _similarity_threshold = 0.8
    
    
    ######################################## Part for synthesizing instructions ########################################
    def postprocess_synthesized_instructions(self, outputs: List):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').cuda()
        
        all_instructions = self._initial_instructions.copy()
        all_instruction_texts = []
        for inst in all_instruction_texts:
            inst = inst[inst.find('"""') + len('"""'): inst.find('>>>')].strip()
            inst += '\n    """'
            all_instruction_texts.append(inst)
        all_instruction_representations = get_text_representation(all_instruction_texts, tokenizer, model)
        
        retained = []
        discarded = []
        for i in trange(0, len(outputs), self.batch_size_compute_text_representations, desc='postprocessing synthesized instructions...'):
            instructions = []
            texts = []
            for j in range(i, min(i + self.batch_size_compute_text_representations, len(outputs))):
                instruction = outputs[j]['instruction']
            
                # only retain those starting with package import statement/function definition with docstring
                if not re.match("(?:import|from|def)", instruction.lstrip()) or not re.search('def.+?:\n\s+"""', instruction):
                    continue
                if (instruction.count('"""') == 2 and not instruction.endswith('"""')) or instruction.count('"""') > 2:
                    continue
                
                if instruction.endswith('"""'):
                    instruction = instruction[:-len('"""')].strip()
                instruction += '\n    """'
                instruction_text = instruction[index: instruction.rfind('"""')].strip()
                instructions.append(instruction)
                texts.append(instruction_text)
                
            if not texts:
                continue
            
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
        system_message = "You are provided with an instruction for code completion. The instruction describes a Python problem with function definition and docstring, but lacks test cases. You are asked to generate a canonical and correct solution to the instruction (import proper packages if necessary). Two test cases to help better understand the instruction are also required, each including an input-output pair. Test cases must accord with your solution. You should put your solution and test cases into a pair of backquotes."
        instruction = re.sub("(\))\s*->.+?(:\n)", r"\1\2", synthesized_instruction['instruction'])
        return {
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': "Provide a solution to the following instruction wrapped in backquotes and also two test cases designed for it: ```def is_not_prime(n: int):\n    \"\"\" Check if the given number is non-prime.\n    \"\"\"```"},
                {'role': 'assistant', 'content': "```Solution:\n\nimport math\n\n\ndef is_not_prime(n: int):\n    result = False\n    for i in range(2, int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result\n\nExamples:\n\nInput: is_not_prime(2)\nOutput: False\n\nInput: is_not_prime(10)\nOutput: True```"},
                {'role': 'user', 'content': f"Provide a solution to the following instruction wrapped in backquotes and also two test cases designed for it: ```{instruction}```"}
            ]
        }
    
    def postprocess_completion(self, synthesized_instruction: Dict, content: str):
        
        def check(content):
            if not (content.startswith('```') and content.endswith('```')):
                return None
            if not(content := content[3:-3].strip()):
                return None
            
            try:
                solution, tests = re.split('Examples:', content, flags=re.I)
                solution = re.split('Solution:', solution, flags=re.I)[1].strip()
            except:
                return None
            
            # now tests is in format "Input: xxx[\n]Output: xxx[\n]Input: xxx[\n]Output: xxx" after strip
            tests = re.split('Input: ', tests.strip(), flags=re.I)
            
            pairs = []
            for test in tests:
                try:
                    input, output = re.split('Output: ', test, flags=re.I)
                except:
                    continue
                pairs.append((input.rstrip(), output.rstrip()))
            if not pairs:
                return None
            
            return solution, pairs
        
        def random_format(pairs):
            tests = ''
            format = random.randrange(2)
            if format == 0:
                # "    >>> is_not_prime(2)\n    False"
                for input, output in pairs:
                    tests += f"    >>> {input}\n    {output}\n"
            else:
                tests += "    Examples:\n"
                if any('\n' in input or '\n' in output for input, output in pairs):
                    for input, output in pairs:
                        tests += f"        Input:\n{input}\n"
                        tests += f"        Output:\n{output}\n"
                else:
                    # "    is_not_prime(2) == False"
                    for input, output in pairs:
                        tests += f"    {input} == {output}\n"
            return tests
                    
        
        result = check(content)
        if result is None:
            return None
        
        solution, pairs = result
        tests = random_format(pairs)
        instruction = synthesized_instruction['instruction']
        import_statement = re.match("((?:from|import).+?)def ", solution, re.DOTALL)
        if import_statement is not None:
            instruction = import_statement.group(1) + instruction
        index = instruction.rfind('\n') + 1
        
        return {
            'instruction': self._template_split + instruction[:index] + tests + instruction[index:] + '\n' + '\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.',
            'output': '[PYTHON]\n' + solution + '\n[/PYTHON]',
            'test': pairs
        }
    
    ######################################## Part for verifying completions ########################################
    def prepare_for_verification(self, completion: Dict):
        output = completion['output'].split('[PYTHON]')[1].split('[/PYTHON]')[0].strip()
        
        test_list = []
        for test_case, result in completion['test']:
            test_list.append(f"assert {test_case} == {result}")
        return {'test_list': test_list}, output
        