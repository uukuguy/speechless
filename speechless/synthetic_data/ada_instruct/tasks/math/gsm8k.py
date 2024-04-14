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


class GSM8K(Task):
    _template = "<s> [INST] You are expert at solving math problems that require multi-step reasoning, and here is your task:\n"
    _template_split = "You are expert at solving math problems that require multi-step reasoning, and here is your task:\n"
    _response_split = "[/INST]"
    
    _initial_instructions = [
        "In Professor Plum's biology class there are 40 students. Of those students, 80 percent have puppies. Of those who have puppies, 25% also have parrots. How many students have both puppies and parrots?",
        "Diane bought twenty more apples than Cecile. If Cecile bought 15 apples, how many apples did they buy altogether?",
        "Ann can skate 6 miles an hour. Her friend Glenda can skate 8 miles an hour. If they start in the same place and skate in straight lines in opposite directions for 3 hours, how many miles apart do they end up?",
        "Running for 2 hours, Jonah burnt 30 calories every hour. How many more calories would he have lost if he would have run for five hours?",
        "The city of Richmond has 1000 more people than Victoria. Victoria has 4 times as many people as Beacon. If Richmond has 3000 people, how many people are there in Beacon?",
        "To get his fill of oysters, Crabby has to eat at least twice as many oysters as Squido does. If Squido eats 200 oysters, how many oysters do they eat altogether?",
        "John sells 20 woodburning for $15 each.  The wood cost $100.  How much does he make in profit?",
        "In a field of 500 clovers, 20% have four leaves and one quarter of these are purple clovers. Assuming these proportions are exactly correct, how many clovers in the field are both purple and four-leaved?",
        "Tony lifts weights as a form of exercise.  He can lift 90 pounds with one arm in the exercise known as \"the curl.\"  In an exercise known as \"the military press,\" he can lift over his head twice the weight that he can curl.  His favorite exercise is known as \"the squat\" and he can squat 5 times the weight that he can lift in the military press.  How much weight, in pounds, can Tony lift in the squat exercise?",
        "Elsa started the day with 40 marbles.  At breakfast, she lost 3 marbles while playing.  At lunchtime, she gave her best friend Susie 5 marbles.  In the afternoon, Elsa's mom bought her a new bag with 12 marbles.  Susie came back and gave Elsa twice as many marbles as she received at lunch.  How many marbles did Elsa end the day with?"
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
                if not instruction.endswith('?'):
                    continue
                
                instruction_text = instruction
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
        system_message = "You are expert at solving math problems that require multi-step reasoning. I will provide you with several grade-school level math questions. For each question, you must answer with multiple steps of detailed reasoning. You should put your reasoning and final answer into a pair of backquotes."
        instruction = synthesized_instruction['instruction']
        return {
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': "Answer the following math question wrapped in backquotes with multiple steps of detailed reasoning. Your final answer should be in pure numerical form without any units. A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem. ```Running for 2 hours, Jonah burnt 30 calories every hour. How many more calories would he have lost if he would have run for five hours?```"},
                {'role': 'assistant', 'content': "```When Jonah ran for 2 hours, burning 30 calories every hour, he burnt a total of 2 * 30 = 60 calories.\nIf he had run for five hours, losing 30 calories every hour, Jonah would have burnt 5 * 30 = 150 calories.\nThe difference in the number of calories Jonah would have burnt if he ran for five hours instead of 2 hours is 150 - 60 = 90 calories. ### The answer is 90```"},
                {'role': 'user', 'content': f"Answer the following math question wrapped in backquotes with multiple steps of detailed reasoning. Your final answer should be in pure numerical form without any units. A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem. ```{instruction}```"}
            ]
        }
    
    def postprocess_completion(self, synthesized_instruction: Dict, content: str):
        
        def check(content):
            if not (content.startswith('```') and content.endswith('```')):
                return None
            if not(content := content[3:-3].strip()):
                return None
            
            try:
                reasoning, answer = re.split('### The answer is', content, flags=re.I)
            except:
                return None
            
            reasoning = reasoning.strip()
            answer = answer.replace(',', '').lstrip().rstrip('. ')
            if answer and answer[0] == '.':
                answer = '0' + answer
            
            try:
                float(answer)
            except:
                return None
            
            return reasoning, answer
        
        result = check(content)
        if result is None:
            return None
        
        reasoning, answer = result
        return {
            'instruction': self._template_split + synthesized_instruction['instruction'],
            'output': "Let's think step by step.\n" + reasoning + '\nThe answer is ' + answer,
            'reasoning': reasoning,
            'answer': answer
        }
    
    ######################################## Part for verifying completions ########################################
    def prepare_for_verification(self, completion: Dict):
        pass
    