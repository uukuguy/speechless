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


class CommonsenseQA(Task):
    _template = "<s> [INST] You are expert at commonsense reasoning, and here is your task: "
    _template_split = "You are expert at commonsense reasoning, and here is your task: "
    _response_split = "[/INST]"
    
    _initial_instructions = [
        "The peanuts were being sold at stalls by the ferris wheel, so where were they being sold?\nA. barn\nB. ballpark\nC. carnival\nD. jar\nE. plane",
        "What might happen to a person after falling?\nA. get up\nB. receive letters\nC. cross street\nD. give gifts\nE. feel embarrassed",
        "James was hunted for stabbing someone to death.  He was afraid that he would be forced to serve what?\nA. knife wounds\nB. law suit\nC. llaughing\nD. mess\nE. jail time",
        "What happens after people perform a concert?\nA. smile\nB. music\nC. medium\nD. applause\nE. accomplish",
        "The story was interesting, the character grew to become a baby boy after starting out a what?\nA. old person\nB. learn to speak\nC. old man\nD. adult\nE. died",
        "What does the policeman gain from discovering the truth in a crime?\nA. enlightenment\nB. confusion\nC. liberating\nD. peace of mind\nE. increased knowledge",
        "John and Grace  live where there are many people, but it's hard for them to move around because of all the what?\nA. race track\nB. populated areas\nC. opera\nD. commotion\nE. roadblock",
        "Where should you keep a pet lizard?\nA. desert\nB. garden\nC. galapagos archipelago\nD. wild\nE. glass cage",
        "An astronout in space needed to know what side of the planet the English Channel was on, where did he aim his spacecraft?\nA. northern hemisphere\nB. atlantic ocean\nC. go boating\nD. england\nE. canada",
        "What negative thing might someone experience if they spend time learning about a subject that is very complex and difficult?\nA. headache\nB. gain knowledge\nC. elation\nD. advance\nE. aids"
    ]
    batch_size_compute_text_representations = 1
    _similarity_threshold = 0.8
    
    ######################################## Part for synthesizing instructions ########################################    
    def postprocess_synthesized_instructions(self, outputs: List):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').cuda()
        
        all_instructions = self._initial_instructions.copy()
        all_instruction_representations = get_text_representation(all_instructions, tokenizer, model)
        options_pattern = re.compile("(A\.)([\s\S]+?)(B\.)([\s\S]+?)(C\.)([\s\S]+?)(D\.)([\s\S]+?)(E\.)([\s\S]+)$")
        
        retained = []
        discarded = []
        for i in trange(0, len(outputs), self.batch_size_compute_text_representations, desc='postprocessing synthesized instructions...'):
            instructions = []
            texts = []
            for j in range(i, min(i + self.batch_size_compute_text_representations, len(outputs))):
                instruction = outputs[j]['instruction']
                if (end := instruction.rfind('F. ')) != -1:
                    instruction = instruction[:end].strip()
                if (question := re.match("([\s\S]+?)A.", instruction)) is None:
                    continue
                if len(question.group(1)) <= 30:
                    continue
                
                # random shuffle the options: "A. a B. b C. c..." => "A. c B. a C. b..."
                if (options := options_pattern.search(instruction)) is None:
                    continue
                trailing_space = options.group(2)[len(options.group(2).rstrip()):]
                options = options.group() + trailing_space
                oid = list(range(2, 11, 2))
                random.shuffle(oid)
                repl = f"\\1\\{oid[0]}\\3\\{oid[1]}\\5\\{oid[2]}\\7\\{oid[3]}\\9\\{oid[4]}"
                options = options_pattern.sub(repl, options).rstrip()
                
                instruction_text = instruction = question.group(1) + options
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
        system_message = "You are expert at commonsense reasoning. I will ask you several multiple-choice questions about commonsense, each with five options A, B, C, D and E. For each question, You must choose the option that best agrees with the commonsense involved in the question. If there is no correct option, respond with 'None'. You should put your reasoning and your choice into a pair of backquotes."
        instruction = synthesized_instruction['instruction']
        return {
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': "Choose from the following multiple-choice question about commonsense wrapped in backquotes. If there is no correct option, respond with 'None'. ```What happens after people perform a concert?\nA. smile\nB. music.\nC. medium\nD. applause\nE. accomplish```"},
                {'role': 'assistant', 'content': "```After people perform a concert, they applaud to show appreciation and enjoyment for the performance. It is a form of positive feedback that musicians and performers often receive after they have finished their show. ### The answer is D```"},
                {'role': 'user', 'content': f"Choose from the following multiple-choice question about commonsense wrapped in backquotes. If there is no correct option, respond with 'None'. ```{instruction}```"}
            ]
        }
    
    def postprocess_completion(self, synthesized_instruction: Dict, content: str):
        def check(content):
            if not (content.startswith('```') and content.endswith('```')):
                return None
            if not(content := content[3:-3].strip()):
                return None
            if 'none' in content.lower():
                return None
            
            try:
                reasoning, answer = re.split('### The answer is', content, flags=re.I)
                reasoning = reasoning.strip()
                answer = answer.rstrip('.').strip().upper()
            except:
                return None
            
            options = "ABCDE"
            if len(answer) != 1 or answer not in options:
                return None
            
            if answer == 'E':
                label_text = re.search("(E\.[\s\S]+)$", synthesized_instruction['instruction']).group(1)
            else:
                next_opt = options[options.index(answer) + 1]
                label_text = re.search(f"({answer}\.[\s\S]+?){next_opt}\.", synthesized_instruction['instruction']).group(1)
            answer = label_text.strip()
            
            return reasoning, answer
        
        result = check(content)
        if result is None:
            return None
        
        reasoning, answer = result
        return {
            'instruction': self._template_split + synthesized_instruction['instruction'],
            'output': "The answer is: " + answer,
            'reasoning': reasoning,
            'answer': answer
        }
        
    ######################################## Part for verifying completions ########################################
    def prepare_for_verification(self, completion: Dict):
        pass
    