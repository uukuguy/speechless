import sys
import os
from abc import ABC, abstractmethod
from typing import List, Dict


class Task(ABC):
    _template: str = None
    _template_split: str = None
    _response_split: str = None
    
    @property
    def template(self):
        return self._template
    
    ######################################## Part for synthesizing instructions ########################################
    def get_prompt(self, prefix: str = ''):
        return self._template + prefix
    
    def get_response(self, generation: str):
        response = generation.split(self._template_split)[1].strip()
        return response.split(self._response_split)[0].strip()
    
    @abstractmethod
    def postprocess_synthesized_instructions(self, outputs: List):
        pass
    
    ######################################## Part for completing synthesized instructions ########################################
    @abstractmethod
    def fill_request_for_completion(self, synthesized_instruction: Dict):
        pass
    
    @abstractmethod
    def postprocess_completion(self, synthesized_instruction: Dict, content: str):
        pass
    
    ######################################## Part for verifying completions ########################################
    @abstractmethod
    def prepare_for_verification(self, completion: Dict):
        pass