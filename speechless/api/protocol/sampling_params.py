from typing import Optional, Union, List
from dataclasses import dataclass

# vllm/sampling_params.py
"""
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        stop: Union[None, str, List[str]] = None,
        stop_token_ids: List[int] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
"""
@dataclass
class VLLMSamplingParams:
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = False
    stop: Union[None, str, List[str]] = None
    stop_token_ids: List[int] = None
    ignore_eos: bool = False
    max_tokens: int = 1024
    logprobs: Optional[int] = None

    # def use_safe_sampling_method(self):
    #     if self.temperature < 1e-5:
    #         self.use_greedy_search_sampling()
    #     elif self.n > 1:
    #         self.use_beam_search_sampling(n=self.n, best_of=self.best_of, early_stopping=self.early_stopping)
    #     else:
    #         self.use_normal_sampling()

    def use_sampling_method(self, sampling_method):
        if sampling_method == 'greedy':
            self.use_greedy_search_sampling()
        elif sampling_method == 'beam_search':
            self.use_beam_search_sampling(n=self.n, best_of=self.best_of, early_stopping=self.early_stopping)
        else:
            self.use_normal_sampling()

    def use_greedy_search_sampling(self):
        self.use_beam_search = False
        self.best_of = 1
        self.top_p = 1.0
        self.top_k = -1

    # def check_beam_search_sampling_params(**kwargs):
    #     n = kwargs.get('n', 1)
    #     best_of = kwargs.get('best_of', n)
    #     early_stopping = kwargs.get('early_stopping', False)
    #     assert n >= 2, f"num_beams must be >= 2, but is {n}"
    #     if best_of is None:
    #         best_of = n
    #     assert best_of >= n, f"best_of must be >= n, but best_of is {best_of} and n is {n}."
    #     # early_stopping must be in [True, False, 'never']
    #     assert early_stopping in [True, False, 'never'], f"early_stopping must be in [True, False, 'never'], but is {early_stopping}."

    # @check_beam_search_sampling_params
    def use_beam_search_sampling(self, n: int, best_of: int = None, early_stopping: bool = False):
        self.use_beam_search = True
        self.n = n
        self.best_of = best_of
        self.early_stopping = early_stopping 

        # Fixed parameters for beam search
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = -1

    def use_normal_sampling(self):
        self.use_beam_search = False
        self.early_stopping = False
        self.length_penalty = 1.0