from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class CompletionParams:
    model: str = None
    prompt: str = None
    suffix: Optional[str] = None 
    max_tokens: Optional[int] = 1024 # The maximum number of tokens to generate in the completion. The token count of your prompt plus max_tokens cannot exceed the model's context length. 
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0 # We generally recommend altering this or temperature but not both. 
    n: Optional[int] = 1 # How many completions to generate for each prompt.
    stream: Optional[bool] = False
    # logprob - Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.
    logprob: Optional[int] = None 
    echo: Optional[bool] = False # Echo back the prompt in addition to the completion.
    stop: Optional[List[str]] = None # Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
    presence_penalty: Optional[float] = 0.0 # Number between -2.0 and 2.0. How much to penalize new tokens based on whether they appear in the text so far. Increases the model's likelihood to talk about new topics.
    frequency_penalty: Optional[float] = 0.0 # Number between -2.0 and 2.0. How much to penalize new tokens based on their existing frequency in the text so far. Decreases the model's likelihood to repeat the same line verbatim.
    best_of: Optional[int] = 1 # Generates best_of completions server-side and returns the "best" (the one with the lowest log probability per token). Results cannot be streamed. best_of must >= n.
    logit_bias: Optional[Dict[str, float]] = None # Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this parameter to bias the model towards generating text with certain characteristics. See the OpenAI documentation for more details.
    user: Optional[str] = None # The user ID for the OpenAI Chat model. This is required for Chat models.

    def __post_init__(self):
        # assert self.temperature is None or self.top_p is None, "We generally recommend altering this or temperature but not both."
        pass

    def get_sampling_params(self):
        sampling_param_keys = ['max_tokens', 'temperature', 'top_p', 'stop', 
                               'n', 'best_of', 
                               'presence_penalty', 'frequency_penalty', 'logit_bias']
        sampling_params = {k: getattr(self, k) for k in sampling_param_keys if getattr(self, k) is not None}
        return sampling_params

    @classmethod
    def from_request(cls, request_dict: Dict):
        completion_params = { k: v for k, v in request_dict.items() if hasattr(cls, k)}
        return CompletionParams(**completion_params)


@dataclass
class CompletionResponse:
    id: str
    object: str # Which is always 'text_completion'
    created: int # The Unix timestamp (in seconds) of when the completion was created.
    model: str # The model used for completion.
    # choices - The list of completion choices the model generated for the input prompt.
    # "choices": [
    #     {
    #     "text": "\n\nThis is indeed a test",
    #     "index": 0,
    #     "logprobs": null,
    #     "finish_reason": "length" # 'stop', 'length' (maximum tokens reached), 'content_filter'
    #     }
    # ]
    choices: List[Dict[str, Any]] 
    # usage - The usage statistics for the model at the time of completion.
    # "usage": {
    #     "prompt_tokens": 5,
    #     "completion_tokens": 7,
    #     "total_tokens": 12
    # }
    usage: Dict[str, int]


@dataclass
class ChatCompletionParams:
    model: str = None
    messages: List[Dict[str, str]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0 # We generally recommend altering this or temperature but not both. 
    n: Optional[int] = 1 # How many completions to generate for each prompt.
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None # Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
    max_tokens: Optional[int] = 2048 # The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    presence_penalty: Optional[float] = 0.0 # Number between -2.0 and 2.0. How much to penalize new tokens based on whether they appear in the text so far. Increases the model's likelihood to talk about new topics.
    frequency_penalty: Optional[float] = 0.0 # Number between -2.0 and 2.0. How much to penalize new tokens based on their existing frequency in the text so far. Decreases the model's likelihood to repeat the same line verbatim.
    logit_bias: Optional[Dict[str, float]] = None # Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this parameter to bias the model towards generating text with certain characteristics. See the OpenAI documentation for more details.
    user: Optional[str] = None # The user ID for the OpenAI Chat model. This is required for Chat models.


@dataclass
class ChatCompletionResponse:
    id: str
    object: str # Which is always 'chat.completion'
    created: int # The Unix timestamp (in seconds) of when the completion was created.
    model: str # The model used for completion.
    # choices - The list of completion choices the model generated for the input prompt.
    # "choices": [{
    #     "index": 0,
    #     "message": {
    #     "role": "assistant",
    #     "content": "\n\nHello there, how may I assist you today?",
    #     "function_call": {'name': 'get_user_name', 'arguments': ""},
    #     },
    #     "finish_reason": "stop" # 'stop', 'length' (maximum tokens reached), 'content_filter', 'function_call'
    # }]
    choices: List[Dict[str, Any]] 
    # usage - The usage statistics for the model at the time of completion.
    # "usage": {
    #     "prompt_tokens": 5,
    #     "completion_tokens": 7,
    #     "total_tokens": 12
    # }
    usage: Dict[str, int]
