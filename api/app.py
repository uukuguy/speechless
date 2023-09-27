"""
Main entry point for LLM-api
https://github.com/1b5d/llm-api/blob/main/app/main.py
"""
import json
from typing import Any, Dict, Optional, AsyncGenerator
import uuid
def random_uuid() -> str:
    return str(uuid.uuid4().hex)

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from sse_starlette.sse import EventSourceResponse
import uvicorn

from settings import Settings
settings = Settings()


# -------------------- logging --------------------
log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
log_config["formatters"]["default"][
    "fmt"
] = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
log_config["loggers"]["llm-api"] = {
    "handlers": ["default"],
    "level": uvicorn.config.LOG_LEVELS[settings.log_level],
}

from logging.config import dictConfig
dictConfig(log_config)


# -------------------- FastAPI --------------------
app = FastAPI(title="llm-api", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """
    A general generate request representation
    """

    prompt: str
    params: Optional[Dict[str, Any]]


class EmbeddingsRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """
    A general embeddings request representation
    """

    text: str


# -------------------- get_llm() --------------------
from llms import HuggingFaceLLM, VllmLLM, ExllamaV2LLM
available_models = {
    'huggingface': HuggingFaceLLM,
    'vllm': VllmLLM,
    'exllamav2': ExllamaV2LLM,
}
assert settings.model_family in available_models, f"Model family {settings.model_family} not supported. Available models: {available_models.keys()}"

def get_llm():
    ModelClass = available_models[settings.model_family]
    llm = ModelClass(settings)

    return llm

llm = None
engine = None

llm = get_llm()
# if settings.model_family != 'vllm':
#     llm = get_llm()
# else:
#     from vllm.engine.arg_utils import AsyncEngineArgs
#     from vllm.engine.async_llm_engine import AsyncLLMEngine
#     from vllm.sampling_params import SamplingParams

#     params = settings.model_params or {}
#     model_name_or_path = settings.setup_params["repo_id"]

#     params['model'] = model_name_or_path
#     params['tensor_parallel_size'] = 2
#     params.pop("device_map", None)
#     params.pop("trust_remote_code", None)
#     engine_args = AsyncEngineArgs(**params)
#     engine = AsyncLLMEngine.from_engine_args(engine_args)

# -------------------- API /v1/completions --------------------
# @app.post("/v1/completions")
async def comletions(request: Request):
    # """
    # Generate text based on a text prompt
    # """
    request_dict = await request.json()
    print(f"{request_dict=}")

    # if settings.model_family == 'vllm':
    #     return await comletions_vllm(request)

    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = dict(**request_dict)
    print(f"{sampling_params=}")

    request_id = random_uuid()

    for k in ['model']:
        sampling_params.pop(k, None)

    # if 'max_new_tokens' not in sampling_params:
    #     max_new_tokens = sampling_params.pop("max_tokens", 1024)
    #     sampling_params['max_new_tokens'] = max_new_tokens

    # for k in ['model', 'max_tokens', 'stop']:
    #     sampling_params.pop(k, None)

    text = llm.generate(prompt=prompt, sampling_params=sampling_params or {})
    print(f"{text=}")

    response_dict = {
        'choices':[
            {'text': text}
        ]
    }
    return JSONResponse(response_dict)

@app.post("/v1/completions")
async def async_comletions(request: Request):
    # """
    # Generate text based on a text prompt
    # """
    request_dict = await request.json()
    # print(f"{request_dict=}")

    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = dict(**request_dict)

    request_id = random_uuid()

    # if 'max_new_tokens' not in sampling_params:
    #     max_new_tokens = sampling_params.pop("max_tokens", 1024)
    #     sampling_params['max_new_tokens'] = max_new_tokens

    # for k in ['model', 'max_tokens', 'stop']:
    #     sampling_params.pop(k, None)

    # sampling_params for vllm
    for k in ['model']:
        sampling_params.pop(k, None)

    # sampling_params for vllm
    #     sampling params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0, top_p=1.0, top_k=-1,
# use_beam_search=False, stop=[], ignore_eos=False, max_tokens=16, logprobs=None


    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # request_dict = await request.json()
    # prompt = request_dict.pop("prompt")
    # stream = request_dict.pop("stream", False)
    # sampling_params = SamplingParams(**request_dict)
    # request_id = random_uuid()

    # sampling_params = SamplingParams(**sampling_params)
    # results_generator = engine.generate(prompt, sampling_params, request_id)

    results_generator = llm.async_generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            # prompt = request_output.prompt
            # text_outputs = [
            #     prompt + output.text for output in request_output.outputs
            # ]
            # ret = {"text": text_outputs}
            # yield (json.dumps(ret) + "\0").encode("utf-8")
            yield request_output

    async def abort_request() -> None:
        # await engine.abort(request_id)
        await llm.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            # await engine.abort(request_id)
            await llm.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    # prompt = final_output.prompt
    # # text = "".join([prompt + output.text for output in final_output.outputs])
    # text = final_output.outputs[0].text

    text = final_output 

    # ret = {"text": text}
    # return JSONResponse(ret)

    # text = llm.generate(prompt=prompt, params=sampling_params or {})
    response_dict = {
        'choices':[
            {'text': text}
        ]
    }
    return JSONResponse(response_dict)


# -------------------- API /generate --------------------
@app.post("/generate")
def generate(payload: GenerateRequest):
    """
    Generate text based on a text prompt
    """
    return llm.generate(prompt=payload.prompt, params=payload.params or {})


# -------------------- API /agenerate --------------------
@app.post("/agenerate")
def agenerate(request: Request, payload: GenerateRequest):
    """
    Generate a stream of text based on a text prompt
    """

    async def event_publisher():
        async for token in llm.agenerate(
            prompt=payload.prompt, params=payload.params or {}
        ):
            if await request.is_disconnected():
                break
            yield token

    return EventSourceResponse(event_publisher())


# -------------------- API /embeddings --------------------
@app.post("/embeddings")
def embeddings(payload: EmbeddingsRequest):
    """
    Generate embeddings for a text input
    """
    return llm.embeddings(payload.text)

# -------------------- API /models --------------------
@app.post("/models")
def models(payload: EmbeddingsRequest):
    """
    Return available models. 
    """
    return "Available models: " + ",".join(available_models.keys())


# -------------------- API /check --------------------
@app.get("/check")
def check():
    """
    Status check
    """
    return "Ok"


# -------------------- Main --------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=log_config,
        log_level=settings.log_level,
    )