"""
Main entry point for LLM-api
https://github.com/1b5d/llm-api/blob/main/app/main.py
"""
import os, json, time
from typing import Any, Dict, Optional, AsyncGenerator
from loguru import logger
import uuid
def random_uuid() -> str:
    return str(uuid.uuid4().hex)

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from sse_starlette.sse import EventSourceResponse
import uvicorn

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
    parser.add_argument("--log_level", type=str, default="info", help="Log level")  
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--model_family", type=str, default="vllm", help="Model family")

    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")

    args = parser.parse_args()
    return args

args = get_args()

from settings import Settings
settings = Settings(model_name_or_path=args.model_name_or_path, 
                    model_family=args.model_family, 
                    stream=args.stream, 
                    host=args.host, 
                    port=args.port, 
                    log_level=args.log_level,
                    )

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
from speechless.api.llms import HuggingFaceLLM, VllmLLM, ExllamaV2LLM

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

llm = get_llm()

# -------------------- API /v1/completions --------------------
from speechless.api.protocol.openai import CompletionResponse

@app.post("/v1/completions")
async def comletions(request: Request):
    """
    Generate text based on a text prompt
    """

    # -------------------- Parameters from request --------------------
    request_dict = await request.json()
    # print(f"{request_dict=}")

    prompt = request_dict.get("prompt")
    model = request_dict.get("model")
    stream = request_dict.get("stream", False)

    # -------------------- generate() --------------------
    request_id = random_uuid()
    generated_text = None

    completion_generator = llm.async_generate(prompt, request_dict, request_id)
    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for generated_output in completion_generator:
                yield generated_output['text']

        async def abort_request() -> None:
            await llm.abort(request_id)

        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        async for generated_output in completion_generator:
            # Abort the request if the client disconnects.
            if await request.is_disconnected():
                await llm.abort(request_id)
                return Response(status_code=499)
            generated_text = generated_output['text']

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = prompt_tokens + completion_tokens
        response_dict = {
            'id': request_id,
            'object': 'text_completion',
            'created': round(time.time()),
            'model': model,
            'choices':[
                {
                    'text': generated_text,
                    'index': 0,
                    'logprobs': None,
                    'finish_reason': 'stop',
                }
            ],
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
            },
        }

        completion_response = CompletionResponse(**response_dict)

        return JSONResponse(completion_response.__dict__)


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