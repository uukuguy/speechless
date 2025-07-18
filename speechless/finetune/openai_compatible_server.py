# openai_compatible_server.py
"""
OpenAI-compatible API server for serving local LLMs.
This server implements the OpenAI API format so it can be used as a drop-in replacement.
"""

import os
import time
import uuid
import json
import logging
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenAI-Compatible API Server",
    description="A server that provides OpenAI-compatible API endpoints for local LLMs",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bearer token authentication
security = HTTPBearer()

# Global variables for models
LOADED_MODELS = {}
API_KEYS = set()


# Pydantic models for API requests and responses
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 2048
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "user"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ModelManager:
    """Manages loading and caching of models."""

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.environ.get("MODEL_DIR", "./models")
        self.models = {}
        self.tokenizers = {}
        self.max_memory = None

    def load_model(self, model_name: str):
        """Load a model by name."""
        if model_name in self.models:
            return self.models[model_name], self.tokenizers[model_name]

        logger.info(f"Loading model: {model_name}")
        
        # Check if model exists locally or on Hugging Face
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            model_path = model_name  # Assume it's a Hugging Face model ID
        
        # Configure quantization if needed
        quantization_config = None
        if os.environ.get("USE_4BIT", "false").lower() == "true":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif os.environ.get("USE_8BIT", "false").lower() == "true":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Determine device mapping
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        
        # Enable flash attention if available
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            try:
                if hasattr(model.config, "attn_implementation"):
                    model.config.attn_implementation = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled")
            except Exception as e:
                logger.warning(f"Failed to enable Flash Attention: {e}")
        
        # Cache model and tokenizer
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        return model, tokenizer

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.models:
            logger.info(f"Unloading model: {model_name}")
            del self.models[model_name]
            del self.tokenizers[model_name]
            torch.cuda.empty_cache()

    def get_available_models(self):
        """Get a list of available models."""
        # Check local models
        local_models = []
        if os.path.exists(self.model_dir):
            local_models = [d for d in os.listdir(self.model_dir) 
                          if os.path.isdir(os.path.join(self.model_dir, d))]
        
        # Include currently loaded models
        loaded_models = list(self.models.keys())
        
        # Combine and deduplicate
        all_models = list(set(local_models + loaded_models))
        
        return all_models


# Initialize model manager
model_manager = ModelManager()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    token = credentials.credentials
    
    # If no API keys are set, allow any token
    if not API_KEYS:
        return True
    
    if token not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    
    return True


def format_prompt(messages: List[Message]) -> str:
    """Format messages into a prompt string."""
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}\n"
        else:
            prompt += f"<|{message.role}|>\n{message.content}\n"
    
    # Add the final assistant prefix
    prompt += "<|assistant|>\n"
    
    return prompt


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))


@app.on_event("startup")
async def startup_event():
    """Load API keys on startup."""
    global API_KEYS
    
    api_keys_file = os.environ.get("API_KEYS_FILE", "api_keys.txt")
    if os.path.exists(api_keys_file):
        with open(api_keys_file, "r") as f:
            API_KEYS = {line.strip() for line in f.readlines() if line.strip()}
        logger.info(f"Loaded {len(API_KEYS)} API keys")
    else:
        logger.warning(f"API keys file {api_keys_file} not found. Authentication disabled.")


@app.get("/v1/models", response_model=ModelList)
async def list_models(_: bool = Depends(verify_token)):
    """List available models."""
    models = model_manager.get_available_models()
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "created": int(time.time()), "owned_by": "user"} 
                for model in models]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_token),
):
    """Create a chat completion."""
    try:
        # Load model and tokenizer
        model, tokenizer = model_manager.load_model(request.model)
        
        # Format prompt
        prompt = format_prompt(request.messages)
        
        # Count prompt tokens
        prompt_tokens = count_tokens(prompt, tokenizer)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        # Handle streaming
        if request.stream:
            # Set up streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Define generation config
            generation_config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Start generation
            generation_kwargs = dict(
                inputs=input_ids,
                generation_config=generation_config,
                streamer=streamer,
            )
            
            async def generate():
                # Run generation in a thread to avoid blocking
                thread = threading.Thread(
                    target=lambda: model.generate(**generation_kwargs)
                )
                thread.start()
                
                # Stream chunks as they're generated
                created_time = int(time.time())
                completion_id = f"chatcmpl-{uuid.uuid4()}"
                
                # Stream header
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # Stream content
                for text in streamer:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Stream done
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Non-streaming case
        generation_config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                generation_config=generation_config,
            )
        
        # Decode the generated text
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove the prompt from the beginning of the output
        output = output[len(prompt):].strip()
        
        # Count completion tokens
        completion_tokens = count_tokens(output, tokenizer)
        
        # Construct response
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_token),
):
    """Create a completion."""
    try:
        # Load model and tokenizer
        model, tokenizer = model_manager.load_model(request.model)
        
        # Get prompt
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0]  # Use the first prompt
        
        # Count prompt tokens
        prompt_tokens = count_tokens(prompt, tokenizer)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        # Handle streaming
        if request.stream:
            # Set up streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Define generation config
            generation_config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Start generation
            generation_kwargs = dict(
                inputs=input_ids,
                generation_config=generation_config,
                streamer=streamer,
            )
            
            async def generate():
                # Run generation in a thread to avoid blocking
                thread = threading.Thread(
                    target=lambda: model.generate(**generation_kwargs)
                )
                thread.start()
                
                # Stream chunks as they're generated
                created_time = int(time.time())
                completion_id = f"cmpl-{uuid.uuid4()}"
                
                for text in streamer:
                    chunk = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created_time,
                        "model": request.model,
                        "choices": [
                            {
                                "text": text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Stream done
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Non-streaming case
        generation_config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                generation_config=generation_config,
            )
        
        # Decode the generated text
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove the prompt from the beginning of the output
        output = output[len(prompt):].strip()
        
        # Count completion tokens
        completion_tokens = count_tokens(output, tokenizer)
        
        # Construct response
        response = {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": output,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/models/{model_name}")
async def unload_model(
    model_name: str,
    _: bool = Depends(verify_token),
):
    """Unload a model from memory."""
    try:
        model_manager.unload_model(model_name)
        return {"message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory containing models")
    parser.add_argument("--api-keys-file", type=str, default="api_keys.txt", help="File containing API keys")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["API_KEYS_FILE"] = args.api_keys_file
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the server
    uvicorn.run(
        "openai_compatible_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()