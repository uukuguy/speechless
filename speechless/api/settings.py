from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import torch

@dataclass
class Settings:
    models_dir: str = "./models"

    model_name_or_path: str = None
    # model_family: str = "huggingface"
    model_family: str = "vllm"
    # model_family: str = "exllamav2"

    setup_params: Dict[str, Any] = field(default_factory=dict)
    vllm_engine_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    model_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    host: str = "0.0.0.0"
    port: int = 5001 
    log_level: str = "info"
    stream: bool = False
    # gpu_split = "18,24" # for exllamav2, 34B GPTQ
    # gpu_split = ""

    def __post_init__(self):
        self.setup_params = dict(
            
            repo_id="/opt/local/llm_models/huggingface.co/speechlessai/speechless-nl2sql-mistral-7b-v0.1",
            # repo_id="/opt/local/llm_models/huggingface.co/speechlessai/speechless-mistral-7b-v0.1",
            # repo_id="/opt/local/llm_models/huggingface.co/mistralai/Mistral-7B-v0.1",
            # repo_id="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-34b-v2.0",
            # repo_id="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-34b",
            # repo_id="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-dolphin-orca-platypus-13b",
            # repo_id="/opt/local/llm_models/huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GPTQ",
            # repo_id="/opt/local/llm_models/huggingface.co/TheBloke/CodeLlama-34B-Instruct-GPTQ",
            # repo_id="/opt/local/llm_models/huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ",
            tokenizer_repo_id=None,
        ) if len(self.setup_params) == 0 else self.setup_params

        if self.model_name_or_path is not None:
            self.setup_params['repo_id'] = self.model_name_or_path

        # --------------------- setup_params ---------------------
        assert self.setup_params['repo_id'] is not None, "setup_params.repo_id must be set"
        if self.setup_params.get("tokenizer_repo_id", None) is None:
            self.setup_params["tokenizer_repo_id"] = self.setup_params["repo_id"]

        repo_id = self.setup_params['repo_id']
        num_gpus = torch.cuda.device_count()
        is_awq = '-awq' in self.setup_params['repo_id'].lower()
        is_gptq = '-gptq' in self.setup_params['repo_id'].lower()
        # --------------------- vllm_engine_params ---------------------
        self.vllm_engine_params = dict(
            model = repo_id,
            trust_remote_code = True,
            # download_dir: Optional[str] = None
            # load_format: str = 'auto'
            dtype = 'float16' if is_gptq or is_awq else 'bfloat16',
            # seed: int = 0
            # max_model_len: Optional[int] = None
            worker_use_ray = True,
            # pipeline_parallel_size = num_gpus,
            tensor_parallel_size = num_gpus,

            # KV cache arguments
            # block_size: int = 16
            # swap_space: int = 4  # GiB CPU swap space size (GiB) per GPU
            # gpu_memory_utilization: float = 0.90
            # max_num_batched_tokens: int = 2560 # maximum number of batched tokens per iteration
            # max_num_seqs: int = 256 # maximum number of sequences per iteration
            # disable_log_stats: bool = False
            # revision: Optional[str] = None

            quantization = 'awq' if is_awq else None # ['awq', Non] Method used to quantize the weights
        )

        # --------------------- model_params ---------------------
        self.model_params = dict(
                device_map = "auto",
        ) if len(self.model_params) == 0 else self.model_params 
