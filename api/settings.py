from typing import Any, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class Settings:
    models_dir: str = "./models"
    # model_family: str = "huggingface"
    model_family: str = "vllm"
    # model_family: str = "exllamav2"
    model_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    setup_params: Dict[str, Any] = field(default_factory=dict)
    host: str = "0.0.0.0"
    port: int = 5001 
    log_level: str = "info"
    stream: bool = False
    # gpu_split = "18,24" # for exllamav2, 34B GPTQ
    gpu_split = ""

    def __post_init__(self):
        self.setup_params = dict(
            repo_id="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-34b",
            # repo_id="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-dolphin-orca-platypus-13b",
            # repo_id="/opt/local/llm_models/huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GPTQ",
            # repo_id="/opt/local/llm_models/huggingface.co/TheBloke/CodeLlama-34B-Instruct-GPTQ",
            # repo_id="/opt/local/llm_models/huggingface.co/TheBloke/Xwin-LM-70B-V0.1-GPTQ",
            tokenizer_repo_id=None,
        ) if len(self.setup_params) == 0 else self.setup_params

        if 'GPTQ' in self.setup_params['repo_id']:
            dtype = "float16"
        else:
            dtype = "bfloat16"

        self.model_params = dict(
                device_map = "auto",
                trust_remote_code = True,
                dtype = dtype,
        ) if len(self.model_params) == 0 else self.model_params 

        assert self.setup_params['repo_id'] is not None, "setup_params.repo_id must be set"
        if self.setup_params.get("tokenizer_repo_id", None) is None:
            self.setup_params["tokenizer_repo_id"] = self.setup_params["repo_id"]
