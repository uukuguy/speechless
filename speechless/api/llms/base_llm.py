import os
from typing import Dict, List, AsyncIterator

# ==================== class BaseLLM ====================
import hashlib
from abc import ABC, abstractmethod
class BaseLLM(ABC):
    """
    A base class for LLMs
    """

    @property
    def model_name(self):
        return os.path.basename(self.model_path)

    def get_model_dir(self, models_dir, model_family, model_name):
        """
        create a base working dir for a certain model
        """
        name_digest = str(int(hashlib.md5(model_name.encode("utf-8")).hexdigest(), 16))[
            0:12
        ]
        dir_name = "_".join([model_family, name_digest])
        return os.path.join(models_dir, dir_name)

    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, str]) -> str:
        """
        generate text using LLM based on an input prompt
        """

    @abstractmethod
    async def agenerate(
        self, prompt: str, params: Dict[str, str], request_id: str
    ) -> AsyncIterator[str]:
        """
        asynchronously generate text using LLM based on an input prompt
        """
        # avoid mypy error https://github.com/python/mypy/issues/5070
        if False:  # pylint: disable=using-constant-test
            yield

    @abstractmethod
    def embeddings(self, text: str) -> List[float]:
        """
        create embeddings from the input text
        """
