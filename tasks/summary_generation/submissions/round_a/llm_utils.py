import os
from loguru import logger


DEFAULT_OPENAI_ENVIRONMENTS = {
    "ZHIPUAI": {
        "BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
        "DEFAULT_MODEL": "glm-4-plus",
        "BASE_URL_NAME": "ZHIPUAI_BASE_URL",
        "API_KEY_NAME": "ZHIPUAI_API_KEY",
    }
}

class LLMClient:
    def __init__(self, model_name: str = None, api_key: str = None, llm_type: str ="ZHIPUAI", generate_params: dict = None):
        openai_envs = DEFAULT_OPENAI_ENVIRONMENTS.get(llm_type, None)
        if openai_envs:
            base_url_name = openai_envs["BASE_URL_NAME"]
            base_url = os.getenv(base_url_name, None)
            if base_url is None:
                base_url = os.getenv("OPENAI_BASE_URL") or openai_envs["BASE_URL"]
                
            if api_key is None:
                api_key_name = openai_envs["API_KEY_NAME"]
                api_key = os.getenv(api_key_name, None) or os.getenv("OPENAI_API_KEY")

            model_name = model_name or (os.getenv("OPENAI_DEFAULT_MODEL") or openai_envs["DEFAULT_MODEL"])
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            model_name = model_name or os.getenv("OPENAI_DEFAULT_MODEL")
            

        from openai import OpenAI
        self.model_name = model_name
        self.generate_params = generate_params if generate_params is not None else {}
        self.client =OpenAI(api_key=api_key, base_url=base_url)



    def generate(self, prompt: str, generate_params: dict = None, system_prompt: str = None, verbose: bool = False) -> str:
        if verbose:
            logger.info(f"Generating text with prompt: {prompt}")

        if generate_params is None:
            generate_params = self.generate_params.copy()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  
                messages=[
                    {"role": "system", "content": system_prompt if system_prompt else "You are an AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                **generate_params
            )
            generated_text = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            generated_text = "error"
        if verbose:
            logger.debug(f"Generated text: {generated_text}")

        generated_text = generated_text.replace("```json", "").replace("```", "")
        return generated_text
