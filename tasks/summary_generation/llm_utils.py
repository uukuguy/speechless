import os
from loguru import logger

class LLMClient:
    def __init__(self, model_name: str = None, generate_params: dict = None):
        # ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
        # self.client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
        from openai import OpenAI
        self.client =  OpenAI()
        self.model_name = model_name or os.getenv("OPENAI_DEFAULT_MODEL")
        self.generate_params = generate_params if generate_params is not None else {}

        self.client =OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))



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
