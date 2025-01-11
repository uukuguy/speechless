import os
from loguru import logger

class LLMClient:
    def __init__(self, model_name: str):
        # ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
        # self.client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
        from openai import OpenAI
        self.client =  OpenAI()
        self.model_name = model_name or "gpt-4o"

        self.client =OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))


    def generate(self, prompt: str, system_prompt: str = None, verbose: bool = False) -> str:
        if verbose:
            logger.info(f"Generating text with prompt: {prompt}")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  
                messages=[
                    {"role": "system", "content": system_prompt if system_prompt else "You are an AI assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            generated_text = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            generated_text = "error"
        if verbose:
            logger.debug(f"Generated text: {generated_text}")

        generated_text = generated_text.replace("```json", "").replace("```", "")
        return generated_text
