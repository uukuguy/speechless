#!/usr/bin/env python:w
from openai import OpenAI

# MODEL="Llama3.2-3B-Instruct:f16"
# MODEL="Llama3.1-8B-Instruct:Q8_0"
MODEL="Llama3.1-70B-Instruct:Q4_K_M"

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)
response = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
generated_text = response.choices[0].message.content
print(f"{generated_text}")

class LLM:
    def __init__(self, model = None):
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.model = model

    def generate(self, prompt, model=None, system_message="You are a helpful assistant."):
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
        )
        generated_text = response.choices[0].message.content
        return generated_text
