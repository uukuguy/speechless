# speechless.generate

A series of functions that support unified interface access to the functionality of LLM generation.

- [generate_llm_prompt()](prompt_utils.py#generate_llm_prompt)
  Function to facilitate generating final prompts for LLM based on different standard prompt templates. Based on fastchat.conversion, and add some general conversation templates, such as "general-llama-2", "general-mistral", "general-chatml".
