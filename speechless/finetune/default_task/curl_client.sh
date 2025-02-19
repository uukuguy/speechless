#!/bin/bash

curl ${OPENAI_BASE_URL}/chat/completions \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-type: application/json" \
  -d '{
  "model": "/opt/local/llm_models/huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Which is bigger, 9.9 or 9.11?"
    },
  ]
}' 