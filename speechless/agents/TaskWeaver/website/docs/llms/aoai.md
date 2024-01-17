---
description: Using LLMs from OpenAI/AOAI
---
# Azure OpenAI

1. Create an account on [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) and get your API key.
2. Add the following to your `taskweaver_config.json` file:
```json showLineNumbers
{
  "llm.api_base":"YOUR_AOAI_ENDPOINT",
  "llm.api_key":"YOUR_API_KEY",
  "llm.api_type":"azure",
  "llm.auth_mode":"api-key",
  "llm.model":"gpt-4-1106-preview",
  "llm.response_format": "json_object"
}
```

:::tip
`llm.model` is the model name you want to use.
You can find the list of models [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models).
:::

:::info
For `gpt-4-1106-preview` and `gpt-3.5-turbo-1106`, `llm.response_format` can be set to `json_object`.
However, for the earlier models, which do not support JSON response explicitly, `llm.response_format` should be set to `null`.
:::

3. Start TaskWeaver and chat with TaskWeaver.
You can refer to the [Quick Start](../quickstart.md) for more details.