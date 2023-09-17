# Speechless

## Data

1. Removed instructions with less than 100 tokens in response.
2. Data deduplication grouped by instruction type using GTE embedding and cosine similarity (threshold>0.95)

### garage-bAInd/Open-Platypus

### Open-Orca/OpenOrca

id.startswith('cot.')

### jondurbin/airoboros-2.2

```json
experts = {
  "qa": [
    "quiz",
    "multiple_choice",
    "contextual",
    "counterfactual_contextual"
  ],
  "creative": [
    "card",
    "writing",
    "experience",
    "song",
    "roleplay",
    "gtkm",
    "rp",
    "detailed_writing",
    "joke"
  ],
  "code": [
    "coding"
  ],
  "reasoning": [
    "cot",
    "theory_of_mind",
    "riddle",
    "orca"
  ],
  "function": [
    "agent",
    "plan"
  ],
  "general": [
    "wordgame",
    "trivia",
    "general"
  ]
}
```