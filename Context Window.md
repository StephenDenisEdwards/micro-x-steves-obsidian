# Context Window

The **context window** is the maximum number of tokens an LLM can process in a single request (prompt + completion). It is the hard constraint that makes [[Compaction Strategies]] necessary for [[Agent Loops]].

## Token Limits by Model

| Model | Context Window | Approx. Words |
|-------|---------------|---------------|
| GPT-4o | 128K tokens | ~96K words |
| GPT-4o-mini | 128K tokens | ~96K words |
| Claude 3.5 Sonnet | 200K tokens | ~150K words |
| Claude 3 Opus | 200K tokens | ~150K words |
| Llama 3.1 405B | 128K tokens | ~96K words |

Even 128K tokens gets consumed quickly in [[Agent Loops]] that involve tool calls. A single tool call + result can be 500-2,000 tokens. After 50 iterations, you've used 25K-100K tokens on history alone, leaving little room for the system prompt and the model's response.

## Why Large Windows Don't Eliminate Compaction

You might think 128K or 200K tokens is "enough." It isn't, for three reasons:

1. **Cost scales with input tokens.** Every turn re-sends the full history. At $2.50/M input tokens (GPT-4o), a 100K-token conversation costs $0.25 per turn just for input. Over 50 turns, that's $12.50 for a single task. [[Summarization-Based Compaction]] can cut this dramatically.

2. **Attention degrades with length.** LLMs perform worse on information buried in the middle of long contexts ("lost in the middle" effect). Compaction keeps the most relevant information close to the model's attention. See [[Selective Pruning]] for importance-based approaches.

3. **Latency increases with context length.** Time-to-first-token grows with input size. For interactive agents, this matters. [[Sliding Window]] gives the fastest compaction with the lowest latency overhead.

## Token Budgeting

A practical approach to managing the context window:

```
Total Budget: 128,000 tokens
- System prompt:    2,000 tokens (reserved)
- Retrieved context: 4,000 tokens (from [[Embedding and Retrieval]])
- Conversation history: 112,000 tokens (managed by compaction)
- Response headroom: 10,000 tokens (reserved for output)
```

[[Token Counting]] covers how to measure actual usage with tiktoken and model-specific tokenizers. [[Compaction Tradeoffs]] discusses when to trigger compaction relative to the budget.

## The Compaction Trigger

Most systems compact when history reaches a threshold — typically 50-75% of the context window. This leaves headroom for:

- The current turn's reasoning
- Tool call results that might be large
- The model's response

See [[Compaction Strategies]] for the full menu of approaches, and [[Compaction in LangChain]] for ready-made implementations.
