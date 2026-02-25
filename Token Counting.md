# Token Counting

Token counting is the foundation of [[Context Window]] management. You can't implement [[Compaction Strategies]] without knowing how many tokens your conversation history consumes.

## What Is a Token?

Tokens are the units LLMs process. They're roughly:
- 1 token ≈ 4 characters in English
- 1 token ≈ ¾ of a word
- 100 tokens ≈ 75 words

But this varies by language, code vs. prose, and the specific tokenizer. Code tends to tokenize less efficiently than English prose.

## tiktoken — The Standard Library

OpenAI's `tiktoken` is the standard tool for counting tokens for GPT models:

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example
text = "What are the main compaction strategies?"
tokens = count_tokens(text)  # ~8 tokens
```

For chat messages, you need to account for message formatting overhead:

```python
def count_chat_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4  # message formatting overhead
        total += len(encoding.encode(msg["content"]))
        total += len(encoding.encode(msg["role"]))
    total += 2  # reply priming
    return total
```

## Model-Specific Tokenizers

Different model families use different tokenizers:

| Model Family | Tokenizer | Library |
|-------------|-----------|---------|
| GPT-4o, GPT-4o-mini | o200k_base | tiktoken |
| GPT-4, GPT-3.5 | cl100k_base | tiktoken |
| Claude 3.x | Custom BPE | anthropic SDK (estimated) |
| Llama 3.x | Custom BPE | transformers / sentencepiece |

For cross-model systems, count tokens with the specific model's tokenizer. Using the wrong tokenizer can be off by 10-30%.

## Budget Management

A token budget system tracks consumption and triggers compaction:

```python
class TokenBudget:
    def __init__(self, context_window: int):
        self.context_window = context_window
        self.reserved_system = 2000
        self.reserved_response = 10000
        self.reserved_retrieval = 4000  # for [[Embedding and Retrieval]]

    @property
    def available_for_history(self) -> int:
        return (self.context_window
                - self.reserved_system
                - self.reserved_response
                - self.reserved_retrieval)

    def should_compact(self, history_tokens: int) -> bool:
        return history_tokens > self.available_for_history * 0.75
```

When `should_compact()` returns True, the system triggers whichever [[Compaction Strategies|strategy]] is configured — [[Sliding Window]], [[Summarization-Based Compaction]], or [[Selective Pruning]].

## Token Counting in Practice

Key gotchas:
- **Tool calls have overhead**: Function names, parameter schemas, and JSON formatting add tokens beyond the visible content. See [[Tool Call History]].
- **System prompts are re-sent every turn**: A 2,000-token system prompt costs 2,000 tokens per API call, not total.
- **Images and multimodal content**: These consume tokens too (e.g., GPT-4o charges ~85 tokens per image tile).

See also: [[Context Window]], [[Compaction Tradeoffs]], [[Compaction in LangChain]]
