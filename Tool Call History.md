# Tool Call History

Tool calls require special handling during compaction because they have unique structure and disproportionate importance in [[Agent Loops]].

## Structure of Tool Calls

A single tool interaction produces multiple messages:

```json
[
  {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\": \"src/auth.py\"}"}}]},
  {"role": "tool", "content": "import flask\nfrom oauth import ...\n\nclass AuthHandler:\n    def __init__(self):\n        self.provider = 'google'\n        ...  (200 lines of code)"},
  {"role": "assistant", "content": "I've read the auth module. It uses Google OAuth with Flask..."}
]
```

That's three messages for one logical operation, and the tool result can be enormous (full file contents, API responses, command output).

## Why Tool Calls Are Special

**High information density**: A tool result containing file contents or database query results carries more factual weight than 10 turns of reasoning. [[Selective Pruning]] should score these highly.

**Structural coupling**: The three messages (call → result → interpretation) must be kept together or removed together. Dropping just the tool result while keeping the assistant's interpretation creates a confusing context.

**Size variance**: Tool results range from 10 tokens ("File deleted successfully") to 50,000+ tokens (large file contents). A single large tool result can consume a significant portion of the [[Context Window]].

## Compaction Strategies for Tool Calls

### 1. Result Truncation

Replace large tool results with truncated versions:

```python
def truncate_tool_result(result: str, max_tokens: int = 500) -> str:
    tokens = count_tokens(result)
    if tokens <= max_tokens:
        return result
    # Keep first and last portions
    lines = result.split('\n')
    head = '\n'.join(lines[:10])
    tail = '\n'.join(lines[-5:])
    return f"{head}\n\n... ({tokens} tokens truncated) ...\n\n{tail}"
```

### 2. Summary Replacement

Replace the entire tool call triplet with a summary:

```
Original (3,000 tokens):
  [Assistant calls read_file("src/auth.py")]
  [Tool returns 200 lines of code]
  [Assistant analyzes the code]

Compacted (200 tokens):
  [Summary: Read src/auth.py — Google OAuth with Flask, AuthHandler
   class, uses refresh tokens, has rate limiting middleware]
```

### 3. Selective Retention

Keep only the most recent tool calls in full, summarize older ones:

```python
def compact_tool_calls(messages, keep_recent=3):
    tool_groups = group_tool_call_triplets(messages)
    for i, group in enumerate(tool_groups):
        if i < len(tool_groups) - keep_recent:
            # Summarize older tool calls
            summary = summarize_tool_call(group)
            replace_messages(messages, group, summary)
```

## Tool Calls in [[Hierarchical Memory]]

- **Layer 0 (working memory)**: Keep the last 2-3 tool calls in full detail
- **Layer 1 (short-term summary)**: Summarize tool calls from the recent past, preserving key outputs
- **Layer 2 (long-term)**: Embed tool results for retrieval via [[Embedding and Retrieval]]

Tool results are especially valuable in Layer 2 because they contain factual data (file contents, API responses) that the agent may need to recall later.

## Common Pitfalls

- **Dropping tool results but keeping the call**: The model sees it called a tool but doesn't see what happened. This confuses it.
- **Keeping all tool results uncompacted**: A few large file reads can consume 50K+ tokens. Use [[Token Counting]] to monitor.
- **Embedding raw tool results**: Large code dumps should be chunked before embedding, not embedded as one massive chunk.

See also: [[Selective Pruning]], [[Compaction Strategies]], [[Multi-Turn Conversation Design]]
