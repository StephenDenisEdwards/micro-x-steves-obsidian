# Compaction Strategies

Compaction strategies are techniques for reducing the size of conversation history in [[Agent Loops]] to fit within the [[Context Window]]. Each strategy makes different tradeoffs between information preservation, latency, and cost.

## Strategy Overview

| Strategy | Information Loss | Latency Cost | Implementation Complexity |
|----------|-----------------|--------------|--------------------------|
| [[Sliding Window]] | High (oldest turns dropped) | None | Very Low |
| [[Summarization-Based Compaction]] | Medium (details compressed) | High (LLM call) | Medium |
| [[Hierarchical Memory]] | Low (layered retention) | Medium | High |
| [[Selective Pruning]] | Low-Medium (scored retention) | Low | Medium |

## Choosing a Strategy

**Use [[Sliding Window]] when:**
- Tasks are short-lived (< 20 turns)
- Older context is genuinely irrelevant
- Latency is critical and you can't afford an extra LLM call
- You're prototyping and need something fast

**Use [[Summarization-Based Compaction]] when:**
- Earlier decisions and reasoning are important for later steps
- The agent is doing multi-phase work (research → plan → implement)
- You can tolerate the latency of an extra LLM call
- You want a balance of simplicity and information retention

**Use [[Hierarchical Memory]] when:**
- Tasks are very long-running (50+ turns)
- You need both recent detail and long-term awareness
- You're building a production system and can invest in complexity
- The agent needs to recall context from much earlier in the conversation

**Use [[Selective Pruning]] when:**
- Some messages are clearly more important than others (e.g., [[Tool Call History]] results vs. reasoning traces)
- You want fine-grained control over what's retained
- The agent's tasks involve a mix of high-value and low-value turns

## Combining Strategies

Production systems often combine multiple strategies:

```
1. Selective Pruning removes low-importance messages first
2. Sliding Window drops anything older than N turns
3. Summarization compresses the dropped window into a summary
4. The summary is stored in Hierarchical Memory
```

This layered approach is what [[Compaction in LangChain]] supports through composable memory classes. See [[Compaction Tradeoffs]] for guidance on tuning the aggressiveness of each layer.

## When to Compact

Compaction should trigger based on [[Token Counting]], not message count. A typical trigger:

```python
if count_tokens(history) > context_window * 0.6:
    compact(history)
```

The 60% threshold leaves room for the current turn, tool results, and the model's response. See [[Context Window]] for detailed budget breakdowns.
