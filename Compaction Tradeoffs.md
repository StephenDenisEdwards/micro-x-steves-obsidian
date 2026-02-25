# Compaction Tradeoffs

Every [[Compaction Strategies|compaction strategy]] makes tradeoffs between three competing concerns: **latency**, **accuracy**, and **cost**. Understanding these tradeoffs is essential for choosing the right approach for your [[Agent Loops|agent system]].

## The Tradeoff Triangle

```
        Accuracy
       /        \
      /   IDEAL  \
     /   (pick 2) \
    /              \
Latency ————————— Cost
```

- **Accuracy**: How much information is preserved after compaction
- **Latency**: Time added per turn by the compaction process
- **Cost**: Dollar cost of compaction (LLM calls, embedding API calls)

## Strategy Comparison

| Strategy | Accuracy | Latency Added | Cost Added | Best For |
|----------|----------|--------------|------------|----------|
| [[Sliding Window]] | Low | 0ms | $0 | Prototypes, chatbots |
| [[Selective Pruning]] | Medium-High | ~10ms (scoring) | $0 | Mixed-importance histories |
| [[Summarization-Based Compaction]] | Medium | 500-2000ms (LLM call) | $0.001-0.01/compact | Multi-phase tasks |
| [[Hierarchical Memory]] | High | 200-1000ms (retrieval) | $0.001-0.02/compact | Long-running agents |
| No compaction | Perfect | 0ms | High (long inputs) | Short tasks (< 10 turns) |

## When to Trigger Compaction

### Too Early (Wasteful)
Compacting at 30% [[Context Window]] usage wastes summarization calls and discards context you had room to keep.

### Too Late (Risky)
Compacting at 90% leaves no headroom. A single large [[Tool Call History|tool result]] could overflow the window.

### Sweet Spot: 50-70%
```python
COMPACT_THRESHOLD = 0.60  # 60% of context window

if token_count(history) > context_window * COMPACT_THRESHOLD:
    history = compact(history)
```

This leaves room for:
- Current turn's reasoning and tool calls (~20%)
- The model's response (~10%)
- Unexpected large tool results (~10%)

## Cost Analysis

### Without Compaction
An agent running 50 turns with GPT-4o:
- Average history at turn 50: ~80K tokens
- Input cost at $2.50/M: $0.20 per turn at the end
- Total input cost across 50 turns: ~$5.00

### With Sliding Window (10 turns)
- History capped at ~15K tokens
- Input cost: ~$0.04 per turn (constant)
- Total input cost across 50 turns: ~$2.00
- **Savings: 60% — but older context is lost**

### With Summarization (every 10 turns)
- History: summary (~1K tokens) + recent window (~15K tokens)
- Summarization cost: 5 calls × ~$0.01 = $0.05
- Input cost: ~$0.04 per turn
- Total cost: ~$2.05
- **Savings: 59% — and context is preserved in summaries**

### With Hierarchical Memory
- History: retrieved chunks (~4K) + summary (~1K) + recent (~10K) = ~15K tokens
- Embedding cost: negligible (~$0.001 total for a 50-turn conversation)
- Retrieval: ~100ms per turn (local ChromaDB)
- Summarization: ~$0.05
- Total cost: ~$2.10
- **Savings: 58% — with best context recall**

## Accuracy Degradation

How each strategy handles a specific scenario — the agent implemented OAuth in turn 5, and at turn 40 the user asks "what auth method are we using?":

| Strategy | Can answer correctly? | Why? |
|----------|----------------------|------|
| No compaction | Yes | Full history available |
| Sliding Window (k=10) | **No** | Turn 5 was dropped |
| Summarization | **Probably** | Summary should mention OAuth (depends on summary quality) |
| Selective Pruning | **Likely** | Tool results from turn 5 scored high and were retained |
| Hierarchical Memory | **Yes** | Turn 5 is in the vector store and retrieved by the auth-related query |

## Decision Framework

```
Is the task short (< 15 turns)?
  → No compaction needed

Is latency critical (interactive UI)?
  → Sliding Window + tolerate some context loss

Is accuracy critical (complex multi-step task)?
  → Hierarchical Memory or Summarization

Is cost the main constraint?
  → Sliding Window (free) or Selective Pruning (free)

Building a production system?
  → Hierarchical Memory with all layers
  → See [[Compaction in LangChain]] for ready-made components
```

## Monitoring Compaction Quality

Track these metrics to know if your compaction is working:

- **Task completion rate**: Does the agent finish tasks successfully after compaction?
- **Repeated work**: Does the agent redo work it already did (sign of lost context)?
- **Token usage per turn**: Is compaction actually reducing costs?
- **Latency per turn**: Is compaction adding unacceptable delay?

See also: [[Compaction Strategies]], [[Token Counting]], [[Context Window]]
