# Selective Pruning

Selective pruning is a [[Compaction Strategies|compaction strategy]] that assigns **importance scores** to messages and removes the least important ones first. Unlike [[Sliding Window]], which drops messages purely by age, pruning keeps high-value messages regardless of when they occurred.

## Importance Scoring

Every message gets a score based on multiple signals:

| Signal | High Importance | Low Importance |
|--------|----------------|----------------|
| Message type | Tool results, user requirements | Acknowledgments ("OK, doing that now") |
| Content density | File paths, code, decisions | Filler reasoning, hedging |
| Recency | Recent turns | Older turns (slight decay) |
| Reference count | Frequently referenced back | Never referenced again |
| [[Tool Call History]] | Tool results with data | Tool calls that failed/retried |

```python
def score_message(msg, position, total):
    score = 0.0
    # Type-based scoring
    if msg.role == "tool":
        score += 0.8  # Tool results are almost always important
    elif msg.role == "user":
        score += 0.6  # User messages contain requirements
    else:
        score += 0.3  # Assistant reasoning — often compressible

    # Recency bonus (linear decay)
    recency = position / total
    score += recency * 0.4

    # Content signals
    if contains_code_or_paths(msg):
        score += 0.3
    if is_acknowledgment(msg):
        score -= 0.5

    return score
```

## Pruning Process

1. Score all messages in the history
2. Sort by score (ascending — lowest first)
3. Remove messages from the bottom until history fits within [[Context Window]] budget
4. Optionally: replace removed messages with a one-line placeholder: `[Pruned: agent acknowledged the request]`

The placeholders help maintain conversational coherence without consuming many tokens.

## What to Always Keep

Certain messages should **never** be pruned:

- **System prompt**: Always needed
- **Last user message**: The current request
- **Last tool call + result**: Agent needs to see what just happened
- **Messages containing key decisions**: "We chose PostgreSQL over MongoDB because..."

## What to Prune First

Safe to prune early:
- Assistant messages that are pure reasoning with no conclusions
- Repeated failed attempts (keep only the successful one)
- Verbose tool results where a summary exists
- "Let me..." planning messages that were already acted upon

## Combining with Other Strategies

Selective pruning works best as the first stage in a pipeline:

```
1. Selective Pruning → remove low-value messages
2. [[Summarization-Based Compaction]] → compress remaining old messages
3. [[Sliding Window]] → hard cap on total size
```

This layered approach is more aggressive than any single strategy alone. [[Hierarchical Memory]] provides the architecture for combining them.

## Limitations

- Scoring heuristics are imperfect — sometimes a "low value" message turns out to be critical later
- Requires custom logic per use case (coding agents vs. research agents have different importance signals)
- Placeholder messages can confuse the model if overused

See [[Compaction Tradeoffs]] for analysis of when pruning's precision is worth the implementation complexity.

See also: [[Compaction Strategies]], [[Tool Call History]], [[Token Counting]]
