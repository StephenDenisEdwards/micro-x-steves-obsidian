# Hierarchical Memory

Hierarchical memory organizes an agent's conversation history into **layers** with different retention policies. Recent turns are kept in full detail, while older context is progressively compressed. It combines the strengths of [[Sliding Window]] and [[Summarization-Based Compaction]].

## The Memory Hierarchy

```
Layer 0: Working Memory (full detail)
├── Last 5-10 turns, verbatim
├── Current tool calls and results
└── ~10,000 tokens

Layer 1: Short-Term Summary
├── Summarized turns from the last ~30 minutes
├── Key decisions, file paths, findings
└── ~2,000 tokens

Layer 2: Long-Term Memory (embeddings)
├── Vector store of all past turns and summaries
├── Retrieved on demand via [[Embedding and Retrieval]]
└── Unlimited capacity, ~2,000 tokens retrieved per query
```

## How It Works

As the [[Agent Loops|agent loop]] runs:

1. New turns enter **Layer 0** (working memory)
2. When Layer 0 exceeds its token budget, oldest turns are summarized and moved to **Layer 1**
3. When Layer 1 exceeds its budget, summaries are embedded and moved to **Layer 2**
4. Before each LLM call, relevant context from Layer 2 is retrieved and injected

```python
def build_context(query, layers):
    context = []
    # Layer 2: retrieve relevant long-term memories
    relevant = vector_store.similarity_search(query, k=3)
    context.extend(relevant)
    # Layer 1: include short-term summary
    context.append(layers.short_term_summary)
    # Layer 0: include full recent turns
    context.extend(layers.working_memory)
    return context
```

## Advantages Over Flat Approaches

| Approach | Recall of Recent | Recall of Distant | Token Efficiency |
|----------|-----------------|-------------------|-----------------|
| Full history | Perfect | Perfect | Poor — grows unbounded |
| [[Sliding Window]] | Perfect | None | Good — fixed size |
| [[Summarization-Based Compaction]] | Reduced | Medium | Good — fixed size |
| **Hierarchical** | Perfect | Good (via retrieval) | Good — bounded with retrieval |

The key advantage: hierarchical memory gives near-perfect recall at any distance while keeping token usage bounded.

## Design Decisions

**Layer 0 size**: Typically 5-10 turns or ~10K tokens. Must include the most recent tool call and its result, since the agent often refers back to it immediately. See [[Tool Call History]].

**Layer 1 summarization frequency**: Summarize when Layer 0 exceeds budget. Use a smaller model (GPT-4o-mini) to reduce cost. See [[Summarization-Based Compaction]] for summary quality tips.

**Layer 2 retrieval**: Use [[Embedding and Retrieval]] with the current user message as the query. Retrieve 3-5 chunks. More than that risks noise; fewer risks missing relevant context.

## Implementation in LangChain

[[Compaction in LangChain]] doesn't have a single "hierarchical memory" class, but you can compose one:
- `ConversationBufferWindowMemory` → Layer 0
- `ConversationSummaryMemory` → Layer 1
- `VectorStoreRetrieverMemory` → Layer 2

Combined with [[Token Counting]] for budget management, this gives a production-grade hierarchical memory system.

See also: [[Compaction Strategies]], [[Selective Pruning]], [[Multi-Turn Conversation Design]]
