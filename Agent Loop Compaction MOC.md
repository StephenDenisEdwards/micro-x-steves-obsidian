# Agent Loop Compaction — Map of Content

This vault explores **agent loop compaction**: the set of techniques used to manage context window consumption in LLM-powered agent loops. As agents iterate through observe → think → act cycles, their conversation history grows until it hits token limits. Compaction solves this.

## Core Concepts

- [[Agent Loops]] — The fundamental observe → think → act cycle that drives autonomous agents
- [[Context Window]] — Token limits and why they force compaction
- [[Compaction Strategies]] — Overview of all major approaches

## Compaction Techniques

- [[Summarization-Based Compaction]] — Using an LLM to compress older turns into summaries
- [[Sliding Window]] — Fixed-size window that drops the oldest messages
- [[Hierarchical Memory]] — Layered memory: recent turns in full, older turns as summaries or embeddings
- [[Selective Pruning]] — Importance scoring to keep high-value messages and discard low-value ones

## Related Systems

- [[Token Counting]] — How to measure and budget token usage with tiktoken and model-specific tokenizers
- [[Embedding and Retrieval]] — Using vector search to recall relevant past context on demand
- [[Tool Call History]] — Special considerations when compacting tool calls and their results
- [[Multi-Turn Conversation Design]] — How conversation structure affects compaction effectiveness

## Practical Implementation

- [[Compaction in LangChain]] — Built-in memory classes: ConversationSummaryMemory, ConversationBufferWindowMemory
- [[Compaction Tradeoffs]] — Latency vs accuracy vs cost, and when to trigger compaction
