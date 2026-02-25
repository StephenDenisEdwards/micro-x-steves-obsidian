# Embedding and Retrieval

Embedding and retrieval uses **vector search** to recall relevant past context on demand, rather than keeping everything in the [[Context Window]]. It's a core component of [[Hierarchical Memory]] and the basis of RAG (Retrieval-Augmented Generation).

## How It Works

```
1. EMBED: Convert text chunks into vector embeddings
   "The agent chose OAuth for auth" → [0.023, -0.041, 0.089, ...]

2. STORE: Save embeddings in a vector database
   ChromaDB, Pinecone, Weaviate, pgvector, etc.

3. QUERY: When the agent needs context, embed the query and find similar chunks
   "What authentication method did we choose?" → top-3 similar chunks

4. INJECT: Add retrieved chunks to the prompt before the LLM call
```

## Embedding Models

| Model | Dimensions | Cost | Quality |
|-------|-----------|------|---------|
| text-embedding-3-small | 1536 | $0.02/M tokens | Good |
| text-embedding-3-large | 3072 | $0.13/M tokens | Better |
| text-embedding-ada-002 | 1536 | $0.10/M tokens | Legacy |
| nomic-embed-text | 768 | Free (local) | Good |

For most agent compaction use cases, `text-embedding-3-small` is the sweet spot — cheap enough to embed everything, good enough for accurate retrieval.

## Chunking for Agent History

When embedding [[Agent Loops|agent loop]] history, the chunking strategy matters:

**Per-turn chunking**: Each message becomes one chunk. Simple but loses conversational context.

**Turn-pair chunking**: User message + agent response become one chunk. Preserves Q&A pairs.

**Semantic chunking**: Group messages by topic (e.g., all messages about "setting up auth" become one chunk). Best quality but requires extra processing.

**Heading-based chunking**: For structured notes (like this vault), split by markdown headings. Each section becomes a self-contained chunk.

## Retrieval in the Compaction Pipeline

In [[Hierarchical Memory]], retrieval serves as **Layer 2** — the long-term memory:

```python
def retrieve_relevant_context(query: str, k: int = 3) -> list[str]:
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def build_prompt(user_message: str, working_memory: list, summary: str):
    # Layer 2: Retrieve from long-term memory
    retrieved = retrieve_relevant_context(user_message)
    # Assemble the prompt
    return [
        system_prompt,
        *retrieved,          # Long-term recall
        summary,             # Layer 1: short-term summary
        *working_memory,     # Layer 0: recent turns
        user_message
    ]
```

## Graph-Aware Retrieval

In interconnected knowledge bases (like an Obsidian vault with `[[wiki links]]`), retrieval can follow links to pull in related context:

```python
def graph_expanded_retrieval(query: str, k: int = 3, expand: int = 2):
    # Get directly relevant chunks
    primary = vector_store.similarity_search(query, k=k)
    # Follow wiki links to get related chunks
    linked_notes = extract_wiki_links(primary)
    secondary = vector_store.get(where={"source": {"$in": linked_notes}})
    return primary + secondary[:expand]
```

This is particularly powerful for vaults where concepts are densely linked — retrieving a chunk about [[Compaction Strategies]] might also pull in the linked [[Sliding Window]] and [[Summarization-Based Compaction]] notes.

## Vector Database Options

For local agent use:
- **ChromaDB**: Lightweight, Python-native, persistent to disk. Great for single-user agents.
- **SQLite-VSS**: If you're already using SQLite.
- **FAISS**: Facebook's library, very fast but no persistence by default.

For production:
- **Pinecone**: Managed, scalable, expensive.
- **Weaviate**: Self-hosted or cloud, good hybrid search.
- **pgvector**: PostgreSQL extension, great if you're already on Postgres.

See also: [[Hierarchical Memory]], [[Compaction Strategies]], [[Tool Call History]]
