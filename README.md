# Obsidian Vault RAG Pipeline — Agent Loop Compaction

A demonstration of Retrieval-Augmented Generation (RAG) over an [Obsidian](https://obsidian.md) knowledge vault. The vault contains interconnected notes about **agent loop compaction** — the techniques used to manage context window consumption in LLM-powered agent systems. A Python pipeline ingests the vault into a vector database and answers natural-language questions grounded in the vault's content.

## What Is Obsidian?

[Obsidian](https://obsidian.md) is a markdown-based knowledge management tool built around three principles:

- **Local-first markdown files.** Every note is a plain `.md` file on disk. No proprietary format, no database — just files you own. This makes Obsidian vaults trivially accessible to external tools like this RAG pipeline.

- **Bidirectional wiki links.** Notes reference each other with `[[wiki links]]` (e.g., `[[Sliding Window]]`). These links create a knowledge graph — a web of connections between concepts. Obsidian visualizes this graph and lets you navigate it, but the links are just plain text in the markdown files.

- **Emergent structure.** Rather than forcing a folder hierarchy upfront, Obsidian encourages organic growth through links. Structure emerges from connections. A **Map of Content (MOC)** note serves as a hub that links to related notes, acting as a table of contents for a topic cluster.

These properties — plain markdown, wiki links as a knowledge graph, and hub-based organization — make Obsidian vaults a natural fit for RAG. The markdown is easy to parse, the wiki links provide graph structure for context expansion, and the MOC pattern gives the retriever a navigational backbone.

## The Example Vault

The vault explores **agent loop compaction**: how to manage the growing conversation history in LLM agent loops (observe → think → act → observe) so it fits within token limits without losing critical context.

### Note inventory

| Category | Note | Description |
|----------|------|-------------|
| Hub | `Agent Loop Compaction MOC.md` | Top-level map of content linking to all other notes |
| Core | `Agent Loops.md` | The observe → think → act cycle, ReAct pattern, loop variants |
| Core | `Context Window.md` | Token limits by model, why large windows aren't enough, budget management |
| Core | `Compaction Strategies.md` | Overview of all approaches with a comparison table and selection guide |
| Technique | `Summarization-Based Compaction.md` | Using LLMs to compress older turns, recursive summarization |
| Technique | `Sliding Window.md` | Fixed-size message window, token vs. message-based sizing |
| Technique | `Hierarchical Memory.md` | Three-layer architecture: working memory, short-term summary, long-term embeddings |
| Technique | `Selective Pruning.md` | Importance scoring, what to always keep vs. prune first |
| System | `Token Counting.md` | tiktoken, model-specific tokenizers, budget management classes |
| System | `Embedding and Retrieval.md` | Vector search for recalling past context, chunking strategies, graph-aware retrieval |
| System | `Tool Call History.md` | Special handling of tool call triplets during compaction |
| System | `Multi-Turn Conversation Design.md` | Conversation structure patterns that make compaction more effective |
| Practical | `Compaction in LangChain.md` | ConversationSummaryMemory, ConversationBufferWindowMemory, composing hierarchical memory |
| Practical | `Compaction Tradeoffs.md` | Latency vs. accuracy vs. cost analysis, decision framework |

Every note contains `[[wiki links]]` to related notes, forming a densely connected graph. For example, `Compaction Strategies.md` links to each individual technique note, and each technique note links back to the strategies overview and to the tradeoffs analysis.

## RAG Pipeline

The `rag/` directory contains a Python pipeline that ingests the vault into a vector database and answers questions using retrieved context.

### Architecture

```
                          ┌─────────────────┐
                          │  Obsidian Vault  │
                          │   (14 .md files) │
                          └────────┬─────────┘
                                   │
                            rag/ingest.py
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
             Split by        Extract          Embed with
             markdown       [[wiki links]]    text-embedding-
             headings       as metadata       3-small
                    │              │               │
                    └──────────────┼───────────────┘
                                   ▼
                          ┌─────────────────┐
                          │    ChromaDB      │
                          │  (rag/chroma_db) │
                          │   98 chunks      │
                          └────────┬─────────┘
                                   │
                            rag/query.py
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
              Similarity     Graph expansion   Answer with
              search         via wiki link     GPT-4o-mini
              (top 4)        metadata          (cite sources)
                    │              │               │
                    └──────────────┼───────────────┘
                                   ▼
                             Final answer
```

### How the vault is used in the RAG

**Ingestion (`rag/ingest.py`)**

1. **Load**: Reads every `.md` file in the vault root.
2. **Split**: Uses `MarkdownHeaderTextSplitter` to split each note by `#`, `##`, and `###` headings. Each section becomes a self-contained chunk with its heading preserved for context.
3. **Extract wiki links**: A regex parser finds all `[[wiki links]]` in each chunk and stores them as metadata (e.g., `"wiki_links": "Sliding Window, Summarization-Based Compaction"`). This preserves the Obsidian graph structure in the vector database.
4. **Embed**: Each chunk is embedded with OpenAI's `text-embedding-3-small` model.
5. **Store**: Chunks and their metadata are persisted to a local ChromaDB database at `rag/chroma_db/`.

**Querying (`rag/query.py`)**

1. **Similarity search**: The user's question is embedded and compared against all chunks. The top 4 most similar chunks are retrieved.
2. **Graph expansion**: The retriever examines the `wiki_links` metadata on the retrieved chunks. If those links point to notes not already in the result set, it fetches additional chunks from those linked notes (up to 2 extra). This exploits Obsidian's link structure — a question about "compaction strategies" might retrieve the overview note, and graph expansion automatically pulls in the specific technique notes it links to.
3. **Answer generation**: The retrieved chunks (primary + graph-expanded) are formatted with source attribution and passed to GPT-4o-mini, which generates an answer citing the vault notes in `[[Note Name]]` format.

### Files

| File | Purpose |
|------|---------|
| `rag/config.py` | Central configuration: vault path, model names, retrieval parameters |
| `rag/ingest.py` | Loads vault, splits by headings, extracts wiki links, embeds into ChromaDB |
| `rag/query.py` | CLI query tool: retrieves chunks, expands via graph, generates answer |
| `rag/requirements.txt` | Python dependencies |

### Setup

```bash
# Install dependencies
pip install -r rag/requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Ingest the vault into ChromaDB
python rag/ingest.py

# Query the vault
python rag/query.py "What are the main compaction strategies?"
python rag/query.py "How does LangChain handle compaction?"
python rag/query.py "What is the difference between sliding window and summarization?"
```

### Configuration

All settings are in `rag/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `VAULT_PATH` | Parent of `rag/` | Root directory of the Obsidian vault |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | Model used for answer generation |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `GRAPH_EXPAND_K` | `2` | Additional chunks pulled via wiki link expansion |
