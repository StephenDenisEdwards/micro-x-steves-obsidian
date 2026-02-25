"""Configuration for the Obsidian vault RAG pipeline."""

from pathlib import Path

# Vault settings
VAULT_PATH = Path(__file__).resolve().parent.parent
VAULT_GLOB = "*.md"

# ChromaDB settings
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "obsidian_vault"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM for answer generation
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2

# Retrieval settings
RETRIEVAL_K = 4            # Number of chunks to retrieve
GRAPH_EXPAND_K = 2         # Additional chunks from linked notes
