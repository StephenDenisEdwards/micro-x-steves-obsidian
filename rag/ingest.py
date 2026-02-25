"""Ingest Obsidian vault markdown files into ChromaDB.

Loads all .md files from the vault, splits by markdown headings,
extracts [[wiki links]] as metadata, and embeds into ChromaDB.

Usage:
    python rag/ingest.py
"""

import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VAULT_GLOB,
    VAULT_PATH,
)

WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def extract_wiki_links(text: str) -> list[str]:
    """Extract all [[wiki link]] targets from markdown text."""
    return list(set(WIKI_LINK_PATTERN.findall(text)))


def load_vault_documents(vault_path: Path) -> list[Document]:
    """Load all markdown files from the vault and split by headings."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    documents = []
    md_files = sorted(vault_path.glob(VAULT_GLOB))

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        note_name = md_file.stem

        # Split by markdown headings
        splits = splitter.split_text(text)

        if not splits:
            # File has no headings — treat the whole file as one chunk
            splits = [Document(page_content=text, metadata={})]

        for i, doc in enumerate(splits):
            # Extract wiki links from this chunk
            links = extract_wiki_links(doc.page_content)

            # Build metadata
            doc.metadata["source"] = note_name
            doc.metadata["file"] = md_file.name
            doc.metadata["chunk_index"] = i
            doc.metadata["wiki_links"] = ", ".join(links) if links else ""

            # Build a section label from header metadata
            section_parts = []
            for key in ("h1", "h2", "h3"):
                if key in doc.metadata:
                    section_parts.append(doc.metadata[key])
            doc.metadata["section"] = " > ".join(section_parts) if section_parts else note_name

            documents.append(doc)

    return documents


def ingest(clear_existing: bool = True) -> int:
    """Ingest vault documents into ChromaDB. Returns chunk count."""

    # Clear existing database if requested
    if clear_existing and CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"Cleared existing database at {CHROMA_PERSIST_DIR}")

    # Load and split documents
    print(f"Loading vault from {VAULT_PATH}...")
    documents = load_vault_documents(VAULT_PATH)
    print(f"Split into {len(documents)} chunks from {len(set(d.metadata['source'] for d in documents))} notes")

    # Create embeddings and store in ChromaDB
    print(f"Embedding with {EMBEDDING_MODEL}...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    print(f"Stored {len(documents)} chunks in {CHROMA_PERSIST_DIR}")
    return len(documents)


if __name__ == "__main__":
    count = ingest()
    print(f"\nDone! Indexed {count} chunks.")
