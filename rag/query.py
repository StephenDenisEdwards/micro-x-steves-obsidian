"""Query the Obsidian vault RAG pipeline.

Retrieves relevant chunks from ChromaDB, expands context via wiki links,
and generates an answer with GPT-4o-mini.

Usage:
    python rag/query.py "What are the main compaction strategies?"
"""

import sys

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GRAPH_EXPAND_K,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVAL_K,
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledgeable assistant answering questions about agent loop "
        "compaction and related AI agent concepts. Use the retrieved context below "
        "to answer the question. Cite the source notes in your answer using "
        "[[Note Name]] format. If the context doesn't contain enough information "
        "to answer, say so.\n\n"
        "Retrieved context:\n{context}"
    )),
    ("human", "{question}"),
])


def get_vectorstore() -> Chroma:
    """Load the persisted ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )


def graph_expanded_retrieval(question: str, vectorstore: Chroma) -> str:
    """Retrieve relevant chunks, then expand by following wiki links."""

    # Primary retrieval
    primary_docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)

    # Collect wiki links from retrieved docs
    linked_notes = set()
    primary_sources = set()
    for doc in primary_docs:
        primary_sources.add(doc.metadata.get("source", ""))
        links_str = doc.metadata.get("wiki_links", "")
        if links_str:
            for link in links_str.split(", "):
                linked_notes.add(link.strip())

    # Remove notes we already have
    linked_notes -= primary_sources

    # Retrieve chunks from linked notes (graph expansion)
    expanded_docs = []
    if linked_notes and GRAPH_EXPAND_K > 0:
        for note_name in linked_notes:
            results = vectorstore.get(
                where={"source": note_name},
                include=["documents", "metadatas"],
            )
            if results["documents"]:
                # Take the first chunk from the linked note
                expanded_docs.append({
                    "content": results["documents"][0],
                    "source": note_name,
                })
            if len(expanded_docs) >= GRAPH_EXPAND_K:
                break

    # Format context
    context_parts = []
    for doc in primary_docs:
        source = doc.metadata.get("source", "Unknown")
        section = doc.metadata.get("section", "")
        header = f"[[{source}]]"
        if section and section != source:
            header += f" > {section}"
        context_parts.append(f"--- {header} ---\n{doc.page_content}")

    for doc in expanded_docs:
        context_parts.append(
            f"--- [[{doc['source']}]] (linked) ---\n{doc['content']}"
        )

    return "\n\n".join(context_parts)


def query(question: str) -> str:
    """Run a RAG query against the vault."""
    vectorstore = get_vectorstore()

    # Build the chain
    chain = (
        {
            "context": lambda q: graph_expanded_retrieval(q, vectorstore),
            "question": RunnablePassthrough(),
        }
        | ANSWER_PROMPT
        | ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        | StrOutputParser()
    )

    return chain.invoke(question)


def main():
    if len(sys.argv) < 2:
        print("Usage: python rag/query.py \"Your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")
    print("Retrieving and generating answer...\n")

    answer = query(question)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
