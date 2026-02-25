# Compaction in LangChain

LangChain provides built-in memory classes that implement several [[Compaction Strategies]]. These are the fastest way to add compaction to an [[Agent Loops|agent loop]] without building everything from scratch.

## Memory Classes Overview

### ConversationBufferWindowMemory

Implements [[Sliding Window]] — keeps only the last K interactions:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10)  # Keep last 10 exchanges
memory.save_context({"input": "Set up auth"}, {"output": "I'll implement OAuth..."})
# After 10+ exchanges, oldest are dropped
```

**Pros**: Zero overhead, dead simple
**Cons**: Loses all older context. See [[Sliding Window]] for full tradeoff analysis.

### ConversationSummaryMemory

Implements [[Summarization-Based Compaction]] — uses an LLM to maintain a running summary:

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")  # Cheap model for summarization
memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "What's OAuth?"}, {"output": "OAuth is..."})
memory.save_context({"input": "Implement it"}, {"output": "Here's the code..."})

print(memory.load_memory_variables({})["history"])
# "The human asked about OAuth. The AI explained it's an authorization
#  framework. The human then asked to implement it, and the AI provided
#  code using the google-auth library."
```

**Pros**: Preserves semantic content across arbitrarily long conversations
**Cons**: Every `save_context` call triggers an LLM call — adds latency and cost. See [[Compaction Tradeoffs]].

### ConversationSummaryBufferMemory

The best of both worlds — implements [[Sliding Window]] + [[Summarization-Based Compaction]]:

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=2000  # Summarize when buffer exceeds this
)
```

Recent messages are kept in full (buffer). When the buffer exceeds `max_token_limit`, older messages are summarized. This is essentially [[Hierarchical Memory]] with two layers.

### VectorStoreRetrieverMemory

Implements [[Embedding and Retrieval]] — stores all messages as embeddings and retrieves relevant ones:

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

**Pros**: Can recall any past message if it's relevant to the current query
**Cons**: Retrieval isn't guaranteed to find what you need. No recency guarantee.

## Composing a Hierarchical Memory

LangChain doesn't have a single hierarchical memory class, but you can compose one using `CombinedMemory`:

```python
from langchain.memory import CombinedMemory

# Layer 0: Recent turns in full
buffer = ConversationBufferWindowMemory(
    k=5,
    memory_key="recent_history",
    input_key="input"
)

# Layer 1+2: Summary + retrieval for older turns
summary_retrieval = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=2000,
    memory_key="summary_history",
    input_key="input"
)

memory = CombinedMemory(memories=[buffer, summary_retrieval])
```

## Custom Compaction with LCEL

For more control, build compaction into a LangChain Expression Language (LCEL) chain:

```python
from langchain_core.runnables import RunnableLambda

def compact_history(inputs):
    history = inputs["history"]
    if count_tokens(history) > 4000:
        summary = summarize_chain.invoke({"history": history})
        return {"history": summary, **inputs}
    return inputs

chain = (
    RunnableLambda(compact_history)
    | prompt
    | llm
    | output_parser
)
```

## Token Counting in LangChain

LangChain's memory classes use [[Token Counting]] internally via `get_num_tokens()` on the LLM:

```python
llm = ChatOpenAI(model="gpt-4o-mini")
token_count = llm.get_num_tokens("How many tokens is this?")  # ~8
```

This drives the `max_token_limit` parameter in `ConversationSummaryBufferMemory`.

See also: [[Compaction Strategies]], [[Hierarchical Memory]], [[Compaction Tradeoffs]]
