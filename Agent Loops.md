# Agent Loops

An **agent loop** is the core execution cycle of an LLM-powered autonomous agent. The agent repeatedly cycles through phases until it completes its task or is interrupted.

## The Basic Loop

```
while not done:
    observation = perceive(environment)
    thought = llm.reason(history + observation)
    action = select_action(thought)
    result = execute(action)
    history.append(observation, thought, action, result)
```

Each iteration adds to the conversation history. This is where [[Context Window]] pressure begins — every loop iteration consumes tokens, and long-running tasks can easily exhaust the available window.

## The ReAct Pattern

The most common agent loop pattern is **ReAct** (Reasoning + Acting):

1. **Observe** — Receive input or tool output from the environment
2. **Think** — The LLM reasons about what to do next (chain-of-thought)
3. **Act** — The agent calls a tool or produces a final answer
4. **Observe** — The tool result comes back, starting the next cycle

ReAct interleaves reasoning traces with actions, which produces better results than pure chain-of-thought or pure action sequences. However, the reasoning traces consume significant tokens, making [[Compaction Strategies]] essential for multi-step tasks.

## Why Loops Need Compaction

A typical agent loop iteration might consume 500–2,000 tokens (thought + action + result). After 20 iterations, that's 10,000–40,000 tokens of history — potentially half of a model's [[Context Window]]. Without compaction:

- The agent hits token limits and crashes
- Earlier context gets truncated unpredictably by the API
- Costs scale linearly with history length (every turn re-sends everything)

This is why [[Compaction Strategies]] are not optional for production agent systems. See [[Compaction Tradeoffs]] for guidance on when and how aggressively to compact.

## Loop Variants

| Pattern | Description | Compaction Pressure |
|---------|-------------|-------------------|
| ReAct | Reasoning + Acting interleaved | High — reasoning traces are verbose |
| Plan-then-Execute | Plan upfront, execute steps | Medium — plan is fixed, only execution grows |
| Reflexion | Self-critique after failure | Very High — adds reflection turns on top of action turns |
| Tree of Thought | Branching reasoning | Very High — multiple reasoning paths stored |

All variants benefit from [[Hierarchical Memory]] and [[Selective Pruning]] to manage their growing histories.

See also: [[Tool Call History]], [[Multi-Turn Conversation Design]]
