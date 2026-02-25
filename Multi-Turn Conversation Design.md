# Multi-Turn Conversation Design

The way you structure conversations in [[Agent Loops]] directly affects how well [[Compaction Strategies]] work. Good conversation design makes compaction easier and less lossy.

## Conversation Structure Patterns

### Flat Conversations
```
User → Agent → User → Agent → User → Agent ...
```
Every message is at the same level. Simple but hard to compact because there's no natural grouping.

### Phase-Based Conversations
```
Phase 1: Research
  User → Agent → Tool → Agent → Tool → Agent
Phase 2: Planning
  Agent → User → Agent
Phase 3: Implementation
  Agent → Tool → Agent → Tool → Agent → Tool → Agent
```
Natural phase boundaries make excellent compaction points. You can summarize an entire phase into a few sentences. [[Summarization-Based Compaction]] works best when phases are clearly delineated.

### Tree-Structured Conversations
```
Main task
├── Subtask A
│   ├── Tool call 1
│   └── Tool call 2
├── Subtask B
│   └── Tool call 3
└── Subtask C
```
Each subtask can be compacted independently. Completed subtasks become summaries while the active subtask stays in full detail.

## Designing for Compaction

### 1. Use Explicit Phase Markers

When building an agent, have it emit phase transitions:

```python
messages.append({
    "role": "assistant",
    "content": "## Phase Complete: Research\n"
               "Key findings: [summary]\n"
               "## Starting Phase: Implementation"
})
```

These markers give [[Selective Pruning]] clear signals about where to draw compaction boundaries.

### 2. Self-Summarizing Turns

Design the agent's prompt to produce self-summarizing responses:

```
System prompt: "After completing each subtask, write a brief
'RESULT:' line summarizing what was accomplished and any key
outputs. This helps with context management."
```

These embedded summaries make [[Summarization-Based Compaction]] more accurate because the model has already identified what's important.

### 3. Minimize Filler Turns

Turns like "Sure, I'll do that now" or "Let me think about this..." add tokens without information. Design prompts that encourage the agent to go directly to action:

```
# Instead of:
"I'll read that file for you now."
[reads file]
"I've read the file. Here's what I found..."

# Prefer:
[reads file]
"The file contains: [analysis]"
```

This reduces the amount [[Selective Pruning]] has to remove and keeps the conversation denser.

### 4. Structured Output for Tool Results

When an agent processes tool results, encourage structured summaries:

```
File: src/auth.py
- Purpose: OAuth authentication handler
- Key classes: AuthHandler, TokenRefresher
- Dependencies: flask, google-auth
- Issues found: No rate limiting on refresh endpoint
```

Structured summaries survive compaction better than prose paragraphs because each line is a standalone fact.

## Impact on [[Context Window]] Usage

| Design Pattern | Tokens per Turn | Compaction Efficiency |
|---------------|----------------|----------------------|
| Verbose, conversational | 500-1,000 | Low — lots of filler |
| Concise, structured | 200-400 | High — most tokens carry information |
| Self-summarizing | 300-500 | Very High — built-in summaries |

Better conversation design means less aggressive compaction is needed, which means less information loss. It's the cheapest optimization available.

## Conversation Design and [[Hierarchical Memory]]

When conversations are phase-based and self-summarizing, hierarchical memory works beautifully:
- Each completed phase naturally compresses into a Layer 1 summary
- Self-summaries at the end of subtasks become the basis for Layer 1 content
- Phase transitions tell the system when to move content from Layer 0 to Layer 1

See also: [[Agent Loops]], [[Compaction Strategies]], [[Tool Call History]], [[Compaction Tradeoffs]]
