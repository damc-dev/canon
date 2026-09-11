---
name: explain-canon
description: This skill should be used when the user asks "why do you believe that", "where did that decision come from", "what superseded this", "show the decision history", or asks for provenance behind Canon context.
version: 0.2.0
---

# Explain Canon

Trace an effective statement back to its Markdown source, metadata, scope, and available Git history.

## Workflow

1. Identify the stable document ID or source path from the current context. Search Canon if necessary.
2. Call `explain_knowledge` with the ID or path.
3. Explain the effective type, scope, generated status, locked status, and explicit supersession links.
4. Include Git provenance when available.
5. When explaining evolution, call `search_knowledge` with `include_superseded=true` and compare the relevant records.
6. Distinguish a local supersession from global obsolescence. A narrower-scope replacement suppresses its target only where that narrower scope applies.

Keep normal answers concise. Reveal detailed provenance when asked rather than polluting every interaction with repository mechanics.
