---
name: explain-canon
description: Trace an established choice back to its source file, scope, and Git history. Use for any question about why the project does something a particular way, what decided it, when or by whom it changed, or what replaced an earlier choice - including task-shaped requests like "why does the payments API deploy to Container Apps rather than AKS", "document which decision applies and cite the commit", or "write up where this constraint came from". Use it rather than reading knowledge files directly, since provenance and supersession live outside the file text.
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
