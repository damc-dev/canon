---
name: canon-context
description: This skill should be used when the user asks to "implement a feature", "design the architecture", "choose a technology", "change infrastructure", or do other consequential project work that should honor established Canon decisions and constraints.
version: 0.2.0
---

# Canon Context

Retrieve effective project context before making consequential implementation or design choices.

## Workflow

1. Determine the most specific known scope for the work. Use `global` only when no narrower scope is known.
2. Call Canon's `get_context` tool with a concise description of the task and the selected scope.
3. Treat `MUST` items as constraints, `DECIDED` items as established choices, `PREFER` items as defaults, and `REFERENCE` items as evidence rather than instructions.
4. Honor context inherited from ancestor scopes. Exclude sibling scopes.
5. Never revive a superseded decision merely because it appears semantically relevant.
6. Surface `UNKNOWN` when Canon has no applicable guidance. Make only reversible assumptions, or ask for a decision when the choice is consequential.
7. Cite the source path when an established item materially affects the answer.

## Authority boundary

Treat human-owned Markdown under `knowledge/` as the source of authority. Treat the SQLite index as disposable. Treat every file marked `generated: true` as reference material, regardless of its folder or declared type.

Do not silently create or change authoritative knowledge. Use the proposal workflow for durable conclusions discovered during work.
