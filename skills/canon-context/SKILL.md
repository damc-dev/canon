---
name: canon-context
description: Load established project knowledge before consequential work, so the answer follows decisions and constraints already made. Use whenever the task names a project scope, or picks or writes down a platform, provider, runtime, database, library, or other technical choice - including plain build requests like "create deployment-plan.md for the payments API", "write the config for this service", "add a Dockerfile", or "how should we deploy this". Also use when asked what has been decided, established, or chosen for a scope. Do not wait for Canon to be mentioned by name.
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
