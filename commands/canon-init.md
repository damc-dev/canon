---
description: Initialize Canon's human-owned knowledge structure in the current project.
---

Initialize Canon in the current project.

1. Create `knowledge/constraints`, `knowledge/decisions`, `knowledge/standards`, `knowledge/preferences`, and `knowledge/reference` when absent.
2. Create `.canon/proposals` when absent.
3. Add `.canon/index.db` to the project's `.gitignore` without removing existing entries.
4. Do not overwrite existing knowledge.
5. Call `rebuild_knowledge_index` after initialization.
6. Report each created path and explain that Markdown under `knowledge/` is human-owned authority while `.canon/proposals/` is non-authoritative review material.
