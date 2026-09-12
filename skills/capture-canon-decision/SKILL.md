---
name: capture-canon-decision
description: Turn a durable decision into a reviewable Canon proposal instead of an ad-hoc note or memory file. Use whenever asked to remember, record, capture, note, or persist something for later or for future sessions, and whenever the user states a decision the project should follow from now on - "we've decided to move to AKS", "we use PostgreSQL for the database", "going forward, deploys go through staging". Applies even when the request sounds like a personal note and never mentions Canon.
version: 0.2.0
---

# Capture a Canon Decision

Turn durable decisions, constraints, standards, or preferences into reviewable proposals without granting AI-generated text authority.

## Workflow

1. Confirm that the statement is durable, project-relevant, and specific enough to record.
2. Classify it as `constraint`, `decision`, `standard`, `preference`, or `reference`.
3. Select the narrowest correct scope.
4. Identify an older document only when the user explicitly states that the new decision replaces it. Pass that stable ID through `supersedes`.
5. Call Canon's `propose_knowledge` tool.
6. Present the proposal path and summarize the human promotion boundary.

## Promotion boundary

Keep proposals under `.canon/proposals/` with `generated: true` and `status: proposed`. Never move or rewrite a proposal directly into authority on the model's own initiative.

Promote knowledge only through a human-reviewed change to a Markdown file under `knowledge/`, followed by the user's normal Git workflow. Preserve AI provenance when reusing proposed wording.
