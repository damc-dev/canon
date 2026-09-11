# Canon

**Canon gives AI coding agents a shared understanding of what your organization has actually decided.**

Coding agents already have search, memory, rules, skills, and retrieval. Those systems are good at finding relevant text. They are much less reliable at answering a harder question:

> When several relevant sources disagree, which knowledge wins?

Canon is a small, local authority layer for Claude Code. It composes human-owned Markdown into effective task context, preserves scope and decision history, and keeps AI-generated proposals outside the authority boundary.

```text
Human-owned Markdown + Git
           │
           ▼
   Canon authority resolver
           │
    ┌──────┼────────┐
    ▼      ▼        ▼
 inherit  supersede explain
    └──────┼────────┘
           ▼
 MUST · DECIDED · PREFER · REFERENCE · UNKNOWN
           │
           ▼
      Claude Code
```

Canon is not another wiki, vector database, or agent memory system. It is the layer that tells an agent how retrieved knowledge should affect its behavior.

## The problem

A search result can be relevant and still be the wrong thing to follow.

```text
Company constraint     Never commit credentials.          MUST
Project decision       Use Container Apps.                 DECIDED
Engineering standard   Prefer managed identity.            PREFER
Research note          AKS may help some batch workloads.  REFERENCE
```

Ordinary retrieval tends to flatten these into equally plausible chunks. Canon keeps their semantics intact. Constraints constrain. Decisions remain in force until explicitly superseded. Standards and preferences guide. Reference material supplies evidence without silently becoming policy.

## What it feels like

Ask Claude Code to add a deployment pipeline. Before choosing an implementation, the `canon-context` skill retrieves the effective context for the task and scope:

```text
MUST
- Production deployments require manual approval.
- Never commit credentials. [locked]

DECIDED
- Use Container Apps for the API workload.

PREFER
- Use managed identity instead of secrets.

UNKNOWN
- No rollback strategy has been established.
```

Claude can work within those boundaries, tell you when an important decision is missing, and answer “Why?” with the source Markdown and Git provenance.

Most sessions should require no knowledge-management work. Canon should be noticeable because the agent makes fewer context-free choices, stops reviving old decisions, and asks before inventing a consequential standard.

## Core rules

1. Human-owned Markdown under `knowledge/` is authoritative.
2. The SQLite full-text index under `.canon/index.db` is disposable.
3. Constraints, decisions, standards, and preferences affect agent behavior; references are evidence.
4. More specific scopes inherit applicable knowledge from ancestor scopes, never siblings.
5. `supersedes` replaces an older stable document ID only where the newer document's scope applies.
6. Locked constraints cannot be treated as locally optional.
7. Anything marked `generated: true` is demoted to reference, even inside an authoritative folder.
8. AI may create proposals, but it never silently creates organizational authority.
9. Every effective statement remains explainable back to Markdown and, when available, Git.

## Install

Requirements:

- Python 3.11+
- Claude Code with plugin and MCP support
- The MCP Python SDK

Install the runtime dependency:

```bash
python3 -m pip install "mcp>=2,<3"
```

Load the plugin directly during development:

```bash
claude --plugin-dir /absolute/path/to/canon
```

The plugin uses `${CLAUDE_PLUGIN_ROOT}` in `.mcp.json`, so the bundled local server remains portable when the plugin is installed elsewhere.

## Initialize a project

In Claude Code, run:

```text
/canon-init
```

This creates the additive project structure without overwriting existing knowledge:

```text
your-project/
├── knowledge/
│   ├── constraints/
│   ├── decisions/
│   ├── standards/
│   ├── preferences/
│   └── reference/
└── .canon/
    ├── index.db        # generated, disposable
    └── proposals/      # generated, non-authoritative
```

Commit `knowledge/`. Ignore `.canon/index.db`. Decide with your team whether proposals belong in Git.

## Write Canon documents

Metadata stays intentionally small. A global locked constraint:

```markdown
---
id: company/security/no-secrets
type: constraint
scope: global
locked: true
---

# Credentials

Never commit credentials to source control.
```

A project decision:

```markdown
---
id: project/runtime
type: decision
scope: client/acme/project/payments
---

# Runtime

Use Container Apps for the payments API.
```

Most documents need only a stable `id` and a `type`. Omit `scope` to use `global`.

Supported types:

| Type | Agent meaning | Context label |
|---|---|---|
| `constraint` | Must comply | `MUST` |
| `decision` | Established choice | `DECIDED` |
| `standard` | Strong default | `PREFER` |
| `preference` | Preferred approach | `PREFER` |
| `reference` | Evidence only | `REFERENCE` |

## Scope inheritance

Scopes are slash-delimited and hierarchical:

```text
global
  └── client/acme
      └── client/acme/project/payments
          └── client/acme/project/payments/workload/api
```

A request at the API workload receives knowledge from all applicable ancestors. It does not receive knowledge from `client/other` or a sibling workload.

Prefer the narrowest scope that accurately describes the knowledge. Scope is an applicability boundary, not a search-ranking score.

## Explicit supersession

Keep old decisions in Markdown and Git. Replace them explicitly using stable IDs:

```markdown
---
id: project/runtime-v2
type: decision
scope: client/acme/project/payments
supersedes: company/runtime-v1
---

# Runtime

Use Container Apps.
```

At `client/acme/project/payments`, `project/runtime-v2` suppresses `company/runtime-v1` from normal context. The company decision remains active elsewhere. Historical queries can still include both records.

Canon does not infer supersession from similarity. Replacement must be declared.

## Human authority and AI proposals

When a durable conclusion emerges, Claude can call `propose_knowledge`. Canon writes the result only to:

```text
.canon/proposals/
```

Every proposal has:

```yaml
status: proposed
generated: true
```

Those files are not indexed as authority. Promotion means creating or editing a human-owned Markdown file under `knowledge/`, reviewing the exact change, and committing it through the normal Git workflow.

This boundary prevents AI knowledge laundering: generated summaries cannot gradually become indistinguishable from decisions established by people.

## MCP tools

Canon exposes five local tools:

| Tool | Purpose |
|---|---|
| `get_context(task, scope)` | Return effective `MUST`, `DECIDED`, `PREFER`, and `REFERENCE` context |
| `search_knowledge(query, scope)` | Search applicable current knowledge |
| `explain_knowledge(identifier)` | Show source metadata and available Git provenance |
| `propose_knowledge(...)` | Create a non-authoritative proposal |
| `rebuild_knowledge_index()` | Rebuild the disposable SQLite FTS5 index |

Use `include_superseded=true` with `search_knowledge` to inspect history.

## Included skills

- `canon-context` retrieves effective context before consequential work.
- `capture-canon-decision` creates reviewable proposals for durable decisions.
- `explain-canon` answers where a belief came from and how it changed.

The plugin also includes `/canon-init` for additive project setup.

## Development and validation

Run the dependency-free core tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

The tests cover classification, generated-content demotion, proposal isolation, index rebuilding, ancestor inheritance, sibling isolation, scope-aware supersession, historical retrieval, and protection against AI-generated supersession.

Validate JSON and compile the Python sources:

```bash
python3 -m json.tool .claude-plugin/plugin.json >/dev/null
python3 -m json.tool .mcp.json >/dev/null
python3 -m compileall -q server tests
```

## Deliberate limits

Canon v0.2.0 is intentionally small:

- local project knowledge only;
- lexical SQLite FTS5 search, not embeddings;
- explicit scope strings, not an organizational ontology;
- explicit supersession, not inferred conflict resolution;
- Git provenance when the project is in Git;
- no remote repository fetching, policy engine, dashboard, or automatic promotion.

Possible future layers include opt-in repository inheritance, decision-gap telemetry, and adapters that turn selected machine-verifiable authorities into enforcement. They are not part of the core resolver.

## Why Canon

Your coding agent already has memory, search, rules, and skills. Canon tells it which knowledge wins.

The goal is an open, version-controlled authority layer that lets AI coding agents inherit what your organization knows without letting AI redefine what your organization believes.
