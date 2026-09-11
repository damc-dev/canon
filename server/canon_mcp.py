from __future__ import annotations

from pathlib import Path

try:
    from mcp.server import MCPServer
except ImportError as exc:  # pragma: no cover - exercised only without the optional runtime
    raise SystemExit("Canon requires the MCP Python SDK. Install it with: python3 -m pip install 'mcp>=2,<3'") from exc

try:
    from .canon_core import (
        explain_knowledge as explain,
        find_project_root,
        get_context as context,
        propose_knowledge as propose,
        rebuild_index,
        search_knowledge as search,
    )
except ImportError:
    from canon_core import (  # type: ignore
        explain_knowledge as explain,
        find_project_root,
        get_context as context,
        propose_knowledge as propose,
        rebuild_index,
        search_knowledge as search,
    )


mcp = MCPServer("Canon")


def _root(project_root: str | None) -> Path:
    return find_project_root(project_root)


@mcp.tool()
def get_context(task: str, scope: str = "global", project_root: str | None = None, limit: int = 12) -> str:
    """Return effective Canon context for a task and hierarchical scope."""
    return context(_root(project_root), task, scope, limit)


@mcp.tool()
def search_knowledge(
    query: str,
    scope: str = "global",
    include_superseded: bool = False,
    project_root: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Search Canon, excluding in-scope superseded documents unless history is requested."""
    return search(_root(project_root), query, scope, include_superseded=include_superseded, limit=limit)


@mcp.tool()
def explain_knowledge(identifier: str, project_root: str | None = None) -> dict:
    """Explain a Canon document by ID or relative path, including available Git provenance."""
    return explain(_root(project_root), identifier)


@mcp.tool()
def propose_knowledge(
    title: str,
    body: str,
    document_type: str = "decision",
    scope: str = "global",
    supersedes: list[str] | None = None,
    project_root: str | None = None,
) -> dict:
    """Create a generated, non-authoritative proposal under .canon/proposals/."""
    return propose(
        _root(project_root),
        title=title,
        body=body,
        document_type=document_type,
        scope=scope,
        supersedes=supersedes or (),
    )


@mcp.tool()
def rebuild_knowledge_index(project_root: str | None = None) -> dict:
    """Rebuild the disposable local SQLite FTS index from human-owned Markdown."""
    return rebuild_index(_root(project_root))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
