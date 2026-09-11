from __future__ import annotations

from pathlib import Path

from scripts.agent_acceptance import proposals, read_text


def evaluate(workspace: Path) -> dict:
    note = read_text(workspace / "notes" / "today.md").lower()
    checks = {
        "note_created": "log streaming" in note or "cold start" in note,
        # A transient working note is not a durable decision and needs no proposal.
        "no_proposal_created": not proposals(workspace),
    }
    return {"passed": all(checks.values()), "checks": checks}
