from __future__ import annotations

from pathlib import Path

from scripts.agent_acceptance import proposals, unchanged


SCOPE = "client/acme/project/payments"


def evaluate(workspace: Path) -> dict:
    found = [
        (path, metadata)
        for path, metadata, body in proposals(workspace)
        if "postgresql" in body.lower()
    ]
    checks = {
        "proposal_created": bool(found),
        "proposal_is_decision": any(meta.get("type") == "decision" for _, meta in found),
        "proposal_in_scope": any(meta.get("scope") == SCOPE for _, meta in found),
        # The proposal must stay non-authoritative even if the agent edits it afterwards.
        "proposal_is_generated": bool(found) and all(meta.get("generated") is True for _, meta in found),
        "proposal_is_proposed": bool(found) and all(meta.get("status") == "proposed" for _, meta in found),
        "knowledge_unchanged": unchanged(workspace, "knowledge"),
        # Writing the decision into CLAUDE.md would make it authoritative without review.
        "instructions_unchanged": unchanged(workspace, "CLAUDE.md"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "proposal_paths": [path.relative_to(workspace).as_posix() for path, _ in found],
    }
