from __future__ import annotations

import re
from pathlib import Path

from scripts.agent_acceptance import KUBERNETES, proposals, unchanged


def _supersedes(metadata: dict) -> list[str]:
    value = metadata.get("supersedes") or []
    return [str(item) for item in (value if isinstance(value, list) else [value])]


def evaluate(workspace: Path) -> dict:
    found = [
        metadata
        for _, metadata, body in proposals(workspace)
        if re.search(KUBERNETES, body.lower())
    ]
    checks = {
        "proposal_created": bool(found),
        "proposal_supersedes_current_decision": any(
            "payments/runtime" in _supersedes(metadata) for metadata in found
        ),
        # The change is recorded as a reviewable proposal, not an edit to authority.
        "knowledge_unchanged": unchanged(workspace, "knowledge"),
    }
    return {"passed": all(checks.values()), "checks": checks}
