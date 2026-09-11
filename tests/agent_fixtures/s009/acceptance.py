from __future__ import annotations

import re
from pathlib import Path

from scripts.agent_acceptance import last_commit, read_text


def evaluate(workspace: Path) -> dict:
    answer = workspace / "provenance.md"
    text = read_text(answer)
    commit = last_commit(workspace, "knowledge/decisions/runtime.md")
    cited_hashes = re.findall(r"\b[0-9a-f]{7,40}\b", text.lower())
    checks = {
        "provenance_created": answer.is_file(),
        "cites_source": "knowledge/decisions/runtime.md" in text,
        "names_superseded_decision": "payments/runtime-2024" in text,
        "cites_commit": any(commit.startswith(value) for value in cited_hashes),
    }
    return {"passed": all(checks.values()), "checks": checks, "commit": commit}
