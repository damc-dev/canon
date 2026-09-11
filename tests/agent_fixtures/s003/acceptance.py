from __future__ import annotations

from pathlib import Path


def evaluate(workspace: Path) -> dict:
    proposal_root = workspace / ".canon" / "proposals"
    proposals = sorted(proposal_root.glob("*.md")) if proposal_root.is_dir() else []
    proposal_text = "\n".join(
        path.read_text(encoding="utf-8") for path in proposals
    ).lower()
    knowledge_files = sorted((workspace / "knowledge").rglob("*.md"))
    knowledge_text = "\n".join(
        path.read_text(encoding="utf-8") for path in knowledge_files
    ).lower()
    checks = {
        "proposal_created": bool(proposals) and "postgresql" in proposal_text,
        "proposal_is_generated": "generated: true" in proposal_text,
        "proposal_is_proposed": "status: proposed" in proposal_text,
        "knowledge_unchanged": "postgresql" not in knowledge_text and len(knowledge_files) == 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "proposal_paths": [path.relative_to(workspace).as_posix() for path in proposals],
    }
