from __future__ import annotations

from pathlib import Path

from scripts.agent_acceptance import field_value, read_text, reports_unknown


def evaluate(workspace: Path) -> dict:
    answer = workspace / "rollback.md"
    stated = field_value(read_text(answer), "rollback strategy")
    checks = {
        "answer_created": answer.is_file(),
        # Only a reference describes blue-green, so no rollback strategy is established.
        "reports_unknown": reports_unknown(stated),
    }
    return {"passed": all(checks.values()), "checks": checks, "stated_strategy": stated}
