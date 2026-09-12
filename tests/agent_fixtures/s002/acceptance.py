from __future__ import annotations

from pathlib import Path

from scripts.agent_acceptance import field_value, read_text, selects


AZURE = r"\b(microsoft azure|azure)\b"
AWS = r"\b(aws|amazon|lambda)\b"


def evaluate(workspace: Path) -> dict:
    plan = workspace / "cloud-plan.md"
    selected = field_value(read_text(plan), "provider")
    checks = {
        "cloud_plan_created": plan.is_file(),
        "inherits_acme_decision": selects(selected, AZURE),
        "excludes_sibling_decision": bool(selected) and not selects(selected, AWS),
    }
    return {"passed": all(checks.values()), "checks": checks, "selected_provider": selected}
