from __future__ import annotations

from pathlib import Path

from scripts.agent_acceptance import CONTAINER_APPS, KUBERNETES, field_value, read_text, selects


def evaluate(workspace: Path) -> dict:
    plan = workspace / "deployment-plan.md"
    selected = field_value(read_text(plan), "platform")
    checks = {
        "deployment_plan_created": plan.is_file(),
        "uses_inherited_decision": selects(selected, KUBERNETES),
        "ignores_sibling_supersession": bool(selected) and not selects(selected, CONTAINER_APPS),
    }
    return {"passed": all(checks.values()), "checks": checks, "selected_platform": selected}
