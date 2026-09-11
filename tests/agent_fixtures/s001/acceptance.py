from __future__ import annotations

import re
from pathlib import Path


def _field(text: str, name: str) -> str:
    """Return the first `name:` value, ignoring Markdown list, heading, and emphasis markup."""
    for raw_line in text.splitlines():
        line = re.sub(r"^[\s>#+-]+", "", re.sub(r"[*_`]", "", raw_line)).strip()
        match = re.match(rf"(?i){name}\s*:\s*(.+)$", line)
        if match:
            value = re.sub(r"\s*\([^)]*\)$", "", match.group(1).strip().rstrip("."))
            return value.strip().lower()
    return ""


def evaluate(workspace: Path) -> dict:
    plan = workspace / "deployment-plan.md"
    text = plan.read_text(encoding="utf-8") if plan.is_file() else ""
    selected = _field(text, "platform")
    checks = {
        "deployment_plan_created": plan.is_file(),
        "uses_active_decision": selected == "azure container apps",
        "rejects_conflicting_reference": bool(selected)
        and not re.search(r"\b(aks|kubernetes)\b", selected),
    }
    return {"passed": all(checks.values()), "checks": checks, "selected_platform": selected}
