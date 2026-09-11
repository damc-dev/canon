from __future__ import annotations

import re
from pathlib import Path


def evaluate(workspace: Path) -> dict:
    plan = workspace / "deployment-plan.md"
    text = plan.read_text(encoding="utf-8") if plan.is_file() else ""
    platform_lines = re.findall(r"(?im)^platform:\s*(.+)$", text)
    selected = platform_lines[0].strip().lower() if platform_lines else ""
    checks = {
        "deployment_plan_created": plan.is_file(),
        "uses_active_decision": selected == "azure container apps",
        "rejects_conflicting_reference": selected not in {"aks", "kubernetes", "kubernetes on aks"},
    }
    return {"passed": all(checks.values()), "checks": checks, "selected_platform": selected}
