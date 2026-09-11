from __future__ import annotations

import re
from pathlib import Path


def evaluate(workspace: Path) -> dict:
    plan = workspace / "cloud-plan.md"
    text = plan.read_text(encoding="utf-8") if plan.is_file() else ""
    provider_lines = re.findall(r"(?im)^provider:\s*(.+)$", text)
    selected = provider_lines[0].strip().lower() if provider_lines else ""
    checks = {
        "cloud_plan_created": plan.is_file(),
        "inherits_acme_decision": selected in {"azure", "microsoft azure"},
        "excludes_sibling_decision": selected not in {"aws", "amazon web services", "aws lambda"},
    }
    return {"passed": all(checks.values()), "checks": checks, "selected_provider": selected}
