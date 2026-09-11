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
    plan = workspace / "cloud-plan.md"
    text = plan.read_text(encoding="utf-8") if plan.is_file() else ""
    selected = _field(text, "provider")
    checks = {
        "cloud_plan_created": plan.is_file(),
        "inherits_acme_decision": selected in {"azure", "microsoft azure"},
        "excludes_sibling_decision": bool(selected)
        and not re.search(r"\b(aws|amazon|lambda)\b", selected),
    }
    return {"passed": all(checks.values()), "checks": checks, "selected_provider": selected}
