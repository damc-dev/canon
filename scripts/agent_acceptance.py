"""Shared helpers for hidden fixture acceptance checks.

Agents format answers freely, so these helpers accept the common Markdown shapes of a
labelled answer and treat an option as selected only when it is not negated.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from server.canon_core import parse_frontmatter


CONTAINER_APPS = r"\b(azure container apps?|container apps?|aca)\b"
KUBERNETES = r"\b(aks|azure kubernetes service|kubernetes|k8s)\b"

_MARKUP = re.compile(r"[*_`]")
_LEADING_MARKUP = re.compile(r"^[\s>#+-]+")
# Clause boundaries: punctuation, spaced dashes, and unspaced em or en dashes.
_CLAUSE_BOUNDARY = re.compile(r"[;,.:()\[\]]|\s[-–—]\s|[–—]")
_NEGATED_BEFORE = re.compile(
    r"\b(not|no|never|instead of|rather than|over|versus|vs|excluding|except|without)\b"
)
_NEGATED_AFTER = re.compile(
    r"\b(excluded|rejected|ruled out|avoided|not (used|selected|chosen|recommended))\b"
)
_UNKNOWN = re.compile(
    r"\b("
    r"unknown|undecided|undetermined|tbd|pending|"
    r"to be (decided|determined|defined|agreed)|"
    r"not (yet )?(been )?(decided|established|defined|documented|specified|agreed|recorded|set)|"
    r"no (established|decided|documented|agreed|defined|recorded|applicable|existing)\b|"
    r"no (\w+ ){0,3}(strategy|decision|guidance)\b.{0,40}\b"
    r"(established|decided|defined|documented|recorded|agreed|exists|found)|"
    r"none (established|decided|defined|documented|recorded|found)|"
    r"open (question|decision)|needs? (a )?decision"
    r")\b"
)


def _plain(line: str) -> str:
    return _LEADING_MARKUP.sub("", _MARKUP.sub("", line)).strip()


def _clean(value: str) -> str:
    return value.strip().rstrip(".").strip().lower()


def field_value(text: str, name: str) -> str:
    """Return the lower-cased answer for a labelled field, or "" when it is absent.

    Accepts `Name: value` lines with list, heading, quote, and emphasis markup,
    `| Name | value |` table rows, and a `Name` heading or bare `Name:` line whose
    value is on the next non-empty line. The first labelled answer wins.
    """
    label = re.escape(name)
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        if raw_line.lstrip().startswith("|"):
            cells = [_plain(cell) for cell in raw_line.strip().strip("|").split("|")]
            if len(cells) >= 2 and re.fullmatch(rf"(?i){label}\s*:?", cells[0]) and cells[1]:
                return _clean(cells[1])
            continue
        line = _plain(raw_line)
        inline = re.match(rf"(?i){label}\s*:\s*(.*)$", line)
        if inline and inline.group(1).strip():
            return _clean(inline.group(1))
        if inline or re.fullmatch(rf"(?i){label}", line):
            following = next((_plain(item) for item in lines[index + 1 :] if _plain(item)), "")
            return _clean(following)
    return ""


def selects(value: str, option: str) -> bool:
    """Return whether `value` names the `option` pattern outside a negated clause.

    "Azure Container Apps, not AKS" selects Azure Container Apps but not AKS, and
    "AKS instead of Azure Container Apps" selects AKS only.
    """
    for match in re.finditer(option, value):
        start = max(
            (boundary.end() for boundary in _CLAUSE_BOUNDARY.finditer(value, 0, match.start())),
            default=0,
        )
        following = _CLAUSE_BOUNDARY.search(value, match.end())
        end = following.start() if following else len(value)
        if not _NEGATED_BEFORE.search(value[start : match.start()]) and not _NEGATED_AFTER.search(
            value[match.end() : end]
        ):
            return True
    return False


def reports_unknown(value: str) -> bool:
    """Return whether an answer says no decision has been established."""
    return bool(_UNKNOWN.search(value))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def proposals(workspace: Path) -> list[tuple[Path, dict, str]]:
    """Return each generated proposal with its parsed frontmatter and body."""
    root = workspace / ".canon" / "proposals"
    found = []
    for path in sorted(root.glob("*.md")) if root.is_dir() else []:
        metadata, body = parse_frontmatter(read_text(path))
        found.append((path, metadata, body))
    return found


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout


def unchanged(workspace: Path, *paths: str) -> bool:
    """Return whether the given paths still match the fixture baseline commit."""
    return not _git(workspace, "status", "--porcelain", "--untracked-files=all", "--", *paths)


def committable_files(workspace: Path) -> list[Path]:
    """Return tracked and untracked files that the workspace's own .gitignore allows.

    A developer's global excludes file is ignored so results do not vary by machine.
    """
    listed = _git(
        workspace,
        "-c",
        f"core.excludesFile={os.devnull}",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    return [workspace / name for name in listed.split("\0") if name]


def last_commit(workspace: Path, path: str) -> str:
    return _git(workspace, "log", "-1", "--format=%H", "--", path).strip()
