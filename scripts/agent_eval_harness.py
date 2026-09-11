from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import mlflow
from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import scorer

from scripts.agent_eval_config import PROJECT_ROOT


FIXTURES_ROOT = PROJECT_ROOT / "tests" / "agent_fixtures"
FIXTURE_ID = re.compile(r"s\d{3}")
CANON_TOOL_NAMES = {
    "get_context",
    "search_knowledge",
    "explain_knowledge",
    "propose_knowledge",
    "rebuild_knowledge_index",
}
ALLOWED_AGENT_TOOLS = ",".join(
    ["Edit", "Glob", "Grep", "Read", "Write"]
    + [f"mcp__canon__{name}" for name in sorted(CANON_TOOL_NAMES)]
)
MAX_CAPTURE_CHARS = 20_000


def _truncate(value: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... truncated {len(value) - limit} characters"


def _fixture_paths(fixture_id: str) -> tuple[Path, Path]:
    if not FIXTURE_ID.fullmatch(fixture_id):
        raise ValueError(f"Invalid fixture ID: {fixture_id!r}")
    fixture = FIXTURES_ROOT / fixture_id
    workspace = fixture / "workspace"
    acceptance = fixture / "acceptance.py"
    if not workspace.is_dir() or not acceptance.is_file():
        raise FileNotFoundError(f"Incomplete agent fixture: {fixture_id}")
    return workspace, acceptance


def inventory_files(root: Path) -> dict[str, dict[str, str]]:
    """Return a stable inventory including untracked files and symlinks."""
    inventory: dict[str, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        if relative == ".canon/index.db":
            continue
        if path.is_symlink():
            inventory[relative] = {"kind": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            inventory[relative] = {
                "kind": "file",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return inventory


def compare_inventories(
    before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]
) -> dict[str, list[str]]:
    before_paths = set(before)
    after_paths = set(after)
    return {
        "created": sorted(after_paths - before_paths),
        "modified": sorted(
            path for path in before_paths & after_paths if before[path] != after[path]
        ),
        "deleted": sorted(before_paths - after_paths),
    }


def parse_claude_stream(stdout: str) -> dict[str, Any]:
    """Parse Claude Code's stream-json output into stable evaluation evidence."""
    events: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_errors.append(_truncate(line, 500))
            continue
        if isinstance(event, dict):
            events.append(event)

    tool_calls: list[dict[str, Any]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    assistant_text: list[str] = []
    result_event: dict[str, Any] = {}
    system_event: dict[str, Any] = {}

    for event in events:
        event_type = event.get("type")
        if event_type == "system" and event.get("subtype") == "init":
            system_event = event
        if event_type == "result":
            result_event = event

        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if event_type == "assistant" and block_type == "text":
                assistant_text.append(str(block.get("text", "")))
            elif event_type == "assistant" and block_type == "tool_use":
                call = {
                    "id": str(block.get("id", "")),
                    "name": str(block.get("name", "unknown_tool")),
                    "input": block.get("input", {}),
                    "output": None,
                    "is_error": False,
                }
                tool_calls.append(call)
                if call["id"]:
                    calls_by_id[call["id"]] = call
            elif block_type == "tool_result":
                call = calls_by_id.get(str(block.get("tool_use_id", "")))
                if call is not None:
                    call["output"] = block.get("content")
                    call["is_error"] = bool(block.get("is_error", False))

    final_response = result_event.get("result")
    if not isinstance(final_response, str):
        final_response = "\n".join(text for text in assistant_text if text).strip()

    canon_calls = [
        call
        for call in tool_calls
        if call["name"].rsplit("__", 1)[-1] in CANON_TOOL_NAMES
    ]
    return {
        "final_response": final_response,
        "tool_calls": tool_calls,
        "canon_calls": canon_calls,
        "event_count": len(events),
        "parse_errors": parse_errors,
        "session_id": result_event.get("session_id") or system_event.get("session_id"),
        "model": system_event.get("model"),
        "mcp_servers": system_event.get("mcp_servers", []),
        "available_tools": system_event.get("tools", []),
        "skills": system_event.get("skills", []),
        "duration_ms": result_event.get("duration_ms"),
        "total_cost_usd": result_event.get("total_cost_usd"),
        "usage": result_event.get("usage", {}),
        "is_error": bool(result_event.get("is_error", False)),
        "terminal_reason": result_event.get("terminal_reason"),
    }


def build_claude_command(
    task: str, plugin_root: Path, mcp_config: Path
) -> list[str]:
    executable = os.getenv("CANON_AGENT_EXECUTABLE", "claude")
    command = [
        executable,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        ALLOWED_AGENT_TOOLS,
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config),
        "--plugin-dir",
        str(plugin_root),
        "--disallowedTools",
        "Bash,WebFetch,WebSearch",
        "--max-budget-usd",
        os.getenv("CANON_AGENT_MAX_BUDGET_USD", "0.50"),
    ]
    model = os.getenv("CANON_AGENT_MODEL")
    if model:
        command.extend(["--model", model])
    command.append(task)
    return command


def validate_live_agent() -> dict[str, Any]:
    """Fail before an evaluation spends work on fixtures when Claude is unavailable."""
    executable = os.getenv("CANON_AGENT_EXECUTABLE", "claude")
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"Coding-agent executable not found: {executable}")

    version = subprocess.run(
        [resolved, "--version"], capture_output=True, text=True, check=False, timeout=10
    )
    if os.getenv("CANON_SKIP_AGENT_AUTH_CHECK") == "1":
        return {"executable": resolved, "version": version.stdout.strip(), "auth": "skipped"}
    if any(
        os.getenv(name)
        for name in (
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        )
    ):
        return {"executable": resolved, "version": version.stdout.strip(), "auth": "environment"}

    status = subprocess.run(
        [resolved, "auth", "status"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    try:
        auth = json.loads(status.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Cannot parse `claude auth status`; run `claude auth login`.") from exc
    if status.returncode != 0 or not auth.get("loggedIn"):
        raise RuntimeError("Claude Code is not authenticated; run `claude auth login`.")
    return {
        "executable": resolved,
        "version": version.stdout.strip(),
        "auth": auth.get("authMethod", "authenticated"),
    }


def _write_mcp_config(path: Path, plugin_root: Path) -> None:
    config = {
        "mcpServers": {
            "canon": {
                "command": "uv",
                "args": [
                    "run",
                    "--project",
                    str(plugin_root),
                    "--frozen",
                    "--no-dev",
                    "canon-mcp",
                ],
            }
        }
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _run_git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )


def _initialize_git(workspace: Path) -> None:
    _run_git(workspace, "init", "--quiet")
    _run_git(workspace, "add", "--all")
    # Isolate the baseline commit from the developer's signing and hook configuration.
    _run_git(
        workspace,
        "-c",
        "user.name=Canon Agent Eval",
        "-c",
        "user.email=canon-eval@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "commit",
        "--no-verify",
        "--quiet",
        "-m",
        "fixture baseline",
    )


def _invoke_claude(
    workspace: Path, task: str, plugin_root: Path, mcp_config: Path
) -> subprocess.CompletedProcess[str]:
    timeout = int(os.getenv("CANON_AGENT_TIMEOUT_SECONDS", "300"))
    return subprocess.run(
        build_claude_command(task, plugin_root, mcp_config),
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _run_acceptance(acceptance_path: Path, workspace: Path) -> dict[str, Any]:
    module_name = f"canon_agent_acceptance_{acceptance_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, acceptance_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load acceptance checks: {acceptance_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate(workspace)
    if not isinstance(result, dict) or "passed" not in result:
        raise TypeError("Acceptance checks must return a dict containing 'passed'")
    return result


Invoker = Callable[[Path, str, Path, Path], subprocess.CompletedProcess[str]]


def execute_scenario(
    fixture_id: str,
    task: str,
    *,
    plugin_root: Path = PROJECT_ROOT,
    invoker: Invoker | None = None,
) -> dict[str, Any]:
    """Run one scenario in a disposable repository and return observed evidence."""
    source_workspace, acceptance_path = _fixture_paths(fixture_id)
    started = time.monotonic()
    invoke = invoker or _invoke_claude

    with tempfile.TemporaryDirectory(prefix="canon-agent-eval-") as temporary:
        scenario_root = Path(temporary)
        workspace = scenario_root / "workspace"
        shutil.copytree(source_workspace, workspace, symlinks=True)
        mcp_config = scenario_root / "mcp.json"
        _write_mcp_config(mcp_config, plugin_root.resolve())
        _initialize_git(workspace)
        before = inventory_files(workspace)

        try:
            completed = invoke(workspace, task, plugin_root.resolve(), mcp_config)
            stdout = completed.stdout
            stderr = completed.stderr
            return_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return_code = 124
            timed_out = True

        parsed = parse_claude_stream(stdout)
        after = inventory_files(workspace)
        status = _run_git(workspace, "status", "--short").stdout
        diff = _run_git(workspace, "diff", "--no-ext-diff", "--").stdout
        acceptance = _run_acceptance(acceptance_path, workspace)

        return {
            "response": parsed["final_response"],
            "tool_calls": parsed["tool_calls"],
            "canon_calls": parsed["canon_calls"],
            "workspace_changes": compare_inventories(before, after),
            "git_status": status.splitlines(),
            "git_diff": _truncate(diff),
            "acceptance": acceptance,
            "agent": {
                "return_code": return_code,
                "timed_out": timed_out,
                "stderr": _truncate(stderr),
                "session_id": parsed["session_id"],
                "model": parsed["model"],
                "mcp_servers": parsed["mcp_servers"],
                "available_tools": parsed["available_tools"],
                "skills": parsed["skills"],
                "duration_ms": parsed["duration_ms"],
                "total_cost_usd": parsed["total_cost_usd"],
                "usage": parsed["usage"],
                "terminal_reason": parsed["terminal_reason"],
                "stream_error": parsed["is_error"],
                "event_count": parsed["event_count"],
                "parse_errors": parsed["parse_errors"],
            },
            "harness_duration_seconds": round(time.monotonic() - started, 3),
        }


def scenario_failures(outputs: dict[str, Any] | None) -> list[str]:
    """Return deterministic reasons a scenario failed, independent of LLM judges."""
    if not isinstance(outputs, dict):
        return ["scenario produced no outputs"]
    failures: list[str] = []
    acceptance = outputs.get("acceptance") or {}
    if not acceptance.get("passed"):
        failed_checks = sorted(
            name for name, passed in (acceptance.get("checks") or {}).items() if not passed
        )
        detail = ", ".join(failed_checks) if failed_checks else "no passing result"
        failures.append(f"acceptance checks failed: {detail}")
    agent = outputs.get("agent") or {}
    if agent.get("timed_out"):
        failures.append("agent timed out")
    elif agent.get("return_code") != 0:
        failures.append(f"agent exited with code {agent.get('return_code')}")
    if agent.get("stream_error"):
        failures.append(f"agent reported an error ({agent.get('terminal_reason')})")
    return failures


@scorer(name="canon_scenario_acceptance")
def scenario_acceptance(outputs: dict[str, Any] | None) -> Feedback:
    """Binary gate on hidden acceptance checks and a clean agent exit."""
    failures = scenario_failures(outputs)
    return Feedback(
        value=not failures,
        rationale="; ".join(failures) or "All acceptance checks passed and the agent exited cleanly.",
    )


@mlflow.trace(name="canon_coding_agent_scenario", span_type=SpanType.AGENT)
def run_scenario(fixture_id: str, task: str) -> dict[str, Any]:
    """MLflow prediction entry point. Its arguments match dataset input keys."""
    # MLflow invokes predict_fn once without tracing while validating the dataset.
    # Guard the trace update so that probe remains useful without emitting a warning.
    if mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(tags={"scenario_id": fixture_id, "agent": "claude-code"})
    result = execute_scenario(fixture_id, task)

    for call in result["tool_calls"]:
        with mlflow.start_span(name=call["name"], span_type=SpanType.TOOL) as span:
            span.set_inputs(call["input"])
            span.set_outputs(
                {"content": call["output"], "is_error": call["is_error"]}
            )

    return result
