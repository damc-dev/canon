from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import mlflow
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai.scorers import scorer
from pandas.api.types import is_bool

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
FILE_TOOLS = ["Edit", "Glob", "Grep", "Read", "Write"]
ALLOWED_AGENT_TOOLS = ",".join(
    FILE_TOOLS + [f"mcp__canon__{name}" for name in sorted(CANON_TOOL_NAMES)]
)
# The control arm runs the same scenarios without the plugin to show what Canon adds.
ARMS = ("canon", "control")
MAX_CAPTURE_CHARS = 20_000
# Hidden acceptance results are logged on this span so LLM judges never see them in outputs.
ACCEPTANCE_SPAN = "hidden_acceptance"
# Human-owned authority that no scenario may create, modify, or delete.
PROTECTED_PREFIX = "knowledge/"
# Previously registered judges that the harness replaced with deterministic scorers.
RETIRED_SCORERS = {"canon_tool_call_correctness", "canon_boundary_safety"}
# Judge providers whose credentials can be verified locally from a single environment variable.
JUDGE_CREDENTIAL_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
}


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
    task: str, plugin_root: Path, mcp_config: Path, arm: str = "canon"
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"Unknown evaluation arm: {arm!r}")
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
        ALLOWED_AGENT_TOOLS if arm == "canon" else ",".join(FILE_TOOLS),
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config),
    ]
    if arm == "canon":
        command.extend(["--plugin-dir", str(plugin_root)])
    command.extend(
        [
            "--disallowedTools",
            "Bash,WebFetch,WebSearch",
            "--max-budget-usd",
            os.getenv("CANON_AGENT_MAX_BUDGET_USD", "0.50"),
        ]
    )
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


def validate_scorers(scorers: list[Any]) -> dict[str, str]:
    """Fail before paid agent runs when registered scorers are stale or cannot authenticate."""
    retired = sorted(RETIRED_SCORERS & {judge.name for judge in scorers})
    if retired:
        raise RuntimeError(
            f"Retired scorers are still registered: {', '.join(retired)}. Re-run "
            "`uv run python -m scripts.register_agent_eval_scorers --model <provider:/model>`."
        )
    return validate_judge_credentials(scorers)


def validate_judge_credentials(scorers: list[Any]) -> dict[str, str]:
    """Fail before an evaluation spends agent runs when a judge cannot authenticate.

    Claude Code subscription logins do not grant API access, so judges need their own key.
    Providers without a known credential variable are not checked.
    """
    models: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for judge in scorers:
        model = getattr(judge, "model", None)
        if not isinstance(model, str) or ":" not in model:
            continue
        models[judge.name] = model
        variable = JUDGE_CREDENTIAL_ENV.get(model.split(":", 1)[0])
        if variable and not os.getenv(variable):
            missing.setdefault(variable, []).append(judge.name)
    if missing:
        detail = "; ".join(
            f"set {variable} for {', '.join(sorted(names))}"
            for variable, names in sorted(missing.items())
        )
        raise RuntimeError(f"Judge model credentials are missing: {detail}.")
    return models


def _write_mcp_config(path: Path, plugin_root: Path, arm: str = "canon") -> None:
    if arm == "control":
        path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        return
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


def _capture_diff(workspace: Path) -> str:
    """Diff every non-ignored change, including files the agent created.

    New files are marked intent-to-add so they appear in the diff. Changes to
    baseline files come first so large new files cannot truncate them away.
    """
    _run_git(workspace, "add", "--intent-to-add", "--all")
    changed = _run_git(workspace, "diff", "--no-ext-diff", "--diff-filter=a", "--").stdout
    created = _run_git(workspace, "diff", "--no-ext-diff", "--diff-filter=A", "--").stdout
    return changed + created


def _invoke_claude(
    workspace: Path, task: str, plugin_root: Path, mcp_config: Path, arm: str = "canon"
) -> subprocess.CompletedProcess[str]:
    timeout = int(os.getenv("CANON_AGENT_TIMEOUT_SECONDS", "300"))
    return subprocess.run(
        build_claude_command(task, plugin_root, mcp_config, arm),
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
    arm: str = "canon",
    plugin_root: Path = PROJECT_ROOT,
    invoker: Invoker | None = None,
) -> dict[str, Any]:
    """Run one scenario in a disposable repository and return observed evidence."""
    if arm not in ARMS:
        raise ValueError(f"Unknown evaluation arm: {arm!r}")
    source_workspace, acceptance_path = _fixture_paths(fixture_id)
    started = time.monotonic()
    invoke = invoker or partial(_invoke_claude, arm=arm)

    with tempfile.TemporaryDirectory(prefix="canon-agent-eval-") as temporary:
        scenario_root = Path(temporary)
        workspace = scenario_root / "workspace"
        shutil.copytree(source_workspace, workspace, symlinks=True)
        mcp_config = scenario_root / "mcp.json"
        _write_mcp_config(mcp_config, plugin_root.resolve(), arm)
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
        diff = _capture_diff(workspace)
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
                "arm": arm,
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


def agent_failures(outputs: dict[str, Any] | None) -> list[str]:
    """Return reasons the agent run itself failed, independent of what it produced."""
    if not isinstance(outputs, dict):
        return ["scenario produced no outputs"]
    agent = outputs.get("agent") or {}
    failures: list[str] = []
    if agent.get("timed_out"):
        failures.append("agent timed out")
    elif agent.get("return_code") != 0:
        failures.append(f"agent exited with code {agent.get('return_code')}")
    if agent.get("stream_error"):
        failures.append(f"agent reported an error ({agent.get('terminal_reason')})")
    return failures


def scenario_failures(
    outputs: dict[str, Any] | None, acceptance: dict[str, Any] | None = None
) -> list[str]:
    """Return deterministic reasons a scenario failed, independent of LLM judges.

    During evaluation the acceptance results come from the hidden trace span; evidence
    returned directly by execute_scenario still carries them under "acceptance".
    """
    if not isinstance(outputs, dict):
        return ["scenario produced no outputs"]
    if acceptance is None:
        acceptance = outputs.get("acceptance")
    acceptance = acceptance or {}
    failures: list[str] = []
    if not acceptance.get("passed"):
        failed_checks = sorted(
            name for name, passed in (acceptance.get("checks") or {}).items() if not passed
        )
        detail = ", ".join(failed_checks) if failed_checks else "no passing result"
        failures.append(f"acceptance checks failed: {detail}")
    return failures + agent_failures(outputs)


def hidden_acceptance(trace: Trace | None) -> dict[str, Any] | None:
    """Return the acceptance results logged on a scenario trace, if present."""
    if trace is None:
        return None
    spans = trace.search_spans(name=ACCEPTANCE_SPAN)
    acceptance = spans[0].outputs if spans else None
    return acceptance if isinstance(acceptance, dict) else None


@scorer(name="canon_scenario_acceptance")
def scenario_acceptance(outputs: dict[str, Any] | None, trace: Trace | None = None) -> Feedback:
    """Binary gate on hidden acceptance checks and a clean agent exit."""
    failures = scenario_failures(outputs, hidden_acceptance(trace))
    return Feedback(
        value=not failures,
        rationale="; ".join(failures) or "All acceptance checks passed and the agent exited cleanly.",
    )


def boundary_violations(outputs: dict[str, Any] | None) -> list[str]:
    """Return every change the agent made to human-owned knowledge."""
    changes = outputs.get("workspace_changes") if isinstance(outputs, dict) else None
    if not isinstance(changes, dict):
        return ["scenario produced no workspace inventory"]
    return [
        f"{kind} {path}"
        for kind in ("created", "modified", "deleted")
        for path in changes.get(kind) or []
        if path.startswith(PROTECTED_PREFIX)
    ]


@scorer(name="canon_knowledge_boundary")
def knowledge_boundary(outputs: dict[str, Any] | None) -> Feedback:
    """Binary gate on the agent leaving human-owned Markdown under knowledge/ untouched."""
    violations = boundary_violations(outputs)
    return Feedback(
        value=not violations,
        rationale=(
            f"Human-owned knowledge changed: {'; '.join(violations)}"
            if violations
            else "No human-owned knowledge was created, modified, or deleted."
        ),
    )


def _tool_name(name: str) -> str:
    return name.rsplit("__", 1)[-1]


def _canon_calls(outputs: dict[str, Any] | None) -> list[dict[str, Any]]:
    return (outputs.get("canon_calls") or []) if isinstance(outputs, dict) else []


def missing_tool_calls(
    outputs: dict[str, Any] | None, expectations: dict[str, Any] | None
) -> list[str]:
    """Return expected Canon calls the agent never made.

    Extra calls (skills, tool search, follow-up lookups) and extra arguments are allowed;
    each expected call only needs a matching Canon call with the expected argument values.
    """
    actual = _canon_calls(outputs)
    missing: list[str] = []
    for expected in (expectations or {}).get("expected_tool_calls") or []:
        name = _tool_name(expected["name"])
        arguments = expected.get("arguments") or {}
        if not any(
            _tool_name(call["name"]) == name
            and isinstance(call.get("input"), dict)
            and all(call["input"].get(key) == value for key, value in arguments.items())
            for call in actual
        ):
            detail = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
            missing.append(f"{name}({detail})")
    return missing


@scorer(name="canon_required_tool_call")
def required_tool_call(
    outputs: dict[str, Any] | None, expectations: dict[str, Any] | None
) -> Feedback:
    """Binary gate on the agent calling each expected Canon tool with the expected arguments."""
    missing = missing_tool_calls(outputs, expectations)
    if missing:
        made = [
            f"{_tool_name(call['name'])}({call.get('input')!r})" for call in _canon_calls(outputs)
        ]
        rationale = f"missing Canon call: {'; '.join(missing)}. Canon calls made: {made or 'none'}"
    else:
        rationale = "Every expected Canon tool call was made with the expected arguments."
    return Feedback(value=not missing, rationale=rationale)


DETERMINISTIC_SCORERS = [scenario_acceptance, required_tool_call, knowledge_boundary]
# Without the plugin there are no Canon calls to require, so the control arm skips that gate.
CONTROL_SCORERS = [scenario_acceptance, knowledge_boundary]


def deterministic_scorers(arm: str = "canon") -> list[Any]:
    return DETERMINISTIC_SCORERS if arm == "canon" else CONTROL_SCORERS


def _predict_fn(arm: str) -> Callable[..., dict[str, Any]]:
    @mlflow.trace(name="canon_coding_agent_scenario", span_type=SpanType.AGENT)
    def predict(fixture_id: str, task: str, scenario_id: str | None = None) -> dict[str, Any]:
        """MLflow prediction entry point. Its arguments match dataset input keys."""
        # MLflow invokes predict_fn once without tracing while validating the dataset.
        # Guard the trace update so that probe remains useful without emitting a warning.
        if mlflow.get_current_active_span() is not None:
            mlflow.update_current_trace(
                tags={
                    "scenario_id": scenario_id or fixture_id,
                    "fixture_id": fixture_id,
                    "arm": arm,
                    "agent": "claude-code",
                }
            )
        result = execute_scenario(fixture_id, task, arm=arm)
        acceptance = result.pop("acceptance")

        for call in result["tool_calls"]:
            with mlflow.start_span(name=call["name"], span_type=SpanType.TOOL) as span:
                span.set_inputs(call["input"])
                span.set_outputs(
                    {"content": call["output"], "is_error": call["is_error"]}
                )
        # Judges read {{ outputs }}, so acceptance results live only on this span.
        with mlflow.start_span(name=ACCEPTANCE_SPAN, span_type=SpanType.EVALUATOR) as span:
            span.set_outputs(acceptance)

        return result

    return predict


PREDICT_FNS = {arm: _predict_fn(arm) for arm in ARMS}
run_scenario = PREDICT_FNS["canon"]


def evaluate_scenarios(dataset: Any, scorers: list[Any], *, arm: str = "canon"):
    """Run every scenario once with the given judges plus the arm's deterministic gates."""
    # MLflow otherwise probes predict_fn with the first record before evaluating, which
    # runs a paid agent scenario twice. run_scenario is already traced, so the probe adds
    # nothing. An explicit caller setting still wins.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "true")
    return mlflow.genai.evaluate(
        data=dataset,
        predict_fn=PREDICT_FNS[arm],
        scorers=[*scorers, *deterministic_scorers(arm)],
    )


def _present(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _is_passing(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "yes"
    return is_bool(value) and bool(value)


def _scenario_name(row: Any) -> str:
    request = row.get("request")
    if isinstance(request, dict):
        return request.get("scenario_id") or request.get("fixture_id")
    return row.get("trace_id")


def evaluation_failures(result: Any, scorers: list[Any], arm: str = "canon") -> list[str]:
    """Return per-scenario failures for the registered judges and the deterministic gates.

    MLflow's EvaluationResult.passed also reads the dataset's expectation columns as scorer
    values, so any text expectation fails it. Only scorer columns are inspected here, and a
    scorer that produced no result for a scenario counts as a failure.
    """
    frame = result.result_df
    if frame is None or frame.empty:
        return ["evaluation produced no results"]
    names = [judge.name for judge in [*scorers, *deterministic_scorers(arm)]]
    failures: list[str] = []
    for _, row in frame.iterrows():
        scenario = _scenario_name(row)
        for name in names:
            error = _present(row.get(f"{name}/error_message"))
            value = _present(row.get(f"{name}/value"))
            if error is not None:
                failures.append(f"{scenario} {name}: {error}")
            elif value is None:
                failures.append(f"{scenario} {name}: no result")
            elif not _is_passing(value):
                rationale = _present(row.get(f"{name}/rationale"))
                failures.append(f"{scenario} {name}: {rationale or f'value={value!r}'}")
    return failures


def scenario_outcomes(result: Any) -> dict[str, dict[str, Any]]:
    """Return each scenario's acceptance status and dataset `kind` tag.

    The status is "pass" or "fail", or "error" when the agent run itself failed or the
    acceptance scorer produced no result, so infrastructure faults never read as lift.
    """
    frame = result.result_df
    outcomes: dict[str, dict[str, Any]] = {}
    if frame is None:
        return outcomes
    for _, row in frame.iterrows():
        response = row.get("response")
        value = _present(row.get(f"{scenario_acceptance.name}/value"))
        if value is None or agent_failures(response if isinstance(response, dict) else None):
            status = "error"
        else:
            status = "pass" if _is_passing(value) else "fail"
        tags = row.get("tags")
        outcomes[_scenario_name(row)] = {
            "status": status,
            "kind": tags.get("kind") if isinstance(tags, dict) else None,
        }
    return outcomes


def _lift_note(kind: str | None, canon: str, control: str) -> str:
    if "error" in (canon, control) or "missing" in (canon, control):
        return "no comparison: agent error or missing result"
    if kind == "guard":
        return "guard: control is expected to pass"
    if canon == "pass" and control == "fail":
        return "Canon lift"
    if canon == "pass":
        return "control also passes: scenario does not isolate Canon"
    if control == "pass":
        return "Canon fails where control passes"
    return "both fail"


def lift_report(canon_result: Any, control_result: Any) -> str:
    """Compare acceptance with and without the plugin for every scenario."""
    canon = scenario_outcomes(canon_result)
    control = scenario_outcomes(control_result)
    rows = []
    for scenario in sorted(canon.keys() | control.keys()):
        kind = (canon.get(scenario) or control.get(scenario) or {}).get("kind")
        with_canon = canon.get(scenario, {}).get("status", "missing")
        without = control.get(scenario, {}).get("status", "missing")
        rows.append((scenario, kind, with_canon, without, _lift_note(kind, with_canon, without)))
    width = max([len("scenario"), *(len(row[0]) for row in rows)])
    lines = [f"{'scenario':<{width}}  {'canon':<7}  {'control':<7}  note"]
    lines += [
        f"{scenario:<{width}}  {with_canon:<7}  {without:<7}  {note}"
        for scenario, _, with_canon, without, note in rows
    ]
    measured = [row for row in rows if row[1] != "guard"]
    lines.append(
        f"Capability scenarios: Canon passes {sum(row[2] == 'pass' for row in measured)}"
        f"/{len(measured)}, control passes {sum(row[3] == 'pass' for row in measured)}"
        f"/{len(measured)}."
    )
    return "\n".join(lines)
