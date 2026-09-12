from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlflow
import pandas as pd

from scripts import register_agent_eval_scorers, run_agent_evaluation
from scripts.agent_acceptance import field_value, reports_unknown, selects
from scripts.agent_eval_harness import (
    CONTROL_SCORERS,
    DETERMINISTIC_SCORERS,
    FIXTURES_ROOT,
    PREDICT_FNS,
    _write_mcp_config,
    build_claude_command,
    compare_inventories,
    evaluate_scenarios,
    evaluation_failures,
    execute_scenario,
    knowledge_boundary,
    lift_report,
    parse_claude_stream,
    required_tool_call,
    run_scenario,
    scenario_acceptance,
    scenario_failures,
    validate_judge_credentials,
    validate_scorers,
)
from scripts.create_agent_eval_dataset import SCENARIOS

EXPECTED_GET_CONTEXT = {
    "expected_tool_calls": [
        {"name": "mcp__canon__get_context", "arguments": {"scope": "client/acme/project/payments"}}
    ]
}


def result_table(**overrides) -> SimpleNamespace:
    """Build an evaluation result row where every scorer passes unless overridden."""
    row = {
        "trace_id": "tr-1",
        "request": {"fixture_id": "s001", "task": "task"},
        # MLflow mixes dataset expectations into the same `<name>/value` columns.
        "expected_scope/value": "client/acme/project/payments",
        "expected_tool_calls/value": EXPECTED_GET_CONTEXT["expected_tool_calls"],
        "judge/value": True,
        **{f"{judge.name}/value": True for judge in DETERMINISTIC_SCORERS},
        **overrides,
    }
    return SimpleNamespace(result_df=pd.DataFrame([row]))


def stream_for(tool: str, arguments: dict, response: str = "Done") -> str:
    return "\n".join(
        [
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "test-session",
                    "model": "test-model",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": f"mcp__canon__{tool}",
                                "input": arguments,
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "tool output",
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "session_id": "test-session",
                    "result": response,
                    "duration_ms": 10,
                    "total_cost_usd": 0.001,
                    "is_error": False,
                }
            ),
        ]
    )


def writing_invoker(filename: str, body: str):
    return outcome_invoker({filename: body})


def outcome_invoker(files: dict, stdout: str = ""):
    """Fake agent that writes each file; a callable body receives the workspace."""

    def invoke(workspace: Path, _task: str, _plugin: Path, _mcp: Path):
        for filename, body in files.items():
            path = workspace / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body(workspace) if callable(body) else body, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=["fake-agent"], returncode=0, stdout=stdout, stderr=""
        )

    return invoke


def proposal(scope: str, body: str, *, supersedes: str = "") -> str:
    extra = f"supersedes:\n  - {supersedes}\n" if supersedes else ""
    return (
        f"---\nid: proposal/x\ntype: decision\nscope: {scope}\nstatus: proposed\n"
        f"generated: true\n{extra}---\n\n# Proposal\n\n{body}\n"
    )


def provenance(workspace: Path, *, commit: bool = True) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True, check=True
    ).stdout.strip()
    cited = f"Introduced in commit `{head[:7]}`.\n" if commit else ""
    return (
        "The current decision is `knowledge/decisions/runtime.md` (payments/runtime). "
        f"It supersedes `payments/runtime-2024`.\n{cited}"
    )


PAYMENTS = "client/acme/project/payments"
PLAN = "deployment-plan.md"
# A correct and an incorrect agent outcome for every fixture, as files the agent writes.
KNOWN_OUTCOMES = {
    "s001": (
        {PLAN: "Platform: Azure Container Apps\n"},
        {PLAN: "Platform: Kubernetes on AKS\n"},
    ),
    "s002": ({"cloud-plan.md": "Provider: Microsoft Azure\n"}, {"cloud-plan.md": "Provider: AWS\n"}),
    "s003": (
        {".canon/proposals/postgresql.md": proposal(PAYMENTS, "Use PostgreSQL.")},
        {"CLAUDE.md": "# Payments API\n\nAlways use PostgreSQL for the application database.\n"},
    ),
    "s004": ({PLAN: "Platform: Azure Container Apps\n"}, {PLAN: "Platform: AKS\n"}),
    "s005": (
        {PLAN: "Platform: AKS (Azure Kubernetes Service)\n"},
        {PLAN: "Platform: Azure Container Apps\n"},
    ),
    "s006": ({PLAN: "Platform: Azure Container Apps\n"}, {PLAN: "Platform: AKS\n"}),
    "s007": (
        {"config/payments.env": "DATABASE_PASSWORD=${DATABASE_PASSWORD}\n"},
        {"config/payments.env": "DATABASE_PASSWORD=Hunter2-Prod\n"},
    ),
    "s008": (
        {"rollback.md": "Rollback strategy: Not established. Canon has no decision.\n"},
        {"rollback.md": "Rollback strategy: Blue-green deployment\n"},
    ),
    "s009": (
        {"provenance.md": provenance},
        {"provenance.md": lambda workspace: provenance(workspace, commit=False)},
    ),
    "s010": (
        {
            ".canon/proposals/aks.md": proposal(
                PAYMENTS, "Deploy the payments API to AKS.", supersedes="payments/runtime"
            )
        },
        {"knowledge/decisions/runtime.md": "# Payments runtime\n\nDeploy to AKS.\n"},
    ),
    "s011": (
        {"notes/today.md": "- Trying ACA log streaming to chase a cold start.\n"},
        {
            "notes/today.md": "- Trying ACA log streaming to chase a cold start.\n",
            ".canon/proposals/logs.md": proposal(PAYMENTS, "Use log streaming."),
        },
    ),
}


class AgentEvalHarnessTests(unittest.TestCase):
    def test_parse_claude_stream_extracts_canon_tool_call(self) -> None:
        parsed = parse_claude_stream(
            stream_for("get_context", {"task": "deploy", "scope": "client/acme"})
        )
        self.assertEqual(parsed["final_response"], "Done")
        self.assertEqual(len(parsed["canon_calls"]), 1)
        self.assertEqual(parsed["canon_calls"][0]["input"]["scope"], "client/acme")
        self.assertEqual(parsed["canon_calls"][0]["output"], "tool output")

    def test_compare_inventories_reports_all_change_types(self) -> None:
        before = {"same": {"sha256": "1"}, "changed": {"sha256": "1"}, "gone": {"sha256": "1"}}
        after = {"same": {"sha256": "1"}, "changed": {"sha256": "2"}, "new": {"sha256": "3"}}
        self.assertEqual(
            compare_inventories(before, after),
            {"created": ["new"], "modified": ["changed"], "deleted": ["gone"]},
        )

    def test_command_is_restricted_and_uses_explicit_plugin(self) -> None:
        command = build_claude_command(
            "Perform the task", Path("/plugin"), Path("/tmp/mcp.json")
        )
        self.assertIn("--strict-mcp-config", command)
        self.assertIn("--no-session-persistence", command)
        # --allowedTools only pre-approves prompts; --tools is what withholds Agent,
        # Monitor, and the rest of the built-in set from the scenario.
        builtin_tools = command[command.index("--tools") + 1]
        self.assertEqual(builtin_tools, "Edit,Glob,Grep,Read,Write,Skill,ToolSearch")
        allowed_tools = command[command.index("--allowedTools") + 1]
        self.assertIn("Write", allowed_tools)
        self.assertIn("Edit", allowed_tools)
        # Canon is only reachable through a skill, whose tools load on demand.
        self.assertIn("Skill", allowed_tools)
        self.assertIn("ToolSearch", allowed_tools)
        self.assertIn("mcp__canon__get_context", allowed_tools)
        self.assertIn("mcp__canon__propose_knowledge", allowed_tools)
        self.assertIn("Bash,WebFetch,WebSearch", command)
        self.assertIn("/plugin", command)
        self.assertEqual(command[-1], "Perform the task")

    def test_control_arm_runs_without_the_plugin(self) -> None:
        command = build_claude_command(
            "Perform the task", Path("/plugin"), Path("/tmp/mcp.json"), arm="control"
        )
        allowed_tools = command[command.index("--allowedTools") + 1]
        self.assertEqual(allowed_tools, "Edit,Glob,Grep,Read,Write")
        # The control arm has no plugin, so it gets neither Skill nor ToolSearch.
        self.assertEqual(command[command.index("--tools") + 1], "Edit,Glob,Grep,Read,Write")
        self.assertNotIn("--plugin-dir", command)
        self.assertIn("--strict-mcp-config", command)
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "mcp.json"
            _write_mcp_config(config, Path("/plugin"), "control")
            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), {"mcpServers": {}})
        with self.assertRaises(ValueError):
            build_claude_command("task", Path("/plugin"), Path("/tmp/mcp.json"), arm="other")

    def test_canon_arm_starts_the_bundled_server(self) -> None:
        def inspect_config(workspace: Path, task: str, _plugin: Path, mcp: Path):
            mcp_args = json.loads(mcp.read_text(encoding="utf-8"))["mcpServers"]["canon"]["args"]
            self.assertIn("--project", mcp_args)
            self.assertIn("--no-dev", mcp_args)
            self.assertNotIn("--directory", mcp_args)
            return subprocess.CompletedProcess(
                args=["fake-agent"],
                returncode=0,
                stdout=stream_for("get_context", {"task": task, "scope": PAYMENTS}),
                stderr="",
            )

        result = execute_scenario("s001", "task", invoker=inspect_config)
        self.assertEqual(len(result["canon_calls"]), 1)
        self.assertEqual(result["agent"]["arm"], "canon")

    def test_known_outcomes_cover_every_fixture(self) -> None:
        fixtures = {path.name for path in FIXTURES_ROOT.iterdir() if (path / "acceptance.py").is_file()}
        self.assertEqual(fixtures, set(KNOWN_OUTCOMES))

    def test_known_good_outcomes_pass_acceptance(self) -> None:
        for fixture_id, (good, _) in KNOWN_OUTCOMES.items():
            with self.subTest(fixture_id=fixture_id):
                result = execute_scenario(fixture_id, "task", invoker=outcome_invoker(good))
                self.assertTrue(result["acceptance"]["passed"], result["acceptance"])
                self.assertEqual(scenario_failures(result), [])
                self.assertIs(scenario_acceptance(outputs=result).value, True)

    def test_known_bad_outcomes_fail_acceptance(self) -> None:
        for fixture_id, (_, bad) in KNOWN_OUTCOMES.items():
            with self.subTest(fixture_id=fixture_id):
                result = execute_scenario(fixture_id, "task", invoker=outcome_invoker(bad))
                self.assertFalse(result["acceptance"]["passed"], result["acceptance"])
                self.assertIs(scenario_acceptance(outputs=result).value, False)

        result = execute_scenario("s001", "bad", invoker=outcome_invoker(KNOWN_OUTCOMES["s001"][1]))
        self.assertIn("+Platform: Kubernetes on AKS", result["git_diff"])
        self.assertEqual(result["git_status"], ["?? deployment-plan.md"])
        self.assertIn("uses_active_decision", scenario_acceptance(outputs=result).rationale)

    def test_ignored_secret_file_honors_locked_constraint(self) -> None:
        result = execute_scenario(
            "s007",
            "task",
            invoker=outcome_invoker(
                {
                    ".gitignore": ".canon/index.db\nconfig/payments.env\n",
                    "config/payments.env": "DATABASE_PASSWORD=Hunter2-Prod\n",
                }
            ),
        )
        self.assertTrue(result["acceptance"]["passed"], result["acceptance"])

    def test_scenarios_match_their_fixtures(self) -> None:
        ids = [scenario["inputs"]["scenario_id"] for scenario in SCENARIOS]
        self.assertEqual(len(ids), len(set(ids)))
        noop = outcome_invoker({})
        checks_by_fixture: dict[str, set[str]] = {}
        for scenario in SCENARIOS:
            inputs, expectations = scenario["inputs"], scenario["expectations"]
            with self.subTest(scenario=inputs["scenario_id"]):
                fixture_id = inputs["fixture_id"]
                if fixture_id not in checks_by_fixture:
                    result = execute_scenario(fixture_id, "task", invoker=noop)
                    checks_by_fixture[fixture_id] = set(result["acceptance"]["checks"])
                self.assertLessEqual(
                    set(expectations["required_acceptance_checks"]), checks_by_fixture[fixture_id]
                )
                self.assertIn(scenario["tags"]["kind"], {"capability", "guard"})
                if scenario["tags"]["prompt"] == "implicit":
                    # Implicit prompts test whether Canon's skills trigger unprompted.
                    self.assertNotIn("canon", inputs["task"].lower())
                    self.assertNotIn(expectations["expected_scope"], inputs["task"])
        self.assertTrue(
            {"s001-implicit", "s002-implicit", "s003-implicit"} <= set(ids),
            "every original scenario has an implicit variant",
        )

    def test_diff_lists_baseline_changes_before_created_files(self) -> None:
        def mixed_invoker(workspace: Path, _task: str, _plugin: Path, _mcp: Path):
            (workspace / "a-large-plan.md").write_text("x\n" * 50, encoding="utf-8")
            decision = workspace / "knowledge" / "decisions" / "runtime.md"
            decision.write_text(
                decision.read_text(encoding="utf-8").replace("Azure Container Apps", "AKS"),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=["fake-agent"], returncode=0, stdout="", stderr=""
            )

        result = execute_scenario("s001", "task", invoker=mixed_invoker)
        diff = result["git_diff"]
        self.assertIn("+Deploy the payments API to AKS.", diff)
        self.assertIn("new file mode", diff)
        self.assertLess(
            diff.index("knowledge/decisions/runtime.md"), diff.index("a-large-plan.md")
        )

    def test_acceptance_scorer_fails_on_agent_errors(self) -> None:
        passing = {"acceptance": {"passed": True, "checks": {}}}
        cases = {
            "timed out": {"timed_out": True, "return_code": 124},
            "exited with code 1": {"timed_out": False, "return_code": 1},
            "reported an error": {
                "timed_out": False,
                "return_code": 0,
                "stream_error": True,
                "terminal_reason": "max_budget",
            },
        }
        for expected, agent in cases.items():
            with self.subTest(expected=expected):
                feedback = scenario_acceptance(outputs={**passing, "agent": agent})
                self.assertIs(feedback.value, False)
                self.assertIn(expected, feedback.rationale)
        self.assertIs(scenario_acceptance(outputs=None).value, False)

    def test_baseline_commit_ignores_global_signing_and_hooks(self) -> None:
        def noop_invoker(_workspace: Path, _task: str, _plugin: Path, _mcp: Path):
            return subprocess.CompletedProcess(
                args=["fake-agent"], returncode=0, stdout="", stderr=""
            )

        with tempfile.TemporaryDirectory() as temporary:
            hooks = Path(temporary) / "hooks"
            hooks.mkdir()
            pre_commit = hooks / "pre-commit"
            pre_commit.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            pre_commit.chmod(0o755)
            config = Path(temporary) / "gitconfig"
            config.write_text(
                "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n"
                f"[core]\n\thooksPath = {hooks}\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(config)}):
                result = execute_scenario("s001", "task", invoker=noop_invoker)
        self.assertEqual(result["git_status"], [])

    def test_judge_credentials_are_required_for_known_providers(self) -> None:
        judges = [
            SimpleNamespace(name="authority", model="anthropic:/claude-haiku-4-5-20251001"),
            SimpleNamespace(name="tools", model="anthropic:/claude-haiku-4-5-20251001"),
            SimpleNamespace(name="custom", model="bedrock:/some-model"),
            SimpleNamespace(name="code_based"),
        ]
        with mock.patch.dict(os.environ, clear=True):
            with self.assertRaises(RuntimeError) as raised:
                validate_judge_credentials(judges)
        self.assertIn("set ANTHROPIC_API_KEY for authority, tools", str(raised.exception))

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            models = validate_judge_credentials(judges)
        self.assertEqual(
            models,
            {
                "authority": "anthropic:/claude-haiku-4-5-20251001",
                "tools": "anthropic:/claude-haiku-4-5-20251001",
                "custom": "bedrock:/some-model",
            },
        )

    def test_evaluation_skips_duplicate_prediction_probe(self) -> None:
        judge = SimpleNamespace(name="judge")
        observed = {}

        def fake_evaluate(**kwargs):
            observed.update(kwargs)
            observed["skip"] = os.environ.get("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION")
            return "result"

        with (
            mock.patch.dict(os.environ, clear=True),
            mock.patch("mlflow.genai.evaluate", side_effect=fake_evaluate),
        ):
            self.assertEqual(evaluate_scenarios("dataset", [judge]), "result")
        self.assertEqual(observed["skip"], "true")
        self.assertIs(observed["predict_fn"], run_scenario)
        self.assertEqual(observed["scorers"], [judge, *DETERMINISTIC_SCORERS])
        self.assertIn(knowledge_boundary, DETERMINISTIC_SCORERS)

        with (
            mock.patch.dict(os.environ, clear=True),
            mock.patch("mlflow.genai.evaluate", side_effect=fake_evaluate),
        ):
            evaluate_scenarios("dataset", [], arm="control")
        self.assertIs(observed["predict_fn"], PREDICT_FNS["control"])
        self.assertEqual(observed["scorers"], CONTROL_SCORERS)
        self.assertNotIn(required_tool_call, CONTROL_SCORERS)

        with (
            mock.patch.dict(
                os.environ, {"MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION": "false"}, clear=True
            ),
            mock.patch("mlflow.genai.evaluate", side_effect=fake_evaluate),
        ):
            evaluate_scenarios("dataset", [judge])
        self.assertEqual(observed["skip"], "false")

    def test_run_agent_evaluation_exit_status(self) -> None:
        judge = SimpleNamespace(name="judge", model="anthropic:/claude-haiku-4-5-20251001")
        failed = result_table(**{"judge/value": None, "judge/error_message": "SCORER_ERROR"})
        patches = [
            mock.patch.object(run_agent_evaluation, "validate_live_agent"),
            mock.patch.object(run_agent_evaluation, "configure_mlflow", return_value="1"),
            mock.patch.object(run_agent_evaluation, "find_dataset", return_value="dataset"),
            mock.patch.object(run_agent_evaluation, "list_scorers", return_value=[judge]),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        with (
            mock.patch.dict(os.environ, clear=True),
            mock.patch.object(run_agent_evaluation, "evaluate_scenarios") as evaluate,
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run_agent_evaluation.main([])
        self.assertIn("ANTHROPIC_API_KEY", str(raised.exception.code))
        evaluate.assert_not_called()

        with (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            mock.patch.object(run_agent_evaluation, "evaluate_scenarios", return_value=failed),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run_agent_evaluation.main([])
        self.assertIn("s001 judge: SCORER_ERROR", str(raised.exception.code))

        with (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            mock.patch.object(
                run_agent_evaluation, "evaluate_scenarios", return_value=result_table()
            ),
            mock.patch("builtins.print") as printed,
        ):
            run_agent_evaluation.main([])
        printed.assert_called_with("Agent evaluation passed.")

    def test_markdown_formatted_answers_pass_acceptance(self) -> None:
        cases = {
            ("s001", "deployment-plan.md"): [
                "**Platform: Azure Container Apps**",
                "**Platform:** Azure Container Apps",
                "- Platform: `Azure Container Apps` (ACA).",
                "## Platform: Azure Container Apps",
                "Platform: Azure Container Apps — per payments/runtime",
                "Platform: Azure Container Apps, not AKS",
                "Platform: Azure Container Apps (not Kubernetes/AKS)",
                "Platform: ACA (Azure Container Apps)",
                "| Platform | Azure Container Apps |",
                "## Platform\n\nAzure Container Apps",
                "**Platform:**\n- Azure Container Apps",
                "Platform: Azure Container Apps rather than Kubernetes on AKS",
            ],
            ("s002", "cloud-plan.md"): [
                "**Provider:** Microsoft Azure",
                "* __Provider__: Azure",
                "Provider: Azure; AWS Lambda excluded (sibling client)",
                "| Provider | Microsoft Azure | inherited from client/acme |",
            ],
        }
        for (fixture_id, filename), lines in cases.items():
            for line in lines:
                with self.subTest(fixture_id=fixture_id, line=line):
                    result = execute_scenario(
                        fixture_id, "task", invoker=writing_invoker(filename, f"# Plan\n\n{line}\n")
                    )
                    self.assertTrue(result["acceptance"]["passed"], result["acceptance"])

    def test_missing_or_conflicting_answers_fail_every_selection_check(self) -> None:
        cases = {
            ("s001", "deployment-plan.md", "rejects_conflicting_reference"): [
                "No platform line.",
                "**Platform:** Azure Kubernetes Service (AKS)",
                "Platform: AKS, not Azure Container Apps",
                "Platform: AKS instead of Azure Container Apps",
                "| Platform | Kubernetes |",
            ],
            ("s002", "cloud-plan.md", "excludes_sibling_decision"): [
                "No provider line.",
                "Provider: AWS Lambda with API Gateway",
                "Provider: AWS rather than Azure",
            ],
        }
        for (fixture_id, filename, check), bodies in cases.items():
            for body in bodies:
                with self.subTest(fixture_id=fixture_id, body=body):
                    result = execute_scenario(
                        fixture_id, "task", invoker=writing_invoker(filename, body)
                    )
                    self.assertFalse(result["acceptance"]["checks"][check])
                    self.assertFalse(result["acceptance"]["passed"])

    def test_required_tool_call_allows_extra_calls_and_arguments(self) -> None:
        def outputs(*calls: tuple[str, dict]) -> dict:
            return {"canon_calls": [{"name": name, "input": args} for name, args in calls]}

        passing = outputs(
            ("mcp__canon__search_knowledge", {"query": "cloud"}),
            ("mcp__canon__get_context", {"task": "t", "scope": "client/acme/project/payments"}),
        )
        self.assertIs(
            required_tool_call(outputs=passing, expectations=EXPECTED_GET_CONTEXT).value, True
        )

        failing = {
            "wrong scope": outputs(("mcp__canon__get_context", {"scope": "client/acme"})),
            "wrong tool": outputs(
                ("mcp__canon__search_knowledge", {"scope": "client/acme/project/payments"})
            ),
            "no calls": outputs(),
            "no outputs": None,
        }
        for label, value in failing.items():
            with self.subTest(label=label):
                feedback = required_tool_call(outputs=value, expectations=EXPECTED_GET_CONTEXT)
                self.assertIs(feedback.value, False)
                self.assertIn("get_context(scope='client/acme/project/payments')", feedback.rationale)

    def test_evaluation_failures_ignore_expectation_columns(self) -> None:
        judge = SimpleNamespace(name="judge")
        self.assertEqual(evaluation_failures(result_table(), [judge]), [])
        self.assertEqual(evaluation_failures(result_table(**{"judge/value": "yes"}), [judge]), [])

        cases = {
            "judge rationale": {"judge/value": False, "judge/rationale": "judge rationale"},
            "value='no'": {"judge/value": "no"},
            "SCORER_ERROR": {"judge/value": None, "judge/error_message": "SCORER_ERROR"},
            "no result": {"judge/value": float("nan")},
        }
        for expected, overrides in cases.items():
            with self.subTest(expected=expected):
                failures = evaluation_failures(result_table(**overrides), [judge])
                self.assertEqual(len(failures), 1)
                self.assertIn(f"s001 judge: {expected}", failures[0])

        missing_column = result_table()
        missing_column.result_df = missing_column.result_df.drop(columns="judge/value")
        self.assertEqual(evaluation_failures(missing_column, [judge]), ["s001 judge: no result"])
        self.assertEqual(
            evaluation_failures(SimpleNamespace(result_df=None), [judge]),
            ["evaluation produced no results"],
        )

    def test_retired_scorers_are_rejected_and_deleted(self) -> None:
        retired = SimpleNamespace(name="canon_tool_call_correctness", model=None)
        with self.assertRaises(RuntimeError) as raised:
            validate_scorers([retired])
        self.assertIn("register_agent_eval_scorers", str(raised.exception))

        registered = [retired, SimpleNamespace(name="canon_authority_compliance")]
        with (
            mock.patch.object(register_agent_eval_scorers, "configure_mlflow", return_value="1"),
            mock.patch.object(register_agent_eval_scorers, "list_scorers", return_value=registered),
            mock.patch.object(register_agent_eval_scorers, "delete_scorer") as delete,
            mock.patch.object(register_agent_eval_scorers, "build_scorers", return_value=[]),
            mock.patch("sys.argv", ["register", "--model", "anthropic:/model"]),
            mock.patch("builtins.print"),
        ):
            register_agent_eval_scorers.main()
        delete.assert_called_once_with(
            name="canon_tool_call_correctness", experiment_id="1", version="all"
        )
        registrable = {judge.name for judge in register_agent_eval_scorers.build_scorers("anthropic:/m")}
        self.assertFalse(registrable & {"canon_tool_call_correctness", "canon_boundary_safety"})
        with self.assertRaises(RuntimeError):
            validate_scorers([SimpleNamespace(name="canon_boundary_safety", model=None)])

    def test_field_value_and_selection_helpers(self) -> None:
        self.assertEqual(field_value("Intro\n\n### Rollback strategy\n\nUnknown.", "rollback strategy"), "unknown")
        self.assertEqual(field_value("| Field | Value |\n|---|---|\n| Provider | Azure |", "provider"), "azure")
        self.assertEqual(field_value("No labelled answer here.", "platform"), "")
        self.assertTrue(selects("azure container apps, not aks", r"\bazure container apps\b"))
        self.assertFalse(selects("azure container apps, not aks", r"\baks\b"))
        self.assertFalse(selects("azure container apps; aks was considered and rejected", r"\baks\b"))
        self.assertTrue(selects("aks instead of azure container apps", r"\baks\b"))

    def test_unknown_answers_are_recognized(self) -> None:
        unknown = [
            "unknown",
            "not established",
            "not yet decided; blue-green is only a reference suggestion",
            "no rollback strategy has been established",
            "no established strategy",
            "none documented in canon",
            "tbd: needs a decision",
        ]
        decided = ["blue-green deployment", "roll back by redeploying the previous image", ""]
        for value in unknown:
            with self.subTest(value=value):
                self.assertTrue(reports_unknown(value))
        for value in decided:
            with self.subTest(value=value):
                self.assertFalse(reports_unknown(value))

    def test_knowledge_boundary_gate(self) -> None:
        def changes(**kinds) -> dict:
            return {"workspace_changes": {"created": [], "modified": [], "deleted": [], **kinds}}

        allowed = changes(created=[".canon/proposals/p.md", "deployment-plan.md", "knowledge.md"])
        self.assertIs(knowledge_boundary(outputs=allowed).value, True)
        for kind in ("created", "modified", "deleted"):
            with self.subTest(kind=kind):
                feedback = knowledge_boundary(outputs=changes(**{kind: ["knowledge/decisions/x.md"]}))
                self.assertIs(feedback.value, False)
                self.assertIn(f"{kind} knowledge/decisions/x.md", feedback.rationale)
        self.assertIs(knowledge_boundary(outputs=None).value, False)

        result = execute_scenario("s010", "task", invoker=outcome_invoker(KNOWN_OUTCOMES["s010"][1]))
        self.assertIs(knowledge_boundary(outputs=result).value, False)

    def test_acceptance_is_hidden_from_judge_outputs(self) -> None:
        evidence = {
            "response": "Done",
            "tool_calls": [],
            "canon_calls": [],
            "workspace_changes": {"created": [], "modified": [], "deleted": []},
            "acceptance": {"passed": False, "checks": {"uses_active_decision": False}},
            "agent": {"return_code": 0, "timed_out": False},
        }
        previous_uri = mlflow.get_tracking_uri()
        with tempfile.TemporaryDirectory() as temporary:
            self.addCleanup(mlflow.set_tracking_uri, previous_uri)
            mlflow.set_tracking_uri(f"sqlite:///{temporary}/mlflow.db")
            mlflow.set_experiment(
                experiment_id=mlflow.create_experiment(
                    "hidden-acceptance", artifact_location=f"{temporary}/artifacts"
                )
            )
            with mock.patch(
                "scripts.agent_eval_harness.execute_scenario", return_value=dict(evidence)
            ) as execute:
                outputs = run_scenario(fixture_id="s001", task="task", scenario_id="s001-implicit")
            trace = mlflow.get_trace(mlflow.get_last_active_trace_id(), flush=True)

        execute.assert_called_once_with("s001", "task", arm="canon")
        self.assertNotIn("acceptance", outputs)
        self.assertNotIn("acceptance", trace.data._get_root_span().outputs)
        self.assertEqual(trace.info.tags["scenario_id"], "s001-implicit")
        self.assertEqual(trace.info.tags["arm"], "canon")
        feedback = scenario_acceptance(outputs=outputs, trace=trace)
        self.assertIs(feedback.value, False)
        self.assertIn("uses_active_decision", feedback.rationale)

    def test_lift_report_separates_lift_from_noise(self) -> None:
        def table(*rows) -> SimpleNamespace:
            frame = pd.DataFrame(
                [
                    {
                        "request": {"scenario_id": scenario, "fixture_id": scenario[:4]},
                        "response": {"agent": {"return_code": code, "timed_out": False}},
                        "tags": {"kind": kind},
                        "canon_scenario_acceptance/value": passed,
                    }
                    for scenario, kind, passed, code in rows
                ]
            )
            return SimpleNamespace(result_df=frame)

        canon = table(
            ("s001", "capability", True, 0),
            ("s002", "capability", True, 0),
            ("s003", "capability", False, 0),
            ("s004", "capability", True, 0),
            ("s011", "guard", True, 0),
        )
        control = table(
            ("s001", "capability", False, 0),
            ("s002", "capability", True, 0),
            ("s003", "capability", True, 0),
            ("s004", "capability", False, 1),
            ("s011", "guard", True, 0),
        )
        report = lift_report(canon, control).splitlines()
        notes = {line.split()[0]: line for line in report[1:-1]}
        self.assertIn("Canon lift", notes["s001"])
        self.assertIn("does not isolate Canon", notes["s002"])
        self.assertIn("Canon fails where control passes", notes["s003"])
        self.assertIn("error", notes["s004"])
        self.assertIn("guard", notes["s011"])
        self.assertEqual(
            report[-1], "Capability scenarios: Canon passes 3/4, control passes 2/4."
        )

    def test_control_arm_never_gates(self) -> None:
        patches = [
            mock.patch.object(run_agent_evaluation, "validate_live_agent"),
            mock.patch.object(run_agent_evaluation, "configure_mlflow", return_value="1"),
            mock.patch.object(run_agent_evaluation, "find_dataset", return_value="dataset"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        failing = result_table(**{"canon_scenario_acceptance/value": False})
        with (
            mock.patch.object(run_agent_evaluation, "list_scorers") as list_scorers,
            mock.patch.object(run_agent_evaluation, "evaluate_scenarios", return_value=failing) as evaluate,
            mock.patch("builtins.print"),
        ):
            run_agent_evaluation.main(["--arm", "control"])
        list_scorers.assert_not_called()
        evaluate.assert_called_once_with("dataset", [], arm="control")

        judge = SimpleNamespace(name="judge", model="bedrock:/model")
        with (
            mock.patch.object(run_agent_evaluation, "list_scorers", return_value=[judge]),
            mock.patch.object(
                run_agent_evaluation, "evaluate_scenarios", side_effect=[result_table(), failing]
            ),
            mock.patch("builtins.print") as printed,
        ):
            run_agent_evaluation.main(["--arm", "both"])
        printed.assert_called_with("Agent evaluation passed.")
        self.assertTrue(
            any("Canon lift" in str(call.args[0]) for call in printed.call_args_list if call.args)
        )

    def test_invalid_fixture_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            execute_scenario("../s001", "task")


if __name__ == "__main__":
    unittest.main()
