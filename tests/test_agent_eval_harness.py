from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from scripts import register_agent_eval_scorers, run_agent_evaluation
from scripts.agent_eval_harness import (
    DETERMINISTIC_SCORERS,
    build_claude_command,
    compare_inventories,
    evaluate_scenarios,
    evaluation_failures,
    execute_scenario,
    parse_claude_stream,
    required_tool_call,
    run_scenario,
    scenario_acceptance,
    scenario_failures,
    validate_judge_credentials,
    validate_scorers,
)

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
    def invoke(workspace: Path, _task: str, _plugin: Path, _mcp: Path):
        (workspace / filename).write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(args=["fake-agent"], returncode=0, stdout="", stderr="")

    return invoke


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
        allowed_tools = command[command.index("--allowedTools") + 1]
        self.assertIn("Write", allowed_tools)
        self.assertIn("Edit", allowed_tools)
        self.assertIn("mcp__canon__get_context", allowed_tools)
        self.assertIn("mcp__canon__propose_knowledge", allowed_tools)
        self.assertIn("Bash,WebFetch,WebSearch", command)
        self.assertIn("/plugin", command)
        self.assertEqual(command[-1], "Perform the task")

    def test_known_good_controls_pass_all_acceptance_checks(self) -> None:
        def good_invoker(workspace: Path, task: str, _plugin: Path, _mcp: Path):
            mcp_args = json.loads(_mcp.read_text(encoding="utf-8"))["mcpServers"]["canon"]["args"]
            self.assertIn("--project", mcp_args)
            self.assertIn("--no-dev", mcp_args)
            self.assertNotIn("--directory", mcp_args)
            if "deployment-plan.md" in task:
                (workspace / "deployment-plan.md").write_text(
                    "# Deployment\n\nPlatform: Azure Container Apps\n", encoding="utf-8"
                )
                tool = "get_context"
            elif "cloud-plan.md" in task:
                (workspace / "cloud-plan.md").write_text(
                    "# Cloud\n\nProvider: Microsoft Azure\n", encoding="utf-8"
                )
                tool = "get_context"
            else:
                proposal = workspace / ".canon" / "proposals" / "postgresql.md"
                proposal.parent.mkdir(parents=True)
                proposal.write_text(
                    "---\nstatus: proposed\ngenerated: true\n---\n\nUse PostgreSQL.\n",
                    encoding="utf-8",
                )
                tool = "propose_knowledge"
            return subprocess.CompletedProcess(
                args=["fake-agent"],
                returncode=0,
                stdout=stream_for(tool, {"task": task, "scope": "client/acme/project/payments"}),
                stderr="",
            )

        tasks = {
            "s001": "Create deployment-plan.md",
            "s002": "Create cloud-plan.md",
            "s003": "Remember PostgreSQL",
        }
        for fixture_id, task in tasks.items():
            with self.subTest(fixture_id=fixture_id):
                result = execute_scenario(fixture_id, task, invoker=good_invoker)
                self.assertTrue(result["acceptance"]["passed"])
                self.assertEqual(len(result["canon_calls"]), 1)
                self.assertEqual(scenario_failures(result), [])
                self.assertIs(scenario_acceptance(outputs=result).value, True)

    def test_known_bad_control_fails_acceptance(self) -> None:
        def bad_invoker(workspace: Path, _task: str, _plugin: Path, _mcp: Path):
            (workspace / "deployment-plan.md").write_text(
                "# Deployment\n\nPlatform: Kubernetes on AKS\n", encoding="utf-8"
            )
            return subprocess.CompletedProcess(
                args=["fake-agent"], returncode=0, stdout=stream_for("get_context", {}), stderr=""
            )

        result = execute_scenario("s001", "bad control", invoker=bad_invoker)
        self.assertFalse(result["acceptance"]["passed"])
        self.assertIn("+Platform: Kubernetes on AKS", result["git_diff"])
        self.assertEqual(result["git_status"], ["?? deployment-plan.md"])
        feedback = scenario_acceptance(outputs=result)
        self.assertIs(feedback.value, False)
        self.assertIn("uses_active_decision", feedback.rationale)

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
        self.assertEqual(observed["scorers"], [judge, scenario_acceptance, required_tool_call])

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
                run_agent_evaluation.main()
        self.assertIn("ANTHROPIC_API_KEY", str(raised.exception.code))
        evaluate.assert_not_called()

        with (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            mock.patch.object(run_agent_evaluation, "evaluate_scenarios", return_value=failed),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run_agent_evaluation.main()
        self.assertIn("s001 judge: SCORER_ERROR", str(raised.exception.code))

        with (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            mock.patch.object(
                run_agent_evaluation, "evaluate_scenarios", return_value=result_table()
            ),
            mock.patch("builtins.print") as printed,
        ):
            run_agent_evaluation.main()
        printed.assert_called_with("Agent evaluation passed.")

    def test_markdown_formatted_answers_pass_acceptance(self) -> None:
        cases = {
            ("s001", "deployment-plan.md"): [
                "**Platform: Azure Container Apps**",
                "**Platform:** Azure Container Apps",
                "- Platform: `Azure Container Apps` (ACA).",
                "## Platform: Azure Container Apps",
            ],
            ("s002", "cloud-plan.md"): [
                "**Provider:** Microsoft Azure",
                "* __Provider__: Azure",
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
            ],
            ("s002", "cloud-plan.md", "excludes_sibling_decision"): [
                "No provider line.",
                "Provider: AWS Lambda with API Gateway",
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
        self.assertNotIn(
            "canon_tool_call_correctness",
            {judge.name for judge in register_agent_eval_scorers.build_scorers("anthropic:/m")},
        )

    def test_invalid_fixture_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            execute_scenario("../s001", "task")


if __name__ == "__main__":
    unittest.main()
