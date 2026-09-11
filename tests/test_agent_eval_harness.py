from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.agent_eval_harness import (
    build_claude_command,
    compare_inventories,
    execute_scenario,
    parse_claude_stream,
    scenario_acceptance,
    scenario_failures,
)


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
        feedback = scenario_acceptance(outputs=result)
        self.assertIs(feedback.value, False)
        self.assertIn("uses_active_decision", feedback.rationale)

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

    def test_invalid_fixture_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            execute_scenario("../s001", "task")


if __name__ == "__main__":
    unittest.main()
