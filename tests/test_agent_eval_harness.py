from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from scripts.agent_eval_harness import (
    build_claude_command,
    compare_inventories,
    execute_scenario,
    parse_claude_stream,
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

    def test_invalid_fixture_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            execute_scenario("../s001", "task")


if __name__ == "__main__":
    unittest.main()
