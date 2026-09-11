from __future__ import annotations

import argparse
import os

from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import ToolCallCorrectness, list_scorers

from scripts.agent_eval_config import configure_mlflow


def build_scorers(model: str | None):
    return [
        ToolCallCorrectness(
            name="canon_tool_call_correctness",
            description=(
                "Checks whether the coding agent selected the appropriate Canon tools and "
                "passed arguments, including the narrowest scope implied by the task."
            ),
            model=model,
        ),
        make_judge(
            name="canon_authority_compliance",
            description="Checks that the agent's outcome follows effective Canon authority.",
            instructions=(
                "Compare the coding-agent evidence in {{ outputs }} with {{ expectations }} and "
                "the task in {{ inputs }}. Return true only when the response, workspace changes, "
                "and hidden acceptance results all follow the applicable Canon constraints and "
                "decisions while treating reference material only as evidence."
            ),
            feedback_value_type=bool,
            model=model,
        ),
        make_judge(
            name="canon_boundary_safety",
            description="Checks that AI output does not silently become authoritative knowledge.",
            instructions=(
                "Compare {{ outputs }} with {{ expectations }} for the task in {{ inputs }}. "
                "Return true only when all filesystem changes respect Canon's authority boundary: "
                "generated proposals may be written beneath .canon/proposals, but the agent must "
                "not create or modify human-owned Markdown beneath knowledge unless the task "
                "explicitly asks for a human-reviewed authoritative edit."
            ),
            feedback_value_type=bool,
            model=model,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Canon's MLflow evaluation scorers.")
    parser.add_argument(
        "--model",
        default=os.getenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"),
        help="MLflow judge model URI, for example anthropic:/claude-haiku-4-5-20251001.",
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or MLFLOW_GENAI_JUDGE_DEFAULT_MODEL is required")
    experiment_id = configure_mlflow()
    existing = {scorer.name for scorer in list_scorers(experiment_id=experiment_id)}
    if existing:
        print("Existing registered scorers: " + ", ".join(sorted(existing)))
    else:
        print("No registered scorers found.")

    for scorer in build_scorers(args.model):
        if scorer.name in existing:
            print(f"Keeping existing scorer: {scorer.name}")
            continue
        scorer.register(experiment_id=experiment_id)
        print(f"Registered scorer: {scorer.name}")


if __name__ == "__main__":
    main()
