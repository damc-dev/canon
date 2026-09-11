from __future__ import annotations

import argparse

from mlflow.genai.scorers import list_scorers

from scripts.agent_eval_config import DEFAULT_DATASET_NAME, configure_mlflow, find_dataset
from scripts.agent_eval_harness import (
    evaluate_scenarios,
    evaluation_failures,
    lift_report,
    validate_live_agent,
    validate_scorers,
)


def _registered_scorers(experiment_id: str) -> list:
    scorers = list_scorers(experiment_id=experiment_id)
    if not scorers:
        raise SystemExit(
            "No registered scorers. Run: uv run python -m "
            "scripts.register_agent_eval_scorers --model <provider:/model>"
        )
    try:
        validate_scorers(scorers)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return scorers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Canon's coding-agent evaluation.")
    parser.add_argument(
        "--arm",
        choices=("canon", "control", "both"),
        default="canon",
        help=(
            "canon runs the gated suite with the plugin; control runs the same scenarios "
            "without it; both adds a per-scenario lift report. Only the canon arm gates."
        ),
    )
    args = parser.parse_args(argv)
    try:
        validate_live_agent()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    experiment_id = configure_mlflow()
    dataset = find_dataset(DEFAULT_DATASET_NAME)
    if dataset is None:
        raise SystemExit(
            "Evaluation dataset is missing. Run: uv run python -m "
            "scripts.create_agent_eval_dataset"
        )

    failures: list[str] = []
    canon_result = None
    if args.arm in ("canon", "both"):
        scorers = _registered_scorers(experiment_id)
        canon_result = evaluate_scenarios(dataset, scorers)
        print(canon_result)
        failures = evaluation_failures(canon_result, scorers)
    if args.arm in ("control", "both"):
        # Judges grade Canon behavior, so the control arm runs only the deterministic gates.
        control_result = evaluate_scenarios(dataset, [], arm="control")
        print(control_result)
        if canon_result is None:
            print("Control arm (does not gate):")
            print("\n".join(evaluation_failures(control_result, [], arm="control")) or "all passed")
        else:
            print("Canon lift over the control arm (does not gate):")
            print(lift_report(canon_result, control_result))

    if failures:
        raise SystemExit(
            f"Agent evaluation failed ({len(failures)}):\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
    if canon_result is not None:
        print("Agent evaluation passed.")


if __name__ == "__main__":
    main()
