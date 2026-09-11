from __future__ import annotations

from mlflow.genai.scorers import list_scorers

from scripts.agent_eval_config import DEFAULT_DATASET_NAME, configure_mlflow, find_dataset
from scripts.agent_eval_harness import (
    evaluate_scenarios,
    evaluation_failures,
    validate_live_agent,
    validate_scorers,
)


def main() -> None:
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

    result = evaluate_scenarios(dataset, scorers)
    print(result)
    failures = evaluation_failures(result, scorers)
    if failures:
        raise SystemExit(
            f"Agent evaluation failed ({len(failures)}):\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
    print("Agent evaluation passed.")


if __name__ == "__main__":
    main()
