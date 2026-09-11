from __future__ import annotations

from mlflow import genai
from mlflow.genai.scorers import list_scorers

from scripts.agent_eval_config import DEFAULT_DATASET_NAME, configure_mlflow, find_dataset
from scripts.agent_eval_harness import run_scenario, scenario_acceptance, validate_live_agent


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

    result = genai.evaluate(
        data=dataset, predict_fn=run_scenario, scorers=[*scorers, scenario_acceptance]
    )
    print(result)


if __name__ == "__main__":
    main()
