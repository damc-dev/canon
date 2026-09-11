from __future__ import annotations

import mlflow
from mlflow.genai.scorers import list_scorers

from scripts.agent_eval_config import configure_mlflow, find_dataset
from scripts.agent_eval_harness import validate_judge_credentials, validate_live_agent


def main() -> None:
    experiment_id = configure_mlflow()
    dataset = find_dataset()
    scorers = list_scorers(experiment_id=experiment_id)
    try:
        agent = validate_live_agent()
    except RuntimeError as exc:
        raise SystemExit(f"Agent environment is incomplete: {exc}") from exc

    if dataset is None:
        raise SystemExit("Agent environment is incomplete: evaluation dataset is missing.")
    if not scorers:
        raise SystemExit("Agent environment is incomplete: registered scorers are missing.")
    try:
        judge_models = validate_judge_credentials(scorers)
    except RuntimeError as exc:
        raise SystemExit(f"Agent environment is incomplete: {exc}") from exc

    print(f"MLflow: {mlflow.__version__}")
    print(f"Experiment: {experiment_id}")
    print(f"Dataset: {dataset.name} ({len(dataset.to_df())} records)")
    print("Scorers: " + ", ".join(sorted(scorer.name for scorer in scorers)))
    print("Judges: " + (", ".join(sorted(set(judge_models.values()))) or "none"))
    print(f"Agent: {agent['version']} ({agent['auth']})")


if __name__ == "__main__":
    main()
