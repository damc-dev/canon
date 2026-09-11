from __future__ import annotations

import os

import mlflow
import pytest
from mlflow.genai.scorers import list_scorers

from scripts.agent_eval_config import DEFAULT_DATASET_NAME, configure_mlflow, find_dataset
from scripts.agent_eval_harness import run_scenario, scenario_acceptance, validate_live_agent


@mlflow.test
@pytest.mark.skipif(
    os.getenv("CANON_RUN_AGENT_EVAL") != "1",
    reason="Set CANON_RUN_AGENT_EVAL=1 to run paid coding-agent regression scenarios.",
)
def test_canon_agent_golden_scenarios() -> None:
    validate_live_agent()
    experiment_id = configure_mlflow()
    dataset = find_dataset(DEFAULT_DATASET_NAME)
    assert dataset is not None, "Create the agent evaluation dataset first."
    scorers = list_scorers(experiment_id=experiment_id)
    assert scorers, "Register the agent evaluation scorers first."
    result = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=run_scenario,
        scorers=[*scorers, scenario_acceptance],
    )
    assert result.passed, result.reason
