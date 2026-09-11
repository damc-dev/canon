from __future__ import annotations

import pytest

from scripts.agent_eval_config import configure_mlflow


@pytest.fixture(autouse=True)
def _canon_mlflow_tracking() -> None:
    # The MLflow pytest plugin opens the @mlflow.test run at the start of the call phase,
    # in whatever store is current. Point MLflow at the evaluation store during setup so
    # that run lives alongside the dataset and scorers instead of in ./mlflow.db.
    configure_mlflow()
