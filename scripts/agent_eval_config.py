from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.genai.datasets import search_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "canon-agent-golden-v1"
DEFAULT_EXPERIMENT_NAME = "canon-agent-evaluation"


def default_tracking_uri() -> str:
    database = (PROJECT_ROOT / ".canon" / "eval" / "mlflow.db").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database}"


def configure_mlflow() -> str:
    """Configure a local SQL-backed MLflow experiment unless the caller overrides it."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    )
    return experiment.experiment_id


def find_dataset(name: str = DEFAULT_DATASET_NAME):
    experiment_id = configure_mlflow()
    matches = search_datasets(
        experiment_ids=[experiment_id],
        filter_string=f"name = '{name}'",
        max_results=1,
    )
    return matches[0] if matches else None
