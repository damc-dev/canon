from __future__ import annotations

import argparse
import json

from mlflow.genai.datasets import search_datasets

from scripts.agent_eval_config import configure_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description="List MLflow datasets for Canon evaluation.")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args()

    experiment_id = configure_mlflow()
    datasets = search_datasets(
        experiment_ids=[experiment_id], order_by=["last_update_time DESC"]
    )
    rows = [
        {
            "name": dataset.name,
            "dataset_id": dataset.dataset_id,
            "records": len(dataset.to_df()),
            "tags": dataset.tags,
        }
        for dataset in datasets
    ]
    if args.format == "json":
        print(json.dumps(rows, indent=2, default=str))
        return
    if not rows:
        print("No evaluation datasets found.")
        return
    for row in rows:
        print(f"{row['name']} ({row['dataset_id']}): {row['records']} record(s)")


if __name__ == "__main__":
    main()
