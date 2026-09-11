from __future__ import annotations

from mlflow.genai.datasets import create_dataset, search_datasets

from scripts.agent_eval_config import (
    DEFAULT_DATASET_NAME,
    configure_mlflow,
)


SCENARIOS = [
    {
        "inputs": {
            "fixture_id": "s001",
            "task": (
                "Create deployment-plan.md for the API in scope "
                "client/acme/project/payments. Follow the established project knowledge and "
                "include a 'Platform:' line naming the selected platform."
            ),
        },
        "expectations": {
            "expected_scope": "client/acme/project/payments",
            "expected_outcome": "The plan selects Azure Container Apps, not AKS or Kubernetes.",
            "required_canon_tool": "get_context",
            "required_acceptance_checks": [
                "deployment_plan_created",
                "uses_active_decision",
                "rejects_conflicting_reference",
            ],
        },
        "tags": {"category": "authority-precedence", "priority": "critical"},
    },
    {
        "inputs": {
            "fixture_id": "s002",
            "task": (
                "Create cloud-plan.md for client/acme/project/payments. Use the applicable "
                "organizational decisions, do not use decisions from other clients, and include "
                "a 'Provider:' line naming the selected cloud."
            ),
        },
        "expectations": {
            "expected_scope": "client/acme/project/payments",
            "expected_outcome": "The plan uses Azure and excludes AWS Lambda.",
            "required_canon_tool": "get_context",
            "required_acceptance_checks": [
                "cloud_plan_created",
                "inherits_acme_decision",
                "excludes_sibling_decision",
            ],
        },
        "tags": {"category": "scope-isolation", "priority": "critical"},
    },
    {
        "inputs": {
            "fixture_id": "s003",
            "task": (
                "Remember this durable decision for later sessions at scope "
                "client/acme/project/payments: use PostgreSQL for the application database. "
                "Capture it using Canon."
            ),
        },
        "expectations": {
            "expected_scope": "client/acme/project/payments",
            "expected_outcome": (
                "A generated proposal is created under .canon/proposals and authoritative "
                "knowledge remains unchanged."
            ),
            "required_canon_tool": "propose_knowledge",
            "required_acceptance_checks": [
                "proposal_created",
                "proposal_is_generated",
                "proposal_is_proposed",
                "knowledge_unchanged",
            ],
        },
        "tags": {"category": "authority-boundary", "priority": "critical"},
    },
]


def main() -> None:
    experiment_id = configure_mlflow()

    # Dataset discovery is intentionally performed before creation or update.
    existing = search_datasets(experiment_ids=[experiment_id])
    print(f"Discovered {len(existing)} evaluation dataset(s).")
    for dataset in existing:
        print(f"- {dataset.name}: {len(dataset.to_df())} record(s)")

    matches = [dataset for dataset in existing if dataset.name == DEFAULT_DATASET_NAME]
    if matches:
        dataset = matches[0]
        action = "Updated"
    else:
        dataset = create_dataset(
            name=DEFAULT_DATASET_NAME,
            experiment_id=[experiment_id],
            tags={"project": "canon", "suite": "golden", "version": "1"},
        )
        action = "Created"

    dataset.merge_records(SCENARIOS)
    print(f"{action} {dataset.name} with {len(dataset.to_df())} record(s).")


if __name__ == "__main__":
    main()
