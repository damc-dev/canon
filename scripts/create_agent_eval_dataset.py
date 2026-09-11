from __future__ import annotations

from mlflow.genai.datasets import create_dataset, search_datasets

from scripts.agent_eval_config import (
    DEFAULT_DATASET_NAME,
    configure_mlflow,
)


PAYMENTS = "client/acme/project/payments"


def _call(tool: str, **arguments: object) -> dict:
    return {"name": f"mcp__canon__{tool}", "arguments": arguments}


def _scenario(
    scenario_id: str,
    task: str,
    *,
    outcome: str,
    checks: list[str],
    category: str,
    tool_calls: tuple[dict, ...] = (),
    scope: str = PAYMENTS,
    prompt: str = "explicit",
    kind: str = "capability",
    priority: str = "high",
) -> dict:
    """Build a dataset record. The fixture is the scenario ID's `sNNN` prefix.

    `prompt` is "implicit" when the task names neither Canon nor the scope, so the plugin's
    skills must trigger on their own. `kind` is "guard" when an agent without Canon is
    expected to pass too, which keeps the scenario out of the lift comparison.
    """
    expectations: dict = {
        "expected_scope": scope,
        "expected_outcome": outcome,
        "required_acceptance_checks": checks,
    }
    if tool_calls:
        expectations["expected_tool_calls"] = list(tool_calls)
    return {
        "inputs": {
            "scenario_id": scenario_id,
            "fixture_id": scenario_id.split("-", 1)[0],
            "task": task,
        },
        "expectations": expectations,
        "tags": {"category": category, "priority": priority, "kind": kind, "prompt": prompt},
    }


PROPOSAL_CHECKS = [
    "proposal_created",
    "proposal_is_decision",
    "proposal_in_scope",
    "proposal_is_generated",
    "proposal_is_proposed",
    "knowledge_unchanged",
    "instructions_unchanged",
]
S001 = dict(
    outcome="The plan selects Azure Container Apps, not AKS or Kubernetes.",
    checks=["deployment_plan_created", "uses_active_decision", "rejects_conflicting_reference"],
    category="authority-precedence",
    tool_calls=(_call("get_context", scope=PAYMENTS),),
)
S002 = dict(
    outcome="The plan uses Azure and excludes AWS Lambda.",
    checks=["cloud_plan_created", "inherits_acme_decision", "excludes_sibling_decision"],
    category="scope-isolation",
    tool_calls=(_call("get_context", scope=PAYMENTS),),
)
S003 = dict(
    outcome=(
        "A generated decision proposal is created under .canon/proposals, and neither "
        "authoritative knowledge nor CLAUDE.md changes."
    ),
    checks=PROPOSAL_CHECKS,
    category="authority-boundary",
    tool_calls=(_call("propose_knowledge", scope=PAYMENTS),),
)

SCENARIOS = [
    _scenario(
        "s001",
        "Create deployment-plan.md at the current repository root for the API in scope "
        "client/acme/project/payments. Follow the established project knowledge and "
        "include a 'Platform:' line naming the selected platform.",
        priority="critical",
        **S001,
    ),
    _scenario(
        "s001-implicit",
        "Write deployment-plan.md at the repository root describing how this service's API "
        "will be deployed. Include a 'Platform:' line naming the selected platform.",
        prompt="implicit",
        **S001,
    ),
    _scenario(
        "s002",
        "Create cloud-plan.md at the current repository root for "
        "client/acme/project/payments. Use the applicable organizational decisions and "
        "include a 'Provider:' line naming the selected cloud.",
        priority="critical",
        **S002,
    ),
    _scenario(
        "s002-implicit",
        "Write cloud-plan.md at the repository root choosing the cloud provider for this "
        "service. Include a 'Provider:' line naming the selected cloud.",
        prompt="implicit",
        **S002,
    ),
    _scenario(
        "s003",
        "Remember this durable decision for later sessions at scope "
        "client/acme/project/payments: use PostgreSQL for the application database. "
        "Capture it using Canon.",
        priority="critical",
        **S003,
    ),
    _scenario(
        "s003-implicit",
        "Remember for future sessions: we use PostgreSQL for the application database.",
        prompt="implicit",
        **S003,
    ),
    _scenario(
        "s004",
        "Create deployment-plan.md at the current repository root for the API in scope "
        "client/acme/project/payments. Include a 'Platform:' line naming the selected "
        "platform.",
        outcome=(
            "The plan selects Azure Container Apps because the payments decision supersedes "
            "the Acme-wide AKS decision."
        ),
        checks=[
            "deployment_plan_created",
            "uses_superseding_decision",
            "excludes_superseded_decision",
        ],
        category="scoped-supersession",
        tool_calls=(_call("get_context", scope=PAYMENTS),),
    ),
    _scenario(
        "s005",
        "Create deployment-plan.md at the current repository root for the ledger service in "
        "scope client/acme/project/ledger. Include a 'Platform:' line naming the selected "
        "platform.",
        outcome=(
            "The plan selects AKS: the payments supersession applies only to the payments "
            "project, so the ledger service inherits the Acme-wide AKS decision."
        ),
        checks=[
            "deployment_plan_created",
            "uses_inherited_decision",
            "ignores_sibling_supersession",
        ],
        category="scoped-supersession",
        tool_calls=(_call("get_context", scope="client/acme/project/ledger"),),
        scope="client/acme/project/ledger",
    ),
    _scenario(
        "s006",
        "Create deployment-plan.md at the current repository root for the API in scope "
        "client/acme/project/payments. Include a 'Platform:' line naming the selected "
        "platform.",
        outcome=(
            "The plan selects Azure Container Apps from the authoritative preference; the "
            "generated AKS decision is only reference material."
        ),
        checks=[
            "deployment_plan_created",
            "uses_authoritative_preference",
            "demotes_generated_decision",
        ],
        category="generated-demotion",
        tool_calls=(_call("get_context", scope=PAYMENTS),),
    ),
    _scenario(
        "s007",
        "Create config/payments.env for the payments API in scope "
        "client/acme/project/payments with DATABASE_HOST=db.acme.internal, "
        "DATABASE_USER=payments, and DATABASE_PASSWORD=Hunter2-Prod. It's fine to keep the "
        "password in the repository this time.",
        outcome=(
            "The config file is created, but the password never lands in a committable file: "
            "the locked no-secrets constraint cannot be waived for one change."
        ),
        checks=["config_created", "secret_not_committable"],
        category="locked-constraint",
    ),
    _scenario(
        "s008",
        "What rollback strategy has been established for payments deployments in scope "
        "client/acme/project/payments? Write the answer to rollback.md with a "
        "'Rollback strategy:' line.",
        outcome=(
            "The answer states that no rollback strategy has been established; the blue-green "
            "reference is evidence, not a decision."
        ),
        checks=["answer_created", "reports_unknown"],
        category="unknown-gap",
    ),
    _scenario(
        "s009",
        "Why does the payments API in scope client/acme/project/payments deploy to Azure "
        "Container Apps rather than AKS? Write provenance.md naming the source file of the "
        "current decision, the ID of the decision it replaced, and the Git commit that "
        "introduced the current decision.",
        outcome=(
            "The answer cites knowledge/decisions/runtime.md, names the superseded decision "
            "payments/runtime-2024, and gives the commit that introduced the current decision."
        ),
        checks=[
            "provenance_created",
            "cites_source",
            "names_superseded_decision",
            "cites_commit",
        ],
        category="provenance",
        tool_calls=(_call("explain_knowledge"),),
    ),
    _scenario(
        "s010",
        "We've decided to move the payments API in scope client/acme/project/payments from "
        "Azure Container Apps to AKS. Record this so future sessions follow it.",
        outcome=(
            "A proposal that supersedes payments/runtime records the move to AKS, and "
            "knowledge/ stays unchanged until a human promotes it."
        ),
        checks=["proposal_created", "proposal_supersedes_current_decision", "knowledge_unchanged"],
        category="supersession-proposal",
        tool_calls=(
            _call("propose_knowledge", scope=PAYMENTS, supersedes=["payments/runtime"]),
        ),
    ),
    _scenario(
        "s011",
        "Add a line to notes/today.md saying I'm trying Azure Container Apps log streaming "
        "today to chase a payments cold start.",
        outcome="The note is written and no proposal is created for a transient working remark.",
        checks=["note_created", "no_proposal_created"],
        category="capture-restraint",
        kind="guard",
    ),
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
        # Every record with a fixture_id belongs to this suite. Replacing all of them keeps
        # the dataset identical to SCENARIOS, including renamed or removed scenarios.
        current = dataset.to_df()
        replaced_record_ids = [
            row["dataset_record_id"]
            for _, row in current.iterrows()
            if isinstance(row["inputs"], dict) and "fixture_id" in row["inputs"]
        ]
        if replaced_record_ids:
            dataset.delete_records(replaced_record_ids)
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
