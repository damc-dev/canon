import sqlite3
import tempfile
import unittest
from pathlib import Path

from server.canon_core import (
    INDEX_SCHEMA_VERSION,
    effective_documents,
    get_context,
    index_path,
    load_documents,
    propose_knowledge,
    rebuild_index,
    search_knowledge,
)


class CanonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_doc(self, relative: str, frontmatter: str, body: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\n{frontmatter.strip()}\n---\n\n{body}\n", encoding="utf-8")

    def test_context_classifies_documents(self) -> None:
        self.write_doc("knowledge/constraints/security.md", "id: no-secrets\ntype: constraint\nlocked: true", "# Credentials\n\nNever commit secrets.")
        self.write_doc("knowledge/decisions/runtime.md", "id: runtime\ntype: decision", "# Runtime\n\nUse App Service.")
        output = get_context(self.root, "secrets and runtime")
        self.assertIn("MUST", output)
        self.assertIn("Never commit secrets.", output)
        self.assertIn("DECIDED", output)
        self.assertIn("Use App Service.", output)

    def test_generated_content_is_demoted_to_reference(self) -> None:
        self.write_doc("knowledge/decisions/ai.md", "id: ai-choice\ntype: decision\ngenerated: true", "# AI choice\n\nUse Kubernetes.")
        document = load_documents(self.root)[0]
        self.assertEqual(document.effective_type, "reference")
        self.assertIn("REFERENCE", get_context(self.root, "Kubernetes"))

    def test_proposal_is_isolated_from_authoritative_index(self) -> None:
        result = propose_knowledge(self.root, title="Production approval", body="Require approval.")
        rebuild_index(self.root)
        self.assertFalse(result["authoritative"])
        self.assertTrue(Path(result["path"]).is_relative_to((self.root / ".canon" / "proposals").resolve()))
        self.assertEqual(search_knowledge(self.root, "approval"), [])

    def test_rebuild_creates_disposable_sqlite_index(self) -> None:
        self.write_doc("knowledge/reference/example.md", "id: example\ntype: reference", "# Example\n\nA deployment example.")
        result = rebuild_index(self.root)
        self.assertEqual(result["documents_indexed"], 1)
        self.assertTrue(index_path(self.root).is_file())

    def test_scope_inherits_all_ancestors(self) -> None:
        self.write_doc("knowledge/constraints/global.md", "id: global-rule\ntype: constraint\nscope: global", "# Global\n\nUse managed identity.")
        self.write_doc("knowledge/standards/client.md", "id: client-rule\ntype: standard\nscope: client/acme", "# Client\n\nUse client tags.")
        self.write_doc("knowledge/decisions/project.md", "id: project-rule\ntype: decision\nscope: client/acme/project/canon", "# Project\n\nUse Bicep.")
        rebuild_index(self.root)
        ids = {item.doc_id for item in effective_documents(self.root, "client/acme/project/canon/workload/api")}
        self.assertEqual(ids, {"global-rule", "client-rule", "project-rule"})

    def test_scope_excludes_siblings(self) -> None:
        self.write_doc("knowledge/decisions/acme.md", "id: acme\ntype: decision\nscope: client/acme", "# Acme\n\nUse Azure.")
        self.write_doc("knowledge/decisions/other.md", "id: other\ntype: decision\nscope: client/other", "# Other\n\nUse AWS.")
        rebuild_index(self.root)
        ids = {item.doc_id for item in effective_documents(self.root, "client/acme/project/canon")}
        self.assertEqual(ids, {"acme"})

    def test_supersession_is_scope_aware(self) -> None:
        self.write_doc("knowledge/decisions/runtime-v1.md", "id: runtime-v1\ntype: decision\nscope: global", "# Runtime v1\n\nUse App Service.")
        self.write_doc("knowledge/decisions/runtime-v2.md", "id: runtime-v2\ntype: decision\nscope: client/acme\nsupersedes: runtime-v1", "# Runtime v2\n\nUse Container Apps.")
        rebuild_index(self.root)
        acme_ids = {item.doc_id for item in effective_documents(self.root, "client/acme/project/canon")}
        other_ids = {item.doc_id for item in effective_documents(self.root, "client/other")}
        self.assertEqual(acme_ids, {"runtime-v2"})
        self.assertEqual(other_ids, {"runtime-v1"})

    def test_history_includes_superseded_and_generated_cannot_supersede(self) -> None:
        self.write_doc("knowledge/decisions/runtime-v1.md", "id: runtime-v1\ntype: decision", "# Runtime v1\n\nUse App Service.")
        self.write_doc("knowledge/decisions/runtime-v2.md", "id: runtime-v2\ntype: decision\nsupersedes: runtime-v1", "# Runtime v2\n\nUse Container Apps.")
        self.write_doc("knowledge/decisions/ai.md", "id: ai\ntype: decision\ngenerated: true\nsupersedes: runtime-v2", "# AI suggestion\n\nUse Kubernetes.")
        rebuild_index(self.root)
        active = {item.doc_id for item in effective_documents(self.root)}
        history = {item.doc_id for item in effective_documents(self.root, include_superseded=True)}
        self.assertEqual(active, {"runtime-v2", "ai"})
        self.assertEqual(history, {"runtime-v1", "runtime-v2", "ai"})

    def test_search_matches_singular_and_plural(self) -> None:
        self.write_doc(
            "knowledge/constraints/secrets.md",
            "id: no-secrets\ntype: constraint\nlocked: true",
            "# Secrets\n\nNever commit credentials, passwords, or other secrets to source control.",
        )
        self.write_doc("knowledge/decisions/branch.md", "id: branch\ntype: decision", "# Branching\n\nUse one branch per change.")
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "password")], ["no-secrets"])
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "branches")], ["branch"])
        output = get_context(self.root, "create config env file with database password", "client/acme/project/payments")
        self.assertIn("MUST", output)
        self.assertIn("Never commit credentials, passwords, or other secrets to source control. [locked]", output)

    def test_search_matches_verb_forms(self) -> None:
        self.write_doc("knowledge/decisions/deploy.md", "id: deploy\ntype: decision", "# Deploy\n\nDeploy through the release pipeline.")
        self.write_doc("knowledge/standards/tests.md", "id: tests\ntype: standard", "# Tests\n\nTests are required before merging.")
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "deploying")], ["deploy"])
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "deployed")], ["deploy"])
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "merge")], ["tests"])

    def test_search_ignores_stop_words(self) -> None:
        self.write_doc("knowledge/standards/style.md", "id: style\ntype: standard", "# Style\n\nPrefer tabs to spaces in the Makefile.")
        self.write_doc("knowledge/decisions/deploy.md", "id: deploy\ntype: decision", "# Deploy\n\nDeploy through a release pipeline.")
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "how to deploy the service")], ["deploy"])
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "the")], ["style"])

    def test_outdated_index_is_rebuilt(self) -> None:
        self.write_doc("knowledge/constraints/secrets.md", "id: no-secrets\ntype: constraint", "# Secrets\n\nNever commit passwords.")
        database = index_path(self.root)
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.executescript(
                """
                CREATE TABLE documents (
                    rowid INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, source_path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL, body TEXT NOT NULL, document_type TEXT NOT NULL,
                    effective_type TEXT NOT NULL, scope TEXT NOT NULL, status TEXT NOT NULL,
                    generated INTEGER NOT NULL, locked INTEGER NOT NULL, supersedes TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE documents_fts USING fts5(title, body);
                INSERT INTO documents VALUES
                    (1, 'no-secrets', 'knowledge/constraints/secrets.md', 'Secrets', 'Never commit passwords.',
                     'constraint', 'constraint', 'global', 'active', 0, 0, '');
                INSERT INTO documents_fts(rowid, title, body) VALUES (1, 'Secrets', 'Never commit passwords.');
                """
            )
        self.assertEqual([item["doc_id"] for item in search_knowledge(self.root, "password")], ["no-secrets"])
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], INDEX_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
