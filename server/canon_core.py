from __future__ import annotations

import re
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


AUTHORITATIVE_TYPES = {"constraint", "decision", "standard", "preference"}
TYPE_LABELS = {
    "constraint": "MUST",
    "decision": "DECIDED",
    "standard": "PREFER",
    "preference": "PREFER",
    "reference": "REFERENCE",
}
TYPE_ORDER = {"constraint": 0, "decision": 1, "standard": 2, "preference": 2, "reference": 3}

# Bump whenever the index schema or tokenizer changes so existing indexes are rebuilt.
INDEX_SCHEMA_VERSION = 2
# Function words dropped from search queries; they match nearly every document and carry no topic.
STOP_WORDS = frozenset(
    """
    a an and any are as at be been but by can could did do does for from had has have how i if in
    into is it its me my no not of on or our should so than that the their them then there these
    they this those to us was we were what when where which while who why will with would you your
    """.split()
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    title: str
    body: str
    document_type: str
    scope: str
    status: str
    generated: bool
    locked: bool
    supersedes: tuple[str, ...]

    @property
    def effective_type(self) -> str:
        if self.generated or self.document_type not in TYPE_LABELS:
            return "reference"
        return self.document_type

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supersedes"] = list(self.supersedes)
        value["effective_type"] = self.effective_type
        value["label"] = TYPE_LABELS[self.effective_type]
        return value


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "knowledge").is_dir() or (candidate / ".git").exists():
            return candidate
    return current


def state_dir(project_root: str | Path) -> Path:
    return Path(project_root) / ".canon"


def index_path(project_root: str | Path) -> Path:
    return state_dir(project_root) / "index.db"


def _scalar(raw: str) -> Any:
    value = raw.strip().strip('"').strip("'")
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return [_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    metadata: dict[str, Any] = {}
    active_list: str | None = None
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("-") and active_list:
            metadata.setdefault(active_list, []).append(_scalar(line.lstrip()[1:]))
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if not raw.strip():
            metadata[key] = []
            active_list = key
        else:
            metadata[key] = _scalar(raw)
            active_list = None
    body_start = end + len("\n---")
    return metadata, text[body_start:].lstrip("\n")


def _values(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def read_document(path: Path, project_root: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(text)
    relative = path.relative_to(project_root).as_posix()
    inferred_type = path.parent.name.rstrip("s")
    document_type = str(metadata.get("type", inferred_type)).lower()
    heading = next((line[2:].strip() for line in body.splitlines() if line.startswith("# ")), path.stem)
    return Document(
        doc_id=str(metadata.get("id", Path(relative).with_suffix("").as_posix())),
        source_path=relative,
        title=heading,
        body=body.strip(),
        document_type=document_type,
        scope=normalize_scope(str(metadata.get("scope", "global"))),
        status=str(metadata.get("status", "active")).lower(),
        generated=bool(metadata.get("generated", False)),
        locked=bool(metadata.get("locked", False)),
        supersedes=_values(metadata.get("supersedes")),
    )


def load_documents(project_root: str | Path) -> list[Document]:
    root = Path(project_root).resolve()
    knowledge = root / "knowledge"
    if not knowledge.is_dir():
        return []
    return [read_document(path, root) for path in sorted(knowledge.rglob("*.md")) if path.is_file()]


def normalize_scope(scope: str | None) -> str:
    value = (scope or "global").strip().strip("/")
    return value or "global"


def scope_applies(document_scope: str, requested_scope: str) -> bool:
    document_scope = normalize_scope(document_scope)
    requested_scope = normalize_scope(requested_scope)
    return document_scope == "global" or requested_scope == document_scope or requested_scope.startswith(document_scope + "/")


def scope_specificity(scope: str) -> int:
    normalized = normalize_scope(scope)
    return 0 if normalized == "global" else len(normalized.split("/"))


def rebuild_index(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    documents = load_documents(root)
    state_dir(root).mkdir(parents=True, exist_ok=True)
    database = index_path(root)
    if database.exists():
        database.unlink()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                rowid INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL,
                source_path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                document_type TEXT NOT NULL,
                effective_type TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                generated INTEGER NOT NULL,
                locked INTEGER NOT NULL,
                supersedes TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(title, body, tokenize='porter unicode61');
            """
        )
        connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
        for document in documents:
            cursor = connection.execute(
                """INSERT INTO documents
                (doc_id, source_path, title, body, document_type, effective_type, scope,
                 status, generated, locked, supersedes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.doc_id,
                    document.source_path,
                    document.title,
                    document.body,
                    document.document_type,
                    document.effective_type,
                    document.scope,
                    document.status,
                    int(document.generated),
                    int(document.locked),
                    "\n".join(document.supersedes),
                ),
            )
            connection.execute(
                "INSERT INTO documents_fts(rowid, title, body) VALUES (?, ?, ?)",
                (cursor.lastrowid, document.title, document.body),
            )
    return {"project_root": str(root), "index": str(database), "documents_indexed": len(documents)}


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        source_path=row["source_path"],
        title=row["title"],
        body=row["body"],
        document_type=row["document_type"],
        scope=row["scope"],
        status=row["status"],
        generated=bool(row["generated"]),
        locked=bool(row["locked"]),
        supersedes=tuple(item for item in row["supersedes"].splitlines() if item),
    )


def _index_is_current(database: Path) -> bool:
    if not database.exists():
        return False
    try:
        with sqlite3.connect(database) as connection:
            return connection.execute("PRAGMA user_version").fetchone()[0] == INDEX_SCHEMA_VERSION
    except sqlite3.DatabaseError:
        return False


def _ensure_index(project_root: Path) -> None:
    if not _index_is_current(index_path(project_root)):
        rebuild_index(project_root)


def _all_documents(project_root: Path) -> list[Document]:
    _ensure_index(project_root)
    with sqlite3.connect(index_path(project_root)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM documents").fetchall()
    return [_row_to_document(row) for row in rows]


def _query_terms(query: str) -> list[str]:
    terms = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", query.lower())))
    content_terms = [term for term in terms if term not in STOP_WORDS]
    # A query made only of stop words still searches for them rather than returning nothing.
    return content_terms or terms


def _fts_rowids(project_root: Path, query: str, limit: int) -> list[int]:
    terms = _query_terms(query)
    if not terms:
        return []
    # The porter tokenizer stems these quoted terms too, so "passwords" also matches "password".
    expression = " OR ".join(f'"{term}"' for term in terms)
    with sqlite3.connect(index_path(project_root)) as connection:
        rows = connection.execute(
            "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT ?",
            (expression, limit),
        ).fetchall()
    return [int(row[0]) for row in rows]


def _documents_by_rowid(project_root: Path, rowids: Iterable[int]) -> list[Document]:
    ids = list(rowids)
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with sqlite3.connect(index_path(project_root)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(f"SELECT * FROM documents WHERE rowid IN ({placeholders})", ids).fetchall()
    by_path = {_row_to_document(row).source_path: _row_to_document(row) for row in rows}
    return list(by_path.values())


def effective_documents(
    project_root: str | Path,
    scope: str = "global",
    *,
    candidates: Iterable[Document] | None = None,
    include_superseded: bool = False,
) -> list[Document]:
    root = Path(project_root).resolve()
    requested_scope = normalize_scope(scope)
    all_applicable = [
        document
        for document in _all_documents(root)
        if document.status == "active" and scope_applies(document.scope, requested_scope)
    ]
    suppressed: set[str] = set()
    if not include_superseded:
        for document in all_applicable:
            if not document.generated and document.effective_type in AUTHORITATIVE_TYPES:
                suppressed.update(document.supersedes)
    pool = list(candidates) if candidates is not None else all_applicable
    result = [
        document
        for document in pool
        if document.status == "active"
        and scope_applies(document.scope, requested_scope)
        and (include_superseded or document.doc_id not in suppressed)
    ]
    return sorted(
        result,
        key=lambda document: (
            TYPE_ORDER.get(document.effective_type, 3),
            -scope_specificity(document.scope),
            document.title.lower(),
        ),
    )


def search_knowledge(
    project_root: str | Path,
    query: str,
    scope: str = "global",
    *,
    include_superseded: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    _ensure_index(root)
    candidates = _documents_by_rowid(root, _fts_rowids(root, query, max(limit * 4, 40)))
    documents = effective_documents(
        root,
        scope,
        candidates=candidates,
        include_superseded=include_superseded,
    )
    return [document.to_dict() for document in documents[:limit]]


def get_context(project_root: str | Path, task: str, scope: str = "global", limit: int = 12) -> str:
    results = search_knowledge(project_root, task, scope, limit=limit)
    if not results:
        return "UNKNOWN\nNo applicable authoritative or reference knowledge was found for this task and scope."
    groups: dict[str, list[dict[str, Any]]] = {label: [] for label in ("MUST", "DECIDED", "PREFER", "REFERENCE")}
    for result in results:
        groups[result["label"]].append(result)
    sections: list[str] = []
    for label, items in groups.items():
        if not items:
            continue
        lines = [label]
        for item in items:
            summary = next((line.strip() for line in item["body"].splitlines() if line.strip() and not line.startswith("#")), item["title"])
            lock = " [locked]" if item["locked"] else ""
            lines.append(f"- {summary}{lock}\n  Source: {item['source_path']} · Scope: {item['scope']} · ID: {item['doc_id']}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def explain_knowledge(project_root: str | Path, identifier: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    documents = _all_documents(root)
    match = next((doc for doc in documents if doc.doc_id == identifier or doc.source_path == identifier), None)
    if not match:
        raise ValueError(f"No Canon document matches {identifier!r}")
    source = root / match.source_path
    history: dict[str, str] | None = None
    try:
        output = subprocess.run(
            ["git", "log", "-1", "--format=%H%n%an%n%aI%n%s", "--", str(source)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if output:
            commit, author, authored_at, subject = (output.splitlines() + ["", "", "", ""])[:4]
            history = {"commit": commit, "author": author, "authored_at": authored_at, "subject": subject}
    except OSError:
        history = None
    result = match.to_dict()
    result["absolute_path"] = str(source)
    result["git"] = history
    return result


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "proposal"


def propose_knowledge(
    project_root: str | Path,
    *,
    title: str,
    body: str,
    document_type: str = "decision",
    scope: str = "global",
    supersedes: Iterable[str] = (),
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    proposal_dir = state_dir(root) / "proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    proposal_id = f"proposal/{stamp}-{_slug(title)}"
    path = proposal_dir / f"{stamp}-{_slug(title)}.md"
    supersedes_values = [item for item in supersedes if item]
    supersedes_yaml = ""
    if supersedes_values:
        supersedes_yaml = "supersedes:\n" + "".join(f"  - {item}\n" for item in supersedes_values)
    content = (
        "---\n"
        f"id: {proposal_id}\n"
        f"type: {document_type}\n"
        f"scope: {normalize_scope(scope)}\n"
        "status: proposed\n"
        "generated: true\n"
        f"{supersedes_yaml}"
        "---\n\n"
        f"# {title}\n\n{body.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return {
        "status": "proposed",
        "authoritative": False,
        "path": str(path),
        "message": "Proposal created. Promote it only by creating or editing a human-owned Markdown file under knowledge/ and reviewing that Git change.",
    }
