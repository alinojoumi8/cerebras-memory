from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from embeddings import HashingEmbedder
from importers.projects import scan_projects
from store import KnowledgeStore, stable_document_id


def test_project_allowlist_and_exclusions(settings_factory, tmp_path):
    root = tmp_path / "projects"
    repo = root / "alpha"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("Allowed documentation", encoding="utf-8")
    (repo / "guide.mdx").write_text("Allowed MDX", encoding="utf-8")
    (repo / "notes.txt").write_text("Allowed notes", encoding="utf-8")
    (repo / "code.py").write_text("print('excluded')", encoding="utf-8")
    (repo / "credentials.txt").write_text("password=should-not-enter", encoding="utf-8")
    (repo / "binary.txt").write_bytes(b"hello\x00world")
    (repo / "oversized.md").write_text("x" * 1025, encoding="utf-8")
    generated = repo / "node_modules"
    generated.mkdir()
    (generated / "package.md").write_text("generated dependency", encoding="utf-8")

    settings = settings_factory(projects_root=root, max_file_bytes=1024)
    result = scan_projects(settings)
    assert result.successful
    assert {Path(document.uri).name for document in result.documents} == {
        "README.md",
        "guide.mdx",
        "notes.txt",
    }
    assert {document.project for document in result.documents} == {"alpha"}


def test_stable_ids_updates_deletions_and_idempotent_refresh(settings_factory, tmp_path):
    root = tmp_path / "projects"
    repo = root / "alpha"
    repo.mkdir(parents=True)
    doc = repo / "README.md"
    doc.write_text("first version searchable", encoding="utf-8")
    settings = settings_factory(projects_root=root)
    store = KnowledgeStore(settings, HashingEmbedder(dimensions=settings.embedding_dimensions))

    first_scan = scan_projects(settings)
    first = store.upsert_document(first_scan.documents[0])
    assert first.status == "created"
    assert first.document_id == stable_document_id("projects", "alpha/readme.md")

    second = store.upsert_document(scan_projects(settings).documents[0])
    assert second.status == "unchanged"
    assert second.document_id == first.document_id

    doc.write_text("second version searchable", encoding="utf-8")
    updated = store.upsert_document(scan_projects(settings).documents[0])
    assert updated.status == "updated"
    assert updated.document_id == first.document_id

    doc.unlink()
    empty_scan = scan_projects(settings)
    assert store.reconcile_source("projects", empty_scan.seen_keys) == 1
    assert store.get_document(first.document_id) is None
