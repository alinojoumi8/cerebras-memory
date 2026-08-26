"""Reconciliation guards, retention on unreadable files, and terminal blocks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from embeddings import HashingEmbedder
from importers.projects import _looks_secret, scan_projects
from models import IngestDocument
from store import ReconcileFloorNotMet, KnowledgeStore


def _document(index: int) -> IngestDocument:
    return IngestDocument(
        source="claude",
        source_key=f"session:{index}",
        title=f"Session {index}",
        text=f"USER [2026-07-01T00:00:00Z]\nbody {index}",
        timestamp=datetime.now(timezone.utc),
    )


def test_an_implausibly_small_scan_refuses_to_delete(settings_factory):
    """Several scan paths fail open.

    A truncated Hermes export still exits 0, and an upstream field rename makes
    every message look older than the cutoff. Either produces a near-empty scan
    that is indistinguishable from "the user deleted everything".
    """

    store = KnowledgeStore(settings_factory(), HashingEmbedder(32))
    for index in range(20):
        store.upsert_document(_document(index))

    with pytest.raises(ReconcileFloorNotMet, match="Refusing to reconcile"):
        store.reconcile_source("claude", {"session:0"})

    # Nothing was removed.
    assert store.stats(recover=False)["documents_by_source"]["claude"] == 20


def test_a_plausible_scan_still_reconciles(settings_factory):
    store = KnowledgeStore(settings_factory(), HashingEmbedder(32))
    for index in range(20):
        store.upsert_document(_document(index))

    surviving = {f"session:{index}" for index in range(18)}
    assert store.reconcile_source("claude", surviving) == 2
    assert store.stats(recover=False)["documents_by_source"]["claude"] == 18


def test_small_sources_are_not_blocked_by_the_floor(settings_factory):
    """The floor targets mass deletion, not ordinary cleanup."""

    store = KnowledgeStore(settings_factory(), HashingEmbedder(32))
    store.upsert_document(_document(0))
    assert store.reconcile_source("claude", set()) == 1


def test_deletion_manifests_record_a_discriminating_reason(settings_factory):
    store = KnowledgeStore(settings_factory(), HashingEmbedder(32))
    store.upsert_document(_document(0))
    store.reconcile_source("claude", set(), reason="size_exceeded")

    with store._connect() as connection:  # noqa: SLF001 - asserting stored audit
        reasons = [
            row["reason"]
            for row in connection.execute("SELECT reason FROM deletion_manifests")
        ]
    assert reasons == ["source_reconciliation:size_exceeded"]


def test_oversized_or_undecodable_files_retain_their_key(settings_factory, tmp_path):
    """A file that grows past the limit is not a deleted file.

    Dropping its key would discard the last good indexed version in favour of
    nothing at all.
    """

    settings = settings_factory(max_file_bytes=2048)
    root = settings.projects_root
    (root / "alpha").mkdir(parents=True)
    good = root / "alpha" / "keep.md"
    good.write_text("# Keep\n\nstill readable", encoding="utf-8")
    oversized = root / "alpha" / "huge.md"
    oversized.write_text("x" * 4096, encoding="utf-8")
    binary = root / "alpha" / "binary.md"
    binary.write_bytes(b"text\x00more")

    scan = scan_projects(settings)

    keys = {document.source_key for document in scan.documents}
    assert "alpha/keep.md" in keys
    assert "alpha/huge.md" not in keys
    # ...but both are retained, so reconciliation leaves them alone.
    assert {"alpha/huge.md", "alpha/binary.md"} <= scan.seen_keys


def test_secret_name_matching_is_word_based_not_substring():
    """``part in name`` silently dropped ordinary documentation."""

    for name in (".env", ".env.local", "credentials.json", "id_rsa", "auth.json"):
        assert _looks_secret(Path(name)) is True, name

    for name in (
        "tokenizer-notes.md",
        "password-policy.md",
        "environment.md",
        "use-case-tokens.txt",
        "token-bucket-design.md",
    ):
        assert _looks_secret(Path(name)) is False, name


def test_policy_blocked_units_are_terminal_not_pending(settings_factory):
    """Blocked units were recorded as 'pending', which reads as retryable work.

    The blocked-decision cache means they are never retried, so the pipeline
    could not converge and stats reported work that would never finish.
    """

    store = KnowledgeStore(settings_factory(), HashingEmbedder(32))
    with store._connect() as connection:  # noqa: SLF001 - schema-level assertion
        statuses = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'distillation_unit_state'"
        ).fetchone()[0]
    assert "policy_blocked" in statuses
    assert store.schema_version() == 5
