"""Project scoping for clients that are not on this machine.

Project inference resolved a client root by containment under ``projects_root``.
That is exact but machine-local: a laptop root can never sit under the hub's
``projects_root``, so every remote root resolved to nothing and the search then
silently inherited the *server process's* working directory.  These tests pin
the cross-machine behaviour and the removal of that fallback.
"""

from __future__ import annotations

from pathlib import Path

from store import NO_PROCESS_CWD


def _seed(store, *projects: str) -> None:
    for name in projects:
        store.save_memory(
            f"{name} note",
            f"content belonging to {name}",
            project=name,
            confirmed_by_user=True,
        )


def test_a_remote_client_root_resolves_by_directory_name(store):
    _seed(store, "Alpha")

    scope = store.resolve_project_scope(
        project=None, global_search=False, roots=[Path("D:/work/Alpha")]
    )

    assert scope == {"project": "Alpha", "origin": "client_root_leaf"}


def test_a_local_client_root_still_resolves_by_containment(store):
    """The canary suite asserts ``client_root``; leaf matching must not steal it."""

    _seed(store, "Alpha")
    root = store.settings.projects_root / "Alpha" / "src"

    scope = store.resolve_project_scope(project=None, global_search=False, roots=[root])

    assert scope == {"project": "Alpha", "origin": "client_root"}


def test_a_containment_match_outranks_a_leaf_match(store):
    _seed(store, "Alpha")

    scope = store.resolve_project_scope(
        project=None,
        global_search=False,
        roots=[store.settings.projects_root / "Alpha", Path("D:/work/Alpha")],
    )

    assert scope == {"project": "Alpha", "origin": "client_root"}


def test_client_roots_naming_different_projects_stay_unscoped(store):
    _seed(store, "Alpha", "Beta")

    scope = store.resolve_project_scope(
        project=None,
        global_search=False,
        roots=[Path("D:/work/Alpha"), Path("E:/other/Beta")],
    )

    assert scope == {"project": None, "origin": "global_ambiguous_roots"}


def test_leaf_matching_stops_before_reaching_distant_ancestors(store):
    """A client sending a home directory must not match a project far above it."""

    _seed(store, "Alpha")

    scope = store.resolve_project_scope(
        project=None, global_search=False, roots=[Path("D:/Alpha/a/b/c/d")]
    )

    assert scope == {"project": None, "origin": "global"}


def test_process_cwd_is_never_inherited_by_a_caller_that_has_none(store, monkeypatch):
    """The highest-consequence case: a remote search must not silently take the
    hub's own project scope and report it as ``process_cwd``."""

    _seed(store, "Alpha")
    project_dir = store.settings.projects_root / "Alpha"
    project_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project_dir)

    # Without the sentinel the process working directory really is inherited,
    # which is exactly the hazard for a long-lived server.
    inherited = store.resolve_project_scope(project=None, global_search=False)
    assert inherited == {"project": "Alpha", "origin": "process_cwd"}

    scoped = store.resolve_project_scope(
        project=None, global_search=False, cwd=NO_PROCESS_CWD
    )
    assert scoped == {"project": None, "origin": "global"}


def test_search_response_threads_the_sentinel_through(store, monkeypatch):
    _seed(store, "Alpha")
    project_dir = store.settings.projects_root / "Alpha"
    project_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project_dir)

    response = store.search_response("content", limit=5, cwd=NO_PROCESS_CWD)

    assert response["scope"] == {"project": None, "origin": "global"}


def test_a_remote_root_scopes_an_actual_search(store):
    """End to end: the leaf match must filter results, not just label them."""

    _seed(store, "Alpha", "Beta")

    response = store.search_response(
        "content belonging", limit=10, roots=[Path("D:/work/Alpha")]
    )

    assert response["scope"] == {"project": "Alpha", "origin": "client_root_leaf"}
    assert response["results"], "expected the Alpha memory to be retrievable"
    assert {item["project"] for item in response["results"]} == {"Alpha"}
