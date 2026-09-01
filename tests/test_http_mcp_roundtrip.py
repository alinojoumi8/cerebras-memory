"""End to end over the network: a real server process, a real MCP client.

Everything else about the hub is tested in-process. This is the one that proves
an agent on another machine can actually reach the knowledge base: a uvicorn
process on a loopback port, driven by the SDK streamable-HTTP client with a
bearer token, exercising the same four tools STDIO exposes.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET = "t" * 48
READ_SECRET = "s" * 48
STARTUP_TIMEOUT = 45.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "hub.sqlite3"),
                "model_cache_dir": str(tmp_path / "models"),
                "embedding_model": "test/hash-v1",
                "embedding_dimensions": 32,
                "projects_root": str(tmp_path / "projects"),
                "reranker": {"enabled": False},
                "sources": {
                    "projects": True,
                    "hermes": False,
                    "claude": False,
                    "codex": False,
                    "grok": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def hub(tmp_path: Path):
    """Start the real hub on loopback and tear it down afterwards."""

    port = _free_port()
    config_path = _write_config(tmp_path)

    env = dict(os.environ)
    env["CEREBRAS_MEMORY_CONFIG"] = str(config_path)
    env["CEREBRAS_MEMORY_TEST_EMBEDDER"] = "1"
    env["CEREBRAS_MEMORY_HTTP_TOKENS"] = f"laptop:rw:{SECRET},ci:ro:{READ_SECRET}"

    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "http_server.py"), "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"hub exited early:{chr(10)}{process.stdout.read()}")
            try:
                response = httpx.get(f"{base}/healthz", timeout=2.0)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError("hub did not become healthy in time")
            time.sleep(0.2)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def _run(coro):
    return asyncio.run(coro)


def test_health_reports_ready_without_a_token(hub):
    response = httpx.get(f"{hub}/healthz", timeout=10.0)

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_an_unauthenticated_client_cannot_open_a_session(hub):
    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp") as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

    with pytest.raises(Exception):
        _run(exercise())


def test_a_wrong_token_cannot_open_a_session(hub):
    headers = {"Authorization": f"Bearer {'x' * len(SECRET)}"}

    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp", headers=headers) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

    with pytest.raises(Exception):
        _run(exercise())


def test_a_remote_agent_can_save_search_and_page_a_memory(hub):
    """The whole point: write from a network client, read it back."""

    headers = {"Authorization": f"Bearer {SECRET}"}

    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp", headers=headers) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == {
                    "kb_search",
                    "kb_get",
                    "kb_save_memory",
                    "kb_stats",
                }

                rejected = await session.call_tool(
                    "kb_save_memory",
                    {
                        "title": "Rejected",
                        "content": "not confirmed",
                        "confirmed_by_user": False,
                    },
                )
                assert rejected.isError

                saved = await session.call_tool(
                    "kb_save_memory",
                    {
                        "title": "Tailnet note",
                        "content": "the hub binds loopback and is published over the tailnet",
                        "tags": ["integration"],
                        "confirmed_by_user": True,
                    },
                )
                assert not saved.isError
                document_id = saved.structuredContent["document_id"]

                found = await session.call_tool(
                    "kb_search", {"query": "tailnet loopback", "global_search": True}
                )
                assert not found.isError
                assert document_id in {
                    item["document_id"] for item in found.structuredContent["results"]
                }

                # A network request has no working directory, so it must never
                # come back scoped to whatever the server was started in.
                assert found.structuredContent["scope"]["origin"] != "process_cwd"

                fetched = await session.call_tool("kb_get", {"document_id": document_id})
                assert fetched.structuredContent["found"] is True
                assert fetched.structuredContent["chunks"]

                stats = await session.call_tool("kb_stats", {})
                assert stats.structuredContent["schema_version"] == 6

    _run(exercise())


def test_a_read_only_token_can_search(hub):
    headers = {"Authorization": f"Bearer {READ_SECRET}"}

    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp", headers=headers) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                found = await session.call_tool(
                    "kb_search", {"query": "anything", "global_search": True}
                )
                assert not found.isError

    _run(exercise())


def test_a_read_only_token_cannot_save(hub):
    """The 403 is raised by the middleware before MCP ever sees the call.

    It surfaces when the client task group unwinds rather than at the call site,
    so the expectation belongs around the whole exchange.
    """

    headers = {"Authorization": f"Bearer {READ_SECRET}"}

    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp", headers=headers) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                await session.call_tool(
                    "kb_save_memory",
                    {
                        "title": "Should not persist",
                        "content": "a read-only client must not write",
                        "confirmed_by_user": True,
                    },
                )

    with pytest.raises(BaseException) as caught:
        _run(exercise())
    assert "403" in str(caught.value) or "403" in repr(caught.value)


def test_a_laptop_local_root_scopes_by_name_across_machines(hub, tmp_path):
    """A client root that cannot exist on the hub must still resolve.

    The root is a path only the calling machine has. Containment under the hub
    projects_root is impossible, so this passes only via leaf-name matching.
    """

    headers = {"Authorization": f"Bearer {SECRET}"}
    remote_root = Path("D:/laptop-workspace/Tailnet")

    async def list_roots(_context):
        return types.ListRootsResult(
            roots=[types.Root(uri=remote_root.as_uri(), name="laptop root")]
        )

    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp", headers=headers) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(
                reader, writer, list_roots_callback=list_roots
            ) as session:
                await session.initialize()

                await session.call_tool(
                    "kb_save_memory",
                    {
                        "title": "Scoped note",
                        "content": "belongs to the Tailnet project",
                        "project": "Tailnet",
                        "confirmed_by_user": True,
                    },
                )

                found = await session.call_tool("kb_search", {"query": "belongs"})
                assert not found.isError
                assert found.structuredContent["scope"] == {
                    "project": "Tailnet",
                    "origin": "client_root_leaf",
                }

    _run(exercise())


def test_access_is_audited_without_recording_the_query(hub, tmp_path):
    """The audit must attribute the caller and not keep what they searched for."""

    import sqlite3

    headers = {"Authorization": f"Bearer {SECRET}"}
    secret_phrase = "supercalifragilistic-marker"

    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp", headers=headers) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                await session.call_tool(
                    "kb_search", {"query": secret_phrase, "global_search": True}
                )

    _run(exercise())

    connection = sqlite3.connect(f"file:{tmp_path / 'hub.sqlite3'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute("SELECT * FROM access_audit")]
    connection.close()

    assert rows, "expected the hub to record network access"
    assert {row["client_label"] for row in rows} == {"laptop"}
    assert all(row["transport"] == "mcp_http" for row in rows)
    assert any(row["tool"] == "kb_search" for row in rows)
    # No column may contain what was actually searched for.
    assert secret_phrase not in json.dumps(rows)
    # Nor may it contain the credential.
    assert SECRET not in json.dumps(rows)


def test_rejected_requests_are_audited(hub, tmp_path):
    """A failed attempt is the security-interesting one and must be recorded."""

    import sqlite3

    wrong = "x" * 48
    httpx.post(
        f"{hub}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": f"Bearer {wrong}"},
        timeout=15.0,
    )
    httpx.post(
        f"{hub}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        timeout=15.0,
    )

    connection = sqlite3.connect(f"file:{tmp_path / 'hub.sqlite3'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM access_audit WHERE client_label = '<rejected>'"
        )
    ]
    connection.close()

    assert len(rows) == 2
    assert {row["error_code"] for row in rows} == {"401"}
    assert all(row["status"] == "error" for row in rows)
    # A presented credential and an absent one are distinguishable, and neither
    # is stored in a reversible form.
    fingerprints = {row["token_fingerprint"] for row in rows}
    assert "none" in fingerprints
    assert len(fingerprints) == 2
    assert wrong not in json.dumps(rows)


def test_a_search_records_a_query_hash_and_not_the_query(hub, tmp_path):
    import sqlite3

    headers = {"Authorization": f"Bearer {SECRET}"}
    phrase = "distinctive-audit-probe-phrase"

    async def exercise():
        async with streamablehttp_client(f"{hub}/mcp", headers=headers) as (
            reader,
            writer,
            _,
        ):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                await session.call_tool(
                    "kb_search", {"query": phrase, "global_search": True}
                )

    _run(exercise())

    connection = sqlite3.connect(f"file:{tmp_path / 'hub.sqlite3'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM access_audit WHERE tool = 'kb_search'"
        )
    ]
    connection.close()

    assert rows, "expected the search to be audited"
    hashes = {row["query_hash"] for row in rows}
    assert hashes and None not in hashes, "query_hash must be populated"
    assert all(len(value) == 32 for value in hashes)
    assert phrase not in json.dumps(rows)
