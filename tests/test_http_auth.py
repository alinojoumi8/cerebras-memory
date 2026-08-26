"""Authentication, authorization and transport hardening for the hub.

Every test here drives the ASGI app in-process, so nothing binds a socket.
``prewarm`` and ``audit`` are off throughout: both reach the real store through
``mcp_server._store()``, which resolves the production configuration, and an
auth test has no business opening the live database.
"""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest

from http_server import (
    HEALTH_PATH,
    MCP_PATH,
    READ_ONLY_TOOLS,
    UNREADABLE_TOOL,
    build_app,
    load_client_tokens,
    tool_from_jsonrpc,
)

WRITER_SECRET = "w" * 48
READER_SECRET = "r" * 48
HOST = "matrix.example.ts.net"

TOKENS = load_client_tokens(
    f"laptop:rw:{WRITER_SECRET},ci-github:ro:{READER_SECRET}"
)


def _app(**overrides):
    options = {
        "tokens": TOKENS,
        "allowed_hosts": [HOST],
        "allowed_origins": [f"https://{HOST}"],
        "prewarm": False,
        "audit": False,
    }
    options.update(overrides)
    return build_app(**options)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=f"https://{HOST}",
        headers={"host": HOST},
    )


def _call(name: str) -> bytes:
    return json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name}}
    ).encode("utf-8")


@contextlib.asynccontextmanager
async def _running(app):
    """Run the real lifespan so the MCP session manager task group exists.

    Starlette does not start a mounted sub-app for you, and neither does httpx
    ASGITransport, so without this every request that actually reaches MCP fails
    with "Task group is not initialized". Running it here means the auth tests
    exercise the same startup path the hub uses.
    """

    async with app.router.lifespan_context(app):
        yield


async def _post(app, body: bytes, **headers) -> httpx.Response:
    async with _running(app), _client(app) as client:
        return await client.post(
            MCP_PATH,
            content=body,
            headers={"content-type": "application/json", **headers},
        )


@pytest.mark.anyio
async def test_a_request_without_a_token_is_rejected():
    response = await _post(_app(), _call("kb_search"))

    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


@pytest.mark.anyio
async def test_a_non_bearer_scheme_is_rejected():
    response = await _post(_app(), _call("kb_search"), authorization="Basic abc123")

    assert response.status_code == 401


@pytest.mark.anyio
async def test_a_wrong_token_of_the_same_length_is_rejected():
    """Same length as a real one, so rejection cannot be a length check."""

    response = await _post(
        _app(), _call("kb_search"), authorization=f"Bearer {'x' * len(WRITER_SECRET)}"
    )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_a_rejection_never_echoes_the_presented_token():
    presented = "z" * len(WRITER_SECRET)

    response = await _post(_app(), _call("kb_search"), authorization=f"Bearer {presented}")

    assert presented not in response.text
    for secret in (WRITER_SECRET, READER_SECRET):
        assert secret not in response.text


@pytest.mark.anyio
async def test_an_unexpected_host_header_is_refused():
    """DNS rebinding: a browser tricked into resolving a name to loopback."""

    app = _app()
    async with _running(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://evil.example"
    ) as client:
        response = await client.post(
            MCP_PATH,
            content=_call("kb_search"),
            headers={
                "content-type": "application/json",
                "host": "evil.example",
                "authorization": f"Bearer {WRITER_SECRET}",
            },
        )

    assert response.status_code == 421


@pytest.mark.anyio
async def test_health_is_reachable_without_a_token():
    app = _app()
    async with _running(app), _client(app) as client:
        response = await client.get(HEALTH_PATH)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    # Liveness only. It must not leak corpus size or contents.
    assert set(payload) == {"ok", "schema_version"}


@pytest.mark.anyio
async def test_a_read_only_token_cannot_call_a_write_tool():
    response = await _post(
        _app(), _call("kb_save_memory"), authorization=f"Bearer {READER_SECRET}"
    )

    assert response.status_code == 403
    assert response.json()["error"] == "read_only_token"


@pytest.mark.anyio
async def test_a_read_only_token_is_refused_a_tool_it_cannot_identify():
    """Fail closed: an unnamed tools/call must not slip past as a non-call."""

    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}}
    ).encode("utf-8")

    response = await _post(_app(), body, authorization=f"Bearer {READER_SECRET}")

    assert response.status_code == 403
    assert response.json()["tool"] == UNREADABLE_TOOL


@pytest.mark.anyio
async def test_a_read_only_token_is_refused_an_unclassified_tool():
    """A tool added later is denied until it is deliberately classified."""

    response = await _post(
        _app(), _call("kb_delete_everything"), authorization=f"Bearer {READER_SECRET}"
    )

    assert response.status_code == 403


@pytest.mark.anyio
@pytest.mark.parametrize("tool", sorted(READ_ONLY_TOOLS))
async def test_a_read_only_token_passes_authorization_for_read_tools(tool):
    """Authorization only. 400 comes from the MCP layer, which is the point:
    the request got past the middleware rather than being refused as 403."""

    response = await _post(_app(), _call(tool), authorization=f"Bearer {READER_SECRET}")

    assert response.status_code != 403
    assert response.status_code != 401


def test_tokens_must_declare_a_known_scope():
    with pytest.raises(ValueError, match="unknown scope"):
        load_client_tokens(f"laptop:admin:{WRITER_SECRET}")


def test_tokens_must_be_long_enough_to_be_worth_having():
    with pytest.raises(ValueError, match="too short"):
        load_client_tokens("laptop:rw:short")


def test_token_entries_must_be_well_formed():
    with pytest.raises(ValueError, match="label:scope:secret"):
        load_client_tokens(f"laptop:{WRITER_SECRET}")


def test_comments_and_blank_entries_are_ignored():
    table = load_client_tokens(
        f"# a comment{chr(10)}{chr(10)}laptop:rw:{WRITER_SECRET}{chr(10)}"
    )

    assert [identity.label for identity in table.values()] == ["laptop"]


def test_each_client_gets_a_distinct_fingerprint():
    labels = {identity.label: identity.fingerprint for identity in TOKENS.values()}

    assert len(set(labels.values())) == 2, "fingerprints must identify a client"
    assert all(len(value) == 12 for value in labels.values())


def test_a_body_that_calls_no_tool_is_not_treated_as_one():
    assert tool_from_jsonrpc(json.dumps({"method": "tools/list"}).encode()) is None
    assert tool_from_jsonrpc(json.dumps({"method": "initialize"}).encode()) is None
    assert tool_from_jsonrpc(b"not json at all") is None


def test_a_batched_write_call_is_still_detected():
    body = json.dumps(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "kb_save_memory"},
            },
        ]
    ).encode("utf-8")

    assert tool_from_jsonrpc(body) == "kb_save_memory"
