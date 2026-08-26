"""Authenticated HTTP entry point for the shared knowledge base.

``mcp_server.py`` stays the STDIO server and is imported here unchanged. This
module adds a second way to reach the *same* FastMCP instance; it does not alter
the first, and the STDIO path must keep working with this file deleted.

The listener binds loopback only. Reaching it from another machine is the job of
the overlay network -- ``tailscale serve`` terminates TLS with a real ts.net
certificate and admits only tailnet peers -- so nothing here is exposed to the
LAN and no firewall rule is needed. Never publish it to the open internet.

Deliberately not built on ``FastMCP.streamable_http_app()``: that helper
constructs the session manager without ``session_idle_timeout``, so sessions
orphaned by a laptop going to sleep leak a task each forever, and it appends
``@mcp.custom_route`` routes outside its auth wrapper. Both matter here.
"""

from __future__ import annotations

import os

# Bound native thread pools before NumPy/ONNX is imported. ``mcp_server`` sets
# these too, but with ``setdefault`` and with a per-client STDIO worker in mind:
# a small cap is right when several idle workers must share one machine. The hub
# is the opposite case -- one process, serving everybody -- so it claims a larger
# share here, before that import runs.
for _name, _value in {
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_WAIT_POLICY": "PASSIVE",
}.items():
    os.environ.setdefault(_name, _value)

# The listener never distills, so it never needs an outbound API key. Set before
# config is imported so ``_load_secret_env`` skips the file entirely and the one
# process reachable from other machines cannot make an outbound call at all.
os.environ.setdefault("CEREBRAS_MEMORY_NO_SECRETS", "1")

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Awaitable, Callable, Iterable

import anyio.to_thread
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from config import load_settings
from mcp_server import _store, mcp
from store import SCHEMA_VERSION

DEFAULT_PORT = 8791
DEFAULT_SESSION_IDLE_TIMEOUT = 1800.0
MCP_PATH = "/mcp"
HEALTH_PATH = "/healthz"

# Only tools that cannot mutate the corpus are reachable with a read-only token.
# Anything not named here is treated as a write, so a tool added later is denied
# to read-only clients until it is deliberately classified.
READ_ONLY_TOOLS = frozenset({"kb_search", "kb_get", "kb_stats"})
UNREADABLE_TOOL = "<unreadable>"


@dataclass(frozen=True)
class ClientIdentity:
    """One authenticated caller: which machine, and what it may do."""

    label: str
    scope: str
    fingerprint: str

    @property
    def may_write(self) -> bool:
        return self.scope == "rw"


def load_client_tokens(raw: str | None = None) -> dict[str, ClientIdentity]:
    """Parse ``CEREBRAS_MEMORY_HTTP_TOKENS`` into digest -> identity.

    Entries are ``label:scope:secret``, separated by commas or newlines. One
    token per client rather than one shared secret, so access can be attributed
    to a machine in the audit and revoked for that machine alone.

    Keying by digest is what makes verification constant-time with respect to
    the secret: the comparison a dict lookup performs is over the SHA-256 of the
    presented value, and timing observations of that reveal nothing invertible.
    """

    value = raw if raw is not None else os.environ.get("CEREBRAS_MEMORY_HTTP_TOKENS", "")
    table: dict[str, ClientIdentity] = {}
    for entry in value.replace(",", chr(10)).split(chr(10)):
        entry = entry.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError("each client token must be label:scope:secret")
        label, scope, secret = (part.strip() for part in parts)
        if not label:
            raise ValueError("client token is missing a label")
        if scope not in {"ro", "rw"}:
            raise ValueError(f"client {label!r} has unknown scope {scope!r}; use ro or rw")
        if len(secret) < 32:
            raise ValueError(f"client {label!r} token is too short; use 32 characters or more")
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        table[digest] = ClientIdentity(label=label, scope=scope, fingerprint=digest[:12])
    return table


def authenticate(
    header: str | None, tokens: dict[str, ClientIdentity]
) -> ClientIdentity | None:
    if not header:
        return None
    prefix, _, presented = header.partition(" ")
    presented = presented.strip()
    if prefix.lower() != "bearer" or not presented:
        return None
    digest = hashlib.sha256(presented.encode("utf-8")).hexdigest()
    return tokens.get(digest)


def tool_from_jsonrpc(body: bytes) -> str | None:
    """Name the tool a JSON-RPC body invokes, if it invokes one.

    Returns None for anything that is not a ``tools/call`` -- initialize,
    notifications, ``tools/list`` -- and a sentinel when the call is one but its
    name cannot be read, so an unreadable call fails closed rather than being
    waved through as a non-call.
    """

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:  # noqa: BLE001 - a malformed body invokes nothing
        return None
    messages = payload if isinstance(payload, list) else [payload]
    for message in messages:
        if not isinstance(message, dict) or message.get("method") != "tools/call":
            continue
        params = message.get("params")
        if isinstance(params, dict) and isinstance(params.get("name"), str):
            return params["name"]
        return UNREADABLE_TOOL
    return None


async def _drain(receive: Callable[[], Awaitable[dict]]) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _replay(
    body: bytes, original: Callable[[], Awaitable[dict]]
) -> Callable[[], Awaitable[dict]]:
    """Hand the buffered body to the app, then get out of the way.

    Every later read must fall through to the real channel rather than
    synthesizing a disconnect. An MCP POST is answered with a long-lived SSE
    stream, and the server polls receive() to notice the client going away: a
    fabricated http.disconnect reads as exactly that, so the response is torn
    down while the client is still waiting for it, and the call hangs.
    """

    delivered = False

    async def receive() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original()

    return receive


async def _send_json(
    send: Callable[[dict], Awaitable[None]], status: int, payload: dict
) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-content-type-options", b"nosniff"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BearerAuthMiddleware:
    """Authenticate, authorize, and audit every request before it reaches MCP.

    Pure ASGI rather than ``BaseHTTPMiddleware`` because the MCP response is a
    long-lived SSE stream and must not be buffered. Only the *request* body is
    read, and only for POST -- those are small JSON-RPC messages, never uploads.
    """

    def __init__(
        self,
        app,
        *,
        tokens: dict[str, ClientIdentity],
        allowed_hosts: Iterable[str] = (),
        audit: bool = True,
    ) -> None:
        self.app = app
        self.tokens = tokens
        self.allowed_hosts = {host.casefold() for host in allowed_hosts}
        self.audit = audit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Liveness must be answerable without a credential so the overlay and
        # any monitor can probe it. It reports no counts and no corpus detail.
        if scope.get("path") == HEALTH_PATH:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)

        # TransportSecuritySettings guards the MCP mount; this covers every
        # other route with the same rule so nothing is left unprotected.
        if self.allowed_hosts:
            host = headers.get("host", "").split(",")[0].strip().casefold()
            if host not in self.allowed_hosts:
                await _send_json(send, 421, {"error": "misdirected_request"})
                return

        identity = authenticate(headers.get("authorization"), self.tokens)
        if identity is None:
            await _send_json(send, 401, {"error": "unauthorized"})
            return

        tool: str | None = None
        if scope.get("method") == "POST":
            body = await _drain(receive)
            tool = tool_from_jsonrpc(body)
            if tool is not None and tool not in READ_ONLY_TOOLS and not identity.may_write:
                await _send_json(send, 403, {"error": "read_only_token", "tool": tool})
                return
            receive = _replay(body, receive)

        started = time.perf_counter()
        observed = {"status": 0}

        async def watched_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                observed["status"] = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, watched_send)
        finally:
            if self.audit:
                await self._record(identity, tool, observed["status"], started)

    async def _record(
        self,
        identity: ClientIdentity,
        tool: str | None,
        status: int,
        started: float,
    ) -> None:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        healthy = 200 <= status < 400
        try:
            await anyio.to_thread.run_sync(
                lambda: _store().record_access(
                    client_label=identity.label,
                    token_fingerprint=identity.fingerprint,
                    transport="mcp_http",
                    # A request that calls no tool is still worth recording: it
                    # is how a session opens, and it attributes the connection.
                    tool=tool or "session",
                    status="ok" if healthy else "error",
                    latency_ms=latency_ms,
                    error_code=None if healthy else str(status),
                )
            )
        except Exception:  # noqa: BLE001 - auditing must not break serving
            pass


async def health(request) -> JSONResponse:
    return JSONResponse({"ok": True, "schema_version": SCHEMA_VERSION})


def build_app(
    *,
    tokens: dict[str, ClientIdentity],
    allowed_hosts: Iterable[str] = (),
    allowed_origins: Iterable[str] = (),
    session_idle_timeout: float = DEFAULT_SESSION_IDLE_TIMEOUT,
    prewarm: bool = True,
    audit: bool = True,
) -> Starlette:
    """Build the ASGI app serving the same FastMCP instance as STDIO."""

    hosts = list(allowed_hosts)
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=bool(hosts),
        allowed_hosts=hosts,
        allowed_origins=list(allowed_origins),
    )
    manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=False,
        # Stateful deliberately. In stateless mode a bare tools/call POST builds
        # a transport with no preceding initialize, so the session never records
        # client capabilities, check_client_capability always returns False, and
        # kb_search silently receives no client roots -- disabling project
        # scoping for every caller. list_roots is a server-to-client request and
        # cannot complete across separate transports either. Sleeping laptops are
        # handled by the idle timeout below plus the spec 404-then-reinitialize.
        stateless=False,
        security_settings=security,
        session_idle_timeout=session_idle_timeout,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        # Starlette does not propagate lifespan into a mounted sub-app, and the
        # session manager's task group is started here rather than by FastMCP,
        # so running it is this app's responsibility. Without it every request
        # fails with "Task group is not initialized".
        async with manager.run():
            if prewarm:
                app.state.prewarm = await anyio.to_thread.run_sync(_store().prewarm)
            yield

    return Starlette(
        routes=[
            Route(HEALTH_PATH, health, methods=["GET"]),
            Route(MCP_PATH, endpoint=StreamableHTTPASGIApp(manager)),
        ],
        middleware=[
            Middleware(
                BearerAuthMiddleware,
                tokens=tokens,
                allowed_hosts=hosts,
                audit=audit,
            )
        ],
        lifespan=lifespan,
    )


def guard_working_directory(projects_root: Path) -> None:
    """Refuse to serve from inside the corpus root.

    ``resolve_project_scope`` falls back to the process working directory when a
    client's roots resolve to nothing. For a long-lived server that would scope
    other people's searches to whatever project it was started in and report it
    as ``process_cwd``. Remote callers pass an explicit no-cwd sentinel, but this
    closes the case where the hub is simply launched from the wrong place.
    """

    try:
        cwd = Path.cwd().resolve()
        root = Path(projects_root).resolve()
    except OSError:
        return
    if cwd == root or root in cwd.parents:
        raise SystemExit(
            f"Refusing to start: working directory {cwd} is inside projects_root {root}. "
            "Start the hub from outside the corpus so client searches cannot inherit its scope."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the knowledge base over authenticated HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        dest="allowed_hosts",
        help="Host header to accept, e.g. matrix.tailnet.ts.net (repeatable)",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        dest="allowed_origins",
        help="Origin header to accept (repeatable)",
    )
    parser.add_argument(
        "--session-idle-timeout", type=float, default=DEFAULT_SESSION_IDLE_TIMEOUT
    )
    parser.add_argument("--no-prewarm", action="store_true")
    args = parser.parse_args(argv)

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error(
            "refusing to bind a routable interface; keep the listener on loopback "
            "and publish it with tailscale serve"
        )

    # Settings first: loading them is also what reads CEREBRAS_MEMORY_HTTP_TOKENS
    # out of .env. The provider API keys stay withheld either way.
    settings = load_settings()
    guard_working_directory(settings.projects_root)

    tokens = load_client_tokens()
    if not tokens:
        parser.error(
            "no client tokens configured; set CEREBRAS_MEMORY_HTTP_TOKENS in .env or the "
            "environment to one or more label:scope:secret entries before starting the hub"
        )

    hosts = list(args.allowed_hosts)
    if not hosts:
        hosts = [f"127.0.0.1:{args.port}", f"localhost:{args.port}"]

    import uvicorn

    uvicorn.run(
        build_app(
            tokens=tokens,
            allowed_hosts=hosts,
            allowed_origins=args.allowed_origins,
            session_idle_timeout=args.session_idle_timeout,
            prewarm=not args.no_prewarm,
        ),
        host=args.host,
        port=args.port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
