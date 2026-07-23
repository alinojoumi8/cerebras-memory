"""UTF-8 process streams for Windows CLI and MCP interoperability."""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    for stream, errors in (
        (sys.stdin, "strict"),
        (sys.stdout, "strict"),
        (sys.stderr, "backslashreplace"),
    ):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors=errors)
