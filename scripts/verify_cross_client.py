"""Verify a confirmed write and read through independent MCP processes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER = PROJECT_ROOT / "mcp_server.py"
VERIFICATION_CONTENT = (
    "Cerebras Memory cross-client verification completed on 2026-07-20; "
    "this explicitly confirmed test memory should retain one stable citation."
)


async def _call(tool: str, arguments: dict[str, Any]):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        cwd=str(PROJECT_ROOT),
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await session.call_tool(tool, arguments)


async def verify(confirmed: bool) -> dict[str, Any]:
    if not confirmed:
        raise PermissionError("Pass --confirmed only after the user explicitly approves the test memory")
    saved = await _call(
        "kb_save_memory",
        {
            "title": "Cross-client verification",
            "content": VERIFICATION_CONTENT,
            "tags": ["verification", "mcp"],
            "project": "cerebras-memory",
            "confirmed_by_user": True,
        },
    )
    if saved.isError or not saved.structuredContent:
        raise RuntimeError("The first MCP process could not save the verification memory")
    document_id = saved.structuredContent["document_id"]

    searched = await _call(
        "kb_search",
        {"query": "cross-client verification stable citation", "sources": ["memory"]},
    )
    if searched.isError or not searched.structuredContent:
        raise RuntimeError("The second MCP process could not search the verification memory")
    match = next(
        (
            item
            for item in searched.structuredContent.get("results", [])
            if item.get("document_id") == document_id
        ),
        None,
    )
    if not match:
        raise RuntimeError("The second MCP process did not observe the first process write")
    fetched = await _call("kb_get", {"document_id": document_id})
    if fetched.isError or not fetched.structuredContent or not fetched.structuredContent.get("found"):
        raise RuntimeError("The third MCP process could not page the saved document")
    return {
        "ok": True,
        "document_id": document_id,
        "save_status": saved.structuredContent["status"],
        "citation": match["citation"],
        "search_source": match["source"],
        "get_found": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmed", action="store_true")
    args = parser.parse_args()
    try:
        result = asyncio.run(verify(args.confirmed))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
