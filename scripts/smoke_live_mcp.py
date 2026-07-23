"""Read-only smoke test against the live Cerebras Memory STDIO server."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def exercise() -> dict[str, object]:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server.py")],
        cwd=str(PROJECT_ROOT),
        env=dict(os.environ),
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listed = await session.list_tools()
            tool_names = sorted(tool.name for tool in listed.tools)
            stats = await session.call_tool("kb_stats", {})
            search = await session.call_tool(
                "kb_search",
                {
                    "query": "Cerebras Memory DeepSeek distillation",
                    "limit": 3,
                    "global_search": True,
                },
            )
            results = search.structuredContent["results"]
            fetched = await session.call_tool(
                "kb_get",
                {"document_id": results[0]["document_id"], "limit": 1},
            )
            return {
                "tools": tool_names,
                "stats_error": bool(stats.isError),
                "search_error": bool(search.isError),
                "get_error": bool(fetched.isError),
                "documents": stats.structuredContent["documents"],
                "chunks": stats.structuredContent["chunks"],
                "distillation_search_enabled": stats.structuredContent["distillation"][
                    "search_enabled"
                ],
                "result_count": len(results),
                "distinct_document_ids": len({item["document_id"] for item in results}),
                "all_citations_stable": all(
                    str(item["citation"]).startswith("cerebras-memory://document/")
                    for item in results
                ),
                "scope": search.structuredContent["scope"],
                "get_found": fetched.structuredContent["found"],
            }


def main() -> int:
    report = asyncio.run(exercise())
    print(json.dumps(report, indent=2, sort_keys=True))
    expected_tools = {"kb_search", "kb_get", "kb_save_memory", "kb_stats"}
    passed = (
        set(report["tools"]) == expected_tools
        and not report["stats_error"]
        and not report["search_error"]
        and not report["get_error"]
        and report["distillation_search_enabled"] is True
        and report["result_count"] == report["distinct_document_ids"]
        and report["all_citations_stable"] is True
        and report["get_found"] is True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
