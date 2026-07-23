from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from config import load_settings
from embeddings import HashingEmbedder
from models import IngestDocument
from store import KnowledgeStore


def test_stdio_server_tools_annotations_calls_and_cross_process_visibility(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    database = tmp_path / "mcp.sqlite3"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(database),
                "model_cache_dir": str(tmp_path / "models"),
                "embedding_model": "test/hash-v1",
                "embedding_dimensions": 32,
                "projects_root": str(tmp_path / "projects"),
                "sources": {"projects": True, "hermes": False, "claude": False, "codex": False, "grok": False},
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["CEREBRAS_MEMORY_CONFIG"] = str(config_path)
    env["CEREBRAS_MEMORY_TEST_EMBEDDER"] = "1"
    parameters = StdioServerParameters(
        command=os.sys.executable,
        args=[str(project_root / "mcp_server.py")],
        cwd=str(project_root),
        env=env,
    )

    async def exercise():
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = {tool.name: tool for tool in listed.tools}
                assert set(tools) == {"kb_search", "kb_get", "kb_save_memory", "kb_stats"}
                assert tools["kb_search"].annotations.readOnlyHint is True
                assert tools["kb_save_memory"].annotations.readOnlyHint is False
                assert tools["kb_save_memory"].annotations.destructiveHint is False
                assert "confirmed_by_user" in tools["kb_save_memory"].inputSchema["properties"]
                assert "global_search" in tools["kb_search"].inputSchema["properties"]
                assert "rerank" in tools["kb_search"].inputSchema["properties"]
                assert "ctx" not in tools["kb_search"].inputSchema["properties"]

                rejected = await session.call_tool(
                    "kb_save_memory",
                    {"title": "Rejected", "content": "not confirmed", "confirmed_by_user": False},
                )
                assert rejected.isError

                saved = await session.call_tool(
                    "kb_save_memory",
                    {
                        "title": "Cross-client prófun",
                        "content": "cross process visible memory á íslensku — ð æ ö",
                        "tags": ["integration"],
                        "confirmed_by_user": True,
                    },
                )
                assert not saved.isError
                document_id = saved.structuredContent["document_id"]

                settings = load_settings(config_path)
                external = KnowledgeStore(settings, HashingEmbedder(32))
                assert external.get_document(document_id) is not None
                external.upsert_document(
                    IngestDocument(
                        source="projects",
                        source_key="external",
                        title="External writer",
                        text="external cross process marker",
                        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                        project="fixture",
                    )
                )

                search = await session.call_tool("kb_search", {"query": "external cross process marker"})
                assert not search.isError
                assert search.structuredContent["scope"]["origin"].startswith("global")
                assert search.structuredContent["retrieval"]["reranker"]["status"] == "fallback"
                assert any(
                    result["document_id"] != document_id
                    for result in search.structuredContent["results"]
                )
                fetched = await session.call_tool("kb_get", {"document_id": document_id})
                assert fetched.structuredContent["found"] is True
                stats = await session.call_tool("kb_stats", {})
                assert stats.structuredContent["documents"] == 2

    asyncio.run(exercise())


def test_stdio_server_infers_an_unambiguous_client_root(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    projects_root = tmp_path / "projects"
    alpha_root = projects_root / "Alpha" / "nested"
    beta_root = projects_root / "Beta"
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    database = tmp_path / "roots.sqlite3"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(database),
                "model_cache_dir": str(tmp_path / "models"),
                "embedding_model": "test/hash-v1",
                "embedding_dimensions": 32,
                "projects_root": str(projects_root),
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
    settings = load_settings(config_path)
    seed = KnowledgeStore(settings, HashingEmbedder(32))
    timestamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    for name in ("Alpha", "Beta"):
        seed.upsert_document(
            IngestDocument(
                source="projects",
                source_key=f"scope-{name}",
                title=f"Scope {name}",
                text="MCP roots scope marker",
                timestamp=timestamp,
                project=name,
            )
        )

    env = dict(os.environ)
    env["CEREBRAS_MEMORY_CONFIG"] = str(config_path)
    env["CEREBRAS_MEMORY_TEST_EMBEDDER"] = "1"
    parameters = StdioServerParameters(
        command=os.sys.executable,
        args=[str(project_root / "mcp_server.py")],
        cwd=str(project_root),
        env=env,
    )

    async def list_roots(_context):
        return types.ListRootsResult(
            roots=[types.Root(uri=alpha_root.as_uri(), name="Alpha fixture")]
        )

    async def exercise():
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(
                reader,
                writer,
                list_roots_callback=list_roots,
            ) as session:
                await session.initialize()
                search = await session.call_tool("kb_search", {"query": "MCP roots scope marker"})
                assert not search.isError
                assert search.structuredContent["scope"] == {
                    "project": "Alpha",
                    "origin": "client_root",
                }
                assert {item["project"] for item in search.structuredContent["results"]} == {
                    "Alpha"
                }

    asyncio.run(exercise())
