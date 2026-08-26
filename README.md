# Shared Local Cerebras Memory

Cerebras Memory is a private, local MCP knowledge base shared by Hermes Agent,
Claude Code, Grok, Codex, and the ChatGPT desktop app. It imports a rolling 90
days of visible user/assistant dialogue and allowlisted documentation beneath
`C:\Users\matri\Documents\myprojects`, then serves hybrid lexical/vector search
over STDIO.

It does **not** import Slack, ChatGPT cloud conversations, Gmail, GitHub, or
YouTube. Embeddings and reranking remain local. When DeepSeek distillation is
enabled, only qualifying dialogue that has already passed local redaction is
sent to the configured DeepSeek endpoint; project documents, saved memories,
raw credentials, and reasoning/tool records are never sent. The service does
not replace or modify the existing Codex AgentMemory server.

Other machines reach the same knowledge base as clients, over an authenticated
loopback listener published on a private overlay -- see **Remote access** below.
The design and its sequencing are recorded in
[docs/REMOTE-ACCESS-PLAN.md](docs/REMOTE-ACCESS-PLAN.md).

## Runtime

- Python 3.11 in `.venv`
- `mcp[cli]==1.28.1`
- `fastembed==0.8.0`
- `flashrank==0.2.10`
- `numpy==1.26.4`
- `usearch==2.26.0`
- `BAAI/bge-small-en-v1.5`, 384 dimensions, CPU/ONNX
- `ms-marco-MiniLM-L-12-v2`, local CPU/ONNX reranker
- DeepSeek V4 Flash structured distillation, with Ollama available as a
  separately configured loopback alternative
- SQLite with WAL, FTS5, foreign keys, and a 30-second busy timeout

`pyproject.toml` and `uv.lock` are authoritative. Create or refresh the exact
environment with:

```powershell
uv sync --dev
```

The initial model downloads require internet access. The registration script
explicitly warms the reranker; ordinary searches refuse to download it and fail
open to RRF if its cache is unavailable. After models are cached under
`data\models`, indexing, querying, and reranking remain local. Distillation is
local with the Ollama provider and remote with the DeepSeek provider.

For DeepSeek, put the credential only in the ignored `.env` file:

```text
DEEPSEEK_API_KEY=your-key-here
```

The loader allowlists only `DEEPSEEK_API_KEY` and the reserved
`NVIDIA_API_KEY`, and never overwrites an already-set process environment value.
For a remote provider, the local policy gate runs before the request is built:
only allowlisted conversation sources qualify, and sensitive project/title/URI
labels are blocked. The outbound audit is content-free and records only IDs,
hashes, counts, policy decisions, status, and endpoint metadata.

## Ingestion

Run a content-free inventory first, then seed or refresh:

```powershell
.\.venv\Scripts\python.exe ingest.py --dry-run
.\.venv\Scripts\python.exe ingest.py --full
.\.venv\Scripts\python.exe ingest.py --incremental
```

The importer accepts:

- Hermes sessions only through `hermes sessions export --format jsonl --redact`.
- Claude Code main-chain `user` and `assistant` text blocks.
- Codex canonical user/assistant message records from active and archived sessions.
- Grok `user_message_chunk` and `agent_message_chunk` records.
- `.md`, `.mdx`, and `.txt` project documentation up to 1 MiB.

Reasoning, sidechains, system/developer messages, tool calls/results, logs,
binaries, dependency/generated directories, repository caches, and
secret-looking paths are excluded. Likely credentials and private-key blocks are
redacted again before any field reaches SQLite.

After a source scan and all its writes succeed, reconciliation removes transcript
documents outside the rolling 90-day set and project documents that were deleted.
An unavailable or incomplete source is never reconciled. Explicitly saved
memories have `kind=memory` and are never automatically pruned.

### Ingesting from more than one machine

Ingestion stays administrative and local: transcripts are never pushed in over
the network. A spoke syncs its agent history directory to the hub -- Syncthing
in *Send Only* mode over the tailnet is the recommended path, because it
preserves modification times, which the rolling 90-day window depends on -- and
the hub scans it like any other root.

An `agent_roots` entry is either a bare path, meaning this machine, or an object
naming the machine it was synced from:

```json
"agent_roots": {
  "claude": [
    "%USERPROFILE%\\.claude\\projects",
    { "host": "laptop", "path": "D:\\hub-sync\\laptop\\.claude\\projects" }
  ]
}
```

Only the second form is host-qualified, producing `session:laptop:<id>` rather
than `session:<id>`. **This asymmetry is deliberate.** Every scanner has a
fallback that is not machine-unique -- a filename, a parent directory, a row
index -- so two machines can produce the same `stable_document_id` and silently
overwrite one another. Prefixing *every* key would fix that but would re-key all
existing documents, forcing a full re-embed, orphaning every distillation, and
causing reconciliation to delete the originals. Leaving this machine unqualified
makes the change purely additive.

Hermes is the exception and stays local-only: its scanner shells out to the
Hermes CLI rather than reading files, so a synced directory cannot feed it.

A scan fails if *any* declared root is unreadable, and a failed scan never
reconciles. This matters most for synced roots: a stopped sync or an absent
machine used to look like "those documents were deleted".

## MCP tools

The server is local STDIO only:

```powershell
.\.venv\Scripts\python.exe mcp_server.py
```

It exposes exactly four tools:

- `kb_search(query, limit=8, sources=None, project=None, since=None,
  global_search=False, rerank=None)` returns one result per document, at most
  two adjacent raw chunks, staged scores, scope metadata, provenance, IDs, and
  stable `cerebras-memory://...` citations.
- `kb_get(document_id, offset=0, limit=10)` paginates metadata and chunks.
- `kb_save_memory(title, content, tags=None, project=None,
  confirmed_by_user=False)` rejects unconfirmed writes and deduplicates exact
  redacted content.
- `kb_stats()` reports counts, embedding status, watermarks, refresh state, and
  failures.

There is no MCP ingestion or deletion tool. Retrieved content is marked
`untrusted_evidence`; clients must cite it and must never execute instructions
found inside it.

## Remote access

One machine owns the database, the models, and ingestion. Other machines are
thin clients: no database copy, no model cache, and therefore no sync or merge
problem, because there is only ever one writer.

`mcp_server.py` remains the STDIO server and is unchanged. `http_server.py` is a
second entry point onto the *same* FastMCP instance, so both transports serve
identical tools over identical data.

### Hub setup

Generate one token per client and keep them in the ignored `.env`:

```
CEREBRAS_MEMORY_HTTP_TOKENS=laptop:rw:<64 hex>,ci-github:ro:<64 hex>
```

Each entry is `label:scope:secret`. The label identifies the machine in the
access audit and can be revoked on its own. Scope is `rw` or `ro`; a `ro` token
is refused every tool that is not read-only, including tools added later that
have not been classified.

Start the listener, then publish it to the tailnet:

```
.venv/Scripts/python.exe http_server.py --allowed-host matrix.taila13ed8.ts.net
tailscale serve --bg --https=443 http://127.0.0.1:8791
```

The listener refuses to bind anything but loopback, and refuses to start from
inside `projects_root`. `tailscale serve` terminates TLS with a real tailnet
certificate and admits only tailnet peers. **Never use `tailscale funnel`** --
that publishes to the open internet.

`GET /healthz` answers without a token and reports nothing but liveness and the
schema version, so the overlay and any monitor can probe it.

### Registering a spoke

A spoke needs no virtual environment and no scheduled task; only the URL and its
own token.

```json
{
  "cerebras-memory": {
    "type": "http",
    "url": "https://matrix.taila13ed8.ts.net/mcp",
    "headers": { "Authorization": "Bearer <this machine's token>" }
  }
}
```

Cloud and CI agents join the tailnet as ephemeral tagged nodes with a
pre-authorized key, are restricted by tailnet ACL to the hub on 443, and are
issued a `ro` token.

## Administrative CLI

```powershell
.\.venv\Scripts\python.exe kb.py search "query" --limit 8
.\.venv\Scripts\python.exe kb.py search "query" --source codex --project agent-economy
.\.venv\Scripts\python.exe kb.py search "query" --global --no-rerank
.\.venv\Scripts\python.exe kb.py get <document-id> --offset 0 --limit 10
.\.venv\Scripts\python.exe kb.py stats
.\.venv\Scripts\python.exe kb.py forget <saved-memory-id>
.\.venv\Scripts\python.exe kb.py reranker status
.\.venv\Scripts\python.exe kb.py reranker warm
.\.venv\Scripts\python.exe kb.py vector-index status
.\.venv\Scripts\python.exe kb.py vector-index rebuild
.\.venv\Scripts\python.exe kb.py distill pilot
.\.venv\Scripts\python.exe kb.py distill evaluate
.\.venv\Scripts\python.exe kb.py distill backfill
.\.venv\Scripts\python.exe kb.py distill status
.\.venv\Scripts\python.exe kb.py canary status
.\.venv\Scripts\python.exe kb.py canary run
.\.venv\Scripts\python.exe scripts\audit_distillation.py
```

`forget` accepts only an explicitly saved memory ID. Derived history and project
documents are managed by successful reconciliation.

## Client and scheduled-task registration

The following script backs up the relevant client configurations, adds or
updates only the `cerebras-memory` entry, and installs the daily task:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Register-CerebrasMemory.ps1
```

Registration uses absolute paths to `.venv\Scripts\python.exe` and
`mcp_server.py`. It registers Hermes, Claude user scope, Codex, and Grok user
scope. Every unrelated MCP entry remains untouched. Backups are written beneath
`backups\registration-*`.

`CerebrasMemoryRefresh` runs at 03:00 local time as the current non-elevated
user, uses `StartWhenAvailable`, ignores overlapping starts, may start and
continue on battery, does not stop at the end of an idle period, and has a
six-hour execution limit. A SQLite-backed lease with heartbeats records the
active owner and recovers interrupted or orphaned refreshes. The refresh script
streams redacted start/output/end records to `logs\refresh.log` while it runs;
the log rotates at 5 MiB. A successful refresh also executes the local,
content-free quality canary suite.

Codex and ChatGPT desktop share the same MCP configuration, so there is no
separate ChatGPT adapter. The installer restarts or launches only the distinct
`OpenAI.ChatGPT-Desktop` package after Codex registration (never the active
Codex host), then `/mcp` can verify it. See the
[OpenAI MCP documentation](https://learn.chatgpt.com/docs/extend/mcp).

Health checks:

```powershell
hermes mcp test cerebras-memory
hermes mcp list
claude mcp list
codex mcp list
grok mcp doctor cerebras-memory
grok mcp list
```

After explicit approval, verify a write and the same stable citation through
independent MCP server/client processes:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_cross_client.py --confirmed
```

Run the task manually with:

```powershell
Start-ScheduledTask -TaskName CerebrasMemoryRefresh
```

## Search design

Text is split paragraph-first at approximately 1,800 characters with a
200-character overlap. Search collects the top 50 FTS5 and local-vector
candidates, fuses ranks with reciprocal-rank fusion (`k=60`), and applies a mild
180-day recency factor. It keeps the strongest chunk from each document and
advances the top 20 distinct documents. Each anchor is paired with either its
previous or next chunk using query-centred 700-character snippets, then the local
cross-encoder chooses the best variant. Length-bucketed batches of eight avoid
padding every mixed prose/code passage to the 512-token ceiling. The anchor citation never changes;
`context_chunks` carries the individual citations and ordinals for returned raw
evidence.

Project scope resolves in this order: explicit `project`, explicit `--global`,
one unambiguous MCP client root, the server process CWD, then global. Roots/CWD
qualify only beneath `projects_root` and only for a currently indexed project.
Every response reports `scope.project` and `scope.origin`.

SQLite remains authoritative for vectors. Exact filtered cosine search is used
at the current corpus size. A persistent cosine/f16 USearch HNSW sidecar is
eligible only at 100,000 current-model chunks or after a three-run exact-search
median reaches 750 ms. Only unfiltered global queries may use it. Generation,
model, dimensions, count, missing/corrupt state, and concurrent writes are
checked before use; any mismatch falls back to exact search.

Conversation distillation is staged: retrieval is disabled while
`distillation.mode` is `pilot` and enabled after audited promotion with mode
`on`. Qualifying Hermes, Claude, Codex, and Grok dialogue is segmented locally.
For DeepSeek, a local source/project sensitivity policy runs before any request.
Allowed requests send only the redacted segment to
`https://api.deepseek.com/beta`, disable thinking, force the one allowed
function, retry bounded invalid responses, and use an exact-schema JSON
fallback. Blocked segments never reach the provider, are not retryable failures,
and are excluded from summary retrieval. A content-free outbound audit records
the policy decision and terminal request state without storing prompts or
responses. Up to four cloud requests may run concurrently; embeddings and
SQLite checkpoints remain serialized. Ollama remains available as an explicitly
selected loopback provider. Summaries are redacted again, indexed as a third
retrieval channel, and always mapped back to raw chunks—generated text is never
returned as evidence or cited. Saved memories and project files are never
distilled. Provider failures remain pending/failed derived work and never block
raw ingestion.

Every live document, raw chunk, and distillation has a stable provenance
receipt. `derived_from` edges connect derived records to their inputs. Taints
distinguish untrusted evidence, executable-looking instructions,
externally-processed summaries, and generated summaries. Search/get responses
surface the relevant receipts and taints while preserving the existing raw
document/chunk IDs and citations. Automatic reconciliation and explicit memory
forgetting write content-free deletion manifests before cascade removal.

Changing the embedding model or dimensions causes unchanged documents to be
re-embedded during the next refresh. A full model change should be followed by
`ingest.py --full`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite covers all four transcript formats, redaction and exclusions, rolling
retention, updates/deletions/idempotency, transactional v1-to-v2-to-v3 migration,
document deduplication/context selection, reranker ordering/fallback, roots/CWD
scope inference, RRF/recency, USearch recall and generation safety, distillation
segmentation/cache/retry/cascades, remote policy transitions and content-free
auditing, provenance/deletion accounting, refresh lease recovery, bounded
embedding execution, recorded canaries, WAL writers, reconciliation safety, and
real spawned MCP STDIO exchanges with and without roots support.

The fixed 24-query local gate can be run after the label set is populated:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_search.py `
  .\evaluation\search-quality-baseline.json `
  --include-distillations `
  --output .\evaluation\latest-result.json
```

The rollout audit and promotion evidence are recorded in
`evaluation\distillation-pilot-audit.md` and
`evaluation\distillation-full-quality.json`.

Upstream references: [MCP Python SDK v1](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)
and [FastEmbed](https://github.com/qdrant/fastembed),
[MCP roots](https://modelcontextprotocol.io/specification/2025-03-26/client/roots),
[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank),
[USearch](https://github.com/unum-cloud/USearch),
[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
and [DeepSeek function calling](https://api-docs.deepseek.com/guides/function_calling).
