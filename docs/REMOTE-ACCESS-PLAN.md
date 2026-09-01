# Cerebras Memory: multi-machine and multi-agent access

## Context

`cerebras-memory` is today a strictly single-machine system. One 860 MB SQLite file on this
desktop (`matrix`) holds 4,309 documents / 60,217 chunks; embeddings and reranking run locally
via ONNX; ingestion scans this machine's agent transcript directories on a nightly scheduled
task. Four clients (Claude Code, Codex, Grok, Hermes) each spawn their own STDIO worker against
that one file. Nothing is reachable from anywhere else.

The goal is to reach the same knowledge base from **other PCs** and from **cloud/CI agents**,
with **full parity** — remote machines can search, save memories, and have their own agent
transcripts ingested into the shared corpus.

**Approach: hub-and-spoke over Tailscale.** This desktop stays the sole owner of the database,
the models, and ingestion. A second entry point serves the *same* FastMCP instance over
authenticated streamable-HTTP, bound to loopback and exposed on the tailnet only. Spokes are
thin clients — no database copy, no model cache, no sync/merge problem.

This was chosen over the alternatives because the code rules them out:

- **Shared SQLite over a network share** — dead on arrival. `_initialize` sets WAL
  (`store.py:378`), and WAL's `-shm` shared memory does not work over SMB. The 860 MB file
  would corrupt.
- **Multi-master with DB sync** — 60,217 float32 embedding BLOBs, no merge story, and
  `vector_index_state.data_generation` is a single-writer counter.
- **Cloud-hosted DB** — ruled out by the corpus itself. `config.json`'s
  `remote_policy.sensitive_project_terms` blocks legal/case/court/privileged material from
  even reaching DeepSeek; 1,926 of 2,435 outbound distillation attempts were policy-blocked.

**This requires deliberately relaxing a stated invariant.** `CLAUDE.md` says *"Keep the MCP
server on STDIO; do not add a network transport"* and `SECURITY.md` says *"No listener or
remote ingestion endpoint is created."* Phase 7 rewrites both precisely, bounding the loosening
rather than deleting it. Every other invariant in `CLAUDE.md` is preserved intact: no ingestion
or deletion tool is added, no cloud embeddings, redaction still runs on every write including
new metadata, retrieved chunks stay untrusted evidence, `kind=memory` pruning is untouched, and
registration still preserves unrelated client entries.

**Two pre-existing bugs surfaced during design.** Neither is caused by this work, but both are
detonated by it, so they are fixed first and standalone (Phases 1 and 3).

---

---

## Execution status (updated 2026-08-26)

Branch `feat/remote-access`, off `kb-upgrade-phases-0-4`. Suite: **184 passed**
(baseline was 125). `mcp_server.py` is byte-identical to its committed state.

| Phase | State | Notes |
|---|---|---|
| 0 Stabilize | done | Pre-existing work checkpointed; 605 MB `VACUUM INTO` backup; stale `index_path` corrected. The stalled refresh was diagnosed as a **missing scheduled task**, not a code fault: `CerebrasMemoryRefresh` no longer exists, the last real run succeeded, and every `ingest_state` row is `ok`. |
| 1 Concurrency | done | Locked `FastEmbedder._cache_query` and `_load`; added `KnowledgeStore.prewarm`. No SQLite changes: `_connect` already returns a fresh per-call connection. |
| 2 Scope | done | `NO_PROCESS_CWD` sentinel plus leaf-name matching reported as `client_root_leaf`. Local STDIO behaviour unchanged; the canary suite still asserts `client_root`. |
| 3 Reconcile guard | done | `_jsonl_files` names the missing roots and fails the scan. A contrast test proves this was not guarding a no-op: with a root present but emptied, a 1-in-5 loss reconciles through silently. |
| 4 HTTP hub | done | `http_server.py`, schema v6 `access_audit`, scoped bearer tokens, DNS-rebinding protection, loopback-only bind, refusal to start inside `projects_root`. Verified end to end against the live 4,310-document corpus. |
| 5 Spokes | hub published | `tailscale serve` is live: `https://matrix.taila13ed8.ts.net/` proxies to the loopback listener, tailnet only, valid TLS. Verified end to end against the live corpus. `Register-CerebrasMemory.ps1 -Remote` is written and its guards tested. Nothing left but to run it on a second machine, and none is online yet. |
| 6 Per-host ingest | done | Remote roots keyed `session:{host}:{id}`; this machine stays unqualified so nothing is re-keyed. Hermes stays local-only. |
| 7 Documentation | done | `CLAUDE.md` invariant rewritten and bounded; `SECURITY.md` network-exposure section; `README.md` remote-access and multi-machine ingestion sections. |
| 8 REST surface | not started | Optional. Only needed if a CI agent cannot speak MCP. |
| 9 Offline fallback | not started | Optional, deliberately deferred. |

### Verified live (2026-08-27)

Over `https://matrix.taila13ed8.ts.net/mcp`, against the real 4,310-document
corpus: no token 401, wrong token 401, read-only token attempting a write 403,
`/healthz` 200 without a credential. A memory was saved from a network client,
searched back, and paged. `scope.origin` was never `process_cwd`. Every call is
in `access_audit`, including the rejections -- an absent credential fingerprints
as `none` and a wrong one fingerprints distinctly, so one token retried is
separable from many guesses.

### What is left

1. Run the spoke registration on a second machine once one is online:

```
$env:CEREBRAS_MEMORY_HTTP_TOKEN = "<the spoke secret from .env>"
.\scripts\Register-CerebrasMemory.ps1 -Remote -HubUrl https://matrix.taila13ed8.ts.net/mcp
```

2. Set up Syncthing for that machine's agent history and add its root to
   `agent_roots` as `{ "host": "<name>", "path": "..." }`.
3. Start the hub automatically. It currently runs only when started by hand;
   register it as a boot task with `-LogonType S4U` so it survives a reboot and
   does not need an interactive logon.
4. Optional phases 8 and 9 remain unstarted.

### Operational notes

- Tokens live in the ignored `.env` as `CEREBRAS_MEMORY_HTTP_TOKENS`: client
  `spoke` with scope `rw`, client `ci` with scope `ro`.
- `CerebrasMemoryRefresh` has been restored and runs daily at 03:00.
- `distillation.mode` was set to `off` so the first catch-up refresh brings the
  corpus current without sending four weeks of backlog to DeepSeek. Turn it back
  on once that run has been seen to succeed. The previous config is at
  `backups/pre-remote-access-20260826/config.json.bak`.

### Deviations from the plan as approved

- Pre-existing work was committed to `kb-upgrade-phases-0-4`, not to `main`, and
  `feat/remote-access` branches from it. Committing to `main` would have merged
  2,665 lines of unreviewed work into the trunk as a side effect of this project,
  and phases 1-3 modify files that work rewrites.
- `config.example.json` was left alone. Adding a sample synced root would point a
  fresh copy at a path that does not exist, which the new guard correctly treats
  as a failed scan.
- Three defects were found and fixed that the plan did not anticipate: `prewarm`
  downloaded the reranker at startup, `CEREBRAS_MEMORY_NO_SECRETS` withheld the
  hub's own bearer tokens, and the access audit recorded successes only.

---

## Branch and merge workflow

All work happens on a feature branch; `main` is only touched by the merge at the end.

Note the real git repo is the **inner** directory, `C:\Users\matri\Documents\cerebras memory\ceribras-memory\`
(remote `alinojoumi8/cerebras-memory`). The outer `cerebras memory\.git` is an empty stub — ignore it.

**Starting state as actually found (2026-08-26):** branch `kb-upgrade-phases-0-4` and `main` both
sit at commit `7dee5f0` — the branch was created but never committed to. All 2,665 lines of the
"kb upgrade phases 0-4" work are uncommitted in the working tree, plus 6 untracked test files.
Baseline suite: **125 passed**.

1. **Commit the pre-existing work to `kb-upgrade-phases-0-4`, not to `main`.** It is a separate
   feature that was deliberately branched, and committing it to `main` would merge unreviewed work
   into the trunk as a side effect of this project. `.codebase-memory/` and `graphify-out/` are
   tooling artifacts and get gitignored rather than committed.
2. Branch `feat/remote-access` **off `kb-upgrade-phases-0-4`**, not off `main`. Phases 1-3 modify
   `embeddings.py`, `store.py`, `agent_history.py`, and `config.py` — all heavily rewritten by that
   uncommitted work, and every line reference in this plan was read against the working tree, not
   against `main`. Branching from `main` would conflict immediately and invalidate the references.
   This means `kb-upgrade-phases-0-4` must merge to `main` first, or both merge together.
3. **One commit per phase**, each independently green under
   `.\.venv\Scripts\python.exe -m pytest`. Phases 1–3 are self-contained bug fixes that stand on
   their own merit; Phases 4–7 build the feature. If the feature stalls, Phases 1–3 are still worth
   merging.
4. Push and open a PR for review before merging to `main`.
5. **Do not register any spoke against the branch.** Client configs point at absolute paths on
   disk, so registering a laptop mid-branch leaves it pointing at code that may be rebased or
   reverted. Register spokes (Phase 5) only after the merge.

---

## Phase 0 — Stabilize before adding load

Nothing below is safe against an unknown baseline.

- **Write this plan into the repo** at
  `ceribras-memory/docs/REMOTE-ACCESS-PLAN.md` so any agent — on this machine or another — can
  pick up execution without this conversation. Committing it to the GitHub remote makes the plan
  reachable from the spokes themselves. Add a short pointer to it from `README.md`.
- **Commit the 24 modified tracked files to `main`,** then branch. You cannot bisect a regression
  or revert a bad phase against a dirty tree.
- **Diagnose the broken refresh.** The last `--full` run *failed* (2026-07-29) and the one
  before was `abandoned`; the scheduled task has not succeeded since. Check `logs/refresh.log`
  and `kb.py stats`.
- **Fix the stale index path.** `vector_index_state.index_path` still points at the old
  `C:\Users\matri\Downloads\cerebras memory\...` location from before the move to `Documents`.
- **Take a `VACUUM INTO` backup** before any schema or ingest change.
- Record a baseline: `.\.venv\Scripts\python.exe -m pytest`.

**Done when:** clean `git status`, green suite, a successful `--incremental` refresh,
`kb_stats` showing no failures.

---

## Phase 1 — Concurrency correctness (no network yet)

Under STDIO one client issues calls serially, so two races in `embeddings.py` never fire. One
process serving many agents will hit both.

`embeddings.py:105-116` — `_cache_query` inserts then evicts with
`self._query_cache.pop(next(iter(self._query_cache)))`. Concurrent mutation during that
iteration raises `RuntimeError: OrderedDict mutated during iteration`; two threads evicting the
same key raise `KeyError`. Add a `threading.Lock` in `FastEmbedder.__init__` (`embeddings.py:51`)
held across insert + eviction. The read at `embeddings.py:119` is a single dict lookup, safe
under the GIL.

`embeddings.py:54-71` — `_load` is an unguarded check-then-set; two cold requests both construct
`TextEmbedding`, and with `lazy_load=True` they race inside fastembed's own loader. Use
double-checked locking — **copy the shape already used correctly in this repo** at
`reranking.py:53-63` (`FlashRankReranker._load`) and `store.py:2224-2240`
(`_load_exact_vector_snapshot`).

**No SQLite changes.** `_connect` (`store.py:363-373`) creates a fresh connection per call and
every call site consumes it within one thread. Adding a pool or global lock would be a
regression — it would serialize reads that WAL already runs concurrently.

Also add a **pre-warm helper** (load store, embed one query, materialize the exact-vector
snapshot) so the first remote search doesn't blow past a client's 30 s default timeout, and so
the `@lru_cache(maxsize=1)` on `_store()` (`mcp_server.py:56`) can't double-construct on a
concurrent cold miss.

**Done when:** new concurrency tests fail before / pass after; STDIO behaviour byte-identical.

---

## Phase 2 — Scope correctness (must land before any spoke connects)

`_project_for_path` (`store.py:2135-2144`) resolves a project by strict path containment:
`candidate.relative_to(self.settings.projects_root)`. A laptop root like `D:\work\Alpha` raises
`ValueError` → returns `None` → the roots set is empty → `resolve_project_scope` falls through
to `self._project_for_path(cwd or Path.cwd(), known)` at `store.py:2176`, where `Path.cwd()` is
**the hub server process's** working directory.

Two failures, the second serious: every remote search silently loses scoping; and if the hub
service ever starts with a cwd under `projects_root`, remote agents silently inherit *that*
project with `origin: "process_cwd"`. Given the corpus holds privileged legal material, a
confidently wrong-project answer is the highest-consequence bug in this plan.

The fix is already implied by the ingest side. `_project_from_cwd` (`agent_history.py:263-269`)
derives every transcript's project from `Path(value).name` — the **basename**, machine-independent
by construction. Retrieval using full-path containment while ingest uses basenames is an existing
asymmetry that cross-machine access merely exposes.

1. **Kill the implicit cwd fallback.** Change `store.py:2176` to a sentinel-aware form so
   "explicitly no cwd" is distinguishable from "not supplied", and have the HTTP path pass that
   sentinel. A network request has no meaningful cwd. Do this even if you do nothing else.
2. **Add a leaf-name fallback** inside `_project_for_path`: after `relative_to` fails, try
   `known.get(candidate.name.casefold())`, then walk ancestors. Accept only an unambiguous
   single match, mirroring the existing ambiguity handling at `store.py:2171-2174`. Return a
   distinct `origin: "client_root_leaf"` so `kb_search`'s `scope` field stays honest about how
   it inferred. `_known_projects()` (`store.py:2122-2133`) already returns the case-folded map,
   so no new query is needed.

**Done when:** a root of `D:/work/Alpha` resolves to project `Alpha` with
`origin: "client_root_leaf"`; an explicit no-cwd request returns `origin: "global"` and **never**
`process_cwd`, even when `Path.cwd()` sits inside `projects_root`.

---

## Phase 3 — Reconciliation guard (protects the existing corpus)

Land this standalone, before touching ingestion config.

`_jsonl_files` (`agent_history.py:281-288`) sets `available = False` only when **all** roots are
missing. If the hub's own `.claude\projects` exists but a synced laptop root has vanished (sync
stopped, laptop off for a week), the scan reports **success** with fewer keys and
`reconcile_source` (`store.py:1725`) deletes every laptop document — cascading through
`ON DELETE CASCADE` to chunks, distillations, and provenance receipts. The safety floor at
`store.py:1755-1763` only trips past `reconcile_min_ratio`, which is **0.5** (`config.py:215`),
so a source can lose up to half its documents silently.

**Fix (~6 lines):** have `_jsonl_files` return the set of *declared but missing* roots; each
scanner sets `successful=False` with `error=f"{source} root unavailable: {missing}"`.
`ingest.py:156-162` then hits `continue` and reconciliation never runs. This converts silent mass
deletion into a loud, recoverable failure — matching the philosophy already stated in the
`reconcile_source` docstring at `store.py:1733-1740`.

Consider raising `reconcile_min_ratio` to 0.8 once multi-host ingest is live.

**Done when:** a deliberately unmounted root produces a `failed` status and **zero** deletions.

---

## Phase 4 — HTTP transport + auth on the hub, loopback only

Install Tailscale first — **it is not currently installed** on this machine.

### New file `http_server.py`; `mcp_server.py` stays byte-identical

`mcp` is a module-level singleton (`mcp_server.py:41-50`) with all four tools already registered,
and `FastMCP.settings` is a mutable pydantic-settings model. So `http_server.py` imports the same
object and serves it — the STDIO path is untouched, satisfying the rewritten invariant.

Do **not** call `mcp.run(transport="streamable-http")` (no seam for auth middleware) and do **not**
use `mcp.streamable_http_app()` unmodified — it builds `StreamableHTTPSessionManager` without
`session_idle_timeout`, so sessions orphaned by a sleeping laptop leak forever, and routes added
via `@mcp.custom_route` are appended *outside* the auth wrapper (its own docstring says so).

Build a Starlette app: set thread env vars *before* importing `mcp_server` (its `os.environ.setdefault`
block at `mcp_server.py:19-27` caps `OMP_NUM_THREADS=4` because "every desktop client launches its
own STDIO worker" — wrong for a single hub process); construct the session manager explicitly with
`session_idle_timeout`; mount it at `/mcp` behind `BearerAuthMiddleware`; add `GET /healthz`.

Two gotchas that will otherwise cost hours:

- **Starlette does not propagate lifespan into `Mount`ed sub-apps.** You must wire
  `manager.run()` into the outer app's lifespan yourself or every request 500s with
  "Task group is not initialized". Pre-warm in that same lifespan.
- **`stateless_http` must be `False`.** In stateless mode a bare `tools/call` POST creates a new
  transport with no preceding `initialize`, so `_client_params` is never populated and
  `check_client_capability` returns `False` unconditionally — which makes `_client_root_paths`
  (`mcp_server.py:76-79`) return `[]` for *every* client and silently disables project scoping
  entirely. `list_roots` is also a server→client request that cannot complete across separate
  transports. Handle laptop sleep with `session_idle_timeout` instead; MCP clients treat the
  manager's spec-compliant 404 as "session expired" and re-initialize.

Dependencies: `mcp[cli]==1.28.1` already pulls in uvicorn 0.51.0 and starlette 1.3.1. Follow the
precedent set by the `onnxruntime` comment in `pyproject.toml` and pin them explicitly rather
than relying on transitives.

### Auth

- **`CEREBRAS_MEMORY_HTTP_TOKENS`**, added to `_ALLOWED_SECRET_ENV_KEYS` (`config.py:20`) so it
  loads from the already-gitignored `.env` via `_load_secret_env` (`config.py:128-145`). Matches
  the existing `CEREBRAS_MEMORY_*` naming convention.
- **One token per client, with a scope.** Format `label:scope:secret` — e.g.
  `laptop-claude:rw:<64 hex>`, `ci-github:ro:<64 hex>`. Per-client identity, individual
  revocation, and least privilege for the cloud/CI principal, which is the least trusted.
  A `ro` token rejects `kb_save_memory` before any store call.
- **Constant-time verification without a timing oracle:** build a `sha256(secret) → (label, scope)`
  dict at startup; hash the presented bearer, look up O(1), confirm with `hmac.compare_digest` on
  the digest. Do *not* loop over tokens with an early break — that leaks which token matched as
  the list grows. Reject with a bare 401; never echo the presented value.
- **Host/Origin validation** against DNS rebinding via the SDK's `TransportSecuritySettings`
  (`enable_dns_rebinding_protection=True`). Critical detail: `tailscale serve` forwards the
  **`*.ts.net` hostname** in `Host`, not `127.0.0.1`, so `allowed_hosts` must include
  `matrix.<tailnet>.ts.net` plus loopback forms for local testing. That setting only covers the
  `/mcp` mount — apply the same check in your middleware. Do **not** add `CORSMiddleware`.
- **Drop unused credentials from the listener process.** `load_settings()` unconditionally calls
  `_load_secret_env()` (`config.py:310-311`), injecting `DEEPSEEK_API_KEY` / `NVIDIA_API_KEY` into
  a process that never needs them — distillation only runs from `ingest.py`/`kb.py`. Add a
  `CEREBRAS_MEMORY_NO_SECRETS=1` opt-out honoured by `_load_secret_env` and set it for the hub
  service, so a compromised listener cannot make an outbound API call at all.

### Access audit (schema v6)

Add an `access_audit` table modelled on the discipline already established by
`outbound_distillation_audit` (`store.py:640-655`) — **content-free**: `client_label`,
`token_fingerprint`, `tool`, `query_hash` (not the query), `result_count`, `applied_project`,
`scope_origin`, `latency_ms`, `status`, `error_code`. No query text, no snippets, no token.
Plumb the label via a `contextvars.ContextVar` set in middleware. Stamp HTTP-originated writes as
`mcp_http:<client_label>` in `provenance_receipts.producer` (`store.py:656-673`) so saved memories
are attributable to a machine.

### Exposure

uvicorn binds `127.0.0.1` only — no firewall rule, nothing on the LAN can reach it. Then:

```bash
tailscale serve --bg --https=443 http://127.0.0.1:8791
```

Real Let's Encrypt cert on `*.ts.net`, reachable only by tailnet nodes. Restrict further with a
tailnet ACL allowing only your spokes to reach the hub on 443. **Never `tailscale funnel`** — that
publishes to the open internet. Put that prohibition in `SECURITY.md` explicitly.

Run the hub as a Scheduled Task at boot with `-LogonType S4U` (runs whether logged on or not),
unlike the refresh task's current `Interactive` logon type.

**Done when:** `curl 127.0.0.1:8791/mcp` without a token → 401; with a valid token an MCP
round-trip succeeds; `tailscale serve status` shows the mapping; `git diff mcp_server.py` is empty.

---

## Phase 5 — Register the spokes

### Other PCs

Extend `scripts\Register-CerebrasMemory.ps1` with `-Remote` / `-HubUrl` / `-TokenEnvVar`. The
existing script already does the right thing — `Backup-Configuration` first, then remove-then-add
only the `cerebras-memory` entry — which satisfies `CLAUDE.md`'s *"Preserve every unrelated client
MCP entry."* **Extend, don't restructure.** In `-Remote` mode skip the venv/`warm_models.py`
preflight and skip the scheduled task entirely; spokes never ingest locally.

For Codex/Grok TOML, add a `Set-CodexRemoteBlock` alongside the existing `Set-CodexServerBlock`,
**keeping the same `(?ms)^\[mcp_servers\.cerebras-memory\]...` regex** so it still removes exactly
one block. Prefer env-var interpolation for the token over plaintext in client config; if a client
doesn't support it, note in `SECURITY.md` that the token sits at rest in
`%USERPROFILE%\.codex\config.toml` and must be rotated on machine decommission.

### Cloud / CI agents

Join them to the tailnet as **ephemeral, tagged nodes** using a pre-authorized auth key
(`tag:ci`) — GitHub Actions has an official `tailscale/github-action`. Ephemeral nodes
self-remove when the job ends. Then:

- A tailnet ACL grants `tag:ci` access to **only** the hub on 443 — nothing else on your network.
- Issue CI a **`ro`-scoped token** (Phase 4), separately revocable.
- Consider extending the token record with a project denylist seeded from the
  `remote_policy.sensitive_project_terms` list already in `config.json`, so a CI token cannot
  retrieve legal/privileged material at all. This reuses an existing, already-tuned list.

If a CI agent is not an MCP client, that is the trigger for the optional REST surface below.

**Done when:** cross-machine `verify_cross_client.py` passes — desktop saves, laptop retrieves,
with `origin: "client_root_leaf"` and never `process_cwd`; a bad token is rejected from the
laptop; every unrelated MCP entry in each client config is byte-identical to its backup.

---

## Phase 6 — Per-host ingestion (full parity)

### Do the two machines collide?

`stable_document_id(source, source_key)` is `sha256(f"{source}\0{source_key}")` (`store.py:238-240`).
Each scanner keys on a session UUID — safe — but **every fallback path collides**:

| Scanner | Primary key | Fallback | Collides? |
|---|---|---|---|
| `scan_claude` (`agent_history.py:316`) | `session:{sessionId}` | `path.stem` | fallback yes |
| `scan_codex` (`agent_history.py:368`) | `session:{id}` | `path.stem` | fallback yes |
| `scan_grok` (`agent_history.py:417`) | `session:{sessionId}` | `path.parent.name` | fallback likely |
| `scan_hermes` (`agent_history.py:499`) | `session:{id}` | `f"row-{scanned}"` | **guaranteed** — positional |

A collision means one machine's transcript silently overwrites the other's on upsert.

### Minimal change: host-qualify remote roots only

**Do not migrate the existing 4,309 documents.** Prefixing every key with the local hostname
would re-key every `stable_document_id`, which means re-embedding 60,217 chunks (hours of ONNX
CPU), orphaning 4,014 distillations and 185k receipts, and — because the old keys vanish from the
scan — `reconcile_source` deleting all 4,309 originals.

Instead the hub's own machine keeps the **unqualified** namespace; every other host gets a prefix:

- Local: `session:{id}` — unchanged, zero churn, zero migration
- Remote: `session:{host}:{id}`

Purely additive. Document the asymmetry in `README.md` as a deliberate compatibility decision.

`agent_roots` values are already parsed into `tuple[Path, ...]` (`config.py:392-394`), so allow
each entry to be a bare string (implicit local) or an object:

```json
"agent_roots": {
  "claude": [
    "%USERPROFILE%\\.claude\\projects",
    { "host": "laptop", "path": "D:\\hub-sync\\laptop\\.claude\\projects" }
  ]
}
```

Add `agent_root_hosts: dict[str, str]` to `Settings`; change `_jsonl_files` to return
`(path, host)` pairs. Add `"host": host` to the metadata passed to `_dialogue_document` — it flows
through `_redact_json_value` in `_prepare_document` (`store.py:1232`), so `CLAUDE.md`'s "redact
every write including metadata" holds automatically.

### Sync mechanism: Syncthing

Laptop folder *Send Only*, hub folder *Receive Only*, over the tailnet.

Why specifically: `_file_fallback` (`agent_history.py:259-260`) uses `path.stat().st_mtime` as the
message-timestamp fallback and `transcript_days: 90` filters on it. A sync that rewrites mtimes to
"now" would resurrect aged-out sessions and corrupt watermarks. Syncthing preserves mtimes,
resumes on its own after sleep, and *Send Only* means the hub can never push deletions back.

Fallback without a new daemon: a laptop scheduled task running
`robocopy <src> <hub-share> /E /XO /COPY:DAT /DCOPY:DAT`. `/XO` with no `/MIR` is strictly
additive — **never `/MIR`**, it mirrors deletions. Costs an SMB share, a bigger surface.

**Hermes is the exception** — `scan_hermes` (`agent_history.py:443-460`) shells out to
`hermes sessions export` and cannot be file-synced. Keep Hermes desktop-only in v1.

**Done when:** `kb_stats` document counts increase rather than churn; laptop and desktop sessions
with identical filenames both exist as distinct documents; unqualified local keys are
byte-identical to their pre-change values.

---

## Phase 7 — Documentation

**`CLAUDE.md`** — replace *"Keep the MCP server on STDIO; do not add a network transport"* with
three bullets that bound the loosening:

```
- Keep `mcp_server.py` on STDIO and unchanged. Any network transport lives in a
  separate entry point that imports the same FastMCP instance; adding a listener
  must never alter the STDIO path.
- A network listener may bind loopback only. Reaching it from another machine goes
  through an authenticated private overlay (Tailscale serve); never bind a routable
  interface, and never expose it publicly (no Tailscale Funnel).
- Every network request is authenticated with a per-client scoped bearer token before
  any store call, with Host/Origin validation and a content-free access audit row.
  Reject anonymous requests; never log tokens, queries, or snippets.
```

**`SECURITY.md`** — replace *"Only STDIO transport is enabled. No listener..."* with a **Network
exposure** section: loopback bind + `tailscale serve` + ACL + Funnel prohibited; threat model
**in scope** (unauthenticated tailnet peer, DNS rebinding from a browser on an authorized machine,
token replay, session exhaustion from sleeping spokes, timing oracle on token comparison) and
**out of scope** (a fully compromised tailnet node holds a valid token by definition; physical
access to the hub); still no remote ingestion or deletion endpoint; audit is content-free;
listener runs without API keys in its environment.

**`README.md`** — new `## Remote access` section between `## MCP tools` and
`## Administrative CLI`: hub setup, `tailscale serve` command, token generation and scopes, spoke
registration JSON, host-qualified `agent_roots` example. Extend `## Ingestion` with the per-host
model and the additive-key asymmetry.

**`config.example.json`** — object-form `agent_roots` entry. The token belongs in `.env`, not here.

**`VERIFICATION.md`** — new `## Remote access verification` section in the existing style.

---

## Optional Phase 8 — REST surface (only if a CI agent can't speak MCP)

Two GET routes on the same Starlette app behind the same middleware, reusing the exact store
methods the MCP tools call:

- `GET /v1/search?q=&limit=&project=&sources=&since=&global=&rerank=` → `_store().search_response`
- `GET /v1/document/{document_id}?offset=&limit=` → `_store().get_document` (`store.py:4836`);
  404 when `None`, mirroring `kb_get`'s `{"found": false}` (`mcp_server.py:167`)

Non-negotiable: route through the existing `_offload` helper (`mcp_server.py:61-73`). Running
`search_response` inline stalls every concurrent MCP session — it blocks for ~290 ms of SQLite
plus a 92 MB NumPy matmul plus cross-encoder inference.

Keep it thin: no `roots` parameter (pass the Phase 2 no-cwd sentinel); echo the same
untrusted-evidence notice from `mcp_server.py:143`, since a REST caller has no server
`instructions` block to carry it; set `X-Content-Type-Options: nosniff`; **no write endpoint** —
`kb_save_memory` requires `confirmed_by_user=true` (`store.py:1580-1581`), an assertion only
meaningful in an interactive client. Do not use `@mcp.custom_route` (bypasses auth).

## Optional Phase 9 — Offline fallback

You chose "desktop, mostly on", so the default is **fail fast**: a sleeping hub gives
connection-refused and the agent surfaces "cerebras-memory unavailable". Zero code.

Cheap mitigation first: set the power plan to never sleep while plugged in, and rely on the S4U
boot task from Phase 4.

Only if that proves painful, add a nightly `VACUUM INTO` snapshot pushed over the tailnet with a
laptop-side read-only server (~900 MB per push, plus a 98 MB model cache and a second venv). Note
this breaks the single-source-of-truth invariant the rest of the design rests on, so make it loud:
`kb_save_memory` hard-disabled on the replica, `kb_stats` reporting `snapshot_age_hours`, and every
result carrying a `stale_replica` taint alongside `untrusted_evidence`. A stale answer about a
legal matter is worse than a clean failure.

---

## Verification

Run `.\.venv\Scripts\python.exe -m pytest` before and after every phase, per `CLAUDE.md`.

New tests, modelled on `tests/test_mcp.py` (which already spawns a real server subprocess with
`CEREBRAS_MEMORY_CONFIG` + `CEREBRAS_MEMORY_TEST_EMBEDDER=1` and has a `list_roots_callback`
fixture in `test_stdio_server_infers_an_unambiguous_client_root`):

- **`tests/test_http_auth.py`** — drive the app with `httpx.ASGITransport`, no network. No header
  → 401; `Basic` → 401; wrong-but-same-length bearer → 401; valid → 200. `Host: evil.example` →
  421; `Origin: https://evil.example` → 403. POST `/mcp` with `Content-Type: text/plain` → 400.
  `/healthz` without a token → 200. An `ro`-scoped token calling `kb_save_memory` → rejected.
  Response bodies never contain the token.
- **`tests/test_http_mcp_roundtrip.py`** — real uvicorn on an ephemeral loopback port +
  `streamablehttp_client` with an auth header. Same four tool names; `confirmed_by_user=False`
  errors and `True` succeeds; `kb_search` finds it; `kb_get` pages it. Assert `scope.origin` is
  **not** `process_cwd`.
- **`tests/test_store_concurrency.py`** — 16 threads × 20 `search_response` calls on a
  `HashingEmbedder`-seeded store, identical deterministic results. Targeted regressions: 8 threads
  × 500 distinct queries against `_cache_query` (raises today, passes after the lock); 8 threads
  on a cold `_load` asserting `TextEmbedding` is constructed exactly once.
- **`tests/test_ingest_multihost.py`** — two roots with identical `path.stem` and no `sessionId`
  produce **two** distinct document IDs; unqualified local keys unchanged from pre-change values;
  and **the most important test in the suite** — a declared-but-missing root makes the scan
  `successful=False`, `reconcile_source` is never called, and the document count is unchanged.
- **`tests/test_scope_remote_roots.py`** — `roots=[Path("D:/work/Alpha")]` → `origin:
  "client_root_leaf"`; ambiguous leaf → `global_ambiguous_roots`; no-cwd sentinel → `global`,
  never `process_cwd`, even with `Path.cwd()` inside `projects_root`.

**End-to-end:** extend `scripts/verify_cross_client.py` (which already spawns independent server
processes to prove write→read visibility) with `--hub-url` / `--token`, swapping `stdio_client`
for `streamablehttp_client` behind one branch — the `ClientSession` half is unchanged. Then run
from the laptop: desktop STDIO saves → laptop HTTP searches → laptop HTTP gets, with a
`list_roots_callback` returning a laptop-local path, asserting `scope.origin == "client_root_leaf"`.
Add `--reject-auth` to verify a bad token fails over the real network. Raise the client timeout
above 30 s or confirm the hub pre-warmed.

---

## Top risks

1. **Silent deletion of the existing corpus.** `_jsonl_files` (`agent_history.py:281-284`) reports
   success when *some* roots exist, and `reconcile_min_ratio` is 0.5 — a source can lose half its
   documents without tripping the floor, cascading to chunks, distillations, and receipts. Adding
   a synced root that later disappears is exactly this shape. *Mitigated by landing Phase 3 first
   and standalone, plus a `VACUUM INTO` backup before Phase 6.*
2. **Silent cross-project exposure via the cwd fallback.** `store.py:2176` inherits the hub's cwd
   whenever roots don't resolve — which is *always*, for a remote client. Results come back; they
   are simply scoped to the wrong thing, with no error. Highest consequence given the privileged
   material in the corpus. *Mitigated by Phase 2 landing before any spoke registers, keeping
   `stateless_http=False`, and asserting `origin != "process_cwd"` in the cross-machine verifier.*
3. **WAL growth and refresh contention.** A long-lived HTTP process holding read snapshots —
   notably the explicit `BEGIN` at `store.py:2242` spanning a 92 MB read — blocks WAL
   checkpointing. Overlap that with the 03:00 `--full` rebuild writing all 60,217 chunks and the
   WAL can grow into the gigabytes on a disk already holding 2.2 GB. *Mitigated by having the hub
   drop its snapshot for the duration of a refresh (the `refresh_lease` machinery at
   `store.py:1913-1960` already exposes the signal), reporting WAL size in `/healthz`, and running
   `PRAGMA wal_checkpoint(TRUNCATE)` at the end of `Refresh-CerebrasMemory.ps1`.*

---

## Critical files

| File | Role in this change |
|---|---|
| `ceribras-memory/http_server.py` | **New.** Starlette app, session manager, auth middleware, lifespan pre-warm |
| `ceribras-memory/mcp_server.py` | Imported unchanged — provides `mcp`, `_store`, `_offload`, `_client_root_paths` |
| `ceribras-memory/store.py` | `resolve_project_scope`/`_project_for_path` (2135-2179), `reconcile_source` (1725+), new `access_audit` table |
| `ceribras-memory/embeddings.py` | `_load` (54-71) and `_cache_query` (105-116) — the two concurrency bugs |
| `ceribras-memory/importers/agent_history.py` | `_jsonl_files` (281-288) guard; host-qualified `source_key` at 317, 369, 418, 500 |
| `ceribras-memory/config.py` | `_ALLOWED_SECRET_ENV_KEYS` (20), `_load_secret_env` (128-145), `agent_roots` parsing (392-394) |
| `ceribras-memory/scripts/Register-CerebrasMemory.ps1` | `-Remote`/`-HubUrl` spoke mode, preserving backup-and-surgical-replace |
| `CLAUDE.md`, `SECURITY.md`, `README.md`, `config.example.json` | The invariant rewrite and remote-access docs |
