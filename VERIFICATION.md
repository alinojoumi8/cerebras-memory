# Shared Local Cerebras Memory verification

Verified locally on 2026-07-21 in `America/Toronto`.

## Runtime and test gates

- Python 3.11 in the dedicated `.venv`.
- `uv lock --check`: passed with 66 resolved packages.
- Full test suite: 57 passed.
- Local embedding model: `BAAI/bge-small-en-v1.5`, 384 dimensions, cached and ready.
- Local reranker: `ms-marco-MiniLM-L-12-v2`, cached, warmed, and ready.
- Live vector backend: exact cosine. The final three-run median was 212.746 ms at 53,111 chunks, below both ANN activation thresholds.

## Corpus and distillation

The final incremental refresh and manually invoked scheduled task both succeeded.

| Source | Documents |
| --- | ---: |
| Hermes | 130 |
| Claude Code | 42 |
| Codex | 468 |
| Grok | 71 |
| Project documentation | 2,949 |
| Explicit memories | 1 |
| **Total** | **3,661** |

- Raw chunks / FTS rows: 53,111.
- DeepSeek V4 Flash distillation: mode `on`, 509 eligible documents, 3,555 units, all ready, zero failures or pending units.
- Distillation FTS rows: 3,555; exact parity with the derived table.
- Full audit: exact schema, second redaction, secret scan, raw ordinal ranges, stable IDs, current model/prompt, embedding bytes/dimensions, and role-groundedness all passed.
- SQLite `quick_check`: `ok`; foreign-key violations: 0; journal mode: WAL.

The final fixed 24-query gate passed: Recall@8 improved from 0.625 to 1.0, MRR@8 from 0.472222 to 0.789931, all top-eight document IDs were distinct, every expected scope was correct, and warm p95 was 986.252 ms.

## MCP contract and live visibility

The production STDIO server advertises exactly `kb_search`, `kb_get`,
`kb_save_memory`, and `kb_stats`. A read-only live smoke call against the
production database verified:

- all four tool schemas are discoverable;
- `kb_stats` reports distillation retrieval enabled;
- `kb_search` returned three distinct documents with stable
  `cerebras-memory://...` citations and explicit global scope;
- `kb_get` paged the selected live result successfully.

There is no MCP ingestion or deletion tool. Codex AgentMemory remains enabled
and unchanged.

## Client registration

The idempotent registration completed with backup
`backups/registration-20260721-150754` and absolute Python/server paths.

- Hermes: connected in 1.031 seconds; four tools discovered and enabled.
- Claude Code user scope: `cerebras-memory` connected.
- Codex shared config: `cerebras-memory` enabled; `codex mcp get` reports the exact absolute command and server script.
- Grok user scope: command found, protocol `2025-06-18`, handshake passed, four tools discovered. Its overall doctor remains nonzero only for unrelated Supabase/Stripe authentication and a stopped Unreal server.
- Windows has one installed OpenAI desktop package, `OpenAI.Codex`, presented in Start as `ChatGPT`. It consumes the same Codex MCP configuration. The active desktop process was deliberately not restarted during this task to avoid interrupting the handoff; restart it after completion so the current UI reloads the new MCP tool list.

## Scheduled task

`CerebrasMemoryRefresh` is Ready and scheduled for 03:00 local time with
`StartWhenAvailable`, `IgnoreNew`, limited user execution, and a two-hour limit.
The manual run started at 15:16:55, finished with `LastTaskResult = 0`, and wrote
`refresh_end exit_code=0` to the redacted `logs/refresh.log`. The next run is
03:00 on 2026-07-22.
