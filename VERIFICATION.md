# Shared Local Cerebras Memory verification

Verified locally on 2026-07-24 in `America/Toronto`.

## Runtime and test gates

- Python 3.11.15 in the dedicated `.venv`.
- `uv lock --check`: passed with 66 resolved packages.
- Full test suite: 65 passed in 7.67 seconds.
- Python bytecode compilation: passed.
- Local embedding model: `BAAI/bge-small-en-v1.5`, 384 dimensions, cached and ready.
- Local reranker: `ms-marco-MiniLM-L-12-v2`, cached and warmed. Its CPU session
  is bounded to 24 intra-op threads; ingestion uses one process, batches of 16,
  and no FastEmbed multiprocessing.
- Live vector backend: exact cosine. The three-run median is 282.198 ms at
  55,718 chunks, below the 100,000-chunk and 750 ms ANN activation thresholds.

## Corpus, distillation, and integrity

The final incremental refresh and manually invoked scheduled task both
succeeded.

| Source | Documents |
| --- | ---: |
| Hermes | 133 |
| Claude Code | 42 |
| Codex | 523 |
| Grok | 72 |
| Project documentation | 3,095 |
| Explicit memories | 51 |
| **Total** | **3,916** |

- Raw chunks / FTS rows: 55,718.
- DeepSeek V4 Flash distillation: mode `on`; 3,661 stored derived rows and FTS
  rows, 3,020 ready units, 645 policy-blocked planned units, zero retryable
  pending units, and zero failures.
- The content-free outbound audit contains 77 succeeded allowed attempts and
  645 blocked decisions, with no pending or failed entries.
- Provenance parity is exact: 3,916 active document receipts, 55,718 active
  chunk receipts, and 3,661 active distillation receipts. No deletion manifests
  were required by this refresh.
- Exact DeepSeek API-key scans of SQLite and the refresh log found zero hits.
- SQLite `quick_check`: `ok`; foreign-key violations: 0; journal mode: WAL.

The fixed 24-query gate passed: Recall@8 improved from 0.625 to 1.0, MRR@8 from
0.425347 to 0.816716, all top-eight document IDs were distinct, every expected
scope was correct, and warm p95 was 1,491.87 ms.

The recorded six-case canary suite also passed all cases with p95
1,401.632 ms. Its latest run ID is
`canary_81cc9cece4e744fb91860904d7af87bd`.

## Refresh and resource behavior

The successful scheduled run used refresh run ID
`run_1d1abd07795549ef9eccc3c7b97b6bf0`. It started at 15:15:10 local,
completed at 15:26:57, and left no active lease.

The original ingestion path was observed creating eight Python embedding
workers at roughly 2.4 GiB private memory each. The replacement performs local
embedding in the refresh process with no multiprocessing. The successful
scheduled refresh had no embedding child workers and peaked at approximately
0.6 GiB resident / 1.85 GiB private memory.

## MCP contract and live visibility

The production STDIO server advertises exactly `kb_search`, `kb_get`,
`kb_save_memory`, and `kb_stats`. Search/get responses now include stable
provenance receipts and taints while retaining the original raw citations,
document IDs, and chunk IDs. `kb_stats` reports schema version 3, refresh lease
state, remote-policy audit counts, provenance parity, quality-gate state, and
the exact/ANN decision.

The read-only production smoke test returned three distinct documents, stable
citations, explicit global scope, and a successful `kb_get`.

There is no MCP ingestion or deletion tool. Codex AgentMemory remains enabled
and unchanged.

## Client registration

The idempotent registration completed with backup
`backups/registration-20260724-153809` and absolute Python/server paths.

- Hermes connected in 1.281 seconds and discovered four enabled tools.
- Claude Code user scope reports `cerebras-memory` connected.
- Codex reports `cerebras-memory` enabled at the exact absolute command and
  retained the unrelated `agentmemory` entry unchanged.
- Grok passed command discovery, process start, protocol `2025-06-18`
  handshake, and discovery of four tools.

## Scheduled task

`CerebrasMemoryRefresh` is Ready and scheduled for 03:00 local time with
`StartWhenAvailable`, `IgnoreNew`, limited non-elevated user execution, battery
start/continuation allowed, idle-end continuation, and a six-hour limit. The
manual run finished with `LastTaskResult = 0` and streamed
`refresh_end exit_code=0` to the redacted `logs/refresh.log`. The next run is
03:00 on 2026-07-25.

Codex and the installed desktop ChatGPT/Codex host consume the same Codex MCP
configuration. Restart desktop clients after deployment so they reload the
updated tool schema.
