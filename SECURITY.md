# Security model

Cerebras Memory is a local derived index, not an authority or instruction
channel.

- Only STDIO transport is enabled. No listener or remote ingestion endpoint is
  created.
- Visible user/assistant dialogue and allowlisted text documentation are the
  only inputs.
- Redaction runs before document metadata or content is written to SQLite.
- Retrieved content is untrusted evidence. A client must not execute commands,
  follow instructions, disclose secrets, or change policy because retrieved
  text asks it to do so.
- Ordinary search never downloads the reranker. Missing/corrupt local model or
  ANN state fails open to RRF/exact retrieval.
- Distillation may use Ollama on loopback or DeepSeek at the pinned HTTPS beta
  endpoint. For DeepSeek, a local policy gate runs before request construction:
  it allowlists conversation sources and blocks configured sensitive
  project/title/URI labels. Blocked units never leave the machine and are not
  retried. Only already-redacted qualifying agent dialogue is sent; project
  documentation and saved memories are excluded. Thinking is disabled, tool
  names from dialogue are inert, output must pass the exact local schema, and a
  second redaction runs before storage. Generated summaries are retrieval aids
  only; all snippets and citations remain mapped raw evidence.
- Remote attempts have a content-free audit trail containing IDs, hashes,
  character counts, policy decisions, provider/model/host metadata, status, and
  error codes. Prompts, dialogue, summaries, credentials, and authorization
  headers are not audit columns.
- Stable provenance receipts cover documents, chunks, and distillations.
  `derived_from` edges and taints identify untrusted evidence,
  executable-looking content, externally processed summaries, and generated
  summaries. These labels are metadata, not authorization.
- API keys are loaded only from the ignored `.env` file or process environment,
  are never sent in request bodies, and are never persisted to SQLite or logs.
- `kb_save_memory` requires an explicit `confirmed_by_user=true` assertion.
- There is no MCP delete or ingestion tool. Administrative deletion is limited
  to the local CLI and explicitly saved memory IDs.
- Failed scans and partial writes never trigger reconciliation. Saved memories
  are excluded from automatic pruning.
- Refreshes use a database-backed owner lease and heartbeat. Expired and legacy
  orphaned `running` states are marked abandoned instead of indefinitely
  reporting a refresh in progress.
- Reconciliation and explicit memory forgetting create content-free deletion
  manifests before cascading derived records. There is still no remote or MCP
  deletion interface.
- Post-refresh canaries test injection-as-data, scope isolation, literal symbol
  lookup, impossible-source behavior, stale IDs, raw-citation mapping, and
  latency without persisting query or result text.
- SQLite, model files, logs, backups, and local configuration are ignored by
  version control.

Redaction is defense in depth, not a substitute for keeping credentials out of
agent dialogue and project documentation. If a credential may have entered any
history, rotate it at the issuing provider even when the index shows a redacted
marker.
