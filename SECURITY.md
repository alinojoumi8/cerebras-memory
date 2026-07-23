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
  endpoint. Only already-redacted qualifying agent dialogue is sent; project
  documentation and saved memories are excluded. Thinking is disabled, tool
  names from dialogue are inert, output must pass the exact local schema, and a
  second redaction runs before storage. Generated summaries are retrieval aids
  only; all snippets and citations remain mapped raw evidence.
- API keys are loaded only from the ignored `.env` file or process environment,
  are never sent in request bodies, and are never persisted to SQLite or logs.
- `kb_save_memory` requires an explicit `confirmed_by_user=true` assertion.
- There is no MCP delete or ingestion tool. Administrative deletion is limited
  to the local CLI and explicitly saved memory IDs.
- Failed scans and partial writes never trigger reconciliation. Saved memories
  are excluded from automatic pruning.
- SQLite, model files, logs, backups, and local configuration are ignored by
  version control.

Redaction is defense in depth, not a substitute for keeping credentials out of
agent dialogue and project documentation. If a credential may have entered any
history, rotate it at the issuing provider even when the index shows a redacted
marker.
