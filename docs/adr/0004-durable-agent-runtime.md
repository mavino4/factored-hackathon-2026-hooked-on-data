# ADR-0004: Durable agent runtime with append-only history

- **Status:** Accepted in reduced form — v1 implements this in-process (see architecture §0)
- **Date:** 2026-09-26

## Context
Agent tasks run many model↔tool iterations over minutes. Worker crashes, deploys and provider failovers must not lose progress or run side effects twice. Newer Claude models also reject requests whose earlier turns (with thinking blocks) were edited, and edits destroy prompt-cache reuse.

## Decision
- Run each agent task as a **durable workflow** (Temporal, which is portable and can be self-hosted or managed). Model calls and tool calls are activities with timeouts, retry policies and **idempotency keys**.
- Use a **manual or SDK tool-runner loop inside the workflow**, not a vendor-hosted agent platform, so the same runtime works on every provider.
- **Conversation history is append-only.** We append full `response.content` (including thinking, tool_use and compaction blocks) and never rewrite earlier turns. Per-turn instructions go in new messages, not edits.
- Tool rules: run parallel tool calls concurrently and return all results in a single message. Report errors as `is_error: true`. Validate inputs against the schema and use `strict: true` tool definitions.
- Guardrails per task: max iterations, max wall-clock, max tokens. Irreversible tools pause the workflow for **human approval** (a workflow signal).
- Tool results are untrusted input. Tools run with the requesting user's permissions.

## Consequences
- ✅ Tasks survive worker loss and resume exactly where they stopped. Complete audit trail.
- ✅ Append-only history keeps caches warm and is compatible with thinking-block validation.
- ❌ Temporal is another stateful system to operate (or pay for).
- ❌ Tool implementations must be idempotent or deduplicated by key.
