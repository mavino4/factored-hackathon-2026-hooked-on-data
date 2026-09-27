# ADR-0002: Decoupled token streaming over SSE

- **Status:** Deferred — not needed at current load (6k–10k/day); see architecture §0 for the trigger to adopt
- **Date:** 2026-09-26

## Context
Chat replies stream for 5–60 s, and agent tasks can stream for minutes. If the pod holding the client connection is also the pod calling the LLM, any pod restart, deploy, or network blip kills the response. Mobile clients also reconnect often.

## Decision
- The client ↔ platform transport is **Server-Sent Events** (HTTP, CDN/proxy friendly, built-in `Last-Event-ID` resume). WebSocket is only for features that need true bidirectional real-time.
- **Generation is separate from delivery.** The AI Gateway / Agent Runtime writes every chunk (text delta, tool-call event, status, final usage) to a **Redis Stream** keyed by `message_id`, with a TTL of about 10 min after completion.
- The Stream Gateway only tails Redis Streams and forwards them. Event IDs are the Redis stream entry IDs, so a reconnecting client resumes from `Last-Event-ID` with nothing lost or duplicated.
- If the *generator* dies mid-stream, the AI Gateway restarts the generation and emits a `restart` event so the client clears the partial text.
- Final messages are persisted to the conversation store with their full content blocks.

## Consequences
- ✅ Gateway pods can be killed or redeployed freely. Multi-device viewing of the same stream works for free.
- ✅ Connection scaling and generation scaling are independent.
- ❌ Redis write throughput is proportional to token throughput. Batch deltas (for example every 50 ms) to cut ops, and size Redis per cell.
- ❌ An extra hop adds a few ms. This is negligible compared to model latency.
