# ADR-0003: Central AI Gateway with multi-provider failover

- **Status:** Accepted in reduced form — v1 implements this in-process (see architecture §0)
- **Date:** 2026-09-26

## Context
LLM capacity (tokens/min) and cost are the dominant constraints. Provider rate limits, regional outages, refusals, and prompt-cache behaviour need consistent handling. Spreading this logic across services leads to retry storms, cache misses, and cost we can't see.

## Decision
All model calls go through one internal **AI Gateway** service (deployed per cell), which owns:
- **Provider adapters** built on the official Anthropic SDK clients: the first-party Claude API as primary, Amazon Bedrock (Mantle client) and Google Vertex AI as secondaries.
- **Route-based config**: route → model, effort, `max_tokens`, provider priority list, failover eligibility.
- **Sticky routing** per conversation (provider + region + model), because prompt caches are scoped per provider and per model.
- **Token-aware quotas** per tenant/user, and concurrency limits per provider-region that match contracted capacity.
- **Retries** with exponential backoff and full jitter, honoring `retry-after`, for 429/5xx/overloaded/connection errors only. There is a retry budget, and the SDK's own retries are disabled or coordinated so we don't retry twice.
- **Circuit breakers** per provider-region, with failover down the priority list.
- **Refusal handling**: always check `stop_reason`. Use server-side `fallbacks` on the Claude API and the SDK's client-side refusal-fallback middleware on Bedrock/Vertex.
- **Metering** of `usage` (including cache read/write tokens) to the event bus.
- **Load shedding** by priority class (see ADR-0005).

Failover-eligible routes use only the feature subset that works on all three providers (Messages, streaming, tool use, structured outputs, adaptive thinking/effort, prompt caching, token counting). Features available only on the Claude API (Batches, Files API, task budgets, MCP connector, code execution, web fetch) are used only on routes marked non-failover, or with an explicit degraded path.

## Alternatives considered
- **Off-the-shelf LLM proxy** (LiteLLM, Envoy AI Gateway, etc.): faster to start, but it usually translates to a lowest-common-denominator or OpenAI-style schema and loses Claude-specific features (cache control, thinking, refusal fallbacks). It could be revisited as a thin layer *under* our gateway.
- **Direct SDK calls from each service**: rejected. Quotas, failover and metering would be inconsistent.

## Consequences
- ✅ One place to tune cost, capacity and reliability. Consistent observability.
- ❌ It is a critical dependency, so it must be horizontally scaled per cell, stateless apart from Redis, and heavily tested.
- ❌ Failover to a secondary provider starts with a cold cache: one full-price prompt write per moved conversation.
