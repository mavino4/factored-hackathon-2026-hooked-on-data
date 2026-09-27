# ADR-0005: Admission control and degradation ladder

- **Status:** Deferred — not needed at current load (6k–10k/day); see architecture §0 for the trigger to adopt
- **Date:** 2026-09-26

## Context
Spiky global traffic can exceed contracted LLM capacity or budget in minutes. Adding compute doesn't raise provider token limits. Without a plan, overload becomes random 429s and timeouts for everyone.

## Decision
- Every request carries a **priority class**: interactive-paid > interactive-free > agent-background > batch.
- The AI Gateway tracks capacity headroom per provider-region (in-flight tokens and 429 rate) and budget burn. As pressure rises, it steps down this ladder, starting with the lowest priority:
  1. Lower `effort` on chat routes.
  2. Pause background and batch work (it's queued, not dropped).
  3. Route *new* free-tier conversations to a cheaper model. Existing conversations stay put to keep their caches.
  4. Admission queue with an honest "high demand" UI and position/ETA.
  5. Reject new sessions for the lowest class. Existing conversations continue.
- Steps are automatic, with hysteresis, and every transition is logged and alerted.
- Known events (launches, campaigns) trigger scheduled pre-scaling and capacity reservations with providers.

## Consequences
- ✅ Degradation is predictable and fair, and paying users are protected.
- ✅ Budget overruns are bounded.
- ❌ Product needs to design and accept the degraded UX states.
- ❌ Thresholds need tuning through load tests and game days.
