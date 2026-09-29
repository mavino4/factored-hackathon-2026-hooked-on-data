# ADR-0006: LangGraph for agent and chat orchestration

- **Status:** Accepted
- **Date:** 2026-09-29

## Context
The agent tool loop (`agent/loop.py`) and chat turns (`chat/service.py`) were hand-written
async generators. We want them on a standard orchestration framework, so new steps
(routing, extra tools, sub-agents) are graph nodes instead of more branches in a loop,
without losing any v1 behavior or the tests that pin it.

## Decision
- **Only orchestration moves to LangGraph** (`agent/graph.py`, `chat/graph.py`). The agent is
  a `StateGraph` with `apply_decision → call_model ⇄ run_tools → finish`; chat is a one-node
  graph. Node logic is the previous loop's code, unchanged.
- **FastAPI stays the HTTP layer** (auth, rate limit, quota, SSE, UI), and **every model call
  still goes through the AI Gateway** (ADR-0003). We don't use LangChain chat models: they
  would drop the breaker, sticky failover, cache breakpoints, first-event timeout and metrics.
- **No checkpointer, no `interrupt()`.** State already lives in `messages` and
  `pending_actions`, so an approval starts a new run at `apply_decision`. Runs stay
  stateless per request and work across replicas with no extra tables or driver.
- **Events are pulled, not pushed.** Nodes emit through `graph_stream.emitter()`, which waits
  for the client to take each event. LangGraph's custom stream doesn't wait, so after a
  client disconnect a node would keep running and store the reply; with the pull model the
  run is cancelled at that point, as before.
- The public interfaces (`AgentRunner.run/decide`, `ChatService.send/regenerate`) and the
  event types are unchanged. `tests/test_agent_equivalence.py` pins exact event sequences,
  stored history, usage, errors and cancellation; it was written against the old loop
  and passes on both.

## Consequences
- ✅ Standard graph structure; new steps are nodes and edges.
- ✅ No behavior change: all existing tests pass unmodified; banking evals on
  `qwen2.5:7b` score the same (76%, 19/25).
- ❌ New dependencies: `langgraph` pulls `langchain-core`, `langgraph-sdk` and `langsmith`
  (which cap `websockets` below 17; the app doesn't use WebSockets). LangSmith tracing
  stays off unless `LANGSMITH_TRACING` is set; don't enable it with customer data without
  a data-processing review.
- ↔ ADR-0004 (durable runtime on Temporal) still stands for long tasks. The same nodes can
  become Temporal activities, or we can add a LangGraph checkpointer, when a task outgrows
  one HTTP request.
