"""Langfuse tracing: every chat or agent run becomes one trace, masked before it leaves.

The trace holds one span per graph step (``traced_node``), and under each step the model
calls it made (``generation``, from the gateway, with tokens and cost) and the tools it
ran. Traces carry
the conversation ID as ``session_id``, so Langfuse's Sessions view shows a conversation
turn by turn. Their ``user_id`` is the bank customer ID when sign-in knows it (password
mode), which names no one; otherwise a keyed hash of the user ID. The username, which
comes from the customer's email, is never sent.

Every payload goes through ``privacy.Masker`` first (see ``privacy.py``). Off unless
``LANGFUSE_TRACING_ENABLED`` is true; the Langfuse server is self-hosted (see
``deploy/langfuse``).
"""

import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any

from langfuse import Langfuse, propagate_attributes

from aiplatform import privacy
from aiplatform.config import Settings

log = logging.getLogger(__name__)

# The Langfuse client and masker of the run in progress (set by Tracing.run).
_active: ContextVar["Tracing | None"] = ContextVar("tracing_active", default=None)


class Tracing:
    def __init__(self, client: Langfuse | None = None, masker: privacy.Masker | None = None):
        self._client = client
        self._masker = masker

    @classmethod
    def from_settings(cls, settings: Settings) -> "Tracing":
        if not settings.langfuse_enabled:
            return cls()
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            log.warning("Langfuse tracing is on but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
                        "are not set: tracing off")
            return cls()
        masker = privacy.Masker(settings.trace_hash_key.get_secret_value().encode(),
                                mask_amounts=settings.trace_mask_amounts)
        client = Langfuse(public_key=settings.langfuse_public_key,
                          secret_key=settings.langfuse_secret_key.get_secret_value(),
                          base_url=settings.langfuse_host, environment=settings.env,
                          mask=masker.mask)
        log.info("Langfuse tracing on", extra={"host": settings.langfuse_host,
                                               "mask_amounts": settings.trace_mask_amounts})
        return cls(client, masker)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def run(self, config: dict) -> Iterator[dict]:
        """Wrap one graph run (see ``run_config``); yields the config to run it with."""
        trace_input = config.get("trace_input")  # what started the run (masked on the way)
        history = config.get("trace_sensitive")  # earlier turns, whose values stay masked
        config = {k: v for k, v in config.items()
                  if k not in ("trace_input", "trace_sensitive")}
        if self._client is None:
            yield config
            return
        metadata = config.get("metadata", {})
        user_id = metadata.get("customer_id") or self._masker.pseudonym(metadata["user_id"])
        name = config.get("run_name", "run")
        details = {k: v for k, v in metadata.items()
                   if k not in ("user_id", "customer_id", "thread_id")}
        with (privacy.run_scope(),
              self._client.start_as_current_observation(
                  as_type="agent", name=name, input=trace_input, metadata=details),
              propagate_attributes(user_id=user_id, session_id=metadata["thread_id"],
                                   trace_name=name, tags=config.get("tags"))):
            # Names, last 4 digits and amounts that tools returned in earlier turns are
            # masked from the first step, not only once a tool runs again in this turn.
            self._masker.register(history)
            token = _active.set(self)
            try:
                yield config
            finally:
                _active.reset(token)

    def flush(self) -> None:
        """Send pending traces (on shutdown)."""
        if self._client is not None:
            self._client.shutdown()


class _NoSpan:
    def update(self, **_: Any) -> None:
        pass


@asynccontextmanager
async def llm_span(name: str, *, model: str, params: dict[str, Any],
                   metadata: dict[str, Any]) -> AsyncIterator[Any]:
    """A ``generation`` for one model call (a no-op outside a traced run)."""
    tracing = _active.get()
    if tracing is None:
        yield _NoSpan()
        return
    inputs = {k: params[k] for k in ("system", "messages", "tools") if k in params}
    with tracing._client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=inputs, metadata=metadata,
            model_parameters={"max_tokens": params.get("max_tokens")}) as span:
        yield span


@asynccontextmanager
async def tool_span(name: str, args: dict[str, Any],
                    metadata: dict[str, Any]) -> AsyncIterator[Any]:
    """A ``tool`` observation for one tool call (a no-op outside a traced run)."""
    tracing = _active.get()
    if tracing is None:
        yield _NoSpan()
        return
    with tracing._client.start_as_current_observation(
            as_type="tool", name=name, input=args, metadata=metadata) as span:
        yield span


def traced_node(name: str, describe: Callable[[Any], Any] | None = None):
    """Give a graph step its own span, so the model calls and tools it makes nest under it.
    ``describe(state)`` is the step's input in the trace (keep it short); the state
    update it returns is its output. A no-op outside a traced run."""
    def wrap(fn: Callable[[Any], Awaitable[dict]]):
        @functools.wraps(fn)
        async def node(state):
            tracing = _active.get()
            if tracing is None:
                return await fn(state)
            with tracing._client.start_as_current_observation(
                    as_type="span", name=name,
                    input=describe(state) if describe else None) as span:
                update = await fn(state)
                span.update(output=update or None)
                return update
        return node
    return wrap


def register_sensitive(data: Any) -> None:
    """Mask the values in ``data`` (a tool output) wherever they appear later in the run."""
    tracing = _active.get()
    if tracing is not None:
        tracing._masker.register(data)


def flag_prompt_injection(reason: str) -> None:
    """Mark the current trace as a manipulation attempt: the step in progress gets level
    WARNING and the trace a ``prompt_injection`` score, both filterable in Langfuse."""
    tracing = _active.get()
    if tracing is None:
        return
    reason = tracing._masker.mask(data=reason)
    tracing._client.update_current_span(level="WARNING", status_message=reason)
    tracing._client.score_current_trace(name="prompt_injection", value=1,
                                        data_type="BOOLEAN", comment=reason)


def run_config(name: str, *, user_id: str, conversation_id: str, trace_input: Any = None,
               trace_sensitive: Any = None, customer_id: str | None = None,
               **metadata) -> dict:
    """LangGraph config naming the run and grouping it into the conversation's session.
    ``trace_input`` is the root span's input (e.g. the customer's message);
    ``trace_sensitive`` is the earlier conversation, whose tool results are masked in
    this run too; ``customer_id``, when known, is the trace's user ID."""
    if customer_id is not None:
        metadata["customer_id"] = customer_id
    return {"run_name": name, "tags": [name], "trace_input": trace_input,
            "trace_sensitive": trace_sensitive,
            "metadata": {"thread_id": conversation_id, "user_id": user_id, **metadata}}
