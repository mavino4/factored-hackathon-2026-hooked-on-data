"""AI Gateway: the single path every model call goes through.

Owns provider selection (sticky per conversation), retries with backoff,
circuit breaking, failover, prompt-cache breakpoints and usage logging.
"""

import asyncio
import copy
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from anthropic.types import Message

from aiplatform.config import Settings
from aiplatform.llm.models import Route
from aiplatform.llm.resilience import (
    CircuitBreaker,
    Disposition,
    backoff_delay,
    classify,
    retry_after_seconds,
)

log = logging.getLogger(__name__)

CACHE = {"type": "ephemeral"}


@dataclass
class TextDelta:
    text: str


@dataclass
class Completed:
    message: Message
    provider: str


class GatewayError(Exception):
    pass


class ModelUnavailable(GatewayError):
    """Every provider failed or is circuit-open."""


class StreamInterrupted(GatewayError):
    """The call failed after text was already streamed; the caller must restart the turn."""


def build_params(route: Route, provider: str, *, system: str, messages: list[dict],
                 tools: list[dict] | None, model: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model or route.model.id_for(provider),
        "max_tokens": route.max_tokens,
        # Breakpoint 1: the frozen system prompt (and the tools, which render before it).
        "system": [{"type": "text", "text": system, "cache_control": CACHE}],
        # Breakpoint 2: the end of the conversation, so the next turn reads it from cache.
        "messages": _with_tail_breakpoint(messages),
    }
    if tools:
        # Deterministic order: any change in the tool list invalidates the whole cache.
        params["tools"] = sorted(tools, key=lambda t: t["name"])
    if route.model.supports_effort and model is None:
        params["thinking"] = {"type": "adaptive"}
        if route.effort:
            params["output_config"] = {"effort": route.effort}
    return params


def _with_tail_breakpoint(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    last = copy.deepcopy(messages[-1])  # never mutate stored history
    content = last["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    content[-1]["cache_control"] = CACHE
    last["content"] = content
    return [*messages[:-1], last]


class AIGateway:
    def __init__(self, clients: dict[str, Any], settings: Settings, *,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 sticky_max: int = 50_000):
        if not clients:
            raise ValueError("at least one provider client is required")
        self._clients = clients
        self._settings = settings
        # Providers that serve one configured model regardless of route (local dev).
        self._model_overrides = {"ollama": settings.ollama_model}
        self._sleep = sleep
        self._breakers = {name: CircuitBreaker() for name in clients}
        # conversation_id -> provider that last served it. Bounded LRU, in-process.
        self._sticky: OrderedDict[str, str] = OrderedDict()
        self._sticky_max = sticky_max

    def _provider_order(self, conversation_id: str | None) -> list[str]:
        order = [p for p in self._settings.providers if p in self._clients]
        sticky = self._sticky.get(conversation_id or "")
        if sticky in order:
            order.remove(sticky)
            order.insert(0, sticky)
        return [p for p in order if self._breakers[p].available()]

    def _remember(self, conversation_id: str, provider: str) -> None:
        self._sticky[conversation_id] = provider
        self._sticky.move_to_end(conversation_id)
        while len(self._sticky) > self._sticky_max:
            self._sticky.popitem(last=False)

    async def stream(self, route: Route, *, system: str, messages: list[dict],
                     tools: list[dict] | None = None,
                     conversation_id: str | None = None) -> AsyncIterator[TextDelta | Completed]:
        """Stream one model turn. Yields text deltas, then exactly one ``Completed``."""
        last_error: BaseException | None = None
        for provider in self._provider_order(conversation_id):
            client = self._clients[provider]
            breaker = self._breakers[provider]
            params = build_params(route, provider, system=system, messages=messages, tools=tools,
                                  model=self._model_overrides.get(provider))
            for attempt in range(self._settings.max_attempts_per_provider):
                if not breaker.acquire():
                    break
                emitted = False
                try:
                    async with client.messages.stream(**params) as stream:
                        async for event in stream:
                            if event.type == "text":
                                emitted = True
                                yield TextDelta(event.text)
                        message = await stream.get_final_message()
                except Exception as exc:  # classified below
                    disposition = classify(exc)
                    if disposition is Disposition.FATAL:
                        raise
                    breaker.record_failure()
                    last_error = exc
                    log.warning("model call failed", extra={
                        "provider": provider, "attempt": attempt, "error": repr(exc)})
                    if emitted:
                        raise StreamInterrupted(str(exc)) from exc
                    if disposition is Disposition.FAILOVER:
                        break
                    if attempt + 1 < self._settings.max_attempts_per_provider:
                        await self._sleep(backoff_delay(attempt, retry_after=retry_after_seconds(exc)))
                    continue
                finally:
                    # Also runs on cancellation (client disconnect), so a probe is never stuck.
                    breaker.release()

                breaker.record_success()
                if conversation_id:
                    self._remember(conversation_id, provider)
                _log_usage(route, provider, message)
                yield Completed(message=message, provider=provider)
                return
        raise ModelUnavailable("no model provider available") from last_error

    async def complete(self, route: Route, **kwargs: Any) -> Completed:
        """Non-interactive helper: stream (for timeout safety) and return the final message."""
        async for event in self.stream(route, **kwargs):
            if isinstance(event, Completed):
                return event
        raise GatewayError("stream ended without a final message")


def _log_usage(route: Route, provider: str, message: Message) -> None:
    usage = message.usage
    log.info("model usage", extra={
        "route": route.name,
        "provider": provider,
        "model": message.model,
        "stop_reason": message.stop_reason,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_input_tokens,
        "cache_write_tokens": usage.cache_creation_input_tokens,
    })
