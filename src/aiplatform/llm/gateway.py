"""AI Gateway: the single path every model call goes through.

Owns provider selection (sticky per conversation), retries with backoff,
circuit breaking, failover, prompt-cache breakpoints and usage logging.
"""

import asyncio
import copy
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from anthropic.types import Message

from aiplatform import metrics
from aiplatform.config import Settings
from aiplatform.llm.models import Route, prices_for
from aiplatform.llm.resilience import (
    CircuitBreaker,
    Disposition,
    backoff_delay,
    classify,
    retry_after_seconds,
)
from aiplatform.tracing import llm_span

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
                 tools: list[dict] | None, model: str | None = None,
                 system_suffix: str | None = None,
                 tool_choice: dict | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model or route.model.id_for(provider),
        "max_tokens": route.max_tokens,
        # Breakpoint 1: the frozen system prompt (and the tools, which render before it).
        "system": [{"type": "text", "text": system, "cache_control": CACHE}],
        # Breakpoint 2: the end of the conversation, so the next turn reads it from cache.
        # Stored messages may carry app-only keys (e.g. "author"); the API gets role/content.
        "messages": _with_tail_breakpoint(
            [{"role": m["role"], "content": m["content"]} for m in messages]),
    }
    if system_suffix:  # per-request text (e.g. reply language), after the cached prefix
        params["system"].append({"type": "text", "text": system_suffix})
    if tools:
        # Deterministic order: any change in the tool list invalidates the whole cache.
        params["tools"] = sorted(tools, key=lambda t: t["name"])
    if tool_choice:
        params["tool_choice"] = tool_choice
    # A forced tool choice can't be combined with thinking.
    if route.model.supports_effort and model is None and not tool_choice:
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

    def _first_event_timeout(self, route: Route, provider: str) -> float:
        if provider == "ollama":  # local model may still be loading onto the GPU
            return max(route.first_event_timeout_s, self._settings.ollama_first_event_timeout_s)
        return route.first_event_timeout_s

    def _remember(self, conversation_id: str, provider: str) -> None:
        self._sticky[conversation_id] = provider
        self._sticky.move_to_end(conversation_id)
        while len(self._sticky) > self._sticky_max:
            self._sticky.popitem(last=False)

    async def stream(self, route: Route, *, system: str, messages: list[dict],
                     tools: list[dict] | None = None,
                     conversation_id: str | None = None,
                     system_suffix: str | None = None,
                     tool_choice: dict | None = None) -> AsyncIterator[TextDelta | Completed]:
        """Stream one model turn. Yields text deltas, then exactly one ``Completed``."""
        last_error: BaseException | None = None
        for provider in self._provider_order(conversation_id):
            client = self._clients[provider]
            breaker = self._breakers[provider]
            params = build_params(route, provider, system=system, messages=messages, tools=tools,
                                  model=self._model_overrides.get(provider),
                                  system_suffix=system_suffix, tool_choice=tool_choice)
            for attempt in range(self._settings.max_attempts_per_provider):
                if not breaker.acquire():
                    break
                emitted = False
                started = time.perf_counter()
                try:
                    # One traced generation per attempt (a no-op unless tracing is on).
                    async with _llm_span(route, provider, params, attempt) as span:
                        # Fail fast if the provider stalls before sending anything; once the
                        # stream has started, the SDK's per-read timeout applies.
                        async with asyncio.timeout(
                                self._first_event_timeout(route, provider)) as deadline:
                            async with client.messages.stream(**params) as stream:
                                async for event in stream:
                                    if deadline.when() is not None:
                                        deadline.reschedule(None)
                                    if event.type == "text":
                                        if not emitted:
                                            metrics.LLM_TTFT.labels(route.name, provider).observe(
                                                time.perf_counter() - started)
                                            # Langfuse's time to first token.
                                            span.update(completion_start_time=datetime.now(UTC))
                                        emitted = True
                                        yield TextDelta(event.text)
                                message = await stream.get_final_message()
                        span.update(output=message.model_dump(mode="json"),
                                    usage_details=_usage_details(message),
                                    cost_details={"total": _cost(message)})
                except Exception as exc:  # classified below
                    disposition = classify(exc)
                    kind = "timeout" if isinstance(exc, TimeoutError) else disposition.value
                    metrics.LLM_ERRORS.labels(provider, kind).inc()
                    if disposition is Disposition.FATAL:
                        raise
                    breaker.record_failure()
                    metrics.LLM_BREAKER_OPEN.labels(provider).set(int(breaker.is_open))
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
                metrics.LLM_BREAKER_OPEN.labels(provider).set(0)
                metrics.LLM_DURATION.labels(route.name, provider).observe(
                    time.perf_counter() - started)
                if conversation_id:
                    self._remember(conversation_id, provider)
                _record_usage(route, provider, message)
                yield Completed(message=message, provider=provider)
                return
        raise ModelUnavailable("no model provider available") from last_error

    async def complete(self, route: Route, **kwargs: Any) -> Completed:
        """Non-interactive helper: stream (for timeout safety) and return the final message."""
        async for event in self.stream(route, **kwargs):
            if isinstance(event, Completed):
                return event
        raise GatewayError("stream ended without a final message")


def _llm_span(route: Route, provider: str, params: dict[str, Any], attempt: int):
    return llm_span(f"{route.name}.{provider}", model=params["model"], params=params,
                    metadata={"route": route.name, "provider": provider, "attempt": attempt})


def _usage_details(message: Message) -> dict[str, int]:
    # Langfuse's Anthropic usage keys. The cost is sent with it (our price table, the
    # same as the metrics), since Langfuse may not know every model name.
    usage = message.usage
    return {"input": usage.input_tokens, "output": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0}


def _cost(message: Message) -> float:
    usage = message.usage
    return prices_for(message.model).cost(
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0)


def _record_usage(route: Route, provider: str, message: Message) -> None:
    usage = message.usage
    tokens = {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens or 0,
        "cache_write": usage.cache_creation_input_tokens or 0,
    }
    for kind, count in tokens.items():
        metrics.LLM_TOKENS.labels(route.name, provider, message.model, kind).inc(count)
    cost = _cost(message)
    metrics.LLM_COST.labels(route.name, provider, message.model).inc(cost)
    prompt_tokens = tokens["input"] + tokens["cache_read"] + tokens["cache_write"]
    if provider != "ollama" and prompt_tokens < route.model.min_cacheable_tokens:
        metrics.LLM_BELOW_CACHE_MIN.labels(route.name).inc()
    log.info("model usage", extra={
        "route": route.name,
        "provider": provider,
        "model": message.model,
        "stop_reason": message.stop_reason,
        "anthropic_request_id": getattr(message, "_request_id", None),
        **{f"{kind}_tokens": count for kind, count in tokens.items()},
        "cost_usd": round(cost, 6),
    })
