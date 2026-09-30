"""A scripted stand-in for an Anthropic SDK async client (no network)."""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
from anthropic.types import Message


def make_message(*blocks: dict, stop_reason: str = "end_turn", model: str = "claude-haiku-4-5") -> Message:
    return Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": model,
        "content": list(blocks) or [{"type": "text", "text": "ok"}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })


def text_reply(text: str) -> tuple[list[str], Message]:
    return [text], make_message({"type": "text", "text": text})


def status_error(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.example/v1/messages")
    response = httpx2.Response(status, request=request, headers={"retry-after": "1"})
    cls = {400: anthropic.BadRequestError, 401: anthropic.AuthenticationError,
           429: anthropic.RateLimitError, 500: anthropic.InternalServerError}.get(
               status, anthropic.APIStatusError)
    return cls(f"status {status}", response=response, body=None)


class MidStreamFailure:
    """Emit ``texts`` then raise ``error``."""

    def __init__(self, texts: list[str], error: Exception):
        self.texts, self.error = texts, error


class Stall:
    """Send nothing for ``seconds`` (a stalled provider), then reply with ``outcome``."""

    def __init__(self, seconds: float, outcome):
        self.seconds, self.outcome = seconds, outcome


class _Stream:
    def __init__(self, outcome):
        self._outcome = outcome

    async def __aenter__(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        if isinstance(self._outcome, Stall):
            await asyncio.sleep(self._outcome.seconds)
            self._outcome = self._outcome.outcome
        if isinstance(self._outcome, MidStreamFailure):
            for t in self._outcome.texts:
                yield SimpleNamespace(type="text", text=t)
            raise self._outcome.error
        for t in self._outcome[0]:
            yield SimpleNamespace(type="text", text=t)

    async def get_final_message(self):
        return self._outcome[1]


def classified(intent: str = "account", *, needs_tools: bool | None = None,
               insistence: bool = False, reason: str = "test") -> tuple[list[str], Message]:
    """The intent classifier's forced tool call."""
    return [], make_message({"type": "tool_use", "id": "tu_classify", "name": "classify_intent",
                             "input": {"intent": intent,
                                       "needs_tools": intent == "account" if needs_tools is None
                                       else needs_tools,
                                       "insistence": insistence, "reason": reason}},
                            stop_reason="tool_use")


def is_classify(params: dict) -> bool:
    return (params.get("tool_choice") or {}).get("name") == "classify_intent"


class FakeClient:
    """Replies with ``outcomes`` in order. The agent's intent classification is answered
    automatically with ``classify`` (and kept out of ``calls``, in ``classify_calls``), so
    tool-loop tests only script the agent's own model calls. ``classify=None`` makes the
    classification take the next scripted outcome like any other call."""

    def __init__(self, *outcomes, classify: str | None = "account"):
        self.outcomes = list(outcomes)
        self.classify = classify
        self.calls: list[dict] = []
        self.classify_calls: list[dict] = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **params):
        if self.classify is not None and is_classify(params):
            self.classify_calls.append(params)
            return _Stream(classified(self.classify))
        self.calls.append(params)
        return _Stream(self.outcomes.pop(0))

    async def close(self):
        pass
