"""A scripted stand-in for an Anthropic SDK async client (no network)."""

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
        if isinstance(self._outcome, MidStreamFailure):
            for t in self._outcome.texts:
                yield SimpleNamespace(type="text", text=t)
            raise self._outcome.error
        for t in self._outcome[0]:
            yield SimpleNamespace(type="text", text=t)

    async def get_final_message(self):
        return self._outcome[1]


class FakeClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **params):
        self.calls.append(params)
        return _Stream(self.outcomes.pop(0))

    async def close(self):
        pass
