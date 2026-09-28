"""Model profiles and per-route configuration.

Everything model-specific lives here, so moving a route from the light model to a
larger one is a one-line change in ``ROUTES``.
"""

from dataclasses import dataclass, field
from typing import Literal

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class Prices:
    """USD per million tokens (Claude API list prices; partner platforms differ)."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float

    def cost(self, *, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0,
             cache_write_tokens: int = 0) -> float:
        return (input_tokens * self.input + output_tokens * self.output
                + cache_read_tokens * self.cache_read
                + cache_write_tokens * self.cache_write_5m) / 1_000_000


@dataclass(frozen=True)
class ModelProfile:
    id: str
    max_output_tokens: int
    # Adaptive thinking + `output_config.effort` (Claude 4.6+ family). Haiku 4.5 has neither.
    supports_effort: bool
    # Shorter prompts are silently not cached; used to warn in logs/metrics.
    min_cacheable_tokens: int
    prices: Prices
    # Provider-specific model IDs. Verify them against your Bedrock/Vertex model catalog.
    provider_ids: dict[str, str] = field(default_factory=dict)

    def id_for(self, provider: str) -> str:
        return self.provider_ids.get(provider, self.id)


HAIKU_4_5 = ModelProfile(
    id="claude-haiku-4-5",
    max_output_tokens=64_000,
    supports_effort=False,
    min_cacheable_tokens=4096,
    prices=Prices(input=1.0, output=5.0, cache_read=0.10, cache_write_5m=1.25),
    provider_ids={
        "bedrock": "anthropic.claude-haiku-4-5",
        "vertex": "claude-haiku-4-5@20251001",
    },
)

SONNET_5 = ModelProfile(
    id="claude-sonnet-5",
    max_output_tokens=128_000,
    supports_effort=True,
    min_cacheable_tokens=512,
    prices=Prices(input=2.0, output=10.0, cache_read=0.20, cache_write_5m=2.5),
    provider_ids={"bedrock": "anthropic.claude-sonnet-5"},
)

OPUS_5 = ModelProfile(
    id="claude-opus-5",
    max_output_tokens=128_000,
    supports_effort=True,
    min_cacheable_tokens=512,
    prices=Prices(input=5.0, output=25.0, cache_read=0.50, cache_write_5m=6.25),
    provider_ids={"bedrock": "anthropic.claude-opus-5"},
)


@dataclass(frozen=True)
class Route:
    """How one kind of traffic (chat, agent, ...) calls the model."""

    name: str
    model: ModelProfile
    max_tokens: int
    effort: Effort | None = None  # ignored when the model doesn't support effort
    # Give up on an attempt (and retry/fail over) if the provider sends nothing for this long.
    first_event_timeout_s: float = 30.0


# First versions run on the light model (Claude Haiku 4.5). To upgrade a route,
# swap the model, e.g. Route("agent", OPUS_5, max_tokens=16_000, effort="high").
ROUTES: dict[str, Route] = {
    "chat": Route("chat", HAIKU_4_5, max_tokens=4_096),
    "agent": Route("agent", HAIKU_4_5, max_tokens=8_192, first_event_timeout_s=60.0),
}


FREE = Prices(input=0, output=0, cache_read=0, cache_write_5m=0)


def prices_for(model_id: str) -> Prices:
    """Prices for a model ID as reported in responses (local models such as Ollama are free)."""
    for profile in (HAIKU_4_5, SONNET_5, OPUS_5):
        # Responses may name a dated snapshot, e.g. claude-haiku-4-5-20251001.
        if (model_id == profile.id or model_id.startswith(f"{profile.id}-")
                or model_id in profile.provider_ids.values()):
            return profile.prices
    return FREE
