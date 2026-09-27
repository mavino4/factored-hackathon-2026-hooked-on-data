"""Model profiles and per-route configuration.

Everything model-specific lives here, so moving a route from the light model to a
larger one is a one-line change in ``ROUTES``.
"""

from dataclasses import dataclass, field
from typing import Literal

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class ModelProfile:
    id: str
    max_output_tokens: int
    # Adaptive thinking + `output_config.effort` (Claude 4.6+ family). Haiku 4.5 has neither.
    supports_effort: bool
    # Shorter prompts are silently not cached; used to warn in logs/metrics.
    min_cacheable_tokens: int
    # Provider-specific model IDs. Verify them against your Bedrock/Vertex model catalog.
    provider_ids: dict[str, str] = field(default_factory=dict)

    def id_for(self, provider: str) -> str:
        return self.provider_ids.get(provider, self.id)


HAIKU_4_5 = ModelProfile(
    id="claude-haiku-4-5",
    max_output_tokens=64_000,
    supports_effort=False,
    min_cacheable_tokens=4096,
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
    provider_ids={"bedrock": "anthropic.claude-sonnet-5"},
)

OPUS_5 = ModelProfile(
    id="claude-opus-5",
    max_output_tokens=128_000,
    supports_effort=True,
    min_cacheable_tokens=512,
    provider_ids={"bedrock": "anthropic.claude-opus-5"},
)


@dataclass(frozen=True)
class Route:
    """How one kind of traffic (chat, agent, ...) calls the model."""

    name: str
    model: ModelProfile
    max_tokens: int
    effort: Effort | None = None  # ignored when the model doesn't support effort


# First versions run on the light model (Claude Haiku 4.5). To upgrade a route,
# swap the model, e.g. Route("agent", OPUS_5, max_tokens=16_000, effort="high").
ROUTES: dict[str, Route] = {
    "chat": Route("chat", HAIKU_4_5, max_tokens=4_096),
    "agent": Route("agent", HAIKU_4_5, max_tokens=8_192),
}
