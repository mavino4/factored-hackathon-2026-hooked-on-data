"""Builds one async Anthropic SDK client per configured provider."""

from typing import Any

from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicVertex,
    DefaultAioHttpClient,
)

from aiplatform.config import Settings


def build_clients(settings: Settings) -> dict[str, Any]:
    """Create clients for ``settings.providers``. Call from inside the event loop.

    SDK retries are disabled: the AI Gateway owns retries and failover, and
    retrying in both places would multiply load during an incident.
    """
    common: dict[str, Any] = {"max_retries": 0, "timeout": settings.request_timeout_s}
    clients: dict[str, Any] = {}
    for name in settings.providers:
        if name == "anthropic":
            # Credentials: ANTHROPIC_API_KEY or an `ant auth login` profile.
            clients[name] = AsyncAnthropic(http_client=DefaultAioHttpClient(), **common)
        elif name == "bedrock":
            if not settings.aws_region:
                raise ValueError("AIP_AWS_REGION is required when bedrock is enabled")
            clients[name] = AsyncAnthropicBedrockMantle(aws_region=settings.aws_region, **common)
        elif name == "vertex":
            if not settings.gcp_project_id:
                raise ValueError("AIP_GCP_PROJECT_ID is required when vertex is enabled")
            clients[name] = AsyncAnthropicVertex(
                project_id=settings.gcp_project_id, region=settings.gcp_region, **common
            )
        elif name == "ollama":
            # Ollama serves the Anthropic Messages API; it ignores the key but the SDK needs one.
            clients[name] = AsyncAnthropic(base_url=settings.ollama_base_url, api_key="ollama",
                                           **common)
    return clients


async def close_clients(clients: dict[str, Any]) -> None:
    for client in clients.values():
        await client.close()
