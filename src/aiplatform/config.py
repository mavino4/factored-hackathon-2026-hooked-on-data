"""Runtime settings, loaded from environment variables (prefix ``AIP_``)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "bedrock", "vertex"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIP_", env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # Provider order for failover. The first healthy provider wins; a conversation
    # sticks to the provider that served it so its prompt cache stays warm.
    providers: list[ProviderName] = ["anthropic"]

    # Amazon Bedrock (Mantle client) - only needed when "bedrock" is in `providers`.
    aws_region: str | None = None
    # Google Vertex AI - only needed when "vertex" is in `providers`.
    gcp_project_id: str | None = None
    gcp_region: str = "global"

    # Per-attempt timeout for one model call, in seconds.
    request_timeout_s: float = 120.0
    # Attempts per provider before failing over to the next one.
    max_attempts_per_provider: int = 3

    # Per-user limits (in-process; move to Redis when running more than one replica).
    user_requests_per_minute: int = 20
    user_tokens_per_day: int = 500_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
