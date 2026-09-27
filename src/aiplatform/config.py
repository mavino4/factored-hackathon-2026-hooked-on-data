"""Runtime settings, loaded from environment variables (prefix ``AIP_``)."""

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "bedrock", "vertex", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIP_", env_file=".env", extra="ignore")

    env: Literal["dev", "production"] = "dev"
    log_level: str = "INFO"

    # Authentication. "oidc": require a valid Bearer JWT from the configured issuer.
    # "dev": trust the X-User-Id header - local development only, refused in production.
    auth_mode: Literal["oidc", "dev"] = "oidc"
    oidc_issuer: str | None = None  # e.g. https://your-tenant.eu.auth0.com/
    oidc_audience: str | None = None  # the API identifier the tokens are issued for
    oidc_jwks_url: str | None = None  # optional; discovered from the issuer when unset

    # e.g. postgresql+asyncpg://user:pass@host:5432/aiplatform. Unset = in-memory storage.
    database_url: str | None = None

    # Provider order for failover. The first healthy provider wins; a conversation
    # sticks to the provider that served it so its prompt cache stays warm.
    providers: list[ProviderName] = ["anthropic"]

    # Amazon Bedrock (Mantle client) - only needed when "bedrock" is in `providers`.
    aws_region: str | None = None
    # Google Vertex AI - only needed when "vertex" is in `providers`.
    gcp_project_id: str | None = None
    gcp_region: str = "global"
    # Local Ollama (Anthropic-compatible /v1/messages). For dev/testing without an API key.
    # Every route uses this one local model when "ollama" is the serving provider.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Per-attempt timeout for one model call, in seconds.
    request_timeout_s: float = 120.0
    # Attempts per provider before failing over to the next one.
    max_attempts_per_provider: int = 3

    # Per-user limits. The request rate is per replica; the token quota is shared via the DB.
    user_requests_per_minute: int = 20
    user_tokens_per_day: int = 500_000


    @model_validator(mode="after")
    def _check_auth(self) -> "Settings":
        if self.auth_mode == "dev" and self.env == "production":
            raise ValueError("AIP_AUTH_MODE=dev is not allowed when AIP_ENV=production")
        if self.auth_mode == "oidc" and not (self.oidc_issuer and self.oidc_audience):
            raise ValueError("AIP_AUTH_MODE=oidc needs AIP_OIDC_ISSUER and AIP_OIDC_AUDIENCE "
                             "(or set AIP_AUTH_MODE=dev for local development)")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
