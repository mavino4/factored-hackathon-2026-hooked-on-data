"""Runtime settings, loaded from environment variables (prefix ``AIP_``)."""

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "bedrock", "vertex", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIP_", env_file=".env", extra="ignore")

    env: Literal["dev", "production"] = "dev"
    log_level: str = "INFO"

    # Authentication. "oidc": require a valid Bearer JWT from the configured issuer.
    # "password": username + password accounts with a session cookie (accounts.py); the
    # accounts are created with scripts/users.py.
    # "dev": trust the X-User-Id header - local development only, refused in production.
    auth_mode: Literal["oidc", "password", "dev"] = "oidc"
    oidc_issuer: str | None = None  # e.g. https://your-tenant.eu.auth0.com/
    oidc_audience: str | None = None  # the API identifier the tokens are issued for
    oidc_jwks_url: str | None = None  # optional; discovered from the issuer when unset
    # Public client ID of the web UI's OIDC application (browser login with PKCE).
    oidc_client_id: str | None = None

    # Password mode. A session ends after this long without a request, and at the latest
    # this long after sign-in.
    session_idle_minutes: int = 30
    session_max_hours: int = 8
    # Consecutive failed sign-ins that lock an account, and for how long.
    login_max_failures: int = 5
    login_lock_minutes: int = 15
    # Sign-in attempts per client address per minute (per replica).
    login_requests_per_minute: int = 10
    # Send the session cookie only over HTTPS. Always on in production; off by default
    # elsewhere so plain-HTTP local setups work.
    session_cookie_secure: bool = False

    # The UI's quick actions ("Frequent questions", agent/quick.py). "model": classified like
    # any message; "intent": the button gives the intent (no classifier), the model answers;
    # "direct": no model at all, a fixed text filled with the customer's products.
    quick_actions: Literal["model", "intent", "direct"] = "model"

    # User IDs (token `sub`, username, or X-User-Id in dev mode) allowed to read /v1/admin/*.
    admin_users: list[str] = []

    # Prometheus metrics on a separate internal port (never through the public API). 0 = off.
    metrics_port: int = 9090

    # e.g. postgresql+asyncpg://user:pass@host:5432/aiplatform. Unset = in-memory storage.
    database_url: str | None = None

    # External core-banking database, read-only role (bank_reader). Row-Level Security in
    # that database limits every query to the customer linked to the session's user.
    # e.g. postgresql+asyncpg://bank_reader:...@host:5432/bank. Unset = no banking tools.
    bank_database_url: SecretStr | None = None

    # Provider order for failover. The first healthy provider wins; a conversation
    # sticks to the provider that served it so its prompt cache stays warm.
    providers: list[ProviderName] = ["anthropic"]

    # Claude API key. Read from ANTHROPIC_API_KEY (environment or .env) and passed to the
    # SDK explicitly: values in .env are not exported to the process environment.
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "AIP_ANTHROPIC_API_KEY"))

    # OpenAI API key, read from OPENAI_API_KEY like the Claude key above. Only the intent
    # classifier bench uses it (text-embedding-3-small, agent/classifiers/embeddings.py).
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "AIP_OPENAI_API_KEY"))

    # TypeSafe AI key, for Jev (agent/classifiers/jev.py); only the classifier bench uses it.
    typesafe_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("TYPESAFE_API_KEY", "AIP_TYPESAFE_API_KEY"))

    # Amazon Bedrock (Mantle client) - only needed when "bedrock" is in `providers`.
    aws_region: str | None = None
    # Google Vertex AI - only needed when "vertex" is in `providers`.
    gcp_project_id: str | None = None
    gcp_region: str = "global"
    # Local Ollama (Anthropic-compatible /v1/messages). For dev/testing without an API key.
    # Every route uses this one local model when "ollama" is the serving provider.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    # Local models can take minutes to load onto the GPU on the first call.
    ollama_first_event_timeout_s: float = 300.0

    # Per-attempt timeout for one model call, in seconds.
    request_timeout_s: float = 120.0
    # Attempts per provider before failing over to the next one.
    max_attempts_per_provider: int = 3

    # Per-user limits. The request rate is per replica; the token quota is shared via the DB.
    user_requests_per_minute: int = 20
    user_tokens_per_day: int = 500_000

    # Langfuse tracing of chat and agent runs, masked before sending (tracing.py, privacy.py).
    # Uses the standard LANGFUSE_* names. Point it at the self-hosted server (deploy/langfuse).
    langfuse_enabled: bool = Field(default=False, validation_alias=AliasChoices(
        "AIP_LANGFUSE_ENABLED", "LANGFUSE_TRACING_ENABLED"))
    langfuse_host: str = Field(default="http://localhost:3000", validation_alias=AliasChoices(
        "AIP_LANGFUSE_HOST", "LANGFUSE_BASE_URL", "LANGFUSE_HOST"))
    langfuse_public_key: str | None = Field(default=None, validation_alias=AliasChoices(
        "AIP_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices(
        "AIP_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY"))
    # Mask balances, limits and amounts in traces. Turn off (dev only) to debug figures.
    trace_mask_amounts: bool = True
    # Key for the user-ID hash in traces (stable pseudonyms). Required in production when
    # tracing is on; in dev an unset key gets a random one per process.
    trace_hash_key: SecretStr | None = None
    # Version of the code that produced a trace (e.g. the git commit), so trace reports can
    # compare before and after a change (evals/traces.py --release).
    release: str | None = None


    @model_validator(mode="after")
    def _check_auth(self) -> "Settings":
        if self.auth_mode == "dev" and self.env == "production":
            raise ValueError("AIP_AUTH_MODE=dev is not allowed when AIP_ENV=production")
        if self.auth_mode == "oidc" and not (self.oidc_issuer and self.oidc_audience):
            raise ValueError("AIP_AUTH_MODE=oidc needs AIP_OIDC_ISSUER and AIP_OIDC_AUDIENCE "
                             "(or set AIP_AUTH_MODE=dev for local development)")
        if self.auth_mode == "password" and self.env == "production":
            if not self.database_url:
                raise ValueError("AIP_AUTH_MODE=password needs AIP_DATABASE_URL in production")
            self.session_cookie_secure = True
        if self.langfuse_enabled and self.trace_hash_key is None:
            if self.env == "production":
                raise ValueError("Langfuse tracing needs AIP_TRACE_HASH_KEY in production")
            self.trace_hash_key = SecretStr(secrets.token_hex(32))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
