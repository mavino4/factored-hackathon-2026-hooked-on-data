import os

from aiplatform.config import Settings

# Most tests don't exercise auth; OIDC tests configure it explicitly.
os.environ.setdefault("AIP_AUTH_MODE", "dev")
os.environ.setdefault("AIP_METRICS_PORT", "0")  # no metrics server in tests

# Never read the developer's .env: it may point at a real database, the bank DB or a
# Langfuse server. Tests configure what they need explicitly.
Settings.model_config["env_file"] = None
