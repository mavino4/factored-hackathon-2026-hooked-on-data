import os

# Most tests don't exercise auth; OIDC tests configure it explicitly.
os.environ.setdefault("AIP_AUTH_MODE", "dev")
os.environ.setdefault("AIP_METRICS_PORT", "0")  # no metrics server in tests
