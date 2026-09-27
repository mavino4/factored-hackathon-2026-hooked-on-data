import os

# Most tests don't exercise auth; OIDC tests configure it explicitly.
os.environ.setdefault("AIP_AUTH_MODE", "dev")
