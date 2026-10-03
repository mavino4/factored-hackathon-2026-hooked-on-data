"""Minimal read client for the self-hosted Langfuse's public API.

The server runs Langfuse v4 in ``events_only`` mode, where the trace, session and v1/v2
score endpoints answer with an error. These do work and are all the trace reports need:

- ``GET /api/public/v2/observations``: every step of every run (cursor pagination);
- ``GET /api/public/v3/scores``: scores, with the trace they belong to.

Scores are written with the SDK (``Langfuse.create_score``), like the app does.
"""

import json
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import httpx

OBSERVATION_FIELDS = "core,basic,time,usage,metadata"  # "time" carries timeToFirstToken
OBSERVATIONS_PAGE = 1000
SCORES_PAGE = 100


class LangfuseAPI:
    def __init__(self, host: str, public_key: str, secret_key: str, *,
                 transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=host.rstrip("/"), auth=(public_key, secret_key),
                                  timeout=60.0, transport=transport)

    @classmethod
    def from_settings(cls, settings) -> "LangfuseAPI":
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            raise SystemExit("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are not set (.env)")
        return cls(settings.langfuse_host, settings.langfuse_public_key,
                   settings.langfuse_secret_key.get_secret_value())

    def close(self) -> None:
        self._http.close()

    def _pages(self, path: str, params: dict[str, Any]) -> Iterator[dict]:
        params = {k: v for k, v in params.items() if v is not None}
        while True:
            response = self._http.get(path, params=params)
            if response.status_code >= 400:
                raise RuntimeError(f"Langfuse {path}: {response.status_code} {response.text[:200]}")
            page = response.json()
            yield from page["data"]
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor or not page["data"]:
                return
            params = {**params, "cursor": cursor}

    def observations(self, since: datetime, until: datetime | None = None, *,
                     type: str | None = None, io: bool = False) -> Iterator[dict]:
        """Observations that started in the window. ``io`` adds their (masked) input and
        output, which are large for model calls: ask for it only with a ``type``."""
        fields = OBSERVATION_FIELDS + (",io" if io else "")
        for row in self._pages("/api/public/v2/observations", {
                "fields": fields, "limit": OBSERVATIONS_PAGE, "type": type,
                "fromStartTime": _iso(since), "toStartTime": _iso(until) if until else None}):
            if io:
                row["input"], row["output"] = decoded(row.get("input")), decoded(row.get("output"))
            yield row

    def scores(self, since: datetime) -> Iterator[dict]:
        """Scores since ``since``; ``subject`` says which trace each one is about."""
        return self._pages("/api/public/v3/scores", {
            "fields": "core,subject", "limit": SCORES_PAGE, "fromTimestamp": _iso(since)})


def decoded(value: Any) -> Any:
    """Inputs and outputs come back as JSON text; plain strings stay as they are."""
    if isinstance(value, str) and value[:1] in "{[\"":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
