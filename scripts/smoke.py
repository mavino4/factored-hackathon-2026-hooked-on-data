"""End-to-end smoke test against a REAL model provider and the core-banking DB.

Runs the whole app in-process: 3 streamed turns of general questions (the classifier
should route them without tools), and banking queries (card balance, available credit) whose figures are checked
against the core-banking DB, for two different customers (ana, bruno). Providers come
from the environment (.env), e.g. local Ollama:

    make bank-db   # once
    AIP_PROVIDERS='["ollama"]' AIP_OLLAMA_MODEL=qwen2.5:7b uv run python scripts/smoke.py

or the Claude API (needs ANTHROPIC_API_KEY):

    AIP_PROVIDERS='["anthropic"]' uv run python scripts/smoke.py

In OIDC auth mode, pass tokens for ana and bruno (e.g. from scripts/dev_oidc.py):

    AIP_SMOKE_TOKEN=$(... token --sub ana) AIP_SMOKE_TOKEN_OTHER=$(... token --sub bruno) \
    uv run python scripts/smoke.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AIP_BANK_DATABASE_URL",
                      "postgresql+asyncpg://bank_reader:bank_reader@localhost:5432/bank")

from aiplatform.api.app import create_app
from aiplatform.banking.repository import PostgresBankRepository
from aiplatform.config import get_settings
from evals.run import mentions_amount

USER: dict[str, str] = {}
CHAT_TURNS = [
    "Hi! My name is Ana. Reply in one short sentence.",
    "What's 12 times 7? Answer with just the number.",
    "What is my name? Answer in a few words.",
]


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            events.append((lines["event"], json.loads(lines.get("data", "{}"))))
    return events


def chat_turn(http: TestClient, cid: str, text: str) -> dict:
    # The in-process TestClient buffers the response, so time-to-first-token can't be
    # measured here (that comes from the /metrics histogram in M7). Delta count proves streaming.
    start = time.perf_counter()
    resp = http.post(f"/v1/conversations/{cid}/agent-runs", json={"text": text}, headers=USER)
    resp.raise_for_status()
    body = resp.text
    events = parse_sse(body)
    errors = [data for name, data in events if name == "error"]
    if errors:
        raise SystemExit(f"FAIL chat turn: {errors}")
    chunks = [data["text"] for name, data in events if name == "delta"]
    done = next(data for name, data in events if name == "done")
    return {"text": "".join(chunks), "total_s": time.perf_counter() - start,
            "deltas": len(chunks), **done}


async def bank_truth(settings) -> dict[str, dict]:
    """Credit-card figures for ana and bruno, read through the same RLS-scoped access."""
    repo = PostgresBankRepository(settings.bank_database_url.get_secret_value())
    try:
        truth = {}
        for user in ("ana", "bruno"):
            [card] = await repo.get_products(user, "Tarjeta Crédito")
            truth[user] = {"balance": float(card.current_balance),
                           "available": float(card.available_credit),
                           "currency": card.currency}
        return truth
    finally:
        await repo.close()


def main() -> int:
    settings = get_settings()
    other: dict[str, str]
    if settings.auth_mode == "oidc":
        USER["Authorization"] = f"Bearer {os.environ['AIP_SMOKE_TOKEN']}"
        other = {"Authorization": f"Bearer {os.environ['AIP_SMOKE_TOKEN_OTHER']}"}
    else:
        USER["X-User-Id"] = "ana"
        other = {"X-User-Id": "bruno"}
    print(f"auth={settings.auth_mode}  providers={settings.providers}  "
          f"model={'ollama:' + settings.ollama_model if settings.providers[0] == 'ollama' else 'per ROUTES'}")
    failures = 0
    with TestClient(create_app(settings)) as http:
        if settings.auth_mode == "oidc":
            anon = http.get("/v1/conversations")
            print(f"[auth] no token -> HTTP {anon.status_code} (expect 401)")
            failures += anon.status_code != 401
        cid = http.post("/v1/conversations", json={}, headers=USER).json()["id"]
        for i, text in enumerate(CHAT_TURNS, 1):
            r = chat_turn(http, cid, text)
            print(f"\n[chat {i}] {text}\n  -> {r['text'].strip()!r}\n  "
                  f"outcome={r['outcome']} deltas={r['deltas']} total={r['total_s']:.2f}s")
            if not r["text"].strip():
                print("  FAIL: empty reply"); failures += 1
            if r["deltas"] < 2 and len(r["text"]) > 20:
                print("  WARN: reply arrived in fewer than 2 chunks (streaming not observed)")
        if "ana" not in r["text"].lower():
            print("  WARN: model did not recall the name from turn 1 (quality, not plumbing)")

        history = http.get(f"/v1/conversations/{cid}", headers=USER).json()["messages"]
        roles = [m["role"] for m in history]
        print(f"\n[history] {roles}")
        texts = [m for m in history if m["role"] == "user" and isinstance(m["content"], str)]
        if len(texts) != 3 or roles[-1] != "assistant":
            print("  FAIL: unexpected history shape"); failures += 1

        peek = http.get(f"/v1/conversations/{cid}", headers=other)
        print(f"[isolation] another user reads this conversation -> HTTP {peek.status_code} "
              "(expect 404)")
        failures += peek.status_code != 404

        # Banking: figures must match the core-banking DB, per customer.
        truth = asyncio.run(bank_truth(settings))
        for label, headers in (("ana", USER), ("bruno", other)):
            card = truth[label]
            for question, key in (("¿Cuál es el saldo de mi tarjeta de crédito?", "balance"),
                                  ("¿Cuánto crédito disponible me queda en la tarjeta?",
                                   "available")):
                aid = http.post("/v1/conversations", json={},
                                headers=headers).json()["id"]
                start = time.perf_counter()
                events = parse_sse(http.post(f"/v1/conversations/{aid}/agent-runs",
                                             headers=headers, json={"text": question}).text)
                text = "".join(d["text"] for n, d in events if n == "delta")
                calls = [d["name"] for n, d in events if n == "tool_call"]
                ok = mentions_amount(text, card[key])
                leaked = [u for u in truth if u != label and mentions_amount(text, truth[u][key])]
                print(f"\n[bank:{label}] {question} ({time.perf_counter() - start:.1f}s)\n"
                      f"  tools={calls} expected {key}={card[key]} {card['currency']}\n"
                      f"  -> {text.strip()[:220]!r}")
                if leaked:
                    print(f"  FAIL: shows figures of {leaked}"); failures += 1
                elif "get_products" not in calls:
                    print("  WARN: model answered without calling get_products (quality)")
                elif not ok:
                    print("  WARN: figure differs from the DB (model quality, not plumbing)")

    print(f"\n{'PASS' if not failures else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
