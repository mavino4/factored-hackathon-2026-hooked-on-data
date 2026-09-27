"""End-to-end smoke test against a REAL model provider (no fakes).

Runs the whole app in-process: 3 streamed chat turns, a regenerate check,
and one agent task that must call a tool. Providers come from the environment
(.env), e.g. local Ollama:

    AIP_PROVIDERS='["ollama"]' uv run python scripts/smoke.py

or the Claude API (needs ANTHROPIC_API_KEY):

    AIP_PROVIDERS='["anthropic"]' uv run python scripts/smoke.py
"""

import json
import sys
import time

from fastapi.testclient import TestClient

from aiplatform.api.app import create_app
from aiplatform.config import get_settings

USER = {"X-User-Id": "smoke-test"}
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
    resp = http.post(f"/v1/conversations/{cid}/messages", json={"text": text}, headers=USER)
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


def main() -> int:
    settings = get_settings()
    print(f"providers={settings.providers}  "
          f"model={'ollama:' + settings.ollama_model if settings.providers[0] == 'ollama' else 'per ROUTES'}")
    failures = 0
    with TestClient(create_app(settings)) as http:
        cid = http.post("/v1/conversations", json={"kind": "chat"}, headers=USER).json()["id"]
        for i, text in enumerate(CHAT_TURNS, 1):
            r = chat_turn(http, cid, text)
            print(f"\n[chat {i}] {text}\n  -> {r['text'].strip()!r}\n  "
                  f"stop={r['stop_reason']} deltas={r['deltas']} "
                  f"total={r['total_s']:.2f}s usage={r['usage']}")
            if not r["text"].strip():
                print("  FAIL: empty reply"); failures += 1
            if r["deltas"] < 2 and r["usage"]["output_tokens"] > 3:
                print("  WARN: reply arrived in fewer than 2 chunks (streaming not observed)")
        if "ana" not in r["text"].lower():
            print("  WARN: model did not recall the name from turn 1 (quality, not plumbing)")

        history = http.get(f"/v1/conversations/{cid}", headers=USER).json()["messages"]
        roles = [m["role"] for m in history]
        print(f"\n[history] {roles}")
        if roles != ["user", "assistant"] * 3:
            print("  FAIL: unexpected history shape"); failures += 1

        regen = http.post(f"/v1/conversations/{cid}/regenerate", headers=USER)
        print(f"[regenerate on answered conversation] HTTP {regen.status_code} (expect 409)")
        failures += regen.status_code != 409

        aid = http.post("/v1/conversations", json={"kind": "agent"}, headers=USER).json()["id"]
        start = time.perf_counter()
        resp = http.post(f"/v1/conversations/{aid}/agent-runs", headers=USER,
                         json={"text": "What is the current UTC date and time? "
                                       "Use the get_current_time tool."})
        result = resp.json()
        print(f"\n[agent] HTTP {resp.status_code} in {time.perf_counter() - start:.2f}s -> {result}")
        if resp.status_code != 200 or result.get("outcome") != "done":
            print("  FAIL: agent run did not finish"); failures += 1
        elif "get_current_time" not in result.get("tool_calls", []):
            print("  WARN: model answered without calling the tool (quality, not plumbing)")

    print(f"\n{'PASS' if not failures else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
