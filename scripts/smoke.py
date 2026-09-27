"""End-to-end smoke test against a REAL model provider (no fakes).

Runs the whole app in-process: 3 streamed chat turns, a regenerate check,
and one agent task that must call a tool. Providers come from the environment
(.env), e.g. local Ollama:

    AIP_PROVIDERS='["ollama"]' uv run python scripts/smoke.py

or the Claude API (needs ANTHROPIC_API_KEY):

    AIP_PROVIDERS='["anthropic"]' uv run python scripts/smoke.py

In OIDC auth mode, pass tokens for two different users (e.g. from scripts/dev_oidc.py):

    AIP_SMOKE_TOKEN=... AIP_SMOKE_TOKEN_OTHER=... uv run python scripts/smoke.py
"""

import json
import os
import sys
import time

from fastapi.testclient import TestClient

from aiplatform.api.app import create_app
from aiplatform.config import get_settings

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
    other: dict[str, str]
    if settings.auth_mode == "oidc":
        USER["Authorization"] = f"Bearer {os.environ['AIP_SMOKE_TOKEN']}"
        other = {"Authorization": f"Bearer {os.environ['AIP_SMOKE_TOKEN_OTHER']}"}
    else:
        USER["X-User-Id"] = "smoke-test"
        other = {"X-User-Id": "someone-else"}
    print(f"auth={settings.auth_mode}  providers={settings.providers}  "
          f"model={'ollama:' + settings.ollama_model if settings.providers[0] == 'ollama' else 'per ROUTES'}")
    failures = 0
    with TestClient(create_app(settings)) as http:
        if settings.auth_mode == "oidc":
            anon = http.get("/v1/conversations")
            print(f"[auth] no token -> HTTP {anon.status_code} (expect 401)")
            failures += anon.status_code != 401
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

        peek = http.get(f"/v1/conversations/{cid}", headers=other)
        print(f"[isolation] another user reads this conversation -> HTTP {peek.status_code} "
              "(expect 404)")
        failures += peek.status_code != 404

        regen = http.post(f"/v1/conversations/{cid}/regenerate", headers=USER)
        print(f"[regenerate on answered conversation] HTTP {regen.status_code} (expect 409)")
        failures += regen.status_code != 409

        aid = http.post("/v1/conversations", json={"kind": "agent"}, headers=USER).json()["id"]
        start = time.perf_counter()
        events = parse_sse(http.post(
            f"/v1/conversations/{aid}/agent-runs", headers=USER,
            json={"text": "What is the current UTC date and time? "
                          "Use the get_current_time tool."}).text)
        done = next((d for n, d in events if n == "done"), {})
        print(f"\n[agent] {time.perf_counter() - start:.2f}s events="
              f"{[n for n, _ in events if n != 'delta']} -> {done}")
        if done.get("outcome") != "done":
            print(f"  FAIL: agent run did not finish: {events[-1:]}"); failures += 1
        elif "get_current_time" not in done.get("tool_calls", []):
            print("  WARN: model answered without calling the tool (quality, not plumbing)")

        # Approval flow: the irreversible tool must wait for the user's decision.
        tid = http.post("/v1/conversations", json={"kind": "agent"}, headers=USER).json()["id"]
        events = parse_sse(http.post(
            f"/v1/conversations/{tid}/agent-runs", headers=USER,
            json={"text": "Open a support ticket titled 'Printer broken' with details "
                          "'Paper jam on floor 2'. Use the create_support_ticket tool."}).text)
        approval = next((d for n, d in events if n == "approval_required"), None)
        if approval is None:
            print("\n[approval] WARN: model did not call create_support_ticket "
                  "(quality, not plumbing)")
        else:
            print(f"\n[approval] pending: {approval['tool_name']} {approval['input']}")
            decided = parse_sse(http.post(
                f"/v1/conversations/{tid}/actions/{approval['action_id']}", headers=USER,
                json={"decision": "approve"}).text)
            result = next((d for n, d in decided if n == "tool_result"), {})
            done = next((d for n, d in decided if n == "done"), {})
            print(f"  approved -> tool_result={result.get('content')!r} done={done}")
            if result.get("is_error") is not False or done.get("outcome") not in (
                    "done", "approval_required"):
                print(f"  FAIL: approval did not execute cleanly: {decided[-2:]}"); failures += 1
            again = http.post(f"/v1/conversations/{tid}/actions/{approval['action_id']}",
                              headers=USER, json={"decision": "approve"})
            print(f"  approve twice -> HTTP {again.status_code} (expect 404)")
            failures += again.status_code != 404

    print(f"\n{'PASS' if not failures else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
