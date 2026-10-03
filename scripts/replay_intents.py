"""Send a sample of the intent test set through the whole assistant (classifier, agent,
bank tools) with Langfuse tracing on, once per classifier mode, to compare them in traces.

    uv run python scripts/replay_intents.py --limit 80               # llm and jev_llm
    uv run python scripts/replay_intents.py --modes jev_llm --limit 20

Each mode runs in its own process (the Langfuse client is a per-process singleton) and its
traces carry the release ``<git sha>-replay<tag>-<mode>`` (environment from .env), so they
can be reported apart and compared:

    make trace-report ARGS="--since 2h --release <sha>-replay-llm"
    make trace-report ARGS="--since 2h --release <sha>-replay-jev_llm"
    make trace-score ARGS="--since 2h --judge-jev --sample 40"

Spends API credits (the agent answers every message; Jev and the LLM classify). Needs
Langfuse (make langfuse-up) and its keys in .env, the bank DB (make bank-db) and, for
jev_llm, TYPESAFE_API_KEY.
"""

import argparse
import asyncio
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("AIP_AUTH_MODE", "dev")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiplatform.agent.actions import InMemoryActionStore
from aiplatform.agent.events import AgentDone
from aiplatform.agent.loop import AgentRunner
from aiplatform.api.app import make_jev
from aiplatform.banking.repository import PostgresBankRepository
from aiplatform.banking.tools import make_bank_tools
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.repository import InMemoryConversationRepository
from aiplatform.config import get_settings
from aiplatform.llm.gateway import AIGateway
from aiplatform.llm.providers import build_clients, close_clients
from aiplatform.tracing import Tracing
from aiplatform.usage import InMemoryUsageStore
from evals.intent_dataset import TEST, read
from evals.run import bank_url


def sample(limit: int, seed: int) -> list[dict]:
    """Up to ``limit`` test cases, the same share of each intent, same cases every time."""
    cases = read(TEST)
    rng = random.Random(seed)
    by_intent: dict[str, list[dict]] = {}
    for c in cases:
        by_intent.setdefault(c["intent"], []).append(c)
    per_intent = max(1, limit // len(by_intent))
    chosen = [c for group in by_intent.values()
              for c in rng.sample(group, min(per_intent, len(group)))]
    return chosen[:limit]


async def replay(mode: str, cases: list[dict], release: str, user: str) -> Counter:
    settings = get_settings().model_copy(update={
        "intent_classifier": mode, "release": release, "langfuse_enabled": True})
    tracing = Tracing.from_settings(settings)
    if not tracing.enabled:
        raise SystemExit("Langfuse tracing is off: set LANGFUSE_* in .env (make langfuse-env)")
    jev = make_jev(settings)
    if mode == "jev_llm" and jev is None:
        raise SystemExit("jev_llm needs TYPESAFE_API_KEY in .env")
    clients = build_clients(settings)
    bank = PostgresBankRepository(bank_url(settings))
    repo = InMemoryConversationRepository()
    runner = AgentRunner(AIGateway(clients, settings), repo,
                         InMemoryUsageStore(), InFlight(), InMemoryActionStore(),
                         make_bank_tools(bank), tracing=tracing, jev=jev,
                         jev_threshold=settings.jev_threshold)
    outcomes: Counter = Counter()
    try:
        for i, case in enumerate(cases, 1):
            conv = await repo.create(user, "agent")
            earlier = [m["content"] for m in case.get("history") or [] if m["role"] == "user"]
            for text in [*earlier, case["text"]]:
                async for event in runner.run(user, conv.id, text, case.get("language")):
                    if isinstance(event, AgentDone) and text == case["text"]:
                        outcomes[event.outcome] += 1
            if i % 10 == 0:
                print(f"  {mode}: {i}/{len(cases)}", flush=True)
    finally:
        tracing.flush()
        await close_clients(clients)
        await bank.close()
        if jev is not None:
            await jev.close()
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modes", nargs="+", choices=["llm", "jev_llm"],
                        default=["llm", "jev_llm"])
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user", default="eval-es", help="a user linked in the bank DB")
    parser.add_argument("--tag", default="", help="suffix for the release, e.g. --tag=-r2")
    args = parser.parse_args()

    if len(args.modes) > 1:
        # One process per mode: the Langfuse client is a per-process singleton (its release
        # is fixed when it starts, and flushing shuts it down), so modes must not share one.
        for mode in args.modes:
            command = [sys.executable, __file__, "--modes", mode, "--limit", str(args.limit),
                       "--seed", str(args.seed), "--user", args.user, f"--tag={args.tag}"]
            subprocess.run(command, check=True)
        return 0
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    cases = sample(args.limit, args.seed)
    mode = args.modes[0]
    print(f"{mode}: {len(cases)} cases {dict(Counter(c['intent'] for c in cases))}")
    release = f"{sha}-replay{args.tag}-{mode}"
    start = time.perf_counter()
    outcomes = asyncio.run(replay(mode, cases, release, args.user))
    print(f"{mode}: {dict(outcomes)} in {time.perf_counter() - start:.0f} s (release {release})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
