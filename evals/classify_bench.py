"""Intent classifiers side by side on the frozen test set (evals/intents/test.jsonl).

    uv run --group classifiers python -m evals.classify_bench            # all, LLM cached
    uv run --group classifiers python -m evals.classify_bench --only rules ml --errors
    uv run --group classifiers python -m evals.classify_bench --refresh-llm   # spends API credits

``rules`` needs nothing; ``ml`` and ``embeddings-*`` are trained here on
evals/intents/train.jsonl only (``embeddings-*`` also need Ollama with bge-m3). ``llm`` is
the production classifier (agent/intent.py through the gateway, same prompt, model and
forced tool); its predictions are cached in evals/intents/llm_predictions.jsonl, so it is
only paid for once per test set. Results go to evals/results/classify-<UTC>.json.

Each classifier sees the last customer message and, for short follow-ups, the previous
one (classifiers.with_follow_up); the LLM sees the transcript, as in production.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("AIP_AUTH_MODE", "dev")

from aiplatform.agent import intent as intents
from aiplatform.agent.classifiers import INTENTS, Prediction, with_follow_up
from aiplatform.agent.classifiers.rules import RuleClassifier
from evals.intent_dataset import TEST, TRAIN, read

ROOT = Path(__file__).parent
LLM_CACHE = ROOT / "intents" / "llm_predictions.jsonl"
ALL = ["llm", "rules", "ml", "embeddings-logreg", "embeddings-knn"]


def history_of(case: dict) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in case.get("history") or []]


# --- The LLM classifier (production) ---------------------------------------------------------

async def llm_predictions(cases: list[dict], refresh: bool) -> dict[str, dict]:
    cached = {} if refresh else {r["id"]: r for r in read(LLM_CACHE)}
    todo = [c for c in cases if c["id"] not in cached]
    if not todo:
        return cached
    from aiplatform.chat.prompts import CLASSIFY_SYSTEM_PROMPT
    from aiplatform.config import get_settings
    from aiplatform.llm.gateway import AIGateway
    from aiplatform.llm.models import ROUTES, prices_for
    from aiplatform.llm.providers import build_clients, close_clients

    settings = get_settings()
    clients = build_clients(settings)
    gateway = AIGateway(clients, settings)
    route, gate = ROUTES["classify"], asyncio.Semaphore(4)

    async def one(case: dict) -> dict:
        messages = [*history_of(case), {"role": "user", "content": case["text"]}]
        async with gate:
            start = time.perf_counter()
            completed = await gateway.complete(
                route, system=CLASSIFY_SYSTEM_PROMPT, messages=intents.classify_messages(messages),
                tools=[intents.CLASSIFY_TOOL], tool_choice=intents.FORCE_CLASSIFY,
                conversation_id=f"bench-{case['id']}")
            latency = time.perf_counter() - start
        usage = completed.message.usage
        cost = prices_for(completed.message.model).cost(
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        result = intents.parse(completed.message)
        return {"id": case["id"], "intent": result.name, "confidence": 1.0,
                "reason": result.reason, "latency_s": round(latency, 4), "cost_usd": cost}

    try:
        print(f"llm: classifying {len(todo)} messages (API credits)...")
        rows = await asyncio.gather(*(one(c) for c in todo))
    finally:
        await close_clients(clients)
    cached.update({r["id"]: r for r in rows})
    LLM_CACHE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                 for r in cached.values()))
    return cached


# --- Metrics ----------------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def score(cases: list[dict], predicted: dict[str, str]) -> dict:
    gold = [c["intent"] for c in cases]
    pred = [predicted[c["id"]] for c in cases]
    per_class = {}
    for label in INTENTS:
        tp = sum(g == p == label for g, p in zip(gold, pred))
        fp = sum(p == label != g for g, p in zip(gold, pred))
        fn = sum(g == label != p for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"n": tp + fn, "precision": round(precision, 3),
                            "recall": round(recall, 3), "f1": round(f1, 3)}
    not_attack = [p for g, p in zip(gold, pred) if g != "attack"]
    return {
        "n": len(cases),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(cases), 3),
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(INTENTS), 3),
        "attack_recall": per_class["attack"]["recall"],
        # Real requests taken for attacks: the customer gets refused.
        "false_attack_rate": round(sum(p == "attack" for p in not_attack) / len(not_attack), 3),
        "per_class": per_class,
        "confusion": {g: dict(Counter(p for gg, p in zip(gold, pred) if gg == g))
                      for g in INTENTS},
    }


def build(name: str, train: list[dict], embedder):
    if name == "rules":
        return RuleClassifier()
    texts, labels = [r["text"] for r in train], [r["intent"] for r in train]
    if name == "ml":
        from aiplatform.agent.classifiers.ml import TfidfClassifier
        return TfidfClassifier().fit(texts, labels)
    from aiplatform.agent.classifiers.embeddings import EmbeddingClassifier
    return EmbeddingClassifier(embedder, method=name.split("-", 1)[1]).fit(texts, labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare intent classifiers on the test set.")
    parser.add_argument("--only", nargs="+", choices=ALL, default=ALL)
    parser.add_argument("--refresh-llm", action="store_true",
                        help="classify the test set with the LLM again (API credits)")
    parser.add_argument("--errors", action="store_true", help="print every wrong prediction")
    args = parser.parse_args()

    test, train = read(TEST), read(TRAIN)
    if not test:
        raise SystemExit("no test set: uv run python -m evals.intent_dataset test")
    if not train and set(args.only) - {"rules", "llm"}:
        raise SystemExit("no training set: uv run python -m evals.intent_dataset train")
    print(f"test {len(test)} messages · train {len(train)} messages")

    embedder = None
    if any(n.startswith("embeddings") for n in args.only):
        from aiplatform.agent.classifiers.embeddings import OllamaEmbedder
        embedder = OllamaEmbedder()
        embedder.embed(["warm up"])  # load the model before timing anything

    results, predictions = {}, {}
    for name in args.only:
        if name == "llm":
            rows = asyncio.run(llm_predictions(test, args.refresh_llm))
            preds = {c["id"]: Prediction(rows[c["id"]]["intent"], 1.0, rows[c["id"]]["reason"])
                     for c in test}
            latencies = [rows[c["id"]]["latency_s"] for c in test]
            cost = sum(rows[c["id"]]["cost_usd"] for c in test)
            train_s = 0.0
        else:
            start = time.perf_counter()
            classifier = build(name, train, embedder)
            train_s = time.perf_counter() - start
            if embedder is not None:
                embedder.clear()  # time real embeddings, not ones another classifier cached
            preds, latencies, cost = {}, [], 0.0
            for case in test:
                t0 = time.perf_counter()
                preds[case["id"]] = with_follow_up(classifier, case["text"], history_of(case))
                latencies.append(time.perf_counter() - t0)
        result = score(test, {k: p.intent for k, p in preds.items()})
        result["latency_ms"] = {q: round(1000 * percentile(latencies, v), 2)
                                for q, v in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}
        result["cost_usd_per_1k"] = round(cost / len(test) * 1000, 4)
        result["train_s"] = round(train_s, 1)
        slices = defaultdict(list)
        for c in test:
            slices[f"source:{c['source']}"].append(c)
            slices[f"lang:{c['language']}"].append(c)
        result["slices"] = {k: score(v, {c["id"]: preds[c["id"]].intent for c in v})["accuracy"]
                            for k, v in sorted(slices.items())}
        results[name], predictions[name] = result, preds

    print(f"\n{'classifier':<19}{'acc':>6}{'macroF1':>9}{'attack R':>10}{'false atk':>11}"
          f"{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'$/1k':>8}")
    for name, r in results.items():
        lat = r["latency_ms"]
        print(f"{name:<19}{r['accuracy']:>6.3f}{r['macro_f1']:>9.3f}{r['attack_recall']:>10.3f}"
              f"{r['false_attack_rate']:>11.3f}{lat['p50']:>9.2f}{lat['p95']:>9.2f}"
              f"{lat['p99']:>9.2f}{r['cost_usd_per_1k']:>8.3f}")
    print("\nF1 per intent")
    print(f"{'classifier':<19}" + "".join(f"{i:>14}" for i in INTENTS))
    for name, r in results.items():
        print(f"{name:<19}" + "".join(f"{r['per_class'][i]['f1']:>14.3f}" for i in INTENTS))
    print("\nAccuracy per slice")
    keys = sorted({k for r in results.values() for k in r["slices"]})
    print(f"{'classifier':<19}" + "".join(f"{k.split(':')[1]:>13}" for k in keys))
    for name, r in results.items():
        print(f"{name:<19}" + "".join(f"{r['slices'][k]:>13.3f}" for k in keys))

    if args.errors:
        for name, preds in predictions.items():
            print(f"\n--- {name}: wrong ---")
            for c in test:
                p = preds[c["id"]]
                if p.intent != c["intent"]:
                    print(f"  {c['intent']:>12} → {p.intent:<12} {c['id']:<34} "
                          f"{c['text'][:90]!r}  [{p.reason}]")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"classify-{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"meta": {"timestamp": stamp, "test": len(test),
                                        "train": len(train)},
                               "results": results,
                               "predictions": {n: {k: p.__dict__ for k, p in ps.items()}
                                               for n, ps in predictions.items()}},
                              indent=2, ensure_ascii=False))
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
