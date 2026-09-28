"""Compare local Ollama models on the same eval dataset.

    uv run python evals/compare.py                                   # banking, both models
    uv run python evals/compare.py --models qwen2.5:7b --save-baselines

Each model is warmed up (loaded onto the GPU) before timing. Results go to
evals/results/compare-<UTC>.json; --save-baselines also writes
evals/baselines/<dataset>__<model>.json.

--rescore RESULTS.json re-applies the current checks to the answers saved in a previous
run (no model calls), e.g. after fixing a check that was too strict.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("AIP_AUTH_MODE", "dev")

from aiplatform.config import get_settings
from evals.run import (
    EVALS_DIR,
    Answer,
    CaseResult,
    check_answer,
    load_dataset,
    run_dataset,
    summarize,
)

DEFAULT_MODELS = ["llama3.2:3b", "qwen2.5:7b"]


def model_settings(model: str):
    return get_settings().model_copy(update={"providers": ["ollama"], "ollama_model": model})


def print_comparison(summaries: dict[str, dict]) -> None:
    models = list(summaries)
    width = max(12, *(len(m) for m in models)) + 2
    header = f"{'':<22}" + "".join(f"{m:>{width}}" for m in models)
    print("\n" + header + "\n" + "-" * len(header))

    def row(label: str, values: list[str]) -> None:
        print(f"{label:<22}" + "".join(f"{v:>{width}}" for v in values))

    row("score", [f"{s['score']}% ({s['passed']}/{s['total']})" for s in summaries.values()])
    tags = sorted({t for s in summaries.values() for t in s["by_tag"]})
    for tag in tags:
        row(f"  {tag}", [f"{s['by_tag'].get(tag, {}).get('passed', 0)}/"
                         f"{s['by_tag'].get(tag, {}).get('total', 0)}"
                         for s in summaries.values()])
    row("latency p50", [f"{s['latency_p50_s']:.1f}s" for s in summaries.values()])
    row("tokens", [str(s["total_tokens"]) for s in summaries.values()])

    print("\nFailures:")
    for model, s in summaries.items():
        failed = [c for c in s["cases"] if not c["passed"]]
        print(f"  [{model}] {len(failed)} failed")
        for c in failed:
            print(f"    - {c['id']}: {'; '.join(c['failures'])[:110]}")


def rescore(saved: dict[str, dict], cases) -> dict[str, dict]:
    by_id = {c.id: c for c in cases}
    summaries = {}
    for model, old in saved.items():
        results = []
        for c in old["cases"]:
            case = by_id[c["id"]]
            answer = Answer(text=c["answer"], tools_called=c["tools_called"])
            failures = check_answer(answer, case.checks)
            results.append(CaseResult(**{**c, "passed": not failures, "failures": failures,
                                         "tags": case.tags}))
        meta = {k: v for k, v in old.items()
                if k not in ("cases", "by_tag", "score", "passed", "total")}
        summaries[model] = summarize(results, {**meta, "rescored": True})
    return summaries


async def main_async(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    cases = load_dataset(dataset)
    summaries = {}
    if args.rescore:
        summaries = rescore(json.loads(Path(args.rescore).read_text()), cases)
    for model in [] if args.rescore else args.models:
        print(f"running {len(cases)} cases on ollama:{model} ...", flush=True)
        settings = model_settings(model)
        results = await run_dataset(settings, cases, warm_up=True)
        summaries[model] = summarize(results, {
            "timestamp": datetime.now(UTC).isoformat(), "providers": ["ollama"],
            "ollama_model": model, "dataset": str(dataset), "judge": False})

    print_comparison(summaries)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"compare-{stamp}.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}")
    if args.save_baselines:
        base_dir = EVALS_DIR / "baselines"
        base_dir.mkdir(exist_ok=True)
        for model, summary in summaries.items():
            path = base_dir / f"{dataset.stem}__{model.replace(':', '-')}.json"
            path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"baseline saved: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Ollama models on an eval dataset.")
    parser.add_argument("--dataset", default=str(EVALS_DIR / "banking.jsonl"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--save-baselines", action="store_true")
    parser.add_argument("--rescore", metavar="RESULTS_JSON",
                        help="re-apply the checks to a previous run's answers (no model calls)")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
