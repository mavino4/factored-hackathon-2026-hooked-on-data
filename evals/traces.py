"""Performance report from the traces already in Langfuse: how fast, how expensive and how
the assistant behaved, per customer turn.

    uv run python -m evals.traces [--since 7d] [--release SHA] [--routing MODE] [--quick]
                                  [--check [SLO_JSON]]
                                  [--compare RESULT_JSON] [--export-cases --outcome X]

Reads only (see ``langfuse_api``). One trace is one customer turn: the ``agent`` run with
its steps (classify, call_model, run_tools...), model calls and tools. The full result goes
to ``evals/results/traces-<UTC timestamp>.json``.

Traces are masked before they leave the app, so this measures speed, cost and behaviour.
Whether a figure is right is the job of ``evals/banking.jsonl``.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("AIP_AUTH_MODE", "dev")  # reports don't serve HTTP

from aiplatform.config import get_settings
from evals.langfuse_api import LangfuseAPI

EVALS_DIR = Path(__file__).parent
DEFAULT_SLO = EVALS_DIR / "trace_slo.json"
# Scores written by evals/score_traces.py; the report shows their pass rate.
QUALITY_SCORES = ("language_match", "no_leak", "used_bank_tool", "helpfulness", "jev_agrees_llm")
BANK_TOOLS = {"get_products", "get_customer_profile"}


@dataclass
class Turn:
    trace_id: str
    start: str
    session_id: str | None = None
    user_id: str | None = None
    release: str | None = None
    language: str | None = None
    # How the turn was routed (AIP_QUICK_ACTIONS): "model" (classifier) or the quick-action
    # mode; and the quick action that sent it, if any. Older traces: "model", None.
    routing: str = "model"
    quick_action: str | None = None
    # Who decided the intent (score "classifier": jev, llm, llm_low_confidence,
    # llm_jev_error) and Jev's confidence. Older traces: "llm".
    classifier: str = "llm"
    jev_confidence: float | None = None
    latency_s: float | None = None
    intent: str | None = None
    outcome: str | None = None
    question: str | None = None
    answer: str | None = None
    tools: list[str] = field(default_factory=list)
    iterations: int = 0
    cost: float = 0.0
    classify_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    errors: int = 0
    retries: int = 0
    classify_s: list[float] = field(default_factory=list)
    model_s: list[float] = field(default_factory=list)
    ttft_s: list[float] = field(default_factory=list)
    tool_s: dict[str, list[float]] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)


def parse_since(text: str, now: datetime | None = None) -> datetime:
    """``24h``, ``7d``, ``30m`` (back from now) or an ISO date/time (UTC)."""
    now = now or datetime.now(UTC)
    if match := re.fullmatch(r"(\d+)([mhd])", text):
        unit = {"m": "minutes", "h": "hours", "d": "days"}[match.group(2)]
        return now - timedelta(**{unit: int(match.group(1))})
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def release_of(obs: dict) -> str | None:
    metadata = obs.get("metadata") or {}
    return (obs.get("release") or obs.get("version")
            or metadata.get("resourceAttributes.langfuse.release") or None)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_turns(observations: list[dict], scores: list[dict] = ()) -> list[Turn]:
    """Group observations into customer turns. What newer traces carry as scores (intent,
    outcome) is read from the steps when the score is missing, so older traces count too."""
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for obs in observations:
        by_trace[obs["traceId"]].append(obs)
    trace_scores: dict[str, dict[str, Any]] = defaultdict(dict)
    for score in scores:
        subject = score.get("subject") or {}
        if subject.get("kind") == "trace":
            trace_scores[subject["id"]][score["name"]] = score.get("value")

    turns = []
    for trace_id, steps in by_trace.items():
        root = next((o for o in steps if o["type"] == "AGENT" and o["name"] == "agent"
                     and not o.get("parentObservationId")), None)
        if root is None:  # not an agent run (or its root is outside the window)
            continue
        metadata = root.get("metadata") or {}
        turn = Turn(trace_id=trace_id, start=root["startTime"], session_id=root.get("sessionId"),
                    user_id=root.get("userId"), release=release_of(root),
                    language=metadata.get("language"), latency_s=root.get("latency"),
                    routing=metadata.get("routing") or "model",
                    quick_action=metadata.get("quick_action") or None,
                    question=root.get("input") if isinstance(root.get("input"), str) else None,
                    answer=root.get("output") if isinstance(root.get("output"), str) else None,
                    scores=dict(trace_scores.get(trace_id, {})))
        names = set()
        for obs in sorted(steps, key=lambda o: o["startTime"]):
            name, kind, latency = obs["name"], obs["type"], obs.get("latency")
            names.add(name)
            if obs.get("level") == "ERROR":
                turn.errors += 1
            if kind == "GENERATION":
                usage = obs.get("usageDetails") or {}
                cost = _number(obs.get("totalCost"))
                turn.cost += cost
                turn.input_tokens += int(_number(usage.get("input")))
                turn.output_tokens += int(_number(usage.get("output")))
                turn.cache_read_tokens += int(_number(usage.get("cache_read_input_tokens")))
                if int(_number((obs.get("metadata") or {}).get("attempt"))) > 0:
                    turn.retries += 1
                if name.startswith("classify."):
                    turn.classify_cost += cost
                    if latency is not None:
                        turn.classify_s.append(latency)
                else:
                    if latency is not None:
                        turn.model_s.append(latency)
                    if obs.get("timeToFirstToken") is not None:
                        turn.ttft_s.append(obs["timeToFirstToken"])
            elif kind == "TOOL":
                turn.tools.append(name)
                if latency is not None:
                    turn.tool_s.setdefault(name, []).append(latency)
            elif name == "call_model":
                turn.iterations += 1
                output = obs.get("output")
                if isinstance(output, dict) and output.get("final_text") and not turn.answer:
                    turn.answer = output["final_text"]
            elif name in ("quick_intent", "quick_answer"):
                turn.intent = "account"
            elif name == "classify" and isinstance(obs.get("output"), dict):
                turn.intent = (obs["output"].get("intent") or {}).get("name")
            elif name == "finish" and isinstance(obs.get("input"), dict):
                turn.outcome = obs["input"].get("outcome")
        turn.intent = turn.scores.get("intent") or turn.intent
        turn.classifier = turn.scores.get("classifier") or turn.classifier
        if turn.scores.get("jev_confidence") is not None:
            turn.jev_confidence = _number(turn.scores["jev_confidence"])
        turn.outcome = turn.scores.get("outcome") or turn.outcome or (
            "blocked" if "refuse_attack" in names
            else "handoff" if names & {"handoff", "wait_for_human"}
            else "incomplete")  # no final step: the run failed or the customer left
        turns.append(turn)
    return sorted(turns, key=lambda t: t.start)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 4)


def spread(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50_s": percentile(values, 0.5), "p95_s": percentile(values, 0.95),
            "p99_s": percentile(values, 0.99),
            "max_s": round(max(values), 4) if values else None}


def rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def mean(values: list[float], digits: int = 6) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


def summarize(turns: list[Turn], meta: dict | None = None) -> dict:
    n = len(turns)
    costs = [t.cost for t in turns]
    by_session: dict[str, float] = defaultdict(float)
    for t in turns:
        by_session[t.session_id or t.trace_id] += t.cost
    tool_s: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        for name, values in t.tool_s.items():
            tool_s[name].extend(values)
    intents = Counter(t.intent or "none" for t in turns)
    intent_s: dict[str, list[float]] = defaultdict(list)
    routing_turns: dict[str, list[Turn]] = defaultdict(list)
    classifier_turns: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        routing_turns[t.routing].append(t)
        if t.classify_s:  # turns that went through a classifier (Jev, the LLM or both)
            classifier_turns[t.classifier].append(t)
        if t.latency_s is not None:
            intent_s[t.intent or "none"].append(t.latency_s)
    outcomes = Counter(t.outcome for t in turns)
    account = [t for t in turns if t.intent == "account" and t.outcome == "done"]
    prompt_tokens = sum(t.input_tokens + t.cache_read_tokens for t in turns)

    quality = {}
    for name in QUALITY_SCORES:
        values = [t.scores[name] for t in turns if name in t.scores]
        if values:
            quality[name] = {"n": len(values),
                             "pass_rate": rate(sum(1 for v in values if v in (True, 1, 1.0)),
                                               len(values))}

    return {
        "meta": meta or {},
        "turns": n,
        "conversations": len(by_session),
        "customers": len({t.user_id for t in turns if t.user_id}),
        "window": {"first": turns[0].start, "last": turns[-1].start} if turns else None,
        "releases": dict(Counter(t.release or "unknown" for t in turns)),
        "latency": {
            "turn": spread([t.latency_s for t in turns if t.latency_s is not None]),
            "classifier": spread([v for t in turns for v in t.classify_s]),
            "model_call": spread([v for t in turns for v in t.model_s]),
            "first_token": spread([v for t in turns for v in t.ttft_s]),
            "tools": {name: spread(values) for name, values in sorted(tool_s.items())},
        },
        "cost": {
            "total_usd": round(sum(costs), 6),
            "per_turn_mean_usd": mean(costs),
            "per_turn_p95_usd": percentile(costs, 0.95),
            "per_turn_p99_usd": percentile(costs, 0.99),
            "per_conversation_mean_usd": mean(list(by_session.values())),
            "classifier_share": rate(round(sum(t.classify_cost for t in turns) * 1e6),
                                     round(sum(costs) * 1e6)),
            "input_tokens_per_turn": mean([t.input_tokens for t in turns], 1),
            "output_tokens_per_turn": mean([t.output_tokens for t in turns], 1),
            "cache_read_share": rate(sum(t.cache_read_tokens for t in turns), prompt_tokens),
        },
        "by_intent": {
            name: {"turns": count, "share": rate(count, n),
                   "p95_s": percentile(intent_s[name], 0.95),
                   "p99_s": percentile(intent_s[name], 0.99),
                   "mean_cost_usd": mean([t.cost for t in turns if (t.intent or "none") == name])}
            for name, count in intents.most_common()},
        "by_routing": {
            name: {"turns": len(group), "share": rate(len(group), n),
                   "quick_actions": sum(1 for t in group if t.quick_action),
                   "latency": spread([t.latency_s for t in group if t.latency_s is not None]),
                   "mean_cost_usd": mean([t.cost for t in group]),
                   "model_calls_per_turn": mean(
                       [len(t.model_s) + len(t.classify_s) for t in group], 2)}
            for name, group in sorted(routing_turns.items())},
        "by_classifier": {
            name: {"turns": len(group),
                   "share": rate(len(group), sum(len(g) for g in classifier_turns.values())),
                   # The classification step: every classify.* call of the turn (Jev, LLM).
                   "classify": spread([sum(t.classify_s) for t in group if t.classify_s]),
                   "classify_cost_mean_usd": mean([t.classify_cost for t in group]),
                   "turn": spread([t.latency_s for t in group if t.latency_s is not None]),
                   "jev_confidence_mean": mean([t.jev_confidence for t in group
                                                if t.jev_confidence is not None], 3),
                   "intents": dict(Counter(t.intent or "none" for t in group)),
                   "blocked_rate": rate(sum(t.outcome == "blocked" for t in group), len(group)),
                   "jev_agrees_with_llm": rate(
                       sum(1 for t in group if t.scores.get("jev_agrees_llm") in (True, 1, 1.0)),
                       sum(1 for t in group if "jev_agrees_llm" in t.scores))}
            for name, group in sorted(classifier_turns.items())},
        "outcomes": {name: {"turns": count, "share": rate(count, n)}
                     for name, count in outcomes.most_common()},
        "behaviour": {
            # Account questions that were answered: did the figures come from the bank?
            "account_used_bank_tool_rate": rate(
                sum(1 for t in account if BANK_TOOLS & set(t.tools)), len(account)),
            "iterations_mean": mean([t.iterations for t in turns if t.iterations], 2),
            "iterations_max": max((t.iterations for t in turns), default=0),
            "blocked_rate": rate(outcomes["blocked"], n),
            "handoff_rate": rate(outcomes["handoff"], n),
            "error_rate": rate(sum(1 for t in turns if t.errors), n),
            "retry_rate": rate(sum(1 for t in turns if t.retries), n),
            "max_iterations_rate": rate(outcomes["max_iterations"], n),
            "incomplete_rate": rate(outcomes["incomplete"], n),
        },
        "quality": quality,
    }


def lookup(summary: dict, path: str) -> Any:
    value: Any = summary
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def check_slo(summary: dict, slo: dict[str, dict]) -> list[str]:
    """Thresholds as ``{"latency.turn.p95_s": {"max": 8}}``. A metric without data is
    reported, not failed. Returns the broken ones."""
    broken = []
    for path, limits in slo.items():
        value = lookup(summary, path)
        if value is None:
            print(f"  SLO {path}: no data")
            continue
        if "max" in limits and value > limits["max"]:
            broken.append(f"{path} = {value} > {limits['max']}")
        if "min" in limits and value < limits["min"]:
            broken.append(f"{path} = {value} < {limits['min']}")
    return broken


def _fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "-"
    if unit == "%":
        return f"{value * 100:.1f}%"
    if unit == "$":
        return f"${value:.5f}"
    return f"{value:.2f}{unit}" if isinstance(value, float) else f"{value}{unit}"


def print_report(s: dict) -> None:
    if not s["turns"]:
        print("no agent turns in this window")
        return
    print(f"{s['turns']} turns · {s['conversations']} conversations · {s['customers']} customers"
          f" · {s['window']['first'][:16]} → {s['window']['last'][:16]} UTC")
    print(f"releases: {s['releases']}")
    print("\nLatency              n      p50      p95      p99      max")
    rows = [("turn", s["latency"]["turn"]), ("classifier", s["latency"]["classifier"]),
            ("model call", s["latency"]["model_call"]),
            ("first token", s["latency"]["first_token"]),
            *((f"tool {name}"[:20], v) for name, v in s["latency"]["tools"].items())]
    for name, v in rows:
        print(f"{name:<18}{v['n']:>5} {_fmt(v['p50_s'], 's'):>8} {_fmt(v['p95_s'], 's'):>8} "
              f"{_fmt(v['p99_s'], 's'):>8} {_fmt(v['max_s'], 's'):>8}")
    c = s["cost"]
    print(f"\nCost: total {_fmt(c['total_usd'], '$')} · per turn {_fmt(c['per_turn_mean_usd'], '$')}"
          f" (p95 {_fmt(c['per_turn_p95_usd'], '$')}, p99 {_fmt(c['per_turn_p99_usd'], '$')})"
          f" · per conversation "
          f"{_fmt(c['per_conversation_mean_usd'], '$')}")
    print(f"      classifier share {_fmt(c['classifier_share'], '%')} · tokens per turn "
          f"{c['input_tokens_per_turn']} in / {c['output_tokens_per_turn']} out · cache reads "
          f"{_fmt(c['cache_read_share'], '%')}")
    print("\nIntent            turns   share      p95      p99  mean cost")
    for name, v in s["by_intent"].items():
        print(f"{name:<16}{v['turns']:>7} {_fmt(v['share'], '%'):>7} {_fmt(v['p95_s'], 's'):>8} "
              f"{_fmt(v['p99_s'], 's'):>8} {_fmt(v['mean_cost_usd'], '$'):>10}")
    print("\nRouting           turns   quick      p50      p95      p99  mean cost  calls")
    for name, v in s["by_routing"].items():
        lat = v["latency"]
        print(f"{name:<16}{v['turns']:>7} {v['quick_actions']:>7} {_fmt(lat['p50_s'], 's'):>8} "
              f"{_fmt(lat['p95_s'], 's'):>8} {_fmt(lat['p99_s'], 's'):>8} "
              f"{_fmt(v['mean_cost_usd'], '$'):>10} {_fmt(v['model_calls_per_turn']):>6}")
    print("\nClassifier         turns  share  classify p50  p95  cost/turn   turn p95  "
          "Jev agrees with LLM")
    for name, v in s["by_classifier"].items():
        c, t = v["classify"], v["turn"]
        print(f"{name:<18}{v['turns']:>6} {_fmt(v['share'], '%'):>6} {_fmt(c['p50_s'], 's'):>13}"
              f" {_fmt(c['p95_s'], 's'):>5} {_fmt(v['classify_cost_mean_usd'], '$'):>10}"
              f" {_fmt(t['p95_s'], 's'):>10}  {_fmt(v['jev_agrees_with_llm'], '%')}")
    print("\nOutcome: " + " · ".join(f"{name} {_fmt(v['share'], '%')} ({v['turns']})"
                                      for name, v in s["outcomes"].items()))
    b = s["behaviour"]
    print(f"Behaviour: account answers that used a bank tool "
          f"{_fmt(b['account_used_bank_tool_rate'], '%')} · iterations mean "
          f"{_fmt(b['iterations_mean'])} max {b['iterations_max']} · errors "
          f"{_fmt(b['error_rate'], '%')} · retries {_fmt(b['retry_rate'], '%')} · incomplete "
          f"{_fmt(b['incomplete_rate'], '%')}")
    if s["quality"]:
        print("Quality (evals/score_traces.py): " + " · ".join(
            f"{name} {_fmt(v['pass_rate'], '%')} of {v['n']}" for name, v in s["quality"].items()))


COMPARED = ["turns", "latency.turn.p50_s", "latency.turn.p95_s", "latency.turn.p99_s",
            "latency.model_call.p95_s",
            "latency.first_token.p95_s", "cost.per_turn_mean_usd", "cost.classifier_share",
            "behaviour.account_used_bank_tool_rate", "behaviour.error_rate",
            "behaviour.blocked_rate", "behaviour.handoff_rate"]


def print_comparison(before: dict, after: dict) -> None:
    print(f"\nvs {before.get('meta', {}).get('timestamp', 'earlier run')}:")
    for path in COMPARED:
        old, new = lookup(before, path), lookup(after, path)
        if old is None or new is None:
            continue
        change = f"{(new - old) / old * 100:+.1f}%" if old else "n/a"
        print(f"  {path:<40} {old:>10} → {new:<10} {change}")


def to_case(turn: Turn) -> dict:
    """An eval case (evals/README.md) from a turn. The text is masked: review it, and
    restore or replace the placeholders, before adding it to a suite."""
    case: dict[str, Any] = {
        "id": f"trace-{turn.trace_id[:12]}", "route": "assistant", "user": "eval-es",
        "input": turn.question, "tags": ["from-trace", turn.intent or "none"],
        "checks": {"outcome": turn.outcome},
        "note": f"masked text from trace {turn.trace_id}; review before use"}
    if turn.language:
        case["language"] = turn.language
        if turn.language in ("es", "pt") and turn.outcome == "done":
            case["checks"]["language"] = turn.language
    return case


def load_turns(api: LangfuseAPI, since: datetime, until: datetime | None = None,
               release: str | None = None, routing: str | None = None,
               quick_only: bool = False, classifier: str | None = None) -> list[Turn]:
    observations = [o for o in api.observations(since, until)]
    # Inputs and outputs only where they are small and needed: steps and the run itself.
    detailed = {o["id"]: o for kind in ("SPAN", "AGENT")
                for o in api.observations(since, until, type=kind, io=True)}
    observations = [detailed.get(o["id"], o) for o in observations]
    turns = build_turns(observations, list(api.scores(since)))
    return [t for t in turns if (release is None or t.release == release)
            and (routing is None or t.routing == routing)
            and (not quick_only or t.quick_action)
            and (classifier is None or t.classifier == classifier)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance report from Langfuse traces.")
    parser.add_argument("--since", default="7d", help="24h, 7d, 30m or an ISO date (UTC)")
    parser.add_argument("--until", help="ISO date (UTC); default: now")
    parser.add_argument("--release", help="only turns served by this release (git commit)")
    parser.add_argument("--routing", help="only turns routed this way: model, intent...")
    parser.add_argument("--classifier",
                        help="only turns whose intent this decided: jev, llm, llm_low_confidence...")
    parser.add_argument("--quick", action="store_true",
                        help="only turns sent by a quick-action button")
    parser.add_argument("--check", nargs="?", const=str(DEFAULT_SLO), metavar="SLO_JSON",
                        help="exit 1 if a threshold is broken (default: evals/trace_slo.json)")
    parser.add_argument("--compare", metavar="RESULT_JSON", help="show changes vs a saved report")
    parser.add_argument("--export-cases", action="store_true",
                        help="write the selected turns as eval cases instead of a report")
    parser.add_argument("--outcome", help="with --export-cases: only turns that ended this way")
    parser.add_argument("--failed", metavar="SCORE",
                        help="with --export-cases: only turns that failed this quality score")
    args = parser.parse_args()

    api = LangfuseAPI.from_settings(get_settings())
    try:
        turns = load_turns(api, parse_since(args.since),
                           parse_since(args.until) if args.until else None, args.release,
                           args.routing, args.quick, args.classifier)
    finally:
        api.close()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    if args.export_cases:
        chosen = [t for t in turns if t.question
                  and (args.outcome is None or t.outcome == args.outcome)
                  and (args.failed is None or t.scores.get(args.failed) in (False, 0, 0.0))]
        out = out_dir / f"cases-{stamp}.jsonl"
        out.write_text("".join(json.dumps(to_case(t), ensure_ascii=False) + "\n" for t in chosen))
        print(f"{len(chosen)} cases (masked text, review before use): {out}")
        return 0

    summary = summarize(turns, {"timestamp": datetime.now(UTC).isoformat(),
                                "since": args.since, "until": args.until,
                                "release": args.release, "routing": args.routing,
                                "quick": args.quick, "classifier": args.classifier})
    print_report(summary)
    out = out_dir / f"traces-{stamp}.json"
    out.write_text(json.dumps({**summary, "turn_rows": [
        {k: v for k, v in asdict(t).items() if k not in ("question", "answer")}
        for t in turns]}, indent=2))
    print(f"\nresults: {out}")
    if args.compare:
        print_comparison(json.loads(Path(args.compare).read_text()), summary)
    if args.check:
        broken = check_slo(summary, json.loads(Path(args.check).read_text()))
        if broken:
            print("FAIL: " + "; ".join(broken))
            return 1
        print("SLO: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
