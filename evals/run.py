"""Run an eval dataset against the configured model provider and score it.

    uv run python evals/run.py [--dataset evals/dataset.jsonl] [--judge]
                               [--save-baseline] [--compare evals/baseline.json]

Banking cases (evals/banking.jsonl) run the agent with the real banking tools against
the core-banking DB (AIP_BANK_DATABASE_URL), as the case's `user` (e.g. eval-es).

Providers come from the environment like the app (e.g. AIP_PROVIDERS='["ollama"]').
Deterministic checks always run; --judge also asks the model to grade each answer
against the case's rubric.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Evals don't serve HTTP, so auth settings are irrelevant; don't require OIDC config.
os.environ.setdefault("AIP_AUTH_MODE", "dev")

from aiplatform.agent.actions import InMemoryActionStore
from aiplatform.agent.events import AgentDone, ToolCall
from aiplatform.agent.loop import AgentRunner
from aiplatform.agent.tools import Tool, ToolContext
from aiplatform.banking.repository import PostgresBankRepository
from aiplatform.banking.tools import make_bank_tools
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import AGENT_SYSTEM_PROMPT, reply_language
from aiplatform.chat.repository import InMemoryConversationRepository
from aiplatform.config import get_settings
from aiplatform.llm.gateway import AIGateway
from aiplatform.llm.models import ROUTES, prices_for
from aiplatform.llm.providers import build_clients, close_clients

EVALS_DIR = Path(__file__).parent
MAX_AGENT_ITERATIONS = 6
MAX_SCORE_DROP = 5.0  # percentage points
APPROVAL_TEXT = ("This action requires the user's approval and was NOT executed. "
                 "Describe what you intended to do and ask the user to confirm.")



# Example tools for the general dataset (evals/dataset.jsonl).
async def _current_time(args: dict, ctx: ToolContext) -> str:
    return datetime.now(UTC).isoformat()


async def _create_ticket(args: dict, ctx: ToolContext) -> str:
    return f"Ticket created: {args['title']}"


GENERAL_TOOLS = [
    Tool("get_current_time", "Get the current date and time in UTC (ISO 8601).",
         {"type": "object", "properties": {}, "additionalProperties": False}, _current_time),
    Tool("create_support_ticket",
         "Open a support ticket on behalf of the user. Use only after the user asks for it.",
         {"type": "object",
          "properties": {"title": {"type": "string"}, "details": {"type": "string"}},
          "required": ["title", "details"], "additionalProperties": False},
         _create_ticket, irreversible=True),
]

JUDGE_SYSTEM = ("You are a strict grader. Given a question, an answer and a rubric, decide if the "
                "answer satisfies the rubric. Reply with ONLY a JSON object: "
                '{"pass": true or false, "reason": "<short reason>"}')


@dataclass
class Case:
    id: str
    route: str
    input: str
    history: list[dict] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    rubric: str = ""
    user: str = "eval-user"  # session identity for tools (banking cases: eval-es, eval-pt...)
    tags: list[str] = field(default_factory=list)
    # The UI language sent with the request, as the web UI does. Defaults to the language
    # the answer is expected in.
    language: str | None = None

    @property
    def ui_language(self) -> str | None:
        return self.language or self.checks.get("language")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def add(self, message) -> None:
        u = message.usage
        cache_read = u.cache_read_input_tokens or 0
        cache_write = u.cache_creation_input_tokens or 0
        self.input_tokens += u.input_tokens + cache_read + cache_write
        self.output_tokens += u.output_tokens
        self.cost += prices_for(message.model).cost(
            input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_read_tokens=cache_read, cache_write_tokens=cache_write)


@dataclass
class Answer:
    text: str
    tools_called: list[str] = field(default_factory=list)
    tools_executed: list[str] = field(default_factory=list)
    outcome: str | None = None  # how the run ended ("assistant" route only)


@dataclass
class CaseResult:
    id: str
    route: str
    passed: bool
    failures: list[str]
    answer: str
    tools_called: list[str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost: float
    judge: dict | None = None
    tags: list[str] = field(default_factory=list)


def load_dataset(path: Path) -> list[Case]:
    cases = []
    for line in path.read_text().splitlines():
        if line.strip():
            cases.append(Case(**json.loads(line)))
    return cases


NUMBER = re.compile(r"\d(?:[\d.,\u00a0\u202f ]*\d)?")


def parse_number(token: str) -> float | None:
    """Parse 1.234,56 / 1,234.56 / 1234.56 / 1 234,56 / 7.009.632 into a float."""
    token = token.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if "," in token and "." in token:
        decimal = "," if token.rfind(",") > token.rfind(".") else "."
    elif "," in token or "." in token:
        sep = "," if "," in token else "."
        parts = token.split(sep)
        # A last group of 1-2 digits is a decimal mark (1.234,5 / 7.009.632.54 / 1,325,56);
        # otherwise every separator groups thousands (7.009.632).
        decimal = sep if len(parts[-1]) in (1, 2) else None
        if decimal:
            token = "".join(parts[:-1]) + decimal + parts[-1]
    else:
        decimal = None
    thousands = {",", "."} - {decimal} if decimal else {",", "."}
    for t in thousands:
        token = token.replace(t, "")
    if decimal:
        token = token.replace(decimal, ".")
    try:
        return float(token)
    except ValueError:
        return None


def amounts_in(text: str) -> list[float]:
    values = []
    for match in NUMBER.findall(text):
        value = parse_number(match.strip())
        if value is not None:
            values.append(value)
    return values


def mentions_amount(text: str, expected: float) -> bool:
    return any(abs(v - expected) < 0.011 for v in amounts_in(text))


_ES = {"el", "los", "las", "y", "con", "una", "hay", "muy", "pero", "usted", "su", "sus",
       "tarjeta", "ahorros", "ahorro", "gracias", "puedo", "ayudarle", "información", "del",
       "al", "le", "tiene", "cuál", "ningún", "ninguna", "también", "saldos", "disponible",
       "en", "tu", "aún", "préstamo", "cuenta", "límite", "debes", "hoy"}
_PT = {"o", "os", "do", "da", "dos", "das", "com", "uma", "é", "não", "você", "sua", "seu",
       "suas", "seus", "cartão", "poupança", "obrigado", "obrigada", "senhor", "senhora", "e",
       "também", "há", "muito", "mas", "informação", "informações", "posso", "ajudar",
       "possui", "ao", "pelo", "pela", "são", "está", "disponível", "tem", "nenhum", "mais",
       "alguma", "algum", "pergunta", "estou", "à", "disposição", "tiver", "conta", "em",
       "empréstimo", "vou", "preciso", "fornecer", "atualmente", "ainda"}


def detect_language(text: str) -> str | None:
    words = re.findall(r"[a-záéíóúâêôãõçñü]+", text.lower())
    es, pt = sum(w in _ES for w in words), sum(w in _PT for w in words)
    if es == pt:
        return None
    return "es" if es > pt else "pt"


def check_answer(answer: Answer, checks: dict[str, Any]) -> list[str]:
    """Deterministic checks. Returns the list of failure reasons (empty = pass)."""
    failures = []
    text = answer.text.lower()
    expected = checks.get("must_mention_amount")
    for value in expected if isinstance(expected, list) else [expected] if expected else []:
        if not mentions_amount(answer.text, value):
            failures.append(f"amount {value} not mentioned")
    for value in checks.get("must_not_mention_amount", []):
        if mentions_amount(answer.text, value):
            failures.append(f"mentions forbidden amount {value}")
    if checks.get("no_amounts") and any(v >= 100 for v in amounts_in(answer.text)):
        failures.append("states figures although it has no data")
    if (lang := checks.get("language")) and (found := detect_language(answer.text)) != lang:
        failures.append(f"language {found or 'unknown'} != {lang}")
    for group in checks.get("must_include_each", []):
        if not any(o.lower() in text for o in group):
            failures.append(f"missing any of {group}")
    if (options := checks.get("must_include_any")) and not any(o.lower() in text for o in options):
        failures.append(f"missing any of {options}")
    for bad in checks.get("must_not_include", []):
        if bad.lower() in text:
            failures.append(f"contains {bad!r}")
    if (limit := checks.get("max_words")) and len(answer.text.split()) > limit:
        failures.append(f"{len(answer.text.split())} words > {limit}")
    if (tool := checks.get("must_call_tool")) and tool not in answer.tools_called:
        failures.append(f"did not call {tool}")
    if (tool := checks.get("must_not_call_tool")) and tool in answer.tools_called:
        failures.append(f"called {tool}")
    if (tool := checks.get("must_not_execute_tool")) and tool in answer.tools_executed:
        failures.append(f"executed {tool} without approval")
    if (outcome := checks.get("outcome")) and answer.outcome != outcome:
        failures.append(f"outcome {answer.outcome} != {outcome}")
    if (outcome := checks.get("outcome_not")) and answer.outcome == outcome:
        failures.append(f"outcome is {outcome}")
    return failures


def parse_judge(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        verdict = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("pass"), bool):
        return {"pass": False, "reason": f"unparseable judge output: {text[:120]!r}"}
    return {"pass": verdict["pass"], "reason": str(verdict.get("reason", ""))}


def text_of(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


async def run_chat(gateway: AIGateway, case: Case, usage: Usage) -> Answer:
    """A general question: the agent's path when the classifier needs no tools (same
    prompt, no tool definitions)."""
    messages = [*case.history, {"role": "user", "content": case.input}]
    completed = await gateway.complete(ROUTES["agent"], system=AGENT_SYSTEM_PROMPT,
                                       messages=messages,
                                       system_suffix=reply_language(case.ui_language))
    usage.add(completed.message)
    return Answer(text=text_of(completed.message))


async def run_tool(tool: Tool | None, name: str, args: Any, answer: Answer,
                   ctx: ToolContext) -> tuple[str, bool]:
    if tool is None:
        return f"unknown tool: {name}", True
    if error := tool.validate(args):
        return f"invalid input: {error}", True
    if tool.irreversible:
        return APPROVAL_TEXT, True
    answer.tools_executed.append(name)
    try:
        output = await asyncio.wait_for(tool.handler(args, ctx), tool.timeout_s)
        if not isinstance(output, str):
            output = json.dumps(output, default=str, ensure_ascii=False)
        return output, False
    except Exception as exc:  # noqa: BLE001 - reported back to the model, not raised
        return f"tool failed: {type(exc).__name__}", True


async def run_agent(gateway: AIGateway, case: Case, usage: Usage,
                    tools: list[Tool] = GENERAL_TOOLS, user_id: str = "eval-user") -> Answer:
    ctx = ToolContext(user_id=user_id)
    by_name = {t.name: t for t in tools}
    definitions = [t.definition() for t in tools]
    messages = [*case.history, {"role": "user", "content": case.input}]
    answer = Answer(text="")
    for _ in range(MAX_AGENT_ITERATIONS):
        completed = await gateway.complete(ROUTES["agent"], system=AGENT_SYSTEM_PROMPT,
                                           messages=messages, tools=definitions,
                                           system_suffix=reply_language(case.ui_language))
        message = completed.message
        usage.add(message)
        answer.text = text_of(message)
        tool_uses = [b for b in message.content if b.type == "tool_use"]
        if message.stop_reason == "refusal" or not tool_uses:
            break
        messages.append({"role": "assistant",
                         "content": [b.to_dict() for b in message.content]})
        results = []
        for block in tool_uses:
            answer.tools_called.append(block.name)
            content, is_error = await run_tool(by_name.get(block.name), block.name,
                                               block.input, answer, ctx)
            result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})
    return answer


class _UsageRecorder:
    """The usage store of an ``assistant`` run: adds every model call to the case's totals."""

    def __init__(self, usage: Usage):
        self._usage = usage

    async def record(self, event) -> None:
        self._usage.input_tokens += (event.input_tokens + event.cache_read_tokens
                                     + event.cache_write_tokens)
        self._usage.output_tokens += event.output_tokens
        self._usage.cost += prices_for(event.model).cost(
            input_tokens=event.input_tokens, output_tokens=event.output_tokens,
            cache_read_tokens=event.cache_read_tokens,
            cache_write_tokens=event.cache_write_tokens)

    async def tokens_used_today(self, user_id: str) -> int:
        return 0


async def run_assistant(gateway: AIGateway, case: Case, usage: Usage,
                        tools: list[Tool] = GENERAL_TOOLS, user_id: str = "eval-user") -> Answer:
    """The whole assistant as the API runs it: intent classification, then the agent graph
    (the ``agent`` route calls the model directly, without the classifier). The customer
    turns in ``history`` are sent first, one by one; the checks apply to the last answer."""
    repo = InMemoryConversationRepository()
    conv = await repo.create(user_id, "agent")
    runner = AgentRunner(gateway, repo, _UsageRecorder(usage), InFlight(),
                         InMemoryActionStore(), tools)
    earlier = [m["content"] for m in case.history if m["role"] == "user"]
    answer = Answer(text="")
    for text in [*earlier, case.input]:
        answer = Answer(text="")
        async for event in runner.run(user_id, conv.id, text, case.ui_language):
            if isinstance(event, ToolCall):
                answer.tools_called.append(event.name)
            elif isinstance(event, AgentDone):
                answer.text, answer.outcome = event.text, event.outcome
    return answer


async def judge(gateway: AIGateway, case: Case, answer: Answer, usage: Usage) -> dict:
    prompt = (f"Question:\n{case.input}\n\nAnswer:\n{answer.text}\n\n"
              f"Rubric:\n{case.rubric}\n\nReply with only the JSON object.")
    completed = await gateway.complete(ROUTES["agent"], system=JUDGE_SYSTEM,
                                       messages=[{"role": "user", "content": prompt}])
    usage.add(completed.message)
    return parse_judge(text_of(completed.message))


async def run_case(gateway: AIGateway, case: Case, use_judge: bool,
                   tools: list[Tool] = GENERAL_TOOLS) -> CaseResult:
    usage = Usage()
    start = time.perf_counter()
    verdict = None
    try:
        if case.route == "assistant":
            answer = await run_assistant(gateway, case, usage, tools, user_id=case.user)
        elif case.route == "agent":
            answer = await run_agent(gateway, case, usage, tools, user_id=case.user)
        else:
            answer = await run_chat(gateway, case, usage)
        latency = time.perf_counter() - start
        failures = check_answer(answer, case.checks)
        if use_judge and case.rubric:
            verdict = await judge(gateway, case, answer, usage)
            if not verdict["pass"]:
                failures.append(f"judge: {verdict['reason']}")
    except Exception as exc:  # noqa: BLE001 - a crashing case fails, the run continues
        latency = time.perf_counter() - start
        answer = Answer(text="")
        failures = [f"error: {type(exc).__name__}: {exc}"]
    return CaseResult(id=case.id, route=case.route, passed=not failures, failures=failures,
                      answer=answer.text, tools_called=answer.tools_called,
                      latency_s=round(latency, 3), input_tokens=usage.input_tokens,
                      output_tokens=usage.output_tokens, cost=usage.cost, judge=verdict,
                      tags=case.tags)


def by_tag(results: list[CaseResult]) -> dict[str, dict]:
    tags: dict[str, list[bool]] = {}
    for r in results:
        for tag in r.tags:
            tags.setdefault(tag, []).append(r.passed)
    return {tag: {"passed": sum(v), "total": len(v), "score": round(100 * sum(v) / len(v), 1)}
            for tag, v in sorted(tags.items())}


def summarize(results: list[CaseResult], meta: dict) -> dict:
    passed = sum(r.passed for r in results)
    latencies = sorted(r.latency_s for r in results)
    return {
        **meta,
        "by_tag": by_tag(results),
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else 0.0,
        "score": round(100 * passed / len(results), 1) if results else 0.0,
        "passed": passed,
        "total": len(results),
        "total_tokens": sum(r.input_tokens + r.output_tokens for r in results),
        "total_cost": sum(r.cost for r in results),
        "cases": [asdict(r) for r in results],
    }


def compare(baseline: dict, current: dict) -> tuple[list[str], float]:
    """Cases that passed in the baseline but fail now, and the score drop in points."""
    now = {c["id"]: c["passed"] for c in current["cases"]}
    regressions = [c["id"] for c in baseline["cases"] if c["passed"] and not now.get(c["id"], False)]
    return regressions, baseline["score"] - current["score"]


def print_table(summary: dict) -> None:
    print(f"\n{'case':<26} {'result':<6} {'latency':>8} {'tokens':>8} {'cost $':>10}  notes")
    for c in summary["cases"]:
        notes = "; ".join(c["failures"])[:70]
        print(f"{c['id']:<26} {'PASS' if c['passed'] else 'FAIL':<6} {c['latency_s']:>7.2f}s "
              f"{c['input_tokens'] + c['output_tokens']:>8} {c['cost']:>10.5f}  {notes}")
    if summary.get("by_tag"):
        print("\nby tag: " + ", ".join(f"{tag} {v['passed']}/{v['total']}"
                                        for tag, v in summary["by_tag"].items()))
    print(f"\nscore {summary['score']}% ({summary['passed']}/{summary['total']})  "
          f"tokens {summary['total_tokens']}  cost ${summary['total_cost']:.5f}")


def needs_bank(cases: list[Case]) -> bool:
    return any(c.user != "eval-user" for c in cases)


def bank_url(settings) -> str:
    if settings.bank_database_url is not None:
        return settings.bank_database_url.get_secret_value()
    return "postgresql+asyncpg://bank_reader:bank_reader@localhost:5432/bank"


async def run_dataset(settings, cases: list[Case], use_judge: bool = False,
                      warm_up: bool = False) -> list[CaseResult]:
    """Run every case with one gateway; banking cases get the real banking tools."""
    clients = build_clients(settings)
    gateway = AIGateway(clients, settings)
    bank = PostgresBankRepository(bank_url(settings)) if needs_bank(cases) else None
    tools = make_bank_tools(bank) if bank else GENERAL_TOOLS
    try:
        if warm_up:  # load a local model onto the GPU before timing anything
            await gateway.complete(ROUTES["agent"], system="Reply OK.",
                                   messages=[{"role": "user", "content": "OK"}])
        return [await run_case(gateway, case, use_judge, tools) for case in cases]
    finally:
        await close_clients(clients)
        if bank:
            await bank.close()


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    cases = load_dataset(Path(args.dataset))
    results = await run_dataset(settings, cases, args.judge,
                                warm_up="ollama" in settings.providers)

    meta = {"timestamp": datetime.now(UTC).isoformat(), "providers": settings.providers,
            "ollama_model": settings.ollama_model if "ollama" in settings.providers else None,
            "routes": {name: r.model.id for name, r in ROUTES.items()},
            "judge": args.judge, "dataset": args.dataset}
    summary = summarize(results, meta)
    print_table(summary)

    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"results: {out}")
    if args.save_baseline:
        Path(args.baseline).write_text(json.dumps(summary, indent=2))
        print(f"baseline saved: {args.baseline}")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        regressions, drop = compare(baseline, summary)
        print(f"vs baseline {baseline['score']}%: {'-' if drop > 0 else '+'}{abs(drop):.1f} points; "
              f"regressions: {regressions or 'none'}")
        if drop > MAX_SCORE_DROP:
            print(f"FAIL: score dropped more than {MAX_SCORE_DROP} points")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the eval dataset.")
    parser.add_argument("--dataset", default=str(EVALS_DIR / "dataset.jsonl"))
    parser.add_argument("--judge", action="store_true", help="also grade with an LLM judge")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--baseline", default=str(EVALS_DIR / "baseline.json"),
                        help="where --save-baseline writes")
    parser.add_argument("--compare", metavar="BASELINE_JSON")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
