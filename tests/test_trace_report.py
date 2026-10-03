from datetime import UTC, datetime

import httpx

from evals.langfuse_api import LangfuseAPI, decoded
from evals.score_traces import deterministic_scores
from evals.traces import (
    Turn,
    build_turns,
    check_slo,
    parse_since,
    percentile,
    summarize,
    to_case,
)

T0 = "2026-10-01T10:00:0{}.000Z"


def obs(trace, name, kind="SPAN", *, parent="root", latency=0.1, n=0, **extra) -> dict:
    return {"id": f"{trace}-{name}-{n}", "traceId": trace, "name": name, "type": kind,
            "parentObservationId": parent, "startTime": T0.format(n), "latency": latency,
            "level": "DEFAULT", "metadata": {}, **extra}


def root(trace, latency, **extra) -> dict:
    return obs(trace, "agent", "AGENT", parent=None, latency=latency, sessionId=f"s-{trace}",
               userId="CLI-1", metadata={"language": "es"}, input="¿Cuál es mi saldo?", **extra)


def generation(trace, name, cost, n, **extra) -> dict:
    return obs(trace, name, "GENERATION", latency=1.0, n=n, totalCost=cost,
               usageDetails={"input": 100, "output": 20, "cache_read_input_tokens": 0},
               metadata={"attempt": 0}, **extra)


def account_turn(trace="t1") -> list[dict]:
    return [
        root(trace, 4.0),
        obs(trace, "classify", n=1, output={"intent": {"name": "account"}, "use_tools": True}),
        generation(trace, "classify.anthropic", 0.001, 1),
        obs(trace, "call_model", n=2, output={"final_text": "", "outcome": None}),
        generation(trace, "agent.anthropic", 0.002, 2),
        obs(trace, "get_products", "TOOL", latency=0.05, n=3),
        obs(trace, "call_model", n=4, output={"final_text": "Su saldo es <AMOUNT>.",
                                              "outcome": "done"}),
        generation(trace, "agent.anthropic", 0.003, 4, timeToFirstToken=0.4),
        obs(trace, "finish", n=5, input={"outcome": "done"}),
    ]


def attack_turn(trace="t2") -> list[dict]:
    return [root(trace, 2.0),
            obs(trace, "classify", n=1, output={"intent": {"name": "attack"}}),
            generation(trace, "classify.anthropic", 0.001, 1),
            obs(trace, "refuse_attack", n=2)]


def test_a_turn_is_read_from_its_steps():
    turn, = build_turns(account_turn())
    assert (turn.intent, turn.outcome, turn.tools) == ("account", "done", ["get_products"])
    assert turn.iterations == 2 and turn.answer == "Su saldo es <AMOUNT>."
    assert round(turn.cost, 6) == 0.006 and round(turn.classify_cost, 6) == 0.001
    assert turn.ttft_s == [0.4] and turn.language == "es"


def test_outcome_without_a_final_step_is_derived():
    blocked, = build_turns(attack_turn())
    assert blocked.outcome == "blocked"
    cut, = build_turns([root("t3", 1.0), obs("t3", "classify", n=1)])
    assert cut.outcome == "incomplete"


def test_scores_win_over_derived_values():
    scores = [{"name": "outcome", "value": "refused", "subject": {"kind": "trace", "id": "t1"}},
              {"name": "no_leak", "value": True, "subject": {"kind": "trace", "id": "t1"}}]
    turn, = build_turns(account_turn(), scores)
    assert turn.outcome == "refused" and turn.scores["no_leak"] is True


def test_runs_that_are_not_agent_turns_are_ignored():
    assert build_turns([obs("old", "__start__", parent=None)]) == []


def test_summary():
    failed_tool = obs("t4", "get_products", "TOOL", n=3, level="ERROR")
    s = summarize(build_turns([*account_turn(), *attack_turn(), root("t4", 6.0), failed_tool]))
    assert s["turns"] == 3 and s["conversations"] == 3 and s["customers"] == 1
    assert s["latency"]["turn"]["p50_s"] == 4.0
    assert s["latency"]["turn"]["p99_s"] == 5.96
    assert s["latency"]["first_token"]["n"] == 1
    assert s["cost"]["total_usd"] == 0.007
    assert s["cost"]["classifier_share"] == round(0.002 / 0.007, 4)
    assert s["outcomes"]["blocked"]["turns"] == 1
    assert s["by_intent"]["account"]["share"] == round(1 / 3, 4)
    assert s["behaviour"]["account_used_bank_tool_rate"] == 1.0
    assert s["behaviour"]["error_rate"] == round(1 / 3, 4)


def test_turns_are_grouped_by_routing():
    quick_root = root("t5", 1.5)
    quick_root["metadata"].update(routing="intent", quick_action="qa_card_balance")
    quick_turn = [quick_root,
                  obs("t5", "quick_intent", n=1),
                  obs("t5", "call_model", n=2),
                  generation("t5", "agent.anthropic", 0.002, 2)]
    turns = build_turns([*account_turn(), *quick_turn])
    assert [(t.routing, t.quick_action, t.intent) for t in turns] == [
        ("model", None, "account"), ("intent", "qa_card_balance", "account")]
    routing = summarize(turns)["by_routing"]
    assert routing["intent"]["turns"] == 1 and routing["intent"]["quick_actions"] == 1
    assert routing["intent"]["model_calls_per_turn"] == 1.0
    assert routing["model"]["model_calls_per_turn"] == 3.0


def test_summary_of_nothing():
    assert summarize([])["turns"] == 0


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile(list(range(101)), 0.99) == 99
    assert percentile([], 0.95) is None


def test_slo():
    s = summarize(build_turns(account_turn()))
    assert check_slo(s, {"latency.turn.p95_s": {"max": 8}}) == []
    assert check_slo(s, {"latency.turn.p95_s": {"max": 1},
                         "behaviour.account_used_bank_tool_rate": {"min": 1.0}}) == [
        "latency.turn.p95_s = 4.0 > 1"]
    assert check_slo(s, {"latency.nothing.p95_s": {"max": 1}}) == []  # no data: not a failure


def test_since():
    now = datetime(2026, 10, 2, 12, tzinfo=UTC)
    assert parse_since("24h", now) == datetime(2026, 10, 1, 12, tzinfo=UTC)
    assert parse_since("7d", now) == datetime(2026, 9, 25, 12, tzinfo=UTC)
    assert parse_since("2026-10-01") == datetime(2026, 10, 1, tzinfo=UTC)


def test_case_from_a_turn():
    case = to_case(build_turns(account_turn())[0])
    assert case["input"] == "¿Cuál es mi saldo?" and case["route"] == "assistant"
    assert case["checks"] == {"outcome": "done", "language": "es"}


def answered(**kwargs) -> Turn:
    return Turn(trace_id="t", start="", language="es", outcome="done", **kwargs)


def test_deterministic_scores():
    good = answered(intent="account", tools=["get_products"],
                    answer="Su saldo disponible en la cuenta es <AMOUNT>, gracias por su consulta.")
    assert {k: v[0] for k, v in deterministic_scores(good).items()} == {
        "no_leak": True, "language_match": True, "used_bank_tool": True}
    bad = answered(intent="account",
                   answer="Você tem saldo, obrigado; uso get_products para não errar com isso.")
    assert {k: v[0] for k, v in deterministic_scores(bad).items()} == {
        "no_leak": False, "language_match": False, "used_bank_tool": False}
    assert deterministic_scores(answered(answer=None)) == {}
    # Too short to tell the language; a general question needs no bank tool.
    assert list(deterministic_scores(answered(intent="general", answer="Claro."))) == ["no_leak"]


def test_api_follows_the_cursor_and_decodes_payloads():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if "cursor" not in request.url.params:
            return httpx.Response(200, json={"data": [{"id": "a", "input": '{"x": 1}',
                                                       "output": "hola"}],
                                             "meta": {"cursor": "next"}})
        return httpx.Response(200, json={"data": [{"id": "b"}], "meta": {}})

    api = LangfuseAPI("http://lf", "pk", "sk", transport=httpx.MockTransport(handler))
    rows = list(api.observations(datetime(2026, 10, 1, tzinfo=UTC), type="SPAN", io=True))
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["input"] == {"x": 1} and rows[0]["output"] == "hola"
    assert seen[0]["fromStartTime"] == "2026-10-01T00:00:00Z" and seen[0]["type"] == "SPAN"
    assert seen[1]["cursor"] == "next"
    assert decoded("not json {") == "not json {"


def test_turns_are_grouped_by_who_decided_the_intent():
    jev_turn = [root("t6", 2.0), obs("t6", "classify", n=1),
                generation("t6", "classify.jev", 0.00001, 1),
                obs("t6", "call_model", n=2), generation("t6", "agent.anthropic", 0.002, 2)]
    scores = [{"name": "classifier", "value": "jev", "subject": {"kind": "trace", "id": "t6"}},
              {"name": "jev_confidence", "value": 0.97, "subject": {"kind": "trace", "id": "t6"}},
              {"name": "jev_agrees_llm", "value": True, "subject": {"kind": "trace", "id": "t6"}}]
    turns = build_turns([*account_turn(), *jev_turn], scores)
    assert [(t.classifier, t.jev_confidence) for t in turns] == [("llm", None), ("jev", 0.97)]
    by = summarize(turns)["by_classifier"]
    assert by["jev"]["turns"] == 1 and by["jev"]["share"] == 0.5
    assert by["jev"]["classify_cost_mean_usd"] == 0.00001
    assert by["jev"]["jev_agrees_with_llm"] == 1.0
    assert by["llm"]["jev_agrees_with_llm"] is None
