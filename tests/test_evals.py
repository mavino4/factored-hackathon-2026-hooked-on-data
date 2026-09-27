from pathlib import Path

from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from evals.run import (
    Answer,
    Case,
    check_answer,
    compare,
    load_dataset,
    parse_judge,
    run_case,
    summarize,
)
from tests.fakes import FakeClient, make_message, text_reply


def test_dataset_loads_and_is_well_formed():
    cases = load_dataset(Path("evals/dataset.jsonl"))
    assert len(cases) >= 10
    assert len({c.id for c in cases}) == len(cases)
    assert {c.route for c in cases} <= {"chat", "agent"}


def test_check_answer_rules():
    ok = Answer(text="Tokyo", tools_called=["get_current_time"])
    assert check_answer(ok, {"must_include_any": ["tokyo"], "max_words": 2,
                             "must_call_tool": "get_current_time"}) == []
    bad = Answer(text="I cannot help with that request today", tools_called=["create_support_ticket"],
                 tools_executed=["create_support_ticket"])
    failures = check_answer(bad, {"must_include_any": ["Tokyo"], "must_not_include": ["cannot help"],
                                  "max_words": 3, "must_call_tool": "get_current_time",
                                  "must_not_call_tool": "create_support_ticket",
                                  "must_not_execute_tool": "create_support_ticket"})
    assert len(failures) == 6


def test_parse_judge_is_robust():
    assert parse_judge('Sure! {"pass": true, "reason": "ok"}')["pass"] is True
    assert parse_judge('{"pass": "yes"}')["pass"] is False
    assert parse_judge("no json here")["pass"] is False


def test_compare_reports_regressions_and_drop():
    base = {"score": 100.0, "cases": [{"id": "a", "passed": True}, {"id": "b", "passed": True}]}
    now = {"score": 50.0, "cases": [{"id": "a", "passed": True}, {"id": "b", "passed": False}]}
    regressions, drop = compare(base, now)
    assert regressions == ["b"] and drop == 50.0


def gateway(client):
    return AIGateway({"anthropic": client}, Settings(providers=["anthropic"]))


async def test_chat_case_scores_and_costs():
    result = await run_case(gateway(FakeClient(text_reply("Tokyo"))),
                            Case(id="c", route="chat", input="capital?",
                                 checks={"must_include_any": ["Tokyo"]}), use_judge=False)
    assert result.passed and result.input_tokens == 10 and result.output_tokens == 5
    assert result.cost > 0  # Haiku 4.5 prices


async def test_agent_case_never_executes_irreversible_tools():
    call = make_message({"type": "tool_use", "id": "t1", "name": "create_support_ticket",
                         "input": {"title": "x", "details": "y"}}, stop_reason="tool_use")
    client = FakeClient(([], call), text_reply("Please confirm the ticket."))
    case = Case(id="a", route="agent", input="open a ticket",
                checks={"must_call_tool": "create_support_ticket",
                        "must_not_execute_tool": "create_support_ticket",
                        "must_include_any": ["confirm"]})
    result = await run_case(gateway(client), case, use_judge=False)
    assert result.passed, result.failures
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] and "NOT executed" in tool_result["content"]


async def test_judge_failure_fails_the_case_and_errors_are_caught():
    client = FakeClient(text_reply("Paris"), text_reply('{"pass": false, "reason": "wrong"}'))
    case = Case(id="j", route="chat", input="q", rubric="must be Tokyo")
    result = await run_case(gateway(client), case, use_judge=True)
    assert not result.passed and result.failures == ["judge: wrong"]
    crashed = await run_case(gateway(FakeClient(RuntimeError("boom"))), case, use_judge=False)
    assert not crashed.passed and "boom" in crashed.failures[0]
    assert summarize([result, crashed], {})["score"] == 0.0
