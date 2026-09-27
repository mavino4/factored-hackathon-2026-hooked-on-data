import json
import logging

from aiplatform.logging import JsonFormatter


def test_json_formatter_includes_extra_fields():
    record = logging.makeLogRecord({"msg": "model usage", "levelname": "INFO",
                                    "name": "x", "input_tokens": 12, "provider": "anthropic"})
    out = json.loads(JsonFormatter().format(record))
    assert out["msg"] == "model usage"
    assert out["input_tokens"] == 12 and out["provider"] == "anthropic"


def test_rate_limiter_prunes_idle_buckets():
    from aiplatform.ratelimit import RateLimiter
    now = [0.0]
    limiter = RateLimiter(60, clock=lambda: now[0], prune_every=1)
    for i in range(100):
        limiter.allow(f"user{i}")
    now[0] = 1000
    limiter.allow("fresh")
    assert len(limiter) == 1


async def test_usage_store_and_quota_reset_on_new_utc_day():
    from aiplatform.usage import InMemoryUsageStore, TokenQuota, UsageEvent
    from tests.fakes import make_message

    now = [0.0]
    store = InMemoryUsageStore(clock=lambda: now[0])
    quota = TokenQuota(store, 15)
    event = UsageEvent.from_message(make_message(), user_id="u", conversation_id=None,
                                    route="chat", provider="anthropic")
    await store.record(event)  # 10 input + 5 output
    assert await quota.exceeded("u")
    now[0] = 86_400
    assert not await quota.exceeded("u") and store._totals == {}
