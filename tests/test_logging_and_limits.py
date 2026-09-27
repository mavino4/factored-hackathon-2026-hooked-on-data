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


def test_daily_quota_resets_on_new_day():
    from aiplatform.ratelimit import DailyTokenQuota
    now = [0.0]
    quota = DailyTokenQuota(100, clock=lambda: now[0])
    quota.charge("u", 100)
    assert quota.exceeded("u")
    now[0] = 86_400
    assert not quota.exceeded("u") and quota._used == {}
