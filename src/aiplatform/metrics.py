"""Prometheus metrics. Served on a separate internal port (AIP_METRICS_PORT), never
through the public API, because they include usage and cost figures."""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "aip_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "aip_http_response_start_seconds",
    "Time until response headers (for SSE: until the stream starts)", ["method", "route"])
REJECTED = Counter(
    "aip_requests_rejected_total", "Requests rejected before reaching the model", ["reason"])

# Customer messages by classified intent; "attack" counts manipulation attempts.
AGENT_INTENTS = Counter(
    "aip_agent_intents_total", "Customer messages by classified intent", ["intent"])

LLM_TTFT = Histogram(
    "aip_llm_time_to_first_token_seconds", "Model call start to first text token",
    ["route", "provider"], buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60))
LLM_DURATION = Histogram(
    "aip_llm_call_duration_seconds", "Full model call duration", ["route", "provider"],
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300))
LLM_TOKENS = Counter(
    "aip_llm_tokens_total", "Tokens by kind (input, output, cache_read, cache_write)",
    ["route", "provider", "model", "kind"])
LLM_COST = Counter(
    "aip_llm_cost_usd_total", "Estimated model cost in USD (list prices)",
    ["route", "provider", "model"])
LLM_ERRORS = Counter(
    "aip_llm_errors_total", "Failed model call attempts", ["provider", "kind"])
LLM_BREAKER_OPEN = Gauge(
    "aip_llm_breaker_open", "1 while a provider's circuit breaker is open", ["provider"])
LLM_BELOW_CACHE_MIN = Counter(
    "aip_llm_prompt_below_cache_minimum_total",
    "Calls whose prompt is too short for the model to cache", ["route"])
