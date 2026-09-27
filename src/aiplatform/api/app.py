"""HTTP API: conversations, streamed chat turns (SSE) and agent runs."""

import asyncio
import json
import logging
import re
import time
import urllib.parse
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import anthropic
import sqlalchemy as sa
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import start_http_server
from pydantic import BaseModel, Field

from aiplatform import metrics
from aiplatform.agent.actions import ActionNotPending, InMemoryActionStore
from aiplatform.agent.loop import (
    AgentDone,
    AgentRunner,
    AgentText,
    ApprovalRequired,
    ToolCall,
    ToolResult,
)
from aiplatform.agent.tools import DEFAULT_TOOLS, Tool
from aiplatform.auth import AuthError, OIDCVerifier
from aiplatform.chat.inflight import ConversationBusy, InFlight
from aiplatform.chat.repository import ConversationNotFound, InMemoryConversationRepository
from aiplatform.chat.service import ChatService, ConversationTooLong, NothingToRegenerate
from aiplatform.config import Settings, get_settings
from aiplatform.llm.gateway import AIGateway, Completed, GatewayError, TextDelta
from aiplatform.llm.models import prices_for
from aiplatform.llm.providers import build_clients, close_clients
from aiplatform.logging import configure_logging, request_id
from aiplatform.ratelimit import RateLimiter
from aiplatform.storage.sql import (
    SqlActionStore,
    SqlConversationRepository,
    SqlUsageStore,
    create_engine,
)
from aiplatform.usage import InMemoryUsageStore, TokenQuota

log = logging.getLogger(__name__)


class NewConversation(BaseModel):
    kind: Literal["chat", "agent"] = "chat"


class UserMessage(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class Decision(BaseModel):
    decision: Literal["approve", "reject"]


def create_app(settings: Settings | None = None, gateway: AIGateway | None = None,
               tools: list[Tool] | None = None,
               verifier: OIDCVerifier | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        clients = None
        gw = gateway
        if gw is None:
            clients = build_clients(settings)
            gw = AIGateway(clients, settings)
        engine = None
        if settings.database_url:
            engine = create_engine(settings.database_url)
            repo, usage = SqlConversationRepository(engine), SqlUsageStore(engine)
            actions = SqlActionStore(engine)
        else:
            log.warning("AIP_DATABASE_URL not set: using in-memory storage (data is lost on restart)")
            repo, usage = InMemoryConversationRepository(), InMemoryUsageStore()
            actions = InMemoryActionStore()
        inflight = InFlight()
        metrics_server = None
        if settings.metrics_port:
            metrics_server, _ = start_http_server(settings.metrics_port)
        app.state.engine = engine
        app.state.usage = usage
        app.state.repo = repo
        app.state.actions = actions
        app.state.inflight = inflight
        app.state.quota = TokenQuota(usage, settings.user_tokens_per_day)
        app.state.limiter = RateLimiter(settings.user_requests_per_minute)
        app.state.chat = ChatService(gw, repo, usage, inflight)
        app.state.agent = AgentRunner(gw, repo, usage, inflight, actions,
                                      tools if tools is not None else DEFAULT_TOOLS)
        yield
        if metrics_server is not None:
            metrics_server.shutdown()
        if clients:
            await close_clients(clients)
        if engine is not None:
            await engine.dispose()

    app = FastAPI(title="AI Platform", version="0.1.0", lifespan=lifespan)
    security_headers = _security_headers(settings)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id.set(rid)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            response.headers.update(security_headers)
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")  # template, not raw path: bounded labels
            metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            metrics.HTTP_LATENCY.labels(request.method, path).observe(
                time.perf_counter() - started)
            request_id.reset(token)

    if settings.auth_mode == "dev":
        verifier = None
    elif verifier is None:
        verifier = OIDCVerifier(settings.oidc_issuer, settings.oidc_audience,
                                jwks_url=settings.oidc_jwks_url)

    async def current_user(request: Request) -> str:
        if verifier is None:  # dev mode: trust a header (refused in production by Settings)
            user_id = request.headers.get("x-user-id")
            if not user_id:
                raise HTTPException(401, "missing X-User-Id header (dev auth mode)")
            return user_id
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "missing bearer token",
                                headers={"WWW-Authenticate": "Bearer"})
        try:
            principal = await verifier.verify(token)
        except AuthError as exc:
            raise HTTPException(401, str(exc), headers={
                "WWW-Authenticate": 'Bearer error="invalid_token"'}) from None
        request.state.principal = principal
        return principal.user_id

    async def admit(request: Request, user_id: Annotated[str, Depends(current_user)]) -> str:
        state = request.app.state
        if not state.limiter.allow(user_id):
            metrics.REJECTED.labels("rate_limit").inc()
            raise HTTPException(429, "too many requests", headers={"Retry-After": "5"})
        if await state.quota.exceeded(user_id):
            metrics.REJECTED.labels("token_quota").inc()
            raise HTTPException(429, "daily token quota exceeded")
        return user_id

    async def get_conversation(request: Request, conversation_id: str, user_id: str,
                               kind: str | None = None):
        try:
            conv = await request.app.state.repo.get(conversation_id, user_id)
        except ConversationNotFound:
            raise HTTPException(404, "conversation not found") from None
        if kind is None:
            return conv
        if conv.kind != kind:
            raise HTTPException(409, f"this is a {conv.kind} conversation")
        if request.app.state.inflight.is_busy(conv.id):
            metrics.REJECTED.labels("busy").inc()
            raise HTTPException(409, "a reply is already in progress")
        return conv

    @app.get("/healthz")
    async def healthz() -> dict:
        """Liveness: the process is up. No dependency checks."""
        return {"status": "ok"}

    @app.get("/config.json")
    async def ui_config() -> dict:
        """Public settings the web UI needs to sign users in (nothing secret)."""
        return {"auth_mode": settings.auth_mode, "oidc_issuer": settings.oidc_issuer,
                "oidc_client_id": settings.oidc_client_id,
                "oidc_audience": settings.oidc_audience}

    @app.get("/readyz")
    async def readyz(request: Request):
        """Readiness: can serve traffic (database reachable when configured)."""
        engine = request.app.state.engine
        if engine is not None:
            try:
                async with asyncio.timeout(2):
                    async with engine.connect() as db:
                        await db.execute(sa.text("SELECT 1"))
            except Exception:
                log.warning("readiness check failed: database unreachable", exc_info=True)
                return JSONResponse({"status": "unavailable", "database": "down"}, 503)
        return {"status": "ok"}

    @app.get("/v1/admin/usage")
    async def admin_usage(request: Request, user_id: Annotated[str, Depends(current_user)],
                          days: int = Query(7, ge=1, le=90)) -> dict:
        if user_id not in settings.admin_users:
            raise HTTPException(403, "admin only")
        rows = await request.app.state.usage.daily_summary(days)
        if rows is None:
            raise HTTPException(501, "usage history needs a database (AIP_DATABASE_URL)")
        for row in rows:
            row["cost_usd"] = round(prices_for(row["model"]).cost(
                input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
                cache_read_tokens=row["cache_read_tokens"],
                cache_write_tokens=row["cache_write_tokens"]), 6)
        return {"days": days, "rows": rows,
                "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
                "note": "Estimated from list prices; local models count as $0."}

    @app.post("/v1/conversations", status_code=201)
    async def create_conversation(body: NewConversation, request: Request,
                                  user_id: Annotated[str, Depends(current_user)]) -> dict:
        conv = await request.app.state.repo.create(user_id, body.kind)
        return _summary(conv)

    @app.get("/v1/conversations")
    async def list_conversations(request: Request,
                                 user_id: Annotated[str, Depends(current_user)],
                                 limit: int = Query(50, ge=1, le=200)) -> dict:
        convs = await request.app.state.repo.list(user_id, limit)
        return {"conversations": [_summary(c) for c in convs]}

    @app.get("/v1/conversations/{conversation_id}")
    async def read_conversation(conversation_id: str, request: Request,
                                user_id: Annotated[str, Depends(current_user)]) -> dict:
        conv = await get_conversation(request, conversation_id, user_id)
        pending = []
        if conv.kind == "agent":
            pending = [{"id": a.id, "tool_name": a.tool_name, "input": a.input}
                       for a in await request.app.state.actions.list_pending(conv.id, user_id)]
        return {**_summary(conv), "messages": conv.messages, "pending_actions": pending}

    @app.post("/v1/conversations/{conversation_id}/messages")
    async def send_message(conversation_id: str, body: UserMessage, request: Request,
                           user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = await get_conversation(request, conversation_id, user_id, "chat")
        return _stream(request.app.state.chat.send(user_id, conv.id, body.text))

    @app.post("/v1/conversations/{conversation_id}/regenerate")
    async def regenerate(conversation_id: str, request: Request,
                         user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = await get_conversation(request, conversation_id, user_id, "chat")
        if not conv.messages or conv.messages[-1]["role"] != "user":
            raise HTTPException(409, "the last message already has a reply")
        return _stream(request.app.state.chat.regenerate(user_id, conv.id))

    @app.post("/v1/conversations/{conversation_id}/agent-runs")
    async def run_agent(conversation_id: str, body: UserMessage, request: Request,
                        user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = await get_conversation(request, conversation_id, user_id, "agent")
        return _stream(request.app.state.agent.run(user_id, conv.id, body.text))

    @app.post("/v1/conversations/{conversation_id}/actions/{action_id}")
    async def decide_action(conversation_id: str, action_id: str, body: Decision,
                            request: Request,
                            user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = await get_conversation(request, conversation_id, user_id, "agent")
        pending = await request.app.state.actions.list_pending(conv.id, user_id)
        if not any(a.id == action_id for a in pending):
            raise HTTPException(404, "no pending action with this id")
        return _stream(request.app.state.agent.decide(
            user_id, conv.id, action_id, approve=body.decision == "approve"))

    # The web UI: static files at "/". Mounted last so API routes take precedence.
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _security_headers(settings: Settings) -> dict[str, str]:
    # The UI talks to this origin and, for OIDC login, to the issuer's token endpoint.
    connect = ["'self'"]
    if settings.auth_mode == "oidc" and settings.oidc_issuer:
        parsed = urllib.parse.urlparse(settings.oidc_issuer)
        connect.append(f"{parsed.scheme}://{parsed.netloc}")
    return {
        "Content-Security-Policy": (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            f"connect-src {' '.join(connect)}; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'self'"),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }


def _summary(conv) -> dict:
    return {"id": conv.id, "kind": conv.kind, "title": conv.title,
            "created_at": conv.created_at.isoformat(), "updated_at": conv.updated_at.isoformat()}


def _stream(events: AsyncIterator) -> StreamingResponse:
    return StreamingResponse(_sse(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


REFUSAL = {"message": "The assistant can't help with that request."}


def _encode(event) -> list[str]:
    """Map chat/agent events to SSE frames."""
    match event:
        case TextDelta(text=text) | AgentText(text=text):
            return [_event("delta", {"text": text})]
        case Completed(message=message):
            frames = [_event("refusal", REFUSAL)] if message.stop_reason == "refusal" else []
            return frames + [_event("done", {
                "stop_reason": message.stop_reason,
                "usage": {"input_tokens": message.usage.input_tokens,
                          "output_tokens": message.usage.output_tokens}})]
        case ToolCall(id=id, name=name, input=args):
            return [_event("tool_call", {"id": id, "name": name, "input": args})]
        case ToolResult(id=id, name=name, is_error=is_error, content=content):
            return [_event("tool_result", {"id": id, "name": name, "is_error": is_error,
                                           "content": content[:2_000]})]
        case ApprovalRequired(action_id=action_id, tool_name=tool_name, input=args):
            return [_event("approval_required", {"action_id": action_id,
                                                 "tool_name": tool_name, "input": args})]
        case AgentDone(outcome=outcome, tool_calls=tool_calls):
            frames = [_event("refusal", REFUSAL)] if outcome == "refused" else []
            return frames + [_event("done", {"outcome": outcome, "tool_calls": tool_calls})]
    raise TypeError(f"unknown event: {event!r}")


async def _sse(events: AsyncIterator) -> AsyncIterator[str]:
    try:
        async for event in events:
            for frame in _encode(event):
                yield frame
    except ConversationTooLong:
        yield _event("error", {"code": "conversation_too_long",
                               "message": "Start a new conversation."})
    except ConversationBusy:
        yield _event("error", {"code": "busy", "message": "A reply is already in progress."})
    except ActionNotPending:
        yield _event("error", {"code": "action_not_pending",
                               "message": "This action was already decided."})
    except NothingToRegenerate:
        yield _event("error", {"code": "nothing_to_regenerate",
                               "message": "The last message already has a reply."})
    except anthropic.BadRequestError:
        log.exception("chat request rejected by the model API")
        yield _event("error", {"code": "invalid_request",
                               "message": "The request could not be processed."})
    except (GatewayError, anthropic.APIError):
        log.exception("chat stream failed")
        yield _event("error", {"code": "unavailable",
                               "message": "The AI service is busy, please retry."})
    except Exception:
        log.exception("unexpected error in chat stream")
        yield _event("error", {"code": "internal", "message": "Something went wrong."})

