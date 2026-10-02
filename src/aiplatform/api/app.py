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
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Literal

import anthropic
import sqlalchemy as sa
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import start_http_server
from pydantic import BaseModel, Field

from aiplatform import metrics
from aiplatform.accounts import (
    MIN_PASSWORD_LENGTH,
    Accounts,
    InMemoryAccountStore,
    InvalidCredentials,
    WeakPassword,
)
from aiplatform.agent.actions import ActionNotPending, InMemoryActionStore
from aiplatform.agent.handoffs import HandoffNotFound, InMemoryHandoffStore
from aiplatform.agent.loop import (
    AgentDone,
    AgentRunner,
    AgentText,
    ApprovalRequired,
    HandoffOffered,
    HandoffStarted,
    HumanWaiting,
    ToolCall,
    ToolResult,
)
from aiplatform.agent.tools import Tool
from aiplatform.auth import AuthError, OIDCVerifier, Principal
from aiplatform.banking.repository import BankRepository, NotLinked, PostgresBankRepository
from aiplatform.banking.tools import make_bank_tools
from aiplatform.chat.history import ConversationTooLong
from aiplatform.chat.inflight import ConversationBusy, InFlight
from aiplatform.chat.repository import ConversationNotFound, InMemoryConversationRepository
from aiplatform.config import Settings, get_settings
from aiplatform.llm.gateway import AIGateway, GatewayError
from aiplatform.llm.models import prices_for
from aiplatform.llm.providers import build_clients, close_clients
from aiplatform.logging import configure_logging, request_id
from aiplatform.ratelimit import RateLimiter
from aiplatform.storage.sql import (
    SqlAccountStore,
    SqlActionStore,
    SqlConversationRepository,
    SqlHandoffStore,
    SqlUsageStore,
    create_engine,
)
from aiplatform.tracing import Tracing
from aiplatform.usage import InMemoryUsageStore, TokenQuota

log = logging.getLogger(__name__)


class NewConversation(BaseModel):
    # Ignored: chat and agent are one flow now. Accepted so older clients keep working.
    kind: Literal["chat", "agent"] | None = None


Language = Literal["es", "pt", "en"]


class Reply(BaseModel):
    # The language the customer sees in the UI; the assistant replies in it.
    language: Language | None = None


class UserMessage(Reply):
    text: str = Field(min_length=1, max_length=20_000)


class Decision(Reply):
    decision: Literal["approve", "reject"]


class HandoffAnswer(Reply):
    handoff_id: str
    accept: bool


class OperatorMessage(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


def create_app(settings: Settings | None = None, gateway: AIGateway | None = None,
               tools: list[Tool] | None = None,
               verifier: OIDCVerifier | None = None,
               bank_repo: BankRepository | None = None) -> FastAPI:
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
            actions, handoffs = SqlActionStore(engine), SqlHandoffStore(engine)
            account_store = SqlAccountStore(engine)
        else:
            log.warning("AIP_DATABASE_URL not set: using in-memory storage (data is lost on restart)")
            repo, usage = InMemoryConversationRepository(), InMemoryUsageStore()
            actions, handoffs = InMemoryActionStore(), InMemoryHandoffStore()
            account_store = InMemoryAccountStore()
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
        app.state.login_limiter = RateLimiter(settings.login_requests_per_minute)
        app.state.accounts = Accounts(
            account_store, idle=timedelta(minutes=settings.session_idle_minutes),
            max_age=timedelta(hours=settings.session_max_hours),
            max_failures=settings.login_max_failures,
            lock_for=timedelta(minutes=settings.login_lock_minutes))
        tracing = Tracing.from_settings(settings)
        bank = bank_repo
        if bank is None and settings.bank_database_url is not None:
            bank = PostgresBankRepository(settings.bank_database_url.get_secret_value())
        app.state.bank = bank
        agent_tools = tools
        if agent_tools is None:
            agent_tools = make_bank_tools(bank) if bank is not None else []
            if bank is None:
                log.warning("AIP_BANK_DATABASE_URL not set: the agent has no banking tools")
        app.state.agent = AgentRunner(gw, repo, usage, inflight, actions, agent_tools,
                                      handoffs=handoffs, tracing=tracing)
        yield
        tracing.flush()
        if bank is not None and bank_repo is None:
            await bank.close()
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
            path = request.url.path
            if path.startswith("/v1/") or path in ("/", "/index.html", "/config.json"):
                # Per-user data, and the page that displays it: never stored by the browser
                # or a proxy (also keeps the back button from restoring a signed-out page).
                response.headers["Cache-Control"] = "no-store"
            elif path.endswith((".js", ".css")):
                # Revalidate on every load (cheap 304 via ETag), so a new release of the UI
                # is never mixed with a cached old script.
                response.headers["Cache-Control"] = "no-cache"
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")  # template, not raw path: bounded labels
            metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            metrics.HTTP_LATENCY.labels(request.method, path).observe(
                time.perf_counter() - started)
            request_id.reset(token)

    if settings.auth_mode != "oidc":
        verifier = None
    elif verifier is None:
        verifier = OIDCVerifier(settings.oidc_issuer, settings.oidc_audience,
                                jwks_url=settings.oidc_jwks_url)

    async def current_user(request: Request) -> str:
        if settings.auth_mode == "dev":  # trust a header (refused in production by Settings)
            user_id = request.headers.get("x-user-id")
            if not user_id:
                raise HTTPException(401, "missing X-User-Id header (dev auth mode)")
            request.state.principal = Principal(user_id)
            return user_id
        if settings.auth_mode == "password":
            token = request.cookies.get(SESSION_COOKIE)
            principal = await request.app.state.accounts.resolve(token) if token else None
            if principal is None:
                raise HTTPException(401, "not signed in")
            request.state.principal = principal
            return principal.user_id
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

    def password_mode() -> None:
        if settings.auth_mode != "password":
            raise HTTPException(404, "password sign-in is not enabled")

    @app.post("/v1/auth/login", dependencies=[Depends(password_mode)])
    async def login(body: Credentials, request: Request, response: Response) -> dict:
        ip = _client_ip(request)
        if not request.app.state.login_limiter.allow(ip):
            metrics.REJECTED.labels("login_rate_limit").inc()
            raise HTTPException(429, "too many sign-in attempts", headers={"Retry-After": "30"})
        try:
            token, principal = await request.app.state.accounts.login(
                body.username, body.password, ip)
        except InvalidCredentials:
            # One answer for every failure: unknown user, wrong password, locked, disabled.
            raise HTTPException(401, "invalid username or password") from None
        # HttpOnly: page scripts can't read it. SameSite=Strict: other sites can't send it.
        response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict",
                            secure=settings.session_cookie_secure, path="/",
                            max_age=settings.session_max_hours * 3600)
        return {"user_id": principal.user_id}

    @app.post("/v1/auth/logout", dependencies=[Depends(password_mode)], status_code=204)
    async def logout(request: Request, response: Response) -> None:
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            await request.app.state.accounts.logout(token, _client_ip(request))
        response.delete_cookie(SESSION_COOKIE, path="/")

    @app.post("/v1/auth/password", dependencies=[Depends(password_mode)], status_code=204)
    async def change_password(body: PasswordChange, request: Request,
                              user_id: Annotated[str, Depends(current_user)]) -> None:
        """Change the signed-in user's password; their other sessions are signed out."""
        ip = _client_ip(request)
        if not request.app.state.login_limiter.allow(ip):
            metrics.REJECTED.labels("login_rate_limit").inc()
            raise HTTPException(429, "too many attempts", headers={"Retry-After": "30"})
        try:
            await request.app.state.accounts.change_password(
                user_id, body.current_password, body.new_password,
                token=request.cookies.get(SESSION_COOKIE), ip=ip)
        except WeakPassword:
            raise HTTPException(422, f"the new password needs at least {MIN_PASSWORD_LENGTH} "
                                     "characters and can't be the username") from None
        except InvalidCredentials:
            raise HTTPException(403, "the current password is wrong") from None

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
        conv = await request.app.state.repo.create(user_id, "agent")
        return _summary(conv)

    @app.get("/v1/me")
    async def me(request: Request, user_id: Annotated[str, Depends(current_user)]) -> dict:
        """Who is signed in, for the UI greeting. The name comes from the bank DB (RLS)."""
        first_name = None
        bank = request.app.state.bank
        if bank is not None:
            try:
                first_name = (await bank.get_customer(user_id)).first_name
            except NotLinked:
                pass
            except Exception:
                log.warning("could not read the customer's name", exc_info=True)
        return {"user_id": user_id, "first_name": first_name}

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
        pending = [{"id": a.id, "tool_name": a.tool_name, "input": a.input}
                   for a in await request.app.state.actions.list_pending(conv.id, user_id)]
        handoff = await request.app.state.agent.handoffs.latest(conv.id)
        return {**_summary(conv), "messages": conv.messages, "pending_actions": pending,
                "handoff": _handoff(handoff) if handoff else None}

    @app.post("/v1/conversations/{conversation_id}/agent-runs")
    async def run_agent(conversation_id: str, body: UserMessage, request: Request,
                        user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = await get_conversation(request, conversation_id, user_id, "agent")
        return _stream(request.app.state.agent.run(
            user_id, conv.id, body.text, body.language,
            customer_id=request.state.principal.customer_id))

    @app.post("/v1/conversations/{conversation_id}/actions/{action_id}")
    async def decide_action(conversation_id: str, action_id: str, body: Decision,
                            request: Request,
                            user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = await get_conversation(request, conversation_id, user_id, "agent")
        pending = await request.app.state.actions.list_pending(conv.id, user_id)
        if not any(a.id == action_id for a in pending):
            raise HTTPException(404, "no pending action with this id")
        return _stream(request.app.state.agent.decide(
            user_id, conv.id, action_id, approve=body.decision == "approve",
            language=body.language, customer_id=request.state.principal.customer_id))

    @app.post("/v1/conversations/{conversation_id}/handoff")
    async def answer_handoff(conversation_id: str, body: HandoffAnswer, request: Request,
                             user_id: Annotated[str, Depends(current_user)]) -> dict:
        """The customer accepts or declines an offered human advisor."""
        conv = await get_conversation(request, conversation_id, user_id)
        try:
            handoff = await request.app.state.agent.answer_offer(
                user_id, conv.id, body.handoff_id, body.accept, body.language)
        except HandoffNotFound:
            raise HTTPException(404, "no offered handoff with this id") from None
        except ConversationBusy:
            raise HTTPException(409, "a reply is already in progress") from None
        return _handoff(handoff)

    # Operators (users in AIP_ADMIN_USERS) attend handed-off conversations. There is no
    # console yet: these endpoints are the operator API.
    def operator(user_id: str) -> str:
        if user_id not in settings.admin_users:
            raise HTTPException(403, "admin only")
        return user_id

    @app.get("/v1/admin/handoffs")
    async def list_handoffs(
            request: Request, user_id: Annotated[str, Depends(current_user)],
            status: Literal["offered", "open", "declined", "closed"] | None = "open",
            limit: int = Query(100, ge=1, le=500)) -> dict:
        operator(user_id)
        items = await request.app.state.agent.handoffs.list(status, limit)
        return {"handoffs": [{**_handoff(h), "conversation_id": h.conversation_id,
                              "user_id": h.user_id, "summary": h.summary,
                              "created_at": h.created_at.isoformat()} for h in items]}

    @app.get("/v1/admin/handoffs/{handoff_id}")
    async def read_handoff(handoff_id: str, request: Request,
                           user_id: Annotated[str, Depends(current_user)]) -> dict:
        operator(user_id)
        try:
            h = await request.app.state.agent.handoffs.get(handoff_id)
        except HandoffNotFound:
            raise HTTPException(404, "handoff not found") from None
        conv = await request.app.state.repo.get(h.conversation_id, h.user_id)
        return {**_handoff(h), "user_id": h.user_id, "summary": h.summary,
                "conversation": {**_summary(conv), "messages": conv.messages}}

    @app.post("/v1/admin/handoffs/{handoff_id}/messages", status_code=201)
    async def operator_message(handoff_id: str, body: OperatorMessage, request: Request,
                               user_id: Annotated[str, Depends(current_user)]) -> dict:
        operator(user_id)
        try:
            await request.app.state.agent.operator_reply(handoff_id, user_id, body.text)
        except HandoffNotFound:
            raise HTTPException(404, "no open handoff with this id") from None
        except ConversationBusy:
            raise HTTPException(409, "the customer is writing; retry") from None
        return {"status": "sent"}

    @app.post("/v1/admin/handoffs/{handoff_id}/close")
    async def close_handoff(handoff_id: str, request: Request,
                            user_id: Annotated[str, Depends(current_user)]) -> dict:
        """Close the case; the bot answers the conversation again."""
        operator(user_id)
        try:
            h = await request.app.state.agent.close_handoff(handoff_id)
        except HandoffNotFound:
            raise HTTPException(404, "no open handoff with this id") from None
        return _handoff(h)

    # The web UI: static files at "/". Mounted last so API routes take precedence.
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
SESSION_COOKIE = "aip_session"
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


def _client_ip(request: Request) -> str:
    # X-Real-IP is set by the load balancer (deploy/nginx*.conf); used for the sign-in rate
    # limit and the audit trail, never for access decisions.
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "")


def _handoff(h) -> dict:
    return {"id": h.id, "status": h.status, "reason": h.reason}


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
    """Map agent events to SSE frames."""
    match event:
        case AgentText(text=text):
            return [_event("delta", {"text": text})]
        # Customers see progress ("consulting...") but not the tools' arguments or raw
        # results; the model's answer is what they read.
        case ToolCall(id=id, name=name):
            return [_event("tool_call", {"id": id, "name": name})]
        case ToolResult(id=id, name=name, is_error=is_error):
            return [_event("tool_result", {"id": id, "name": name, "is_error": is_error})]
        case ApprovalRequired(action_id=action_id, tool_name=tool_name, input=args):
            return [_event("approval_required", {"action_id": action_id,
                                                 "tool_name": tool_name, "input": args})]
        case HandoffOffered(handoff_id=handoff_id):
            return [_event("handoff_offer", {"handoff_id": handoff_id})]
        case HandoffStarted(handoff_id=handoff_id):
            return [_event("handoff", {"handoff_id": handoff_id})]
        case HumanWaiting(handoff_id=handoff_id):
            return [_event("human_waiting", {"handoff_id": handoff_id})]
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

