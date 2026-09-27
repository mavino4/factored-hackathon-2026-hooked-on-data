"""HTTP API: conversations, streamed chat turns (SSE) and agent runs."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aiplatform.agent.loop import AgentRunner
from aiplatform.agent.tools import DEFAULT_TOOLS, Tool
from aiplatform.chat.inflight import ConversationBusy, InFlight
from aiplatform.chat.repository import ConversationNotFound, InMemoryConversationRepository
from aiplatform.chat.service import ChatService, ConversationTooLong, NothingToRegenerate
from aiplatform.config import Settings, get_settings
from aiplatform.llm.gateway import AIGateway, Completed, GatewayError, TextDelta
from aiplatform.llm.providers import build_clients, close_clients
from aiplatform.logging import configure_logging
from aiplatform.ratelimit import DailyTokenQuota, RateLimiter

log = logging.getLogger(__name__)


class NewConversation(BaseModel):
    kind: Literal["chat", "agent"] = "chat"


class UserMessage(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


def create_app(settings: Settings | None = None, gateway: AIGateway | None = None,
               tools: list[Tool] | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        clients = None
        gw = gateway
        if gw is None:
            clients = build_clients(settings)
            gw = AIGateway(clients, settings)
        repo = InMemoryConversationRepository()
        quota = DailyTokenQuota(settings.user_tokens_per_day)
        inflight = InFlight()
        app.state.repo = repo
        app.state.inflight = inflight
        app.state.quota = quota
        app.state.limiter = RateLimiter(settings.user_requests_per_minute)
        app.state.chat = ChatService(gw, repo, quota, inflight)
        app.state.agent = AgentRunner(gw, repo, quota, inflight,
                                        tools if tools is not None else DEFAULT_TOOLS)
        yield
        if clients:
            await close_clients(clients)

    app = FastAPI(title="AI Platform", version="0.1.0", lifespan=lifespan)

    def current_user(x_user_id: Annotated[str, Header()]) -> str:
        # PLACEHOLDER auth: trusts a header. Replace with OIDC JWT verification
        # before exposing this service outside a trusted network.
        return x_user_id

    def admit(request: Request, user_id: Annotated[str, Depends(current_user)]) -> str:
        state = request.app.state
        if not state.limiter.allow(user_id):
            raise HTTPException(429, "too many requests", headers={"Retry-After": "5"})
        if state.quota.exceeded(user_id):
            raise HTTPException(429, "daily token quota exceeded")
        return user_id

    def get_conversation(request: Request, conversation_id: str, user_id: str, kind: str):
        try:
            conv = request.app.state.repo.get(conversation_id, user_id)
        except ConversationNotFound:
            raise HTTPException(404, "conversation not found") from None
        if conv.kind != kind:
            raise HTTPException(409, f"this is a {conv.kind} conversation")
        if request.app.state.inflight.is_busy(conv.id):
            raise HTTPException(409, "a reply is already in progress")
        return conv

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/conversations", status_code=201)
    async def create_conversation(body: NewConversation, request: Request,
                                  user_id: Annotated[str, Depends(current_user)]) -> dict:
        conv = request.app.state.repo.create(user_id, body.kind)
        return {"id": conv.id, "kind": conv.kind, "created_at": conv.created_at.isoformat()}

    @app.get("/v1/conversations/{conversation_id}")
    async def read_conversation(conversation_id: str, request: Request,
                                user_id: Annotated[str, Depends(current_user)]) -> dict:
        try:
            conv = request.app.state.repo.get(conversation_id, user_id)
        except ConversationNotFound:
            raise HTTPException(404, "conversation not found") from None
        return {"id": conv.id, "kind": conv.kind, "messages": conv.messages}

    @app.post("/v1/conversations/{conversation_id}/messages")
    async def send_message(conversation_id: str, body: UserMessage, request: Request,
                           user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = get_conversation(request, conversation_id, user_id, "chat")
        return _stream(request.app.state.chat.send(user_id, conv.id, body.text))

    @app.post("/v1/conversations/{conversation_id}/regenerate")
    async def regenerate(conversation_id: str, request: Request,
                         user_id: Annotated[str, Depends(admit)]) -> StreamingResponse:
        conv = get_conversation(request, conversation_id, user_id, "chat")
        if not conv.messages or conv.messages[-1]["role"] != "user":
            raise HTTPException(409, "the last message already has a reply")
        return _stream(request.app.state.chat.regenerate(user_id, conv.id))

    @app.post("/v1/conversations/{conversation_id}/agent-runs")
    async def run_agent(conversation_id: str, body: UserMessage, request: Request,
                        user_id: Annotated[str, Depends(admit)]) -> dict:
        conv = get_conversation(request, conversation_id, user_id, "agent")
        try:
            result = await request.app.state.agent.run(user_id, conv.id, body.text)
        except ConversationBusy:
            raise HTTPException(409, "a reply is already in progress") from None
        except ConversationTooLong:
            raise HTTPException(422, "conversation is too long; start a new one") from None
        except anthropic.BadRequestError:
            log.exception("agent run rejected by the model API")
            raise HTTPException(422, "the request could not be processed") from None
        except (GatewayError, anthropic.APIError):
            log.exception("agent run failed")
            raise HTTPException(503, "the AI service is busy, please retry") from None
        return {"outcome": result.outcome, "text": result.text, "tool_calls": result.tool_calls}

    return app


def _stream(events: AsyncIterator[TextDelta | Completed]) -> StreamingResponse:
    return StreamingResponse(_sse(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


async def _sse(events: AsyncIterator[TextDelta | Completed]) -> AsyncIterator[str]:
    try:
        async for event in events:
            if isinstance(event, TextDelta):
                yield _event("delta", {"text": event.text})
            else:
                message = event.message
                if message.stop_reason == "refusal":
                    yield _event("refusal", {"message": "The assistant can't help with that request."})
                yield _event("done", {
                    "stop_reason": message.stop_reason,
                    "usage": {"input_tokens": message.usage.input_tokens,
                              "output_tokens": message.usage.output_tokens},
                })
    except ConversationTooLong:
        yield _event("error", {"code": "conversation_too_long",
                               "message": "Start a new conversation."})
    except ConversationBusy:
        yield _event("error", {"code": "busy", "message": "A reply is already in progress."})
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


app = create_app()
