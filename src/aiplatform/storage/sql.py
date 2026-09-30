"""SQL implementations of the conversation repository and usage store (Postgres in
production; SQLite works too, which the tests use)."""

import time
import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aiplatform.agent.actions import ActionNotPending, PendingAction
from aiplatform.agent.handoffs import Handoff, HandoffNotFound, Reason, Status
from aiplatform.chat.inflight import ConversationBusy
from aiplatform.chat.repository import Conversation, ConversationNotFound, Kind, title_from
from aiplatform.storage.tables import (
    conversations,
    handoffs,
    messages,
    pending_actions,
    usage_events,
)
from aiplatform.usage import UsageEvent, utc_day_start


def create_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        return create_async_engine(url)
    return create_async_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True)


def _as_utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes; everything we store is UTC.
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _row_to_message(row) -> dict:
    message = {"role": row.role, "content": row.content}
    if row.author:
        message["author"] = row.author
    return message


def _row_to_conversation(row) -> Conversation:
    return Conversation(id=str(row.id), user_id=row.user_id, kind=row.kind, title=row.title,
                        created_at=_as_utc(row.created_at), updated_at=_as_utc(row.updated_at))


class SqlConversationRepository:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def create(self, user_id: str, kind: Kind) -> Conversation:
        now = datetime.now(UTC)
        conv = Conversation(id=str(uuid.uuid4()), user_id=user_id, kind=kind,
                            created_at=now, updated_at=now)
        async with self._engine.begin() as db:
            await db.execute(conversations.insert().values(
                id=conv.id, user_id=user_id, kind=kind, created_at=now, updated_at=now))
        return conv

    async def get(self, conversation_id: str, user_id: str) -> Conversation:
        try:
            uuid.UUID(conversation_id)
        except ValueError:
            raise ConversationNotFound(conversation_id) from None
        async with self._engine.connect() as db:
            row = (await db.execute(sa.select(conversations).where(
                conversations.c.id == conversation_id,
                conversations.c.user_id == user_id))).first()
            if row is None:
                raise ConversationNotFound(conversation_id)
            conv = _row_to_conversation(row)
            result = await db.execute(
                sa.select(messages.c.role, messages.c.content, messages.c.author)
                .where(messages.c.conversation_id == conversation_id)
                .order_by(messages.c.seq))
            conv.messages = [_row_to_message(r) for r in result]
        return conv

    async def append(self, conv: Conversation, message: dict) -> None:
        now = datetime.now(UTC)
        values: dict = {"updated_at": now}
        if conv.title is None and (title := title_from(message)):
            values["title"] = title
        try:
            async with self._engine.begin() as db:
                await db.execute(messages.insert().values(
                    conversation_id=conv.id, seq=len(conv.messages), role=message["role"],
                    content=message["content"], author=message.get("author"),
                    created_at=now))
                await db.execute(conversations.update()
                                 .where(conversations.c.id == conv.id).values(**values))
        except IntegrityError as exc:
            raise ConversationBusy(conv.id) from exc
        conv.messages.append(message)
        conv.updated_at = now
        conv.title = values.get("title", conv.title)

    async def list(self, user_id: str, limit: int = 50) -> list[Conversation]:
        async with self._engine.connect() as db:
            result = await db.execute(
                sa.select(conversations).where(conversations.c.user_id == user_id)
                .order_by(conversations.c.updated_at.desc()).limit(limit))
            return [_row_to_conversation(r) for r in result]


class SqlUsageStore:
    def __init__(self, engine: AsyncEngine, clock=time.time):
        self._engine = engine
        self._clock = clock

    async def record(self, event: UsageEvent) -> None:
        async with self._engine.begin() as db:
            await db.execute(usage_events.insert().values(
                user_id=event.user_id, conversation_id=event.conversation_id,
                route=event.route, provider=event.provider, model=event.model,
                input_tokens=event.input_tokens, output_tokens=event.output_tokens,
                cache_read_tokens=event.cache_read_tokens,
                cache_write_tokens=event.cache_write_tokens,
                created_at=datetime.fromtimestamp(self._clock(), UTC)))

    async def daily_summary(self, days: int) -> list[dict]:
        since = datetime.fromtimestamp(utc_day_start(self._clock()) - (days - 1) * 86_400, UTC)
        u = usage_events.c
        day = sa.func.date(u.created_at).label("day")
        async with self._engine.connect() as db:
            result = await db.execute(
                sa.select(day, u.route, u.provider, u.model,
                          sa.func.count().label("requests"),
                          sa.func.sum(u.input_tokens).label("input_tokens"),
                          sa.func.sum(u.output_tokens).label("output_tokens"),
                          sa.func.sum(u.cache_read_tokens).label("cache_read_tokens"),
                          sa.func.sum(u.cache_write_tokens).label("cache_write_tokens"))
                .where(u.created_at >= since)
                .group_by(day, u.route, u.provider, u.model)
                .order_by(day.desc(), u.route, u.provider, u.model))
            return [{**row._mapping, "day": str(row.day)} for row in result]

    async def tokens_used_today(self, user_id: str) -> int:
        since = datetime.fromtimestamp(utc_day_start(self._clock()), UTC)
        total = (usage_events.c.input_tokens + usage_events.c.output_tokens
                 + usage_events.c.cache_read_tokens + usage_events.c.cache_write_tokens)
        async with self._engine.connect() as db:
            value = await db.scalar(sa.select(sa.func.coalesce(sa.func.sum(total), 0)).where(
                usage_events.c.user_id == user_id, usage_events.c.created_at >= since))
        return int(value)


def _row_to_action(row) -> PendingAction:
    return PendingAction(
        id=str(row.id), conversation_id=str(row.conversation_id), user_id=row.user_id,
        tool_use_id=row.tool_use_id, tool_name=row.tool_name, input=row.input,
        status=row.status, created_at=_as_utc(row.created_at),
        decided_at=_as_utc(row.decided_at) if row.decided_at else None)


class SqlActionStore:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def create(self, *, conversation_id: str, user_id: str, tool_use_id: str,
                     tool_name: str, input: dict) -> PendingAction:
        action = PendingAction(id=str(uuid.uuid4()), conversation_id=conversation_id,
                               user_id=user_id, tool_use_id=tool_use_id, tool_name=tool_name,
                               input=input, status="pending", created_at=datetime.now(UTC))
        async with self._engine.begin() as db:
            await db.execute(pending_actions.insert().values(
                id=action.id, conversation_id=conversation_id, user_id=user_id,
                tool_use_id=tool_use_id, tool_name=tool_name, input=input,
                status="pending", created_at=action.created_at))
        return action

    async def list_pending(self, conversation_id: str, user_id: str) -> list[PendingAction]:
        async with self._engine.connect() as db:
            result = await db.execute(
                sa.select(pending_actions).where(
                    pending_actions.c.conversation_id == conversation_id,
                    pending_actions.c.user_id == user_id,
                    pending_actions.c.status == "pending")
                .order_by(pending_actions.c.created_at))
            return [_row_to_action(r) for r in result]

    async def decide(self, action_id: str, *, user_id: str, conversation_id: str,
                     approve: bool) -> PendingAction:
        try:
            uuid.UUID(action_id)
        except ValueError:
            raise ActionNotPending(action_id) from None
        now = datetime.now(UTC)
        async with self._engine.begin() as db:
            result = await db.execute(
                pending_actions.update()
                .where(pending_actions.c.id == action_id,
                       pending_actions.c.user_id == user_id,
                       pending_actions.c.conversation_id == conversation_id,
                       pending_actions.c.status == "pending")
                .values(status="approved" if approve else "rejected", decided_at=now))
            if result.rowcount != 1:
                raise ActionNotPending(action_id)
            row = (await db.execute(
                sa.select(pending_actions).where(pending_actions.c.id == action_id))).one()
        return _row_to_action(row)


def _row_to_handoff(row) -> Handoff:
    return Handoff(id=str(row.id), conversation_id=str(row.conversation_id),
                   user_id=row.user_id, status=row.status, reason=row.reason,
                   summary=row.summary, message_index=row.message_index,
                   created_at=_as_utc(row.created_at),
                   closed_at=_as_utc(row.closed_at) if row.closed_at else None)


class SqlHandoffStore:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def create(self, *, conversation_id: str, user_id: str, status: Status,
                     reason: Reason, summary: str, message_index: int) -> Handoff:
        handoff = Handoff(id=str(uuid.uuid4()), conversation_id=conversation_id,
                          user_id=user_id, status=status, reason=reason, summary=summary,
                          message_index=message_index, created_at=datetime.now(UTC))
        async with self._engine.begin() as db:
            await db.execute(handoffs.insert().values(
                id=handoff.id, conversation_id=conversation_id, user_id=user_id,
                status=status, reason=reason, summary=summary, message_index=message_index,
                created_at=handoff.created_at))
        return handoff

    async def latest(self, conversation_id: str) -> Handoff | None:
        async with self._engine.connect() as db:
            row = (await db.execute(
                sa.select(handoffs).where(handoffs.c.conversation_id == conversation_id)
                .order_by(handoffs.c.created_at.desc()).limit(1))).first()
        return _row_to_handoff(row) if row else None

    async def get(self, handoff_id: str) -> Handoff:
        try:
            uuid.UUID(handoff_id)
        except ValueError:
            raise HandoffNotFound(handoff_id) from None
        async with self._engine.connect() as db:
            row = (await db.execute(
                sa.select(handoffs).where(handoffs.c.id == handoff_id))).first()
        if row is None:
            raise HandoffNotFound(handoff_id)
        return _row_to_handoff(row)

    async def list(self, status: Status | None = None, limit: int = 100) -> list[Handoff]:
        query = sa.select(handoffs).order_by(handoffs.c.created_at).limit(limit)
        if status is not None:
            query = query.where(handoffs.c.status == status)
        async with self._engine.connect() as db:
            return [_row_to_handoff(r) for r in await db.execute(query)]

    async def transition(self, handoff_id: str, *, expected: Status, to: Status,
                         conversation_id: str | None = None,
                         user_id: str | None = None) -> Handoff:
        try:
            uuid.UUID(handoff_id)
        except ValueError:
            raise HandoffNotFound(handoff_id) from None
        where = [handoffs.c.id == handoff_id, handoffs.c.status == expected]
        if conversation_id is not None:
            where.append(handoffs.c.conversation_id == conversation_id)
        if user_id is not None:
            where.append(handoffs.c.user_id == user_id)
        values: dict = {"status": to}
        if to in ("closed", "declined"):
            values["closed_at"] = datetime.now(UTC)
        async with self._engine.begin() as db:
            result = await db.execute(handoffs.update().where(*where).values(**values))
            if result.rowcount != 1:
                raise HandoffNotFound(handoff_id)
            row = (await db.execute(
                sa.select(handoffs).where(handoffs.c.id == handoff_id))).one()
        return _row_to_handoff(row)
