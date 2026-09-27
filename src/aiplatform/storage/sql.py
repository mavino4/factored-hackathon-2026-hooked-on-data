"""SQL implementations of the conversation repository and usage store (Postgres in
production; SQLite works too, which the tests use)."""

import time
import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aiplatform.chat.inflight import ConversationBusy
from aiplatform.chat.repository import Conversation, ConversationNotFound, Kind, title_from
from aiplatform.storage.tables import conversations, messages, usage_events
from aiplatform.usage import UsageEvent, utc_day_start


def create_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        return create_async_engine(url)
    return create_async_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True)


def _as_utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes; everything we store is UTC.
    return value if value.tzinfo else value.replace(tzinfo=UTC)


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
                sa.select(messages.c.role, messages.c.content)
                .where(messages.c.conversation_id == conversation_id)
                .order_by(messages.c.seq))
            conv.messages = [{"role": r.role, "content": r.content} for r in result]
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
                    content=message["content"], created_at=now))
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

    async def tokens_used_today(self, user_id: str) -> int:
        since = datetime.fromtimestamp(utc_day_start(self._clock()), UTC)
        total = (usage_events.c.input_tokens + usage_events.c.output_tokens
                 + usage_events.c.cache_read_tokens + usage_events.c.cache_write_tokens)
        async with self._engine.connect() as db:
            value = await db.scalar(sa.select(sa.func.coalesce(sa.func.sum(total), 0)).where(
                usage_events.c.user_id == user_id, usage_events.c.created_at >= since))
        return int(value)
