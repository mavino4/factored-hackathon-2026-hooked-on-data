"""Contract tests: every repository/usage-store implementation must pass these.

Runs against in-memory and SQLite by default. Set AIP_TEST_DATABASE_URL
(postgresql+asyncpg://...) and run `make test-postgres` for Postgres.
"""

import os

import pytest
from alembic import command
from alembic.config import Config

from aiplatform.agent.actions import ActionNotPending, InMemoryActionStore
from aiplatform.agent.handoffs import HandoffNotFound, InMemoryHandoffStore
from aiplatform.chat.inflight import ConversationBusy
from aiplatform.chat.repository import ConversationNotFound, InMemoryConversationRepository
from aiplatform.storage.sql import (
    SqlActionStore,
    SqlConversationRepository,
    SqlHandoffStore,
    SqlUsageStore,
    create_engine,
)
from aiplatform.storage.tables import metadata
from aiplatform.usage import InMemoryUsageStore, UsageEvent

PG_URL = os.environ.get("AIP_TEST_DATABASE_URL")

BACKENDS = [
    "memory",
    "sqlite",
    pytest.param("postgres", marks=[
        pytest.mark.postgres,
        pytest.mark.skipif(not PG_URL, reason="AIP_TEST_DATABASE_URL not set")]),
]


@pytest.fixture(params=BACKENDS)
async def stores(request, tmp_path):
    if request.param == "memory":
        yield InMemoryConversationRepository(), InMemoryUsageStore(), InMemoryActionStore()
        return
    url = PG_URL if request.param == "postgres" else f"sqlite+aiosqlite:///{tmp_path}/t.db"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield SqlConversationRepository(engine), SqlUsageStore(engine), SqlActionStore(engine)
    await engine.dispose()


@pytest.fixture(params=BACKENDS)
async def desk(request, tmp_path):
    """A conversation repository and a handoff store on the same backend."""
    if request.param == "memory":
        yield InMemoryConversationRepository(), InMemoryHandoffStore()
        return
    url = PG_URL if request.param == "postgres" else f"sqlite+aiosqlite:///{tmp_path}/h.db"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield SqlConversationRepository(engine), SqlHandoffStore(engine)
    await engine.dispose()


async def test_create_get_and_append_in_order(stores):
    repo, _, _ = stores
    conv = await repo.create("alice", "agent")
    await repo.append(conv, {"role": "user", "content": "Hello there"})
    await repo.append(conv, {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]})

    loaded = await repo.get(conv.id, "alice")
    assert loaded.messages == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
    ]
    assert loaded.title == "Hello there" and loaded.kind == "agent"


async def test_other_users_and_bad_ids_are_not_found(stores):
    repo, _, _ = stores
    conv = await repo.create("alice", "agent")
    for cid, user in [(conv.id, "bob"), ("not-a-uuid", "alice"),
                      ("00000000-0000-0000-0000-000000000000", "alice")]:
        with pytest.raises(ConversationNotFound):
            await repo.get(cid, user)


async def test_list_is_per_user_newest_first(stores):
    repo, _, _ = stores
    first = await repo.create("alice", "agent")
    second = await repo.create("alice", "agent")
    await repo.create("bob", "agent")
    await repo.append(first, {"role": "user", "content": "bump"})  # now most recent
    listed = await repo.list("alice")
    assert [c.id for c in listed] == [first.id, second.id]
    assert listed[0].messages == []


async def test_concurrent_writers_cannot_interleave(stores, request):
    repo, _, _ = stores
    if isinstance(repo, InMemoryConversationRepository):
        pytest.skip("in-memory store relies on the in-process InFlight guard")
    conv = await repo.create("alice", "agent")
    replica_a = await repo.get(conv.id, "alice")
    replica_b = await repo.get(conv.id, "alice")
    await repo.append(replica_a, {"role": "user", "content": "from A"})
    with pytest.raises(ConversationBusy):
        await repo.append(replica_b, {"role": "user", "content": "from B"})
    assert [m["content"] for m in (await repo.get(conv.id, "alice")).messages] == ["from A"]


async def test_usage_totals_per_user(stores):
    _, usage, _ = stores
    event = UsageEvent(user_id="alice", conversation_id=None, route="chat", provider="anthropic",
                       model="claude-haiku-4-5", input_tokens=100, output_tokens=20,
                       cache_read_tokens=5, cache_write_tokens=1)
    await usage.record(event)
    await usage.record(event)
    assert await usage.tokens_used_today("alice") == 252
    assert await usage.tokens_used_today("bob") == 0


async def test_actions_are_decided_exactly_once(stores):
    repo, _, actions = stores
    conv = await repo.create("alice", "agent")
    action = await actions.create(conversation_id=conv.id, user_id="alice", tool_use_id="tu_1",
                                  tool_name="create_support_ticket", input={"title": "t"})
    assert [a.id for a in await actions.list_pending(conv.id, "alice")] == [action.id]
    assert await actions.list_pending(conv.id, "bob") == []
    with pytest.raises(ActionNotPending):  # someone else's
        await actions.decide(action.id, user_id="bob", conversation_id=conv.id, approve=True)
    decided = await actions.decide(action.id, user_id="alice", conversation_id=conv.id,
                                   approve=True)
    assert decided.status == "approved" and decided.input == {"title": "t"}
    assert decided.decided_at is not None
    with pytest.raises(ActionNotPending):  # a second decision is refused
        await actions.decide(action.id, user_id="alice", conversation_id=conv.id, approve=False)
    assert await actions.list_pending(conv.id, "alice") == []
    with pytest.raises(ActionNotPending):
        await actions.decide("not-a-uuid", user_id="alice", conversation_id=conv.id, approve=True)


async def test_operator_messages_keep_their_author(desk):
    repo, _ = desk
    conv = await repo.create("alice", "agent")
    await repo.append(conv, {"role": "user", "content": "I want a person"})
    await repo.append(conv, {"role": "assistant", "author": "operator",
                             "content": [{"type": "text", "text": "Hi, I'm Ana"}]})
    loaded = await repo.get(conv.id, "alice")
    assert "author" not in loaded.messages[0]
    assert loaded.messages[1]["author"] == "operator"


async def test_handoffs_move_exactly_once(desk):
    repo, handoffs = desk
    conv = await repo.create("alice", "agent")
    assert await handoffs.latest(conv.id) is None
    offer = await handoffs.create(conversation_id=conv.id, user_id="alice", status="offered",
                                  reason="insistence", summary="keeps asking",
                                  message_index=4)
    assert (await handoffs.latest(conv.id)).id == offer.id
    with pytest.raises(HandoffNotFound):  # someone else's
        await handoffs.transition(offer.id, expected="offered", to="open", user_id="bob")
    opened = await handoffs.transition(offer.id, expected="offered", to="open",
                                       conversation_id=conv.id, user_id="alice")
    assert opened.status == "open" and opened.message_index == 4 and opened.closed_at is None
    with pytest.raises(HandoffNotFound):  # a second accept is refused
        await handoffs.transition(offer.id, expected="offered", to="open")
    assert [h.id for h in await handoffs.list("open")] == [offer.id]
    closed = await handoffs.transition(offer.id, expected="open", to="closed")
    assert closed.closed_at is not None and await handoffs.list("open") == []
    assert (await handoffs.get(offer.id)).status == "closed"
    with pytest.raises(HandoffNotFound):
        await handoffs.get("not-a-uuid")


def test_migration_turns_chats_into_queries(tmp_path):
    import sqlalchemy as sa
    cfg = Config("alembic.ini")
    cfg.attributes["url"] = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "0002")
    engine = sa.create_engine(f"sqlite:///{tmp_path}/c.db")
    with engine.begin() as db:
        db.execute(sa.text(
            "INSERT INTO conversations (id, user_id, kind, created_at, updated_at) VALUES "
            "('00000000-0000-0000-0000-000000000001', 'ana', 'chat', '2026-09-01', '2026-09-01')"))
    command.upgrade(cfg, "head")
    with engine.connect() as db:
        assert db.scalar(sa.text("SELECT kind FROM conversations")) == "agent"


def test_migrations_create_the_same_tables_as_the_metadata(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/m.db"
    cfg = Config("alembic.ini")
    cfg.attributes["url"] = url
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")

    import sqlalchemy as sa
    engine = sa.create_engine(f"sqlite:///{tmp_path}/m.db")
    inspector = sa.inspect(engine)
    for table in metadata.sorted_tables:
        migrated = {c["name"] for c in inspector.get_columns(table.name)}
        assert migrated == set(table.columns.keys()), table.name
    command.downgrade(cfg, "base")
    assert not set(sa.inspect(engine).get_table_names()) & set(metadata.tables)
