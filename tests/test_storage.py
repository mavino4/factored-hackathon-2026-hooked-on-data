"""Contract tests: every repository/usage-store implementation must pass these.

Runs against in-memory and SQLite by default. Set AIP_TEST_DATABASE_URL
(postgresql+asyncpg://...) and run `make test-postgres` for Postgres.
"""

import os

import pytest
from alembic import command
from alembic.config import Config

from aiplatform.chat.inflight import ConversationBusy
from aiplatform.chat.repository import ConversationNotFound, InMemoryConversationRepository
from aiplatform.storage.sql import SqlConversationRepository, SqlUsageStore, create_engine
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
        yield InMemoryConversationRepository(), InMemoryUsageStore()
        return
    url = PG_URL if request.param == "postgres" else f"sqlite+aiosqlite:///{tmp_path}/t.db"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield SqlConversationRepository(engine), SqlUsageStore(engine)
    await engine.dispose()


async def test_create_get_and_append_in_order(stores):
    repo, _ = stores
    conv = await repo.create("alice", "chat")
    await repo.append(conv, {"role": "user", "content": "Hello there"})
    await repo.append(conv, {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]})

    loaded = await repo.get(conv.id, "alice")
    assert loaded.messages == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
    ]
    assert loaded.title == "Hello there" and loaded.kind == "chat"


async def test_other_users_and_bad_ids_are_not_found(stores):
    repo, _ = stores
    conv = await repo.create("alice", "chat")
    for cid, user in [(conv.id, "bob"), ("not-a-uuid", "alice"),
                      ("00000000-0000-0000-0000-000000000000", "alice")]:
        with pytest.raises(ConversationNotFound):
            await repo.get(cid, user)


async def test_list_is_per_user_newest_first(stores):
    repo, _ = stores
    first = await repo.create("alice", "chat")
    second = await repo.create("alice", "agent")
    await repo.create("bob", "chat")
    await repo.append(first, {"role": "user", "content": "bump"})  # now most recent
    listed = await repo.list("alice")
    assert [c.id for c in listed] == [first.id, second.id]
    assert listed[0].messages == []


async def test_concurrent_writers_cannot_interleave(stores, request):
    repo, _ = stores
    if isinstance(repo, InMemoryConversationRepository):
        pytest.skip("in-memory store relies on the in-process InFlight guard")
    conv = await repo.create("alice", "chat")
    replica_a = await repo.get(conv.id, "alice")
    replica_b = await repo.get(conv.id, "alice")
    await repo.append(replica_a, {"role": "user", "content": "from A"})
    with pytest.raises(ConversationBusy):
        await repo.append(replica_b, {"role": "user", "content": "from B"})
    assert [m["content"] for m in (await repo.get(conv.id, "alice")).messages] == ["from A"]


async def test_usage_totals_per_user(stores):
    _, usage = stores
    event = UsageEvent(user_id="alice", conversation_id=None, route="chat", provider="anthropic",
                       model="claude-haiku-4-5", input_tokens=100, output_tokens=20,
                       cache_read_tokens=5, cache_write_tokens=1)
    await usage.record(event)
    await usage.record(event)
    assert await usage.tokens_used_today("alice") == 252
    assert await usage.tokens_used_today("bob") == 0


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
