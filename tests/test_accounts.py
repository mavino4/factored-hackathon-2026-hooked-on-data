"""Password accounts: usernames from emails, sign-in, lockout, sessions.

Every store implementation must pass the same tests (memory and SQLite by default;
Postgres with AIP_TEST_DATABASE_URL and `make test-postgres`).
"""

import os
from datetime import UTC, datetime, timedelta

import pytest

from aiplatform.accounts import (
    PASSWORD_ALPHABET,
    Accounts,
    InMemoryAccountStore,
    InvalidCredentials,
    WeakPassword,
    derive_usernames,
    generate_password,
    hash_password,
    token_hash,
    verify_password,
)
from aiplatform.storage.sql import SqlAccountStore, create_engine
from aiplatform.storage.tables import auth_events, metadata

PG_URL = os.environ.get("AIP_TEST_DATABASE_URL")
BACKENDS = [
    "memory",
    "sqlite",
    pytest.param("postgres", marks=[
        pytest.mark.postgres,
        pytest.mark.skipif(not PG_URL, reason="AIP_TEST_DATABASE_URL not set")]),
]


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


@pytest.fixture(params=BACKENDS)
async def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAccountStore()
        return
    url = PG_URL if request.param == "postgres" else f"sqlite+aiosqlite:///{tmp_path}/a.db"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield SqlAccountStore(engine)
    await engine.dispose()


@pytest.fixture
async def accounts(store):
    clock = Clock()
    service = Accounts(store, max_failures=3, clock=clock)
    await service.create_user("maria.gomez", "correct horse battery", "CLI-1")
    return service, clock


async def events(store) -> list[str]:
    if isinstance(store, InMemoryAccountStore):
        return [event for _, event, _ in store.events]
    async with store._engine.connect() as db:
        return list(await db.scalars(
            auth_events.select().with_only_columns(auth_events.c.event)
            .order_by(auth_events.c.id)))


# --- Usernames -------------------------------------------------------------------------

def test_username_is_the_first_part_of_the_email():
    assigned, skipped = derive_usernames([("CLI-1", "Maria.Gomez@Gmail.com")])
    assert assigned == {"CLI-1": "maria.gomez"} and skipped == {}


def test_shared_first_parts_are_numbered_in_customer_order():
    assigned, _ = derive_usernames([("CLI-3", "ana.p@live.com"), ("CLI-1", "ana.p@gmail.com"),
                                    ("CLI-2", "ana.p@yahoo.com")])
    assert assigned == {"CLI-1": "ana.p", "CLI-2": "ana.p2", "CLI-3": "ana.p3"}


def test_customers_without_a_usable_email_get_no_username():
    assigned, skipped = derive_usernames([("CLI-1", None), ("CLI-2", "josé@x.com"),
                                          ("CLI-3", "no-at-sign"), ("CLI-4", float("nan"))])
    assert assigned == {}
    assert skipped == {"CLI-1": "no_email", "CLI-2": "invalid", "CLI-3": "no_email",
                       "CLI-4": "no_email"}


def test_reserved_and_numbered_names_are_never_reused():
    assigned, _ = derive_usernames(
        [("CLI-1", "operador@x.com"), ("CLI-2", "ana@x.com"), ("CLI-3", "ana2@x.com")],
        taken={"operador", "ana"})
    assert assigned == {"CLI-1": "operador2", "CLI-2": "ana2", "CLI-3": "ana22"}


def test_running_again_keeps_existing_usernames():
    first, _ = derive_usernames([("CLI-2", "ana.p@x.com")])
    again, _ = derive_usernames([("CLI-1", "ana.p@x.com"), ("CLI-2", "ana.p@x.com")],
                                existing=first)
    assert first == {"CLI-2": "ana.p"} and again == {"CLI-1": "ana.p2"}


def test_long_usernames_stay_within_the_limit():
    email = "a" * 40 + "@x.com"
    assigned, _ = derive_usernames([("CLI-1", email), ("CLI-2", email)])
    assert assigned == {"CLI-1": "a" * 40, "CLI-2": "a" * 39 + "2"}


# --- Passwords -------------------------------------------------------------------------

def test_generated_passwords_are_random_and_unambiguous():
    passwords = {generate_password() for _ in range(200)}
    assert len(passwords) == 200
    assert all(len(p) == 16 and set(p) <= set(PASSWORD_ALPHABET) for p in passwords)
    assert not set("0O1lI") & set(PASSWORD_ALPHABET)


def test_passwords_are_stored_as_salted_argon2id_hashes():
    one, two = hash_password("secret-secret"), hash_password("secret-secret")
    assert one.startswith("$argon2id$") and one != two and "secret" not in one
    assert verify_password(one, "secret-secret") and not verify_password(one, "other")
    assert not verify_password("not-a-hash", "secret-secret")


# --- Sign-in ---------------------------------------------------------------------------

async def test_login_opens_a_session_for_the_right_customer(accounts, store):
    service, _ = accounts
    token, principal = await service.login(" Maria.Gomez ", "correct horse battery", "10.0.0.7")
    assert (principal.user_id, principal.customer_id) == ("maria.gomez", "CLI-1")
    assert await service.resolve(token) == principal
    # Only a hash of the token is stored.
    assert await store.get_session(token) is None
    assert (await store.get_session(token_hash(token))).username == "maria.gomez"
    assert await events(store) == ["user_created", "login_ok"]


async def test_wrong_password_and_unknown_user_fail_the_same_way(accounts):
    service, _ = accounts
    with pytest.raises(InvalidCredentials) as wrong:
        await service.login("maria.gomez", "nope")
    with pytest.raises(InvalidCredentials) as unknown:
        await service.login("nobody", "nope")
    assert str(wrong.value) == str(unknown.value) == ""
    assert await service.resolve("made-up-token") is None


async def test_account_locks_after_repeated_failures_then_unlocks(accounts, store):
    service, clock = accounts
    for _ in range(3):
        with pytest.raises(InvalidCredentials):
            await service.login("maria.gomez", "nope")
    with pytest.raises(InvalidCredentials):  # locked: even the right password is refused
        await service.login("maria.gomez", "correct horse battery")
    assert "locked" in await events(store)

    clock.advance(minutes=16)
    token, _ = await service.login("maria.gomez", "correct horse battery")
    assert await service.resolve(token)
    assert (await store.get("maria.gomez")).failed_attempts == 0


async def test_a_success_resets_the_failure_count(accounts, store):
    service, _ = accounts
    for _ in range(2):
        with pytest.raises(InvalidCredentials):
            await service.login("maria.gomez", "nope")
    await service.login("maria.gomez", "correct horse battery")
    for _ in range(2):
        with pytest.raises(InvalidCredentials):
            await service.login("maria.gomez", "nope")
    await service.login("maria.gomez", "correct horse battery")


async def test_sessions_end_when_idle_and_at_their_maximum_age(accounts):
    service, clock = accounts
    token, _ = await service.login("maria.gomez", "correct horse battery")
    clock.advance(minutes=29)
    assert await service.resolve(token)  # activity keeps it alive
    clock.advance(minutes=29)
    assert await service.resolve(token)
    clock.advance(minutes=31)
    assert await service.resolve(token) is None  # idle too long

    token, _ = await service.login("maria.gomez", "correct horse battery")
    for _ in range(17):  # active the whole time, still ends after 8 hours
        clock.advance(minutes=29)
        alive = await service.resolve(token)
    assert alive is None


async def test_logout_ends_the_session(accounts, store):
    service, _ = accounts
    token, _ = await service.login("maria.gomez", "correct horse battery")
    await service.logout(token)
    assert await service.resolve(token) is None
    assert (await events(store))[-1] == "logout"


async def test_changing_the_password_signs_out_other_sessions(accounts):
    service, _ = accounts
    here, _ = await service.login("maria.gomez", "correct horse battery")
    elsewhere, _ = await service.login("maria.gomez", "correct horse battery")
    with pytest.raises(InvalidCredentials):
        await service.change_password("maria.gomez", "nope", "a-brand-new-password")
    with pytest.raises(WeakPassword):
        await service.change_password("maria.gomez", "correct horse battery", "short")
    with pytest.raises(WeakPassword):
        await service.change_password("maria.gomez", "correct horse battery", "maria.gomez")

    await service.change_password("maria.gomez", "correct horse battery",
                                  "a-brand-new-password", token=here)
    assert await service.resolve(here) and await service.resolve(elsewhere) is None
    with pytest.raises(InvalidCredentials):
        await service.login("maria.gomez", "correct horse battery")
    await service.login("maria.gomez", "a-brand-new-password")


async def test_duplicate_usernames_are_rejected(accounts):
    service, _ = accounts
    with pytest.raises(ValueError, match="exists"):
        await service.create_user("Maria.Gomez", "another-password")
    with pytest.raises(ValueError, match="invalid"):
        await service.create_user("maría gómez", "another-password")
