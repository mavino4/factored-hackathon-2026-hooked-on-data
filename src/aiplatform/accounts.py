"""Username + password sign-in (``AIP_AUTH_MODE=password``).

Accounts are created by the operator script (``scripts/users.py``): one per bank customer,
named after the first part of the customer's email, with a random password. What is kept:

- passwords only as Argon2id hashes; the plain text exists once, in the operator's
  credentials file;
- sessions as the SHA-256 of an opaque random token, which the browser holds in an
  HttpOnly cookie. A copy of the database can't be replayed as a session;
- every account event (created, sign-in, failure, lock, password change...) in the
  append-only ``auth_events`` table, never with a password.

Sign-in gives the same answer for an unknown user, a wrong password, a locked account and
a disabled one, and takes about the same time, so usernames can't be probed.
"""

import asyncio
import hashlib
import re
import secrets
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from aiplatform.auth import Principal

USERNAME = re.compile(r"[a-z0-9._-]{1,40}")
# No look-alikes (0/O, 1/l/I): passwords are read from a file and typed by hand.
PASSWORD_ALPHABET = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
PASSWORD_LENGTH = 16  # ~93 bits
MIN_PASSWORD_LENGTH = 12
# Sessions unused for less than this aren't re-stamped, to spare a write per request.
TOUCH_INTERVAL = timedelta(minutes=1)

# Argon2id with OWASP's minimum parameters (19 MiB, 2 passes, 1 lane).
_hasher = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1)

Event = Literal["user_created", "login_ok", "login_failed", "locked", "logout",
                "password_changed", "password_rotated", "unlocked", "disabled", "enabled"]


class InvalidCredentials(Exception):
    """Sign-in refused. Deliberately carries no reason."""


class WeakPassword(Exception):
    """The new password is too short or equal to the username."""


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def generate_password() -> str:
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_LENGTH))


def normalize_username(username: str) -> str:
    return username.strip().lower()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def derive_usernames(customers: Iterable[tuple[str, str | None]], *,
                     taken: Iterable[str] = (),
                     existing: dict[str, str] | None = None,
                     ) -> tuple[dict[str, str], dict[str, str]]:
    """Give each ``(customer_id, email)`` a username: the email's first part, lowercased.

    The first customer (by ID) with a given first part gets it as is, the next ones get a
    number (``maria.gomez``, ``maria.gomez2``, ...). Names in ``taken`` are never assigned.
    Customers in ``existing`` (customer_id -> username) keep their username, so running
    this again after adding customers never renames anyone.

    Returns ``(assigned, skipped)``: customer_id -> username for the new accounts, and
    customer_id -> reason (``no_email`` / ``invalid``) for the ones left without one.
    """
    existing = existing or {}
    used = set(taken) | set(existing.values())
    assigned: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for customer_id, email in sorted(customers, key=lambda c: c[0]):
        if customer_id in existing:
            continue
        if not isinstance(email, str) or "@" not in email:
            skipped[customer_id] = "no_email"
            continue
        base = email.split("@", 1)[0].strip().lower()
        if not USERNAME.fullmatch(base):
            skipped[customer_id] = "invalid"
            continue
        name, n = base, 1
        while name in used or len(name) > 40:
            n += 1
            suffix = str(n)
            name = base[:40 - len(suffix)] + suffix
        used.add(name)
        assigned[customer_id] = name
    return assigned, skipped


@dataclass(frozen=True)
class Account:
    username: str
    password_hash: str
    customer_id: str | None = None
    failed_attempts: int = 0
    locked_until: datetime | None = None
    disabled: bool = False


@dataclass(frozen=True)
class Session:
    token_hash: str
    username: str
    last_seen_at: datetime
    expires_at: datetime


class AccountStore(Protocol):
    async def get(self, username: str) -> Account | None: ...

    async def create(self, username: str, password_hash: str,
                     customer_id: str | None = None) -> None: ...

    async def record_failure(self, username: str, *, max_failures: int,
                             lock_until: datetime) -> bool:
        """Count one failed sign-in, atomically; lock the account when it reaches
        ``max_failures``. Returns whether the account is now locked."""
        ...

    async def reset_failures(self, username: str) -> None: ...

    async def set_password(self, username: str, password_hash: str, now: datetime) -> None: ...

    async def add_session(self, session: Session) -> None: ...

    async def get_session(self, token_hash: str) -> Session | None: ...

    async def touch_session(self, token_hash: str, now: datetime) -> None: ...

    async def delete_session(self, token_hash: str) -> None: ...

    async def delete_sessions(self, username: str, *, keep: str | None = None) -> None:
        """Drop the user's sessions, except the one with token hash ``keep``."""
        ...

    async def record_event(self, username: str, event: Event, ip: str | None,
                           now: datetime) -> None: ...


class InMemoryAccountStore:
    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._sessions: dict[str, Session] = {}
        self.events: list[tuple[str, str, str | None]] = []

    async def get(self, username: str) -> Account | None:
        return self._accounts.get(username)

    async def create(self, username: str, password_hash: str,
                     customer_id: str | None = None) -> None:
        if username in self._accounts:
            raise ValueError(f"user exists: {username}")
        self._accounts[username] = Account(username, password_hash, customer_id)

    async def record_failure(self, username: str, *, max_failures: int,
                             lock_until: datetime) -> bool:
        account = self._accounts[username]
        failures = account.failed_attempts + 1
        locked = failures >= max_failures
        self._accounts[username] = replace(
            account, failed_attempts=failures,
            locked_until=lock_until if locked else account.locked_until)
        return locked

    async def reset_failures(self, username: str) -> None:
        self._accounts[username] = replace(self._accounts[username], failed_attempts=0,
                                           locked_until=None)

    async def set_password(self, username: str, password_hash: str, now: datetime) -> None:
        self._accounts[username] = replace(self._accounts[username],
                                           password_hash=password_hash)

    async def add_session(self, session: Session) -> None:
        self._sessions[session.token_hash] = session

    async def get_session(self, token_hash: str) -> Session | None:
        return self._sessions.get(token_hash)

    async def touch_session(self, token_hash: str, now: datetime) -> None:
        if token_hash in self._sessions:
            self._sessions[token_hash] = replace(self._sessions[token_hash], last_seen_at=now)

    async def delete_session(self, token_hash: str) -> None:
        self._sessions.pop(token_hash, None)

    async def delete_sessions(self, username: str, *, keep: str | None = None) -> None:
        self._sessions = {h: s for h, s in self._sessions.items()
                          if s.username != username or h == keep}

    async def record_event(self, username: str, event: Event, ip: str | None,
                           now: datetime) -> None:
        self.events.append((username, event, ip))


class Accounts:
    """Sign-in, sessions and password changes, on top of an ``AccountStore``."""

    def __init__(self, store: AccountStore, *, idle: timedelta = timedelta(minutes=30),
                 max_age: timedelta = timedelta(hours=8), max_failures: int = 5,
                 lock_for: timedelta = timedelta(minutes=15),
                 clock=lambda: datetime.now(UTC)):
        self.store = store
        self._idle = idle
        self._max_age = max_age
        self._max_failures = max_failures
        self._lock_for = lock_for
        self._clock = clock
        # Checked for unknown users, so they take as long as a wrong password.
        self._dummy_hash = hash_password(secrets.token_urlsafe(16))

    async def create_user(self, username: str, password: str,
                          customer_id: str | None = None) -> None:
        username = normalize_username(username)
        if not USERNAME.fullmatch(username):
            raise ValueError(f"invalid username: {username!r}")
        await self.store.create(username, await asyncio.to_thread(hash_password, password),
                                customer_id)
        await self.store.record_event(username, "user_created", None, self._clock())

    async def login(self, username: str, password: str,
                    ip: str | None = None) -> tuple[str, Principal]:
        """Check the credentials and open a session. Returns its token (shown to the
        browser once, as a cookie) and who signed in. Raises InvalidCredentials."""
        username = normalize_username(username)
        now = self._clock()
        account = await self.store.get(username)
        # Hashing is slow on purpose: off the event loop.
        matches = await asyncio.to_thread(
            verify_password, account.password_hash if account else self._dummy_hash, password)
        if account is None:
            raise InvalidCredentials
        locked = account.locked_until is not None and account.locked_until > now
        if locked or account.disabled or not matches:
            if not locked and not account.disabled:
                now_locked = await self.store.record_failure(
                    username, max_failures=self._max_failures, lock_until=now + self._lock_for)
                if now_locked:
                    await self.store.record_event(username, "locked", ip, now)
            await self.store.record_event(username, "login_failed", ip, now)
            raise InvalidCredentials
        if account.failed_attempts:
            await self.store.reset_failures(username)
        if _hasher.check_needs_rehash(account.password_hash):  # parameters were raised
            await self.store.set_password(
                username, await asyncio.to_thread(hash_password, password), now)
        token = secrets.token_urlsafe(32)
        await self.store.add_session(Session(token_hash(token), username, now,
                                             now + self._max_age))
        await self.store.record_event(username, "login_ok", ip, now)
        return token, Principal(username, customer_id=account.customer_id)

    async def resolve(self, token: str) -> Principal | None:
        """Who a session token belongs to, or None if it's unknown, expired or idle."""
        now = self._clock()
        hashed = token_hash(token)
        session = await self.store.get_session(hashed)
        if session is None:
            return None
        if now >= session.expires_at or now - session.last_seen_at >= self._idle:
            await self.store.delete_session(hashed)
            return None
        account = await self.store.get(session.username)
        if account is None or account.disabled:
            await self.store.delete_session(hashed)
            return None
        if now - session.last_seen_at >= TOUCH_INTERVAL:
            await self.store.touch_session(hashed, now)
        return Principal(account.username, customer_id=account.customer_id)

    async def logout(self, token: str, ip: str | None = None) -> None:
        hashed = token_hash(token)
        session = await self.store.get_session(hashed)
        if session is not None:
            await self.store.delete_session(hashed)
            await self.store.record_event(session.username, "logout", ip, self._clock())

    async def change_password(self, username: str, current: str, new: str, *,
                              token: str | None = None, ip: str | None = None) -> None:
        """Set a new password and sign the user out everywhere else (the session with
        ``token`` stays). Raises InvalidCredentials if ``current`` is wrong."""
        if len(new) < MIN_PASSWORD_LENGTH or normalize_username(new) == username:
            raise WeakPassword
        now = self._clock()
        account = await self.store.get(username)
        if account is None or not await asyncio.to_thread(
                verify_password, account.password_hash, current):
            await self.store.record_event(username, "login_failed", ip, now)
            raise InvalidCredentials
        await self.store.set_password(username, await asyncio.to_thread(hash_password, new),
                                      now)
        await self.store.delete_sessions(username, keep=token_hash(token) if token else None)
        await self.store.record_event(username, "password_changed", ip, now)
