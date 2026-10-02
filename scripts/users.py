"""Operator tool for the password sign-in accounts (AIP_AUTH_MODE=password).

    uv run --with pandas --with pyarrow python scripts/users.py provision   # make users
    uv run python scripts/users.py create operador
    uv run python scripts/users.py create ana --customer-id CLI-00R1HD1YN0Z6
    uv run python scripts/users.py rotate maria.gomez2
    uv run python scripts/users.py unlock|disable|enable maria.gomez2
    uv run python scripts/users.py relink        # after `make bank-db`, which empties the links

`provision` creates one account per Active customer of the Datathon `customers` table that
is in the bank database: the username is the first part of the customer's email (numbered
when several customers share it), the password is random. Running it again only adds the
customers that have no account yet; nobody is renamed and no password changes.

Where things end up:
- app database: `users` (Argon2id hash only) and `auth_events` (audit trail);
- bank database: `bank.customer_logins` (username -> customer), which Row-Level Security
  uses to show each user only their own data;
- `credentials/users-<timestamp>.csv` (mode 600, not in git): the only copy of the plain
  passwords. Hand it over through a safe channel and delete it. A lost password can't be
  recovered, only replaced (`rotate`).

The email itself is read here and never stored.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import asyncpg

from aiplatform.accounts import (
    USERNAME,
    derive_usernames,
    generate_password,
    hash_password,
    normalize_username,
)

ROOT = Path(__file__).resolve().parent.parent
DEMO_LOGINS = ROOT / "deploy/bankdb/demo_logins.json"
CREDENTIALS_DIR = ROOT / "credentials"
# Never given to a customer: operators and the demo users (deploy/bankdb/demo_logins.json).
RESERVED = {"operador", "admin", "root", "demo"}


def reserved_names() -> set[str]:
    demo = set(json.loads(DEMO_LOGINS.read_text())) if DEMO_LOGINS.exists() else set()
    return RESERVED | demo


def read_customers(data_dir: Path) -> list[tuple[str, str | None]]:
    """(customer_id, email) of the Active customers."""
    import pandas as pd  # only `provision` needs it

    files = sorted(glob.glob(str(data_dir / "customers" / "**/*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files for customers under {data_dir}")
    df = pd.concat((pd.read_parquet(f, columns=["customer_id", "email", "customer_status"])
                    for f in files), ignore_index=True)
    df = df[df.customer_status == "Active"]
    return [(r.customer_id, r.email if isinstance(r.email, str) else None)
            for r in df.itertuples()]


def hash_many(passwords: list[str]) -> list[str]:
    return [hash_password(p) for p in passwords]


def hash_in_parallel(passwords: list[str], workers: int) -> list[str]:
    size = 500
    chunks = [passwords[i:i + size] for i in range(0, len(passwords), size)]
    hashes: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, part in enumerate(pool.map(hash_many, chunks), 1):
            hashes.extend(part)
            print(f"\r  hashing passwords: {min(done * size, len(passwords)):,}"
                  f"/{len(passwords):,}", end="", flush=True)
    print()
    return hashes


def write_credentials(rows: list[tuple[str, str, str]]) -> Path:
    """Write username, customer_id, password to a new file only its owner can read."""
    CREDENTIALS_DIR.mkdir(mode=0o700, exist_ok=True)
    path = CREDENTIALS_DIR / f"users-{datetime.now(UTC):%Y%m%d-%H%M%S}.csv"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["username", "customer_id", "password"])
        writer.writerows(rows)
    if path.stat().st_mode & 0o077:
        print(f"WARNING: {path} is readable by other users (this filesystem ignores "
              "permissions); move it somewhere private", file=sys.stderr)
    return path


async def link(bank: asyncpg.Connection, pairs: list[tuple[str, str]]) -> None:
    await bank.executemany(
        "INSERT INTO bank.customer_logins (subject, customer_id) VALUES ($1, $2) "
        "ON CONFLICT (subject) DO UPDATE SET customer_id = EXCLUDED.customer_id", pairs)


async def provision(args) -> None:
    customers = read_customers(args.data_dir)
    app, bank = await asyncpg.connect(args.app_url), await asyncpg.connect(args.bank_url)
    try:
        in_bank = {r["customer_id"] for r in await bank.fetch(
            "SELECT customer_id FROM bank.customers")}
        known = [c for c in customers if c[0] in in_bank]
        rows = await app.fetch("SELECT username, customer_id FROM users")
        existing = {r["customer_id"]: r["username"] for r in rows if r["customer_id"]}
        taken = {r["username"] for r in rows} | reserved_names()
        assigned, skipped = derive_usernames(known, taken=taken, existing=existing)
        already = sum(1 for cid, _ in known if cid in existing)
        first_part = {cid: email.split("@", 1)[0].lower() for cid, email in known if email}
        todo = sorted(assigned.items())
        if args.sample:
            todo = todo[:args.sample]

        print(f"Active customers in the data: {len(customers):,} "
              f"({len(customers) - len(known):,} not in the bank database)")
        print(f"  already have an account:   {already:,}")
        print(f"  without an email:          "
              f"{sum(1 for r in skipped.values() if r == 'no_email'):,}")
        print(f"  email not usable as a username: "
              f"{sum(1 for r in skipped.values() if r == 'invalid'):,}")
        print(f"  numbered (shared first part):   "
              f"{sum(1 for cid, name in todo if name != first_part[cid]):,}")
        print(f"  accounts to create:        {len(todo):,}")
        if not todo:
            return

        passwords = [generate_password() for _ in todo]
        hashes = hash_in_parallel(passwords, args.workers)
        path = write_credentials([(name, cid, pw)
                                  for (cid, name), pw in zip(todo, passwords, strict=True)])
        now = datetime.now(UTC)
        try:
            async with app.transaction():
                await app.copy_records_to_table(
                    "users", records=[(name, cid, h, 0, False, now, now)
                                      for (cid, name), h in zip(todo, hashes, strict=True)],
                    columns=["username", "customer_id", "password_hash", "failed_attempts",
                             "disabled", "password_changed_at", "created_at"])
                await app.copy_records_to_table(
                    "auth_events", records=[(name, "user_created", None, now)
                                            for _, name in todo],
                    columns=["username", "event", "ip", "created_at"])
        except BaseException:
            path.unlink()  # these passwords belong to accounts that were not created
            raise
        await link(bank, [(name, cid) for cid, name in todo])
        print(f"created {len(todo):,} accounts")
        print(f"passwords: {path.relative_to(ROOT)}  (the only copy; mode 600, not in git)")
    finally:
        await app.close()
        await bank.close()


async def event(app: asyncpg.Connection, username: str, name: str) -> None:
    await app.execute("INSERT INTO auth_events (username, event, ip, created_at) "
                      "VALUES ($1, $2, NULL, $3)", username, name, datetime.now(UTC))


def show_password(username: str, password: str) -> None:
    print(f"username: {username}\npassword: {password}\n"
          "Shown once: it is stored only as a hash.")


async def create(args) -> None:
    username = normalize_username(args.username)
    if not USERNAME.fullmatch(username):
        raise SystemExit("usernames are 1-40 characters of a-z 0-9 . _ -")
    app, bank = await asyncpg.connect(args.app_url), await asyncpg.connect(args.bank_url)
    try:
        if args.customer_id and not await bank.fetchval(
                "SELECT 1 FROM bank.customers WHERE customer_id = $1", args.customer_id):
            raise SystemExit(f"no such customer: {args.customer_id}")
        password, now = generate_password(), datetime.now(UTC)
        try:
            async with app.transaction():
                await app.execute(
                    "INSERT INTO users (username, customer_id, password_hash, failed_attempts,"
                    " disabled, password_changed_at, created_at) "
                    "VALUES ($1, $2, $3, 0, false, $4, $4)",
                    username, args.customer_id, hash_password(password), now)
                await event(app, username, "user_created")
        except asyncpg.UniqueViolationError:
            raise SystemExit("that username exists, or the customer already has an "
                             "account (use `rotate` for a new password)") from None
        if args.customer_id:
            await link(bank, [(username, args.customer_id)])
        show_password(username, password)
    finally:
        await app.close()
        await bank.close()


async def update_user(args, sql: str, event_name: str, *params,
                      end_sessions: bool = False) -> str:
    username = normalize_username(args.username)
    app = await asyncpg.connect(args.app_url)
    try:
        async with app.transaction():
            if await app.execute(sql, username, *params) != "UPDATE 1":
                raise SystemExit(f"no such user: {username}")
            if end_sessions:
                await app.execute("DELETE FROM sessions WHERE username = $1", username)
            await event(app, username, event_name)
    finally:
        await app.close()
    return username


async def rotate(args) -> None:
    password = generate_password()
    username = await update_user(
        args, "UPDATE users SET password_hash = $2, password_changed_at = $3, "
              "failed_attempts = 0, locked_until = NULL WHERE username = $1",
        "password_rotated", hash_password(password), datetime.now(UTC), end_sessions=True)
    show_password(username, password)


async def unlock(args) -> None:
    await update_user(args, "UPDATE users SET failed_attempts = 0, locked_until = NULL "
                            "WHERE username = $1", "unlocked")
    print("unlocked")


async def disable(args) -> None:
    await update_user(args, "UPDATE users SET disabled = true WHERE username = $1",
                      "disabled", end_sessions=True)
    print("disabled; open sessions ended")


async def enable(args) -> None:
    await update_user(args, "UPDATE users SET disabled = false WHERE username = $1",
                      "enabled")
    print("enabled")


async def relink(args) -> None:
    app, bank = await asyncpg.connect(args.app_url), await asyncpg.connect(args.bank_url)
    try:
        in_bank = {r["customer_id"] for r in await bank.fetch(
            "SELECT customer_id FROM bank.customers")}
        rows = await app.fetch("SELECT username, customer_id FROM users "
                               "WHERE customer_id IS NOT NULL")
        pairs = [(r["username"], r["customer_id"]) for r in rows
                 if r["customer_id"] in in_bank]
        await link(bank, pairs)
        print(f"linked {len(pairs):,} accounts "
              f"({len(rows) - len(pairs):,} whose customer is not in the bank database)")
    finally:
        await app.close()
        await bank.close()


def main() -> None:
    base = os.environ.get("AIP_USERS_ADMIN_URL",
                          "postgresql://aiplatform:aiplatform@localhost:5432")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app-url", default=f"{base}/aiplatform",
                        help="app database, as a role that can write to it")
    parser.add_argument("--bank-url", default=f"{base}/bank",
                        help="bank database, as its owner (not bank_reader)")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("provision", help="create the customers' accounts")
    p.add_argument("--data-dir", type=Path, default=ROOT.parent / "data_clean")
    p.add_argument("--sample", type=int, default=0, help="create at most N accounts")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.set_defaults(run=provision)

    p = commands.add_parser("create", help="create one account (an operator, a demo user)")
    p.add_argument("username")
    p.add_argument("--customer-id", help="bank customer this user signs in as")
    p.set_defaults(run=create)

    for name, run, text in [
            ("rotate", rotate, "new random password; ends the user's sessions"),
            ("unlock", unlock, "clear a lock from failed sign-ins"),
            ("disable", disable, "block sign-in and end the user's sessions"),
            ("enable", enable, "allow sign-in again")]:
        p = commands.add_parser(name, help=text)
        p.add_argument("username")
        p.set_defaults(run=run)

    commands.add_parser("relink", help="rebuild bank.customer_logins from the accounts"
                        ).set_defaults(run=relink)

    args = parser.parse_args()
    asyncio.run(args.run(args))


if __name__ == "__main__":
    main()
