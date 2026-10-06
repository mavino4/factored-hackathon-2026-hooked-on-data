"""Create the demo password accounts once.

Reads deploy/bankdb/demo_logins.json (copied to /app/demo_logins.json in the image).
Existing usernames are left alone, so a redeploy does not rotate passwords. New
passwords are appended to $SECRETS_FILE (mode 600) and are not printed.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import asyncpg

from aiplatform.accounts import generate_password, hash_password

DEMOS = Path(os.environ.get("DEMO_LOGINS", "/app/demo_logins.json"))
SECRETS = Path(os.environ.get("SECRETS_FILE", "/secrets/demo-passwords.txt"))


async def main() -> None:
    demos: dict[str, dict[str, str]] = json.loads(DEMOS.read_text())
    app = await asyncpg.connect(os.environ["APP_URL"])
    bank = await asyncpg.connect(os.environ["BANK_URL"])
    created = 0
    try:
        existing = {row["username"] for row in await app.fetch("SELECT username FROM users")}
        lines: list[str] = []
        now = datetime.now(UTC)
        for username, info in demos.items():
            customer_id = info["customer_id"]
            await bank.execute(
                "INSERT INTO bank.customer_logins (subject, customer_id) VALUES ($1, $2) "
                "ON CONFLICT (subject) DO UPDATE SET customer_id = EXCLUDED.customer_id",
                username, customer_id)
            if username in existing:
                continue
            password = generate_password()
            await app.execute(
                "INSERT INTO users (username, customer_id, password_hash, failed_attempts, "
                "disabled, password_changed_at, created_at) "
                "VALUES ($1, $2, $3, 0, false, $4, $4)",
                username, customer_id, hash_password(password), now)
            await app.execute(
                "INSERT INTO auth_events (username, event, ip, created_at) "
                "VALUES ($1, 'user_created', NULL, $2)",
                username, now)
            lines.append(f"{username} {password}\n")
            created += 1
        if lines:
            SECRETS.parent.mkdir(parents=True, exist_ok=True)
            with SECRETS.open("a", encoding="utf-8") as handle:
                handle.writelines(lines)
            os.chmod(SECRETS, 0o600)
    finally:
        await app.close()
        await bank.close()
    print(f"demo accounts ready; new passwords: {created}")


if __name__ == "__main__":
    asyncio.run(main())
