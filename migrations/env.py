"""Alembic environment. Uses AIP_DATABASE_URL (async driver) unless a URL is passed
programmatically via ``config.attributes["url"]``."""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from aiplatform.storage.tables import metadata

config = context.config
if config.config_file_name and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)


def _url() -> str:
    url = config.attributes.get("url") or os.environ.get("AIP_DATABASE_URL")
    if not url:
        raise RuntimeError("set AIP_DATABASE_URL to run migrations")
    return url


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = create_async_engine(_url())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=_url(), target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(_run_async())
