"""Isolation tests against real Postgres: Row-Level Security must confine the app's
read-only role to the customer linked to the session subject, whatever the query.

Needs a superuser URL (AIP_TEST_BANK_ADMIN_URL, e.g. the Compose Postgres). Creates its
own `bank_test` database with the production schema; never touches the real `bank` DB.
"""

import asyncio
import importlib.util
import os
from pathlib import Path

import asyncpg
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from aiplatform.banking.repository import NotLinked, PostgresBankRepository

ADMIN_URL = os.environ.get("AIP_TEST_BANK_ADMIN_URL")
pytestmark = [pytest.mark.postgres,
              pytest.mark.skipif(not ADMIN_URL, reason="AIP_TEST_BANK_ADMIN_URL not set")]

ROOT = Path(__file__).resolve().parent.parent
DB = "bank_test"
READER_PASSWORD = os.environ.get("AIP_BANK_READER_PASSWORD", "bank_reader")


def reader_url() -> str:
    host = ADMIN_URL.split("@", 1)[1].rsplit("/", 1)[0]
    return f"postgresql+asyncpg://bank_reader:{READER_PASSWORD}@{host}/{DB}"


def load_loader():
    spec = importlib.util.spec_from_file_location("load_bank_db", ROOT / "scripts/load_bank_db.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def setup_database():
    await load_loader().ensure_database_and_role(ADMIN_URL, DB, READER_PASSWORD)
    conn = await asyncpg.connect(ADMIN_URL.rsplit("/", 1)[0] + f"/{DB}")
    try:
        await conn.execute((ROOT / "deploy/bankdb/schema.sql").read_text())
        await conn.execute("TRUNCATE bank.customer_logins, bank.products, bank.customers")
        await conn.executemany(
            "INSERT INTO bank.customers (customer_id, first_name, country) VALUES ($1, $2, $3)",
            [("C-ANA", "Ana", "Colombia"), ("C-BRU", "Bruno", "México")])
        await conn.executemany(
            "INSERT INTO bank.products (product_id, customer_id, product_type, "
            "product_number_last4, currency, current_balance, credit_limit, product_status) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, 'Active')",
            [("P-A1", "C-ANA", "Cuenta Ahorro", "1111", "COP", 1000, None),
             ("P-A2", "C-ANA", "Tarjeta Crédito", "2222", "COP", 300, 5000),
             ("P-B1", "C-BRU", "Tarjeta Crédito", "3333", "USD", 50, 700)])
        await conn.executemany("INSERT INTO bank.customer_logins VALUES ($1, $2)",
                               [("ana", "C-ANA"), ("bruno", "C-BRU")])
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def database():
    asyncio.run(setup_database())


async def raw(sql: str, subject: str | None = None, before: str | None = None):
    """Run a query as the app's role, optionally with a session subject."""
    engine = create_async_engine(reader_url())
    try:
        async with engine.begin() as conn:
            if subject is not None:
                await conn.execute(sa.text("SELECT set_config('app.subject', :s, true)"),
                                   {"s": subject})
            if before:
                await conn.execute(sa.text(before))
            return (await conn.execute(sa.text(sql))).all()
    finally:
        await engine.dispose()


async def test_select_without_where_only_sees_own_rows(database):
    assert await raw("SELECT count(*) FROM bank.products") == [(0,)]
    assert await raw("SELECT count(*) FROM bank.customers") == [(0,)]
    rows = await raw("SELECT product_id FROM bank.products ORDER BY product_id", "ana")
    assert [r[0] for r in rows] == ["P-A1", "P-A2"]
    assert await raw("SELECT product_id FROM bank.products WHERE product_id = 'P-B1'",
                     "ana") == []
    assert await raw("SELECT count(*) FROM bank.customer_logins", "ana") == [(1,)]
    assert await raw("SELECT count(*) FROM bank.products", "mallory") == [(0,)]
    assert await raw("SELECT count(*) FROM bank.products", "") == [(0,)]


async def test_app_role_cannot_write_or_disable_rls(database):
    for sql in ("UPDATE bank.products SET current_balance = 0",
                "INSERT INTO bank.customer_logins VALUES ('mallory', 'C-ANA')",
                "DELETE FROM bank.customers"):
        with pytest.raises(Exception, match="permission denied|read-only|row-level security"):
            await raw(sql, "ana")
    with pytest.raises(Exception, match="row-level security"):
        await raw("SELECT count(*) FROM bank.products", before="SET LOCAL row_security = off")


async def test_repository_is_scoped_to_the_subject(database):
    repo = PostgresBankRepository(reader_url())
    try:
        ana = await repo.get_products("ana")
        assert [(p.number_last4, p.current_balance) for p in ana] == [("1111", 1000), ("2222", 300)]
        assert ana[1].available_credit == 4700  # generated column: limit - debt
        assert [p.number_last4 for p in await repo.get_products("ana", "Tarjeta Crédito")] == [
            "2222"]
        assert (await repo.get_customer("bruno")).first_name == "Bruno"
        for subject in ("mallory", ""):
            with pytest.raises(NotLinked):
                await repo.get_products(subject)
            with pytest.raises(NotLinked):
                await repo.get_customer(subject)
    finally:
        await repo.close()


async def test_pooled_connections_never_leak_the_previous_subject(database):
    # One pooled connection shared by everyone: SET LOCAL must not survive a transaction.
    engine = create_async_engine(reader_url(), pool_size=1, max_overflow=0)
    repo = PostgresBankRepository("unused", engine=engine)
    try:
        await repo.get_products("ana")
        async with engine.begin() as conn:  # same physical connection, no subject set
            assert (await conn.execute(sa.text("SELECT count(*) FROM bank.products"))).all() == [
                (0,)]
        # Many concurrent requests for different users on a tiny pool never mix results.
        subjects = ["ana", "bruno"] * 20
        results = await asyncio.gather(*(repo.get_products(s) for s in subjects))
        for subject, products in zip(subjects, results, strict=True):
            expected = {"ana": {"1111", "2222"}, "bruno": {"3333"}}[subject]
            assert {p.number_last4 for p in products} == expected
    finally:
        await repo.close()
