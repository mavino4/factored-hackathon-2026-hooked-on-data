"""Read access to the external core-banking database, scoped to one authenticated user.

Every query runs in its own transaction that first sets `app.subject` to the user's
verified token `sub`. The database's Row-Level Security policies (deploy/bankdb/
schema.sql) then only expose the customer linked to that subject, so isolation does
not depend on this code adding the right WHERE clause.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

PRODUCT_TYPES = ["Cuenta Ahorro", "Cuenta Corriente", "Tarjeta Crédito", "Tarjeta Débito",
                 "Préstamo Personal", "Préstamo Hipotecario", "Inversión", "Seguro"]


class NotLinked(Exception):
    """The authenticated user isn't linked to any bank customer."""


@dataclass(frozen=True)
class CustomerProfile:
    first_name: str
    country: str | None
    city: str | None
    segment: str | None
    status: str | None
    tenure_years: Decimal | None


@dataclass(frozen=True)
class Product:
    product_type: str
    number_last4: str | None
    currency: str
    current_balance: Decimal
    credit_limit: Decimal | None
    available_credit: Decimal | None
    interest_rate: Decimal | None
    status: str
    days_past_due: int | None
    opening_date: date | None
    expiration_date: date | None


class BankRepository(Protocol):
    async def get_customer(self, subject: str) -> CustomerProfile:
        """The customer linked to `subject`. Raises NotLinked if there is none."""
        ...

    async def get_products(self, subject: str,
                           product_type: str | None = None) -> list[Product]:
        """Products of the customer linked to `subject`. Raises NotLinked if there is none."""
        ...

    async def close(self) -> None: ...


PRODUCT_COLUMNS = ("product_type, product_number_last4, currency, current_balance, "
                   "credit_limit, available_credit, interest_rate, product_status, "
                   "days_past_due, opening_date, expiration_date")


def _product(row) -> Product:
    return Product(product_type=row.product_type, number_last4=row.product_number_last4,
                   currency=row.currency.strip(), current_balance=row.current_balance,
                   credit_limit=row.credit_limit, available_credit=row.available_credit,
                   interest_rate=row.interest_rate, status=row.product_status,
                   days_past_due=row.days_past_due, opening_date=row.opening_date,
                   expiration_date=row.expiration_date)


class PostgresBankRepository:
    def __init__(self, url: str, *, engine: AsyncEngine | None = None):
        # Read-only role (bank_reader), small pool, bounded queries, read-only transactions.
        self._engine = engine or create_async_engine(
            url, pool_size=5, max_overflow=5, pool_pre_ping=True,
            connect_args={"server_settings": {"statement_timeout": "5000",
                                              "default_transaction_read_only": "on"}})

    @asynccontextmanager
    async def _as(self, subject: str) -> AsyncIterator[AsyncConnection]:
        """A transaction in which the database only shows `subject`'s customer."""
        if not subject:
            raise NotLinked()
        async with self._engine.begin() as conn:
            # set_config(..., is_local=true) == SET LOCAL: reset at the end of the
            # transaction, so a pooled connection never keeps a previous user's identity.
            await conn.execute(sa.text("SELECT set_config('app.subject', :subject, true)"),
                               {"subject": subject})
            yield conn

    async def get_customer(self, subject: str) -> CustomerProfile:
        async with self._as(subject) as conn:
            row = (await conn.execute(sa.text(
                "SELECT first_name, country, city, segment, customer_status, tenure_years "
                "FROM bank.customers"))).first()
        if row is None:
            raise NotLinked()
        return CustomerProfile(first_name=row.first_name, country=row.country, city=row.city,
                               segment=row.segment, status=row.customer_status,
                               tenure_years=row.tenure_years)

    async def get_products(self, subject: str,
                           product_type: str | None = None) -> list[Product]:
        async with self._as(subject) as conn:
            if (await conn.execute(sa.text("SELECT 1 FROM bank.customer_logins"))).first() is None:
                raise NotLinked()
            query = f"SELECT {PRODUCT_COLUMNS} FROM bank.products"
            params = {}
            if product_type:
                query += " WHERE product_type = :product_type"
                params["product_type"] = product_type
            query += " ORDER BY product_type, product_number_last4"
            rows = (await conn.execute(sa.text(query), params)).all()
        return [_product(r) for r in rows]

    async def close(self) -> None:
        await self._engine.dispose()


class InMemoryBankRepository:
    """Same contract without a database (unit tests, demos)."""

    def __init__(self, customers: dict[str, CustomerProfile],
                 products: dict[str, list[Product]], logins: dict[str, str]):
        self._customers = customers  # customer_id -> profile
        self._products = products  # customer_id -> products
        self._logins = logins  # subject -> customer_id

    def _customer_id(self, subject: str) -> str:
        if subject not in self._logins:
            raise NotLinked()
        return self._logins[subject]

    async def get_customer(self, subject: str) -> CustomerProfile:
        return self._customers[self._customer_id(subject)]

    async def get_products(self, subject: str,
                           product_type: str | None = None) -> list[Product]:
        products = self._products.get(self._customer_id(subject), [])
        return [p for p in products if product_type in (None, p.product_type)]

    async def close(self) -> None:
        pass
