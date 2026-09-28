"""Load the simulated core-banking database from the Datathon customers/products tables.

Creates (idempotently) the `bank` database, the read-only `bank_reader` role and the
schema with Row-Level Security (deploy/bankdb/schema.sql), then reloads the data.

Only non-sensitive columns are loaded: no documents, last names, birth dates, emails,
phones, addresses, income or credit scores; product numbers keep only the last 4 digits.

    uv run --with pandas --with pyarrow python scripts/load_bank_db.py            # full data
    uv run --with pandas --with pyarrow python scripts/load_bank_db.py --sample 5000

Demo/test users (token `sub` -> customer) are kept in deploy/bankdb/demo_logins.json.
If the file is missing (or with --repick) they are chosen deterministically so each one
covers a different case, and the file is written for review.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import asyncpg

try:  # Only needed to read the Parquet data; the DB-setup helpers work without it.
    import pandas as pd
except ImportError:  # pragma: no cover - run with `uv run --with pandas --with pyarrow`
    pd = None

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "deploy/bankdb/schema.sql"
LOGINS = ROOT / "deploy/bankdb/demo_logins.json"

CUSTOMER_COLUMNS = ["customer_id", "first_name", "country", "city", "segment",
                    "customer_status", "tenure_years"]
PRODUCT_COLUMNS = ["product_id", "customer_id", "product_type", "product_number_last4",
                   "currency", "current_balance", "credit_limit", "interest_rate",
                   "opening_date", "expiration_date", "product_status", "days_past_due",
                   "has_linked_app", "last_transaction_date"]
# Never loaded. Checked again right before writing, as a safety net.
FORBIDDEN = {"document_number", "document_type", "last_name", "date_of_birth", "gender",
             "email", "mobile_phone", "landline_phone", "address", "postal_code",
             "credit_score", "estimated_monthly_income", "occupation", "marital_status",
             "education_level", "product_number"}

SAVINGS, CREDIT, LOAN = "Cuenta Ahorro", "Tarjeta Crédito", "Préstamo Personal"


def read_table(data_dir: Path, name: str, columns: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / name / "**/*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files for {name} under {data_dir}")
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files), ignore_index=True)


def load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = read_table(data_dir, "customers", CUSTOMER_COLUMNS)
    products = read_table(data_dir, "products", [
        "product_id", "customer_id", "product_type", "product_number", "currency",
        "current_balance", "credit_limit", "interest_rate", "opening_date", "expiration_date",
        "product_status", "days_past_due", "has_linked_app", "last_transaction_date"])
    # Mask immediately: the full number never leaves this function.
    products["product_number_last4"] = products.pop("product_number").astype(str).str[-4:]
    return customers, products[PRODUCT_COLUMNS]


# --- Demo users --------------------------------------------------------------

def pick_demo_customers(customers: pd.DataFrame, products: pd.DataFrame) -> dict[str, dict]:
    """Deterministically pick one customer per demo user, each covering a different case."""
    active = customers[customers.customer_status == "Active"]
    per = products.groupby("customer_id")
    counts = products.pivot_table(index="customer_id", columns="product_type",
                                  values="product_id", aggfunc="count", fill_value=0)

    def one_each(cid, *types):  # exactly one product of each type, so answers are unambiguous
        return cid in counts.index and all(counts.at[cid, t] == 1 for t in types)

    def products_of(cid):
        return per.get_group(cid)

    def card(cid):
        p = products_of(cid)
        return p[p.product_type == CREDIT].iloc[0]

    def first(pred, exclude):
        for row in active.sort_values("customer_id").itertuples():
            cid = row.customer_id
            if cid not in exclude and cid in per.groups and pred(cid, row):
                return cid
        raise SystemExit("no customer matches a demo profile")

    def healthy_pair(cid):  # savings + credit card, both active, card not past due
        if not one_each(cid, SAVINGS, CREDIT):
            return False
        p = products_of(cid)
        return (p.product_status.eq("Active").all() and (card(cid).days_past_due or 0) == 0
                and pd.notna(card(cid).credit_limit))

    chosen: dict[str, dict] = {}
    rules = [
        ("eval-es", "Colombia, COP: savings + credit card with days past due (eval, Spanish)",
         lambda cid, r: r.country == "Colombia" and one_each(cid, SAVINGS, CREDIT)
         and (card(cid).days_past_due or 0) > 0 and card(cid).product_status == "Active"
         and products_of(cid).currency.eq("COP").all()),
        ("eval-pt", "México, USD: savings + credit card + personal loan (eval, Portuguese)",
         lambda cid, r: r.country == "México" and healthy_pair(cid)
         and one_each(cid, SAVINGS, CREDIT, LOAN)),
        ("ana", "Colombia, COP: savings + credit card, all active",
         lambda cid, r: r.country == "Colombia" and healthy_pair(cid)),
        ("bruno", "México, USD: credit card + personal loan",
         lambda cid, r: r.country == "México" and healthy_pair(cid) and one_each(cid, LOAN)),
        ("carol", "Argentina, ARS: a Blocked credit card",
         lambda cid, r: r.country == "Argentina" and one_each(cid, CREDIT)
         and card(cid).product_status == "Blocked"),
        ("dave", "Any country: an investment and a savings account",
         lambda cid, r: one_each(cid, SAVINGS, "Inversión")),
    ]
    used: set[str] = set()
    for subject, why, pred in rules:
        cid = first(pred, used)
        used.add(cid)
        chosen[subject] = {"customer_id": cid, "profile": why}
    return chosen


def demo_logins(customers, products, repick: bool) -> dict[str, dict]:
    if LOGINS.exists() and not repick:
        return json.loads(LOGINS.read_text())
    logins = pick_demo_customers(customers, products)
    LOGINS.write_text(json.dumps(logins, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {LOGINS.relative_to(ROOT)}")
    return logins


# --- Conversion to database records -------------------------------------------

def _num(value) -> Decimal | None:
    return None if pd.isna(value) else Decimal(str(round(float(value), 2)))


def _date(value) -> date | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, datetime | pd.Timestamp):
        return value.date()
    return date.fromisoformat(str(value)[:10])


def customer_records(df: pd.DataFrame) -> list[tuple]:
    return [(r.customer_id, r.first_name, r.country, r.city, r.segment, r.customer_status,
             _num(r.tenure_years)) for r in df.itertuples()]


def product_records(df: pd.DataFrame) -> list[tuple]:
    return [(r.product_id, r.customer_id, r.product_type, r.product_number_last4, r.currency,
             _num(r.current_balance), _num(r.credit_limit), _num(r.interest_rate),
             _date(r.opening_date), _date(r.expiration_date), r.product_status,
             None if pd.isna(r.days_past_due) else int(r.days_past_due),
             bool(r.has_linked_app), _date(r.last_transaction_date)) for r in df.itertuples()]


# --- Database -----------------------------------------------------------------

async def ensure_database_and_role(admin_url: str, db: str, reader_password: str) -> None:
    conn = await asyncpg.connect(admin_url)
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db):
            await conn.execute(f'CREATE DATABASE "{db}"')
            print(f"created database {db}")
        exists = await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = 'bank_reader'")
        verb = "ALTER" if exists else "CREATE"
        # Passwords can't be bound as parameters in DDL; quote_literal escapes it safely.
        literal = await conn.fetchval("SELECT quote_literal($1)", reader_password)
        await conn.execute(f"{verb} ROLE bank_reader WITH LOGIN NOSUPERUSER NOBYPASSRLS "
                           f"NOCREATEDB NOCREATEROLE PASSWORD {literal}")
        await conn.execute(f'REVOKE CONNECT ON DATABASE "{db}" FROM PUBLIC')
        await conn.execute(f'GRANT CONNECT ON DATABASE "{db}" TO bank_reader')
    finally:
        await conn.close()


async def load(bank_url: str, customers: pd.DataFrame, products: pd.DataFrame,
               logins: dict[str, dict]) -> None:
    leaked = FORBIDDEN & (set(customers.columns) | set(products.columns))
    if leaked:
        raise SystemExit(f"refusing to load sensitive columns: {sorted(leaked)}")
    conn = await asyncpg.connect(bank_url)
    try:
        await conn.execute(SCHEMA.read_text())
        async with conn.transaction():
            await conn.execute("TRUNCATE bank.customer_logins, bank.products, bank.customers")
            await conn.copy_records_to_table("customers", schema_name="bank",
                                             records=customer_records(customers),
                                             columns=CUSTOMER_COLUMNS)
            await conn.copy_records_to_table("products", schema_name="bank",
                                             records=product_records(products),
                                             columns=PRODUCT_COLUMNS)
            await conn.executemany(
                "INSERT INTO bank.customer_logins (subject, customer_id) VALUES ($1, $2)",
                [(subject, v["customer_id"]) for subject, v in logins.items()])
        n_c = await conn.fetchval("SELECT count(*) FROM bank.customers")
        n_p = await conn.fetchval("SELECT count(*) FROM bank.products")
        print(f"loaded {n_c:,} customers, {n_p:,} products, {len(logins)} demo logins")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data_clean")
    parser.add_argument("--admin-url", default=os.environ.get(
        "AIP_BANK_ADMIN_URL", "postgresql://aiplatform:aiplatform@localhost:5432/postgres"))
    parser.add_argument("--db", default="bank")
    parser.add_argument("--reader-password",
                        default=os.environ.get("AIP_BANK_READER_PASSWORD", "bank_reader"))
    parser.add_argument("--sample", type=int, default=0,
                        help="load only N customers (demo users always included)")
    parser.add_argument("--repick", action="store_true", help="choose demo users again")
    args = parser.parse_args()
    if pd is None:
        raise SystemExit("pandas/pyarrow needed: uv run --with pandas --with pyarrow python "
                         "scripts/load_bank_db.py")

    customers, products = load_frames(args.data_dir)
    logins = demo_logins(customers, products, args.repick)
    if args.sample:
        keep = {v["customer_id"] for v in logins.values()}
        others = customers[~customers.customer_id.isin(keep)].sample(
            n=max(args.sample - len(keep), 0), random_state=42)
        customers = pd.concat([customers[customers.customer_id.isin(keep)], others])
        products = products[products.customer_id.isin(customers.customer_id)]

    asyncio.run(ensure_database_and_role(args.admin_url, args.db, args.reader_password))
    bank_url = args.admin_url.rsplit("/", 1)[0] + f"/{args.db}"
    asyncio.run(load(bank_url, customers, products, logins))


if __name__ == "__main__":
    main()
