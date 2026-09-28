import asyncio
import importlib.util
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from aiplatform.agent.tools import ToolContext
from aiplatform.banking.repository import (
    CustomerProfile,
    InMemoryBankRepository,
    NotLinked,
    Product,
)
from aiplatform.banking.tools import make_bank_tools

ROOT = Path(__file__).resolve().parent.parent


def product(ptype, last4, balance, currency="COP", limit=None, available=None, **kw):
    return Product(product_type=ptype, number_last4=last4, currency=currency,
                   current_balance=Decimal(balance), credit_limit=limit and Decimal(limit),
                   available_credit=available and Decimal(available),
                   interest_rate=kw.get("rate") and Decimal(kw["rate"]),
                   status=kw.get("status", "Active"), days_past_due=kw.get("dpd"),
                   opening_date=None, expiration_date=kw.get("expires"))


def repo():
    return InMemoryBankRepository(
        customers={
            "C-ANA": CustomerProfile("Ana", "Colombia", "Bogotá", "Plus", "Active", Decimal("4.2")),
            "C-BRU": CustomerProfile("Bruno", "México", "Guadalajara", "Basic", "Active", None),
        },
        products={
            "C-ANA": [product("Cuenta Ahorro", "4112", "9358916.31"),
                      product("Tarjeta Crédito", "4140", "5469094.29", limit="153943385.93",
                              available="148474291.64", rate="31.5", dpd=12,
                              expires=date(2029, 1, 31))],
            "C-BRU": [product("Tarjeta Crédito", "9999", "100.00", currency="USD",
                              limit="1000", available="900", status="Blocked")],
        },
        logins={"ana": "C-ANA", "bruno": "C-BRU"},
    )


def tools(r=None):
    return {t.name: t for t in make_bank_tools(r or repo())}


async def call(tool, args, user):
    return await tool.handler(args, ToolContext(user_id=user))


async def test_products_come_from_the_session_user_only():
    t = tools()["get_products"]
    ana = await call(t, {}, "ana")
    bruno = await call(t, {}, "bruno")
    assert {p["number_last4"] for p in ana["products"]} == {"4112", "4140"}
    assert {p["number_last4"] for p in bruno["products"]} == {"9999"}


async def test_product_view_explains_balances_and_filters_by_type():
    result = await call(tools()["get_products"], {"product_type": "Tarjeta Crédito"}, "ana")
    [card] = result["products"]
    assert card == {
        "type": "Tarjeta Crédito", "number_last4": "4140", "currency": "COP",
        "status": "Active", "balance": 5469094.29,
        "balance_meaning": "amount currently owed on the card (debt)",
        "limit": 153943385.93, "limit_meaning": "credit limit",
        "available_credit": 148474291.64, "interest_rate_pct": 31.5, "days_past_due": 12,
        "expiration_date": "2029-01-31",
    }
    savings = (await call(tools()["get_products"], {"product_type": "Cuenta Ahorro"}, "ana"))
    assert savings["products"][0]["balance_meaning"] == "available funds in the account"


async def test_unlinked_user_gets_no_data():
    for name in ("get_products", "get_customer_profile"):
        result = await call(tools()[name], {}, "mallory")
        assert result["status"] == "not_linked" and "products" not in result


async def test_profile_has_no_sensitive_fields():
    profile = await call(tools()["get_customer_profile"], {}, "ana")
    assert profile == {"status": "ok", "first_name": "Ana", "country": "Colombia",
                       "city": "Bogotá", "segment": "Plus", "customer_status": "Active",
                       "tenure_years": 4.2}


def test_model_cannot_choose_the_customer():
    t = tools()
    # No tool accepts a customer or account identifier: identity comes only from the session.
    for tool in t.values():
        props = tool.input_schema.get("properties", {})
        assert not {"customer_id", "user_id", "account", "product_id"} & set(props)
    assert t["get_products"].validate({"customer_id": "C-BRU"}) == "unexpected field: customer_id"
    assert "must be one of" in t["get_products"].validate({"product_type": "Bitcoin"})


async def test_in_memory_repository_contract():
    r = repo()
    with pytest.raises(NotLinked):
        await r.get_products("nobody")
    assert (await r.get_customer("bruno")).first_name == "Bruno"


# --- Loader safety -------------------------------------------------------------

def load_loader():
    spec = importlib.util.spec_from_file_location("load_bank_db", ROOT / "scripts/load_bank_db.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_refuses_sensitive_columns():
    pd = pytest.importorskip("pandas")
    loader = load_loader()
    customers = pd.DataFrame([{"customer_id": "C1", "first_name": "A", "email": "a@x.com"}])
    products = pd.DataFrame([{"product_id": "P1"}])
    with pytest.raises(SystemExit, match="sensitive columns"):
        asyncio.run(loader.load("postgresql://unused/none", customers, products, {}))
    assert {"email", "document_number", "product_number", "last_name"} <= loader.FORBIDDEN
    assert not loader.FORBIDDEN & set(loader.CUSTOMER_COLUMNS + loader.PRODUCT_COLUMNS)


def test_demo_logins_are_reviewable_ids_only():
    logins = json.loads((ROOT / "deploy/bankdb/demo_logins.json").read_text())
    assert {"eval-es", "eval-pt", "ana", "bruno"} <= set(logins)
    for entry in logins.values():
        assert set(entry) == {"customer_id", "profile"}
        assert entry["customer_id"].startswith("CLI-")
