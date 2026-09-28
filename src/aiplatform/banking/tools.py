"""Banking tools for the agent: read-only queries about the authenticated customer.

The customer is always resolved from the session (ToolContext.user_id); the model can
never pass a customer ID or an account number. Balances are returned with an explicit
meaning, because "balance" means available funds on an account but debt on a card.
"""

from decimal import Decimal
from typing import Any

from aiplatform.agent.tools import Tool, ToolContext
from aiplatform.banking.repository import PRODUCT_TYPES, BankRepository, NotLinked, Product

BALANCE_MEANING = {
    "Cuenta Ahorro": "available funds in the account",
    "Cuenta Corriente": "available funds in the account",
    "Tarjeta Débito": "available funds of the linked account",
    "Tarjeta Crédito": "amount currently owed on the card (debt)",
    "Préstamo Personal": "outstanding loan debt",
    "Préstamo Hipotecario": "outstanding mortgage debt",
    "Inversión": "current value of the investment",
    "Seguro": "insurance balance (usually 0)",
}
LIMIT_MEANING = {
    "Tarjeta Crédito": "credit limit",
    "Préstamo Personal": "amount originally granted",
    "Préstamo Hipotecario": "amount originally granted",
}

NOT_LINKED = {
    "status": "not_linked",
    "message": ("This user is not linked to any bank customer, so no account data is "
                "available. Do not provide or guess any figures; suggest contacting the "
                "bank to link their account."),
}


def _money(value: Decimal | None) -> float | None:
    return None if value is None else float(round(value, 2))


def product_view(p: Product) -> dict[str, Any]:
    view: dict[str, Any] = {
        "type": p.product_type,
        "number_last4": p.number_last4,
        "currency": p.currency,
        "status": p.status,
        "balance": _money(p.current_balance),
        "balance_meaning": BALANCE_MEANING.get(p.product_type, "balance"),
    }
    if p.credit_limit is not None:
        view["limit"] = _money(p.credit_limit)
        view["limit_meaning"] = LIMIT_MEANING.get(p.product_type, "limit")
    if p.available_credit is not None:
        view["available_credit"] = _money(p.available_credit)
    if p.interest_rate is not None:
        view["interest_rate_pct"] = _money(p.interest_rate)
    if p.days_past_due is not None:
        view["days_past_due"] = p.days_past_due
    if p.expiration_date is not None:
        view["expiration_date"] = p.expiration_date.isoformat()
    return view


def make_bank_tools(repo: BankRepository) -> list[Tool]:
    async def get_customer_profile(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        try:
            c = await repo.get_customer(ctx.user_id)
        except NotLinked:
            return NOT_LINKED
        return {"status": "ok", "first_name": c.first_name, "country": c.country,
                "city": c.city, "segment": c.segment, "customer_status": c.status,
                "tenure_years": _money(c.tenure_years)}

    async def get_products(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        try:
            products = await repo.get_products(ctx.user_id, args.get("product_type"))
        except NotLinked:
            return NOT_LINKED
        return {"status": "ok", "count": len(products),
                "products": [product_view(p) for p in products]}

    return [
        Tool(
            name="get_customer_profile",
            description=("Get the authenticated customer's profile: first name, country, "
                         "city, segment, status and tenure. Takes no input; the customer is "
                         "always the signed-in user."),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=get_customer_profile,
        ),
        Tool(
            name="get_products",
            description=(
                "Get the authenticated customer's bank products with balances: type, last 4 "
                "digits, currency, status, balance (see balance_meaning: available funds for "
                "accounts, amount owed for credit cards, outstanding debt for loans), limit, "
                "available credit, interest rate, days past due and expiration date. Use it "
                "for ANY question about balances, limits, available credit or product status. "
                "Optionally filter by product_type. The customer is always the signed-in user."),
            input_schema={
                "type": "object",
                "properties": {
                    "product_type": {
                        "type": "string", "enum": PRODUCT_TYPES,
                        "description": "Only return products of this type.",
                    },
                },
                "additionalProperties": False,
            },
            handler=get_products,
        ),
    ]
