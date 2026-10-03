"""The quick actions of the web UI ("Frequent questions"): one click sends a fixed question.

With ``AIP_QUICK_ACTIONS=intent`` the intent of such a turn comes from the button, not
from the classifier: it is always an account question. With ``direct`` the answer does
not come from the model either: ``render`` fills a fixed text with the ``get_products``
result (agent/graph.py, ``quick_answer``). The button is trusted only when
the text sent is exactly that button's text, in any UI language: anything else (a key
with text the customer wrote) goes through the classifier, which is where manipulation
attempts are caught. The texts must match ``web/i18n.js`` (tests check it).
"""

from dataclasses import dataclass
from typing import Any, Literal

QuickMode = Literal["model", "intent", "direct"]


@dataclass(frozen=True)
class QuickAction:
    key: str
    product_type: str | None  # the products the question is about (None: all)
    texts: dict[str, str]  # UI language -> the button's text


QUICK_ACTIONS: dict[str, QuickAction] = {a.key: a for a in (
    QuickAction("qa_card_balance", "Tarjeta Crédito", {
        "es": "Saldo de mi tarjeta de crédito",
        "pt": "Saldo do meu cartão de crédito",
        "en": "My credit card balance"}),
    QuickAction("qa_savings_balance", "Cuenta Ahorro", {
        "es": "Saldo de mi cuenta de ahorros",
        "pt": "Saldo da minha conta poupança",
        "en": "My savings account balance"}),
    QuickAction("qa_available", "Tarjeta Crédito", {
        "es": "¿Cuánto crédito disponible tengo?",
        "pt": "Quanto crédito disponível eu tenho?",
        "en": "How much credit do I have available?"}),
    QuickAction("qa_overdue", None, {
        "es": "¿Tengo pagos atrasados?",
        "pt": "Tenho pagamentos em atraso?",
        "en": "Do I have overdue payments?"}),
    QuickAction("qa_products", None, {
        "es": "¿Qué productos tengo?",
        "pt": "Quais produtos eu tenho?",
        "en": "What products do I have?"}),
)}


def _canonical(text: str) -> str:
    return " ".join(text.split()).casefold()


def match(key: str | None, text: str) -> QuickAction | None:
    """The quick action ``key`` names, if ``text`` is exactly one of its texts."""
    action = QUICK_ACTIONS.get(key or "")
    if action is None or _canonical(text) not in {_canonical(t) for t in action.texts.values()}:
        return None
    return action


# --- Direct answers (AIP_QUICK_ACTIONS=direct) -------------------------------------------

PRODUCT_NAMES = {
    "es": {"Cuenta Ahorro": "Cuenta de ahorros", "Cuenta Corriente": "Cuenta corriente",
           "Tarjeta Crédito": "Tarjeta de crédito", "Tarjeta Débito": "Tarjeta de débito",
           "Préstamo Personal": "Préstamo personal",
           "Préstamo Hipotecario": "Préstamo hipotecario", "Inversión": "Inversión",
           "Seguro": "Seguro"},
    "pt": {"Cuenta Ahorro": "Conta poupança", "Cuenta Corriente": "Conta corrente",
           "Tarjeta Crédito": "Cartão de crédito", "Tarjeta Débito": "Cartão de débito",
           "Préstamo Personal": "Empréstimo pessoal",
           "Préstamo Hipotecario": "Financiamento imobiliário", "Inversión": "Investimento",
           "Seguro": "Seguro"},
    "en": {"Cuenta Ahorro": "Savings account", "Cuenta Corriente": "Checking account",
           "Tarjeta Crédito": "Credit card", "Tarjeta Débito": "Debit card",
           "Préstamo Personal": "Personal loan", "Préstamo Hipotecario": "Mortgage",
           "Inversión": "Investment", "Seguro": "Insurance"},
}

TEXTS = {
    "es": {
        "not_linked": ("Su usuario no está vinculado a un cliente del banco, así que no puedo "
                       "consultar sus productos. Comuníquese con el banco para vincularlo."),
        "none": "No figura ningún producto de tipo «{type}» a su nombre.",
        "no_products": "No figura ningún producto a su nombre.",
        "card": "{name} {last4}: deuda actual {balance}",
        "limit": "cupo {amount}",
        "available": "disponible {amount}",
        "savings": "{name} {last4}: saldo disponible {balance}",
        "credit": "{name} {last4}: crédito disponible {available}",
        "overdue": "{name} {last4}: {days} días de atraso",
        "no_overdue": "No tiene pagos atrasados en sus productos.",
        "products": "Estos son sus productos:",
        "more": "¿Le ayudo con algo más?",
    },
    "pt": {
        "not_linked": ("Seu usuário não está vinculado a um cliente do banco, então não posso "
                       "consultar seus produtos. Entre em contato com o banco para vinculá-lo."),
        "none": "Não consta nenhum produto do tipo «{type}» em seu nome.",
        "no_products": "Não consta nenhum produto em seu nome.",
        "card": "{name} {last4}: dívida atual {balance}",
        "limit": "limite {amount}",
        "available": "disponível {amount}",
        "savings": "{name} {last4}: saldo disponível {balance}",
        "credit": "{name} {last4}: crédito disponível {available}",
        "overdue": "{name} {last4}: {days} dias de atraso",
        "no_overdue": "O senhor não tem pagamentos em atraso nos seus produtos.",
        "products": "Estes são os seus produtos:",
        "more": "Posso ajudar com mais alguma coisa?",
    },
    "en": {
        "not_linked": ("Your user isn't linked to a bank customer, so I can't look up your "
                       "products. Please contact the bank to link it."),
        "none": "There is no product of type “{type}” in your name.",
        "no_products": "There are no products in your name.",
        "card": "{name} {last4}: current debt {balance}",
        "limit": "limit {amount}",
        "available": "available {amount}",
        "savings": "{name} {last4}: available balance {balance}",
        "credit": "{name} {last4}: available credit {available}",
        "overdue": "{name} {last4}: {days} days past due",
        "no_overdue": "You have no overdue payments on your products.",
        "products": "These are your products:",
        "more": "Can I help you with anything else?",
    },
}


def money(value: float | None, currency: str, language: str) -> str:
    if value is None:
        return "-"
    text = f"{value:,.2f}"
    if language in ("es", "pt"):  # 1.234,56
        text = text.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{currency} {text}"


def render(action: QuickAction, result: dict[str, Any], language: str | None) -> str:
    """The answer to ``action`` from a ``get_products`` result, in the customer's language."""
    lang = language if language in TEXTS else "es"
    texts, names = TEXTS[lang], PRODUCT_NAMES[lang]
    if result.get("status") != "ok":
        return texts["not_linked"]
    products = result.get("products") or []
    if not products:
        if action.product_type is None:
            return texts["no_products"]
        return texts["none"].format(type=names.get(action.product_type, action.product_type))

    def fields(p: dict) -> dict[str, Any]:
        cur = p.get("currency") or ""
        return {"name": names.get(p["type"], p["type"]), "last4": f"•••• {p['number_last4']}",
                "balance": money(p.get("balance"), cur, lang),
                "available": money(p.get("available_credit"), cur, lang),
                "days": p.get("days_past_due")}

    lines: list[str] = []
    if action.key == "qa_card_balance":
        for p in products:
            parts = [texts["card"].format(**fields(p))]
            if p.get("limit") is not None:
                parts.append(texts["limit"].format(amount=money(p["limit"], p["currency"], lang)))
            if p.get("available_credit") is not None:
                parts.append(texts["available"].format(amount=fields(p)["available"]))
            lines.append(" · ".join(parts))
    elif action.key == "qa_savings_balance":
        lines = [texts["savings"].format(**fields(p)) for p in products]
    elif action.key == "qa_available":
        lines = [texts["credit"].format(**fields(p)) for p in products]
    elif action.key == "qa_overdue":
        lines = [texts["overdue"].format(**fields(p)) for p in products
                 if (p.get("days_past_due") or 0) > 0]
        if not lines:
            return f"{texts['no_overdue']} {texts['more']}"
    else:  # qa_products
        lines = [f"{f['name']} {f['last4']}" for f in map(fields, products)]
    head = [texts["products"]] if action.key == "qa_products" else []
    return "\n".join([*head, *(f"- {line}" for line in lines), "", texts["more"]])
