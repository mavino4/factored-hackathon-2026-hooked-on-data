"""The quick actions of the web UI ("Frequent questions"): one click sends a fixed question.

With ``AIP_QUICK_ACTIONS=intent`` the intent of such a turn comes from the button, not
from the classifier: it is always an account question. The button is trusted only when
the text sent is exactly that button's text, in any UI language: anything else (a key
with text the customer wrote) goes through the classifier, which is where manipulation
attempts are caught. The texts must match ``web/i18n.js`` (tests check it).
"""

from dataclasses import dataclass
from typing import Literal

QuickMode = Literal["model", "intent"]


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
