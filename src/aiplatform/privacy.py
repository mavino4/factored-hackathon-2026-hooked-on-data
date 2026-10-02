"""Masking of customer data before it leaves the app in a trace.

``Masker.mask`` takes any trace payload (prompts, replies, tool input and output,
LangGraph state with the full history) and returns a JSON-safe copy where:

- known fields are replaced: first name, city, card/account last 4 digits and, with
  ``mask_amounts``, balances, limits and available credit;
- the values of those fields seen during the run are also replaced wherever they
  appear in free text (e.g. the model's reply "Hola María, tu saldo es 1.234,56");
- free-text PII patterns are replaced: cards (Luhn-checked), CPF/CNPJ, DNI/RUT, IBAN,
  CBU/CLABE and other long digit runs, emails, phones and, with ``mask_amounts``,
  money amounts;
- user IDs (OIDC subjects, usernames) become a keyed hash: stable, so one customer's
  sessions can be grouped, but not reversible without the key. (In password mode the
  trace's own user is the bank customer ID instead, see ``tracing.py``.)

Values seen in a run live in a ContextVar opened with ``run_scope()`` around each graph
run, so runs never share them.
"""

import dataclasses
import hashlib
import hmac
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

NAME, CITY, LAST4, AMOUNT = "<NAME>", "<CITY>", "<LAST4>", "<AMOUNT>"

IDENTITY_KEYS = {"first_name": NAME, "city": CITY, "number_last4": LAST4}
AMOUNT_KEYS = {"balance", "limit", "available_credit", "current_balance", "credit_limit"}
USER_ID_KEYS = {"user_id", "sub", "subject"}

_run_values: ContextVar[dict[str, str] | None] = ContextVar("trace_sensitive_values",
                                                            default=None)


@contextmanager
def run_scope() -> Iterator[None]:
    """Collect sensitive values for one run (see module docstring)."""
    token = _run_values.set({})
    try:
        yield
    finally:
        _run_values.reset(token)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "<CARD>" if _luhn_ok(digits) else "<ID>"  # a mistyped card is still sensitive


SECRET = "<SECRET>"

# Order matters: specific formats first, generic digit runs and amounts last.
_PATTERNS: list[tuple[re.Pattern, Any]] = [
    # Secrets a customer types although asked not to, recognized by the word before them:
    # "mi PIN es 4821", "el código que me llegó por SMS es 884213", "senha: Banco2026!".
    (re.compile(r"(?i)\b(pin|cvv2?|cvc|otp|c[oó]digo|code|token|clave)\b([^\d\n]{0,40}?)"
                r"\d{3,8}(?!\d)"), rf"\1\2{SECRET}"),
    (re.compile(r"(?i)\b(contrase[ñn]a|senha|password|passwd|clave)\b"
                r"(\s*(?:es|é|is|:|=)\s*)\S+"), rf"\1\2{SECRET}"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<EMAIL>"),
    (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"), "<IBAN>"),
    (re.compile(r"(?<![\w-])\d{4}(?:[ -]?\d{4}){2}[ -]?\d{1,7}(?![\w-])"), _card),
    (re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"), "<CNPJ>"),
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "<CPF>"),
    (re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"), "<RUT>"),
    # Not the start of an amount ("2.456.357,90" is money, masked below).
    (re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}\b(?![.,]\d)"), "<DNI>"),
    (re.compile(r"\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){1,3}\d{3,5}[\s.-]?\d{4}\b"), "<PHONE>"),
    (re.compile(r"(?<![\w.,])\(?\d{2,4}\)?[\s.-]?\d{4,5}[\s-]\d{4}\b"), "<PHONE>"),
    (re.compile(r"\b\d{8,}\b"), "<ID>"),  # CBU (22), CLABE (18), unformatted CPF/DNI...
]
_AMOUNT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:R\$|US\$|S/|\$|€|£)\s?-?\d[\d.,]*\d|\b(?:ARS|BRL|CLP|COP|EUR|MXN|PEN|"
                r"USD|UYU)\s?-?\d[\d.,]*\d?"), AMOUNT),
    # 1.234,56  1,234.56  1234.56  1234,5 - not percentages ("12,50 %").
    (re.compile(r"(?<![\w.,])-?\d{1,3}(?:[.,]\d{3})*[.,]\d{1,2}(?![\d.,]*\s?%)(?![\w])"
                r"|(?<![\w.,])-?\d+[.,]\d{1,2}(?![\d.,]*\s?%)(?![\w])"), AMOUNT),
]


def _amount_forms(value: float) -> set[str]:
    """How an amount may be written in es/pt/en text."""
    en = f"{value:,.2f}"  # 1,234.56
    forms = {en, en.replace(",", ""), en.translate(str.maketrans(",.", ".,")),
             en.replace(",", "").replace(".", ",")}
    if value == int(value):
        whole = f"{int(value):,}"
        forms |= {whole, whole.replace(",", ""), whole.replace(",", ".")}
    return {f for f in forms if len(f.lstrip("-")) >= 3}  # never mask tiny numbers like "0"


class Masker:
    def __init__(self, hash_key: bytes, *, mask_amounts: bool = True):
        self._hash_key = hash_key
        self._mask_amounts = mask_amounts

    def pseudonym(self, user_id: str) -> str:
        digest = hmac.new(self._hash_key, user_id.encode(), hashlib.sha256).hexdigest()
        return f"u_{digest[:16]}"

    def mask(self, *, data: Any, **_: Any) -> Any:
        """Langfuse ``mask`` hook. If it raises, Langfuse drops the payload (fail closed)."""
        data = _plain(data)
        values = _run_values.get()
        if values is None:
            values = {}
        self._collect(data, values)
        return self._mask(data, _compile(values))

    def register(self, data: Any) -> None:
        """Remember the sensitive values in a tool output for the rest of the run."""
        values = _run_values.get()
        if values is not None:
            self._collect(_plain(data), values)

    def _collect(self, data: Any, values: dict[str, str]) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                if key in IDENTITY_KEYS and isinstance(value, str) and len(value) >= 2:
                    values[value] = IDENTITY_KEYS[key]
                elif (key in AMOUNT_KEYS and self._mask_amounts
                      and isinstance(value, int | float) and not isinstance(value, bool)):
                    for form in _amount_forms(float(value)):
                        values[form] = AMOUNT
                else:
                    self._collect(value, values)
        elif isinstance(data, list):
            for item in data:
                self._collect(item, values)
        elif isinstance(data, str) and (parsed := _json(data)) is not None:
            self._collect(parsed, values)

    def _mask(self, data: Any, known: re.Pattern | None, key: str | None = None) -> Any:
        if key in USER_ID_KEYS and isinstance(data, str):
            return self.pseudonym(data)
        if key == "model":  # a model ID's date is not an ID number
            return data
        if key in IDENTITY_KEYS and data is not None:
            return IDENTITY_KEYS[key]
        if key in AMOUNT_KEYS and self._mask_amounts and data is not None:
            return AMOUNT
        if isinstance(data, dict):
            return {k: self._mask(v, known, k) for k, v in data.items()}
        if isinstance(data, list):
            return [self._mask(v, known) for v in data]
        if isinstance(data, str):
            if (parsed := _json(data)) is not None:  # e.g. a tool result stored as JSON
                return json.dumps(self._mask(parsed, known), ensure_ascii=False)
            return self._mask_text(data, known)
        return data

    def _mask_text(self, text: str, known: re.Pattern | None) -> str:
        if known is not None:
            text = known.sub(lambda m: _KNOWN_LABELS[m.lastgroup], text)
        for pattern, replacement in _PATTERNS:
            text = pattern.sub(replacement, text)
        if self._mask_amounts:
            for pattern, replacement in _AMOUNT_PATTERNS:
                text = pattern.sub(replacement, text)
        return text


_KNOWN_LABELS = {"name": NAME, "city": CITY, "last4": LAST4, "amount": AMOUNT}
_GROUPS = {NAME: "name", CITY: "city", LAST4: "last4", AMOUNT: "amount"}


def _compile(values: dict[str, str]) -> re.Pattern | None:
    """One regex with a named group per kind of value (longest values first)."""
    by_label: dict[str, list[str]] = {}
    for value, label in values.items():
        by_label.setdefault(label, []).append(value)
    parts = []
    for label, items in by_label.items():
        alternatives = "|".join(re.escape(v) for v in sorted(items, key=len, reverse=True))
        # Whole words / whole numbers only.
        parts.append(rf"(?P<{_GROUPS[label]}>(?<![\w.,])(?:{alternatives})(?![\w]|[.,]\d))")
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


def _json(text: str) -> Any:
    if not text[:1] in ("{", "["):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict | list) else None


def _plain(data: Any) -> Any:
    """Dataclasses, pydantic models and tuples as plain JSON-like values."""
    if isinstance(data, dict):
        return {str(k): _plain(v) for k, v in data.items()}
    if isinstance(data, list | tuple | set):
        return [_plain(v) for v in data]
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        return _plain({f.name: getattr(data, f.name) for f in dataclasses.fields(data)})
    if hasattr(data, "model_dump"):
        return _plain(data.model_dump(mode="json"))
    if data is None or isinstance(data, str | int | float | bool):
        return data
    return str(data)
