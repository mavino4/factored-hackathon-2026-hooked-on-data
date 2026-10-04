"""What the voice says for the figures in an answer. The screen shows the answer as written
("1.200.000,00 COP"); just before it is synthesized, amounts, rates, dates and other
numbers become words in the customer's language ("un millón doscientos mil pesos"), so
the speech model never reads them digit by digit or guesses which separator is the
decimal one. The last digits of a product are the exception: people say them in pairs
("terminada en cincuenta y ocho cincuenta y nueve").
"""

import re

from num2words import num2words

LANGS = {"es": "es", "pt": "pt_BR", "en": "en"}

# (singular, plural) per currency and language.
CURRENCIES = {
    "es": {"COP": ("peso", "pesos"), "USD": ("dólar", "dólares"), "BRL": ("real", "reales"),
           "EUR": ("euro", "euros"), "MXN": ("peso mexicano", "pesos mexicanos")},
    "pt": {"COP": ("peso colombiano", "pesos colombianos"), "USD": ("dólar", "dólares"),
           "BRL": ("real", "reais"), "EUR": ("euro", "euros"),
           "MXN": ("peso mexicano", "pesos mexicanos")},
    "en": {"COP": ("Colombian peso", "Colombian pesos"), "USD": ("dollar", "dollars"),
           "BRL": ("real", "reais"), "EUR": ("euro", "euros"),
           "MXN": ("Mexican peso", "Mexican pesos")},
}
SYMBOLS = {"US$": "USD", "R$": "BRL", "€": "EUR"}
# A bare "$" is the local currency of the language.
DOLLAR_SIGN = {"es": "COP", "pt": "BRL", "en": "USD"}
CENTS = {"es": ("centavo", "centavos"), "pt": ("centavo", "centavos"), "en": ("cent", "cents")}
AND_CENTS = {"es": "con", "pt": "e", "en": "and"}
POINT = {"es": "coma", "pt": "vírgula", "en": "point"}
PERCENT = {"es": "por ciento", "pt": "por cento", "en": "percent"}
MONTHS = {
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"],
    "pt": ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
           "setembro", "outubro", "novembro", "dezembro"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
}

_NUMBER = r"\d{1,3}(?:[.,  ]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
_CODES = "|".join(CURRENCIES["es"])
_MONEY = re.compile(
    rf"(?P<symbol>US\$|R\$|\$|€)\s?(?P<a>{_NUMBER})(?:\s?(?P<code1>{_CODES})\b)?"
    rf"|(?<![\w.,])(?P<b>{_NUMBER})\s?(?P<code2>{_CODES})\b")
_LAST_DIGITS = re.compile(
    r"(?P<lead>\b(?:terminad[oa]s?|termina|finalizad[oa]|acabad[oa]|ending|ends)\s+"
    r"(?:en|em|in)\s+|\bfinal\s+|\búltimos\s+(?:4|cuatro|quatro)\s+(?:dígitos\s+)?)"
    r"(?P<digits>\d{4})\b", re.IGNORECASE)
_PERCENT = re.compile(r"(?<![\w.,])(?P<n>\d+(?:[.,]\d+)?)\s?%")
_DATE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_PLAIN = re.compile(rf"(?<![\w.,])(?:{_NUMBER})(?![\w])")


def spoken(text: str, language: str | None = None) -> str:
    """``text`` with its figures in words, ready to be read aloud."""
    lang = language if language in LANGS else "es"
    text = _LAST_DIGITS.sub(lambda m: m["lead"] + _pairs(m["digits"], lang), text)
    text = _DATE.sub(lambda m: _date(m, lang) or m.group(), text)
    text = _MONEY.sub(lambda m: _money(m, lang), text)
    text = _PERCENT.sub(lambda m: f"{_decimal(m['n'], lang)} {PERCENT[lang]}", text)
    return _PLAIN.sub(lambda m: _plain(m, lang), text)


def split_number(number: str) -> tuple[int, str]:
    """``"1.234.567,89"`` -> (1234567, "89"). With both separators the last one is the
    decimal one; with one, groups of exactly three digits after it mean thousands."""
    number = re.sub(r"[  ]", "", number)
    seps = [c for c in number if c in ".,"]
    if not seps:
        return int(number), ""
    if len(set(seps)) == 2:
        decimal = seps[-1]
    else:
        parts = number.split(seps[0])
        thousands = all(len(p) == 3 for p in parts[1:]) and len(parts[0]) <= 3 \
            and (len(parts) > 2 or len(parts[0]) > 0 and parts[0] != "0")
        decimal = None if thousands else seps[0]
    if decimal is None:
        return int(re.sub(r"[.,]", "", number)), ""
    whole, _, fraction = number.rpartition(decimal)
    return int(re.sub(r"[.,]", "", whole) or 0), fraction


def words(n: int, lang: str) -> str:
    said = num2words(n, lang=LANGS[lang]).replace(",", "")
    if lang == "es":  # "veintiún millones", "un millón", not "veintiuno millones"
        said = re.sub(r"\b(veinti)?uno(?= (?:mil|millón|millones|billón|billones)\b)",
                      lambda m: "veintiún" if m[1] else "un", said)
    return said


def _money(m: re.Match, lang: str) -> str:
    number = m["a"] or m["b"]
    code = m["code1"] or m["code2"] or SYMBOLS.get(m["symbol"] or "") or DOLLAR_SIGN[lang]
    whole, fraction = split_number(number)
    cents = int(fraction.ljust(2, "0")) if fraction else 0
    singular, plural = CURRENCIES[lang][code]
    cent_one, cent_many = CENTS[lang]
    if whole == 0 and cents:  # "cinco centavos", not "cero pesos con cinco centavos"
        return f"{_before_noun(words(cents, lang), lang)} {cent_one if cents == 1 else cent_many}"
    amount = _before_noun(words(whole, lang), lang)
    if lang in ("es", "pt") and whole >= 1_000_000 and whole % 1_000_000 == 0:
        amount += " de"  # "un millón de pesos", "dois milhões de reais"
    said = f"{amount} {singular if whole == 1 else plural}"
    if cents:
        said += (f" {AND_CENTS[lang]} {_before_noun(words(cents, lang), lang)} "
                 f"{cent_one if cents == 1 else cent_many}")
    return said


def _before_noun(number: str, lang: str) -> str:
    """Spanish shortens "uno" before a noun: "un peso", "veintiún pesos"."""
    if lang != "es":
        return number
    return re.sub(r"veintiuno$", "veintiún", re.sub(r"\buno$", "un", number))


def _plain(m: re.Match, lang: str) -> str:
    said = _decimal(m.group(), lang)
    # Before a word it counts something: "un producto", "veintiún días".
    return _before_noun(said, lang) if re.match(r" [^\W\d]", m.string[m.end():]) else said


def _decimal(number: str, lang: str) -> str:
    whole, fraction = split_number(number)
    said = words(whole, lang)
    if fraction:
        leading = len(fraction) - len(fraction.lstrip("0"))
        rest = fraction.lstrip("0")
        tail = [words(0, lang)] * leading + ([words(int(rest), lang)] if rest else [])
        said += f" {POINT[lang]} " + " ".join(tail)
    return said


def _pairs(digits: str, lang: str) -> str:
    """Last digits as people say them: "5859" -> "58 59" in words; a leading zero is said."""
    groups = [digits[:2], digits[2:]]
    return " ".join(" ".join(words(int(d), lang) for d in g) if g.startswith("0")
                    else words(int(g), lang) for g in groups)


def _date(m: re.Match, lang: str) -> str | None:
    year, month, day = int(m["y"]), int(m["m"]), int(m["d"])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    name = MONTHS[lang][month - 1]
    if lang == "en":
        return f"{name} {num2words(day, lang='en', to='ordinal')}, {words(year, lang)}"
    joiner = "de"
    return f"{words(day, lang)} {joiner} {name} {joiner} {words(year, lang)}"
