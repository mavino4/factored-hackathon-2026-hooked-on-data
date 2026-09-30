import pytest

from aiplatform.privacy import Masker, run_scope

M = Masker(b"test-key")
PROFILE = {"status": "ok", "first_name": "María", "city": "Córdoba", "segment": "Premium",
           "products": [{"type": "Tarjeta Crédito", "number_last4": "4821", "currency": "BRL",
                         "balance": 1500.0, "limit": 20000.0, "interest_rate_pct": 12.5,
                         "balance_meaning": "amount currently owed on the card (debt)"}]}


@pytest.mark.parametrize("text, expected", [
    ("CPF 123.456.789-09", "CPF <CPF>"),
    ("CNPJ 12.345.678/0001-90", "CNPJ <CNPJ>"),
    ("RUT 12.345.678-5", "RUT <RUT>"),
    ("DNI 30.123.456", "DNI <DNI>"),
    ("CBU 0110599520000001234567", "CBU <ID>"),
    ("CLABE 002010077777777771", "CLABE <ID>"),
    ("IBAN ES91 2100 0418 4502 0005 1332", "IBAN <IBAN>"),
    ("tarjeta 4111 1111 1111 1111", "tarjeta <CARD>"),  # Luhn-valid
    ("tarjeta 1234 5678 9012 3456", "tarjeta <ID>"),  # mistyped cards are masked too
    ("maria.p@gmail.com", "<EMAIL>"),
    ("tel +55 11 91234-5678", "tel <PHONE>"),
    ("tel (11) 91234-5678", "tel <PHONE>"),
    ("paguei R$ 1.234,56", "paguei <AMOUNT>"),
    ("debo 1,234.56 dólares", "debo <AMOUNT> dólares"),
])
def test_free_text_patterns(text, expected):
    assert M.mask(data=text) == expected


def test_ordinary_text_is_kept():
    text = "¿Cuál es mi saldo? Hoy es 2026-09-29 a las 12:00; tengo 3 cuentas y 12,50 % de tasa."
    assert M.mask(data=text) == text


def test_structured_fields():
    masked = M.mask(data=PROFILE)
    assert masked["first_name"] == "<NAME>" and masked["city"] == "<CITY>"
    product = masked["products"][0]
    assert product["number_last4"] == "<LAST4>"
    assert product["balance"] == "<AMOUNT>" and product["limit"] == "<AMOUNT>"
    # Useful for debugging, not identifying: kept.
    assert masked["segment"] == "Premium"
    assert product["interest_rate_pct"] == 12.5 and product["currency"] == "BRL"


def test_amounts_visible_when_not_masked():
    masked = Masker(b"k", mask_amounts=False).mask(data=PROFILE)
    assert masked["products"][0]["balance"] == 1500.0
    assert masked["first_name"] == "<NAME>"
    assert Masker(b"k", mask_amounts=False).mask(data="R$ 1.234,56") == "R$ 1.234,56"


def test_values_from_a_tool_are_masked_in_later_text():
    with run_scope():
        M.register(PROFILE)
        reply = "Olá María, a senhora deve 1.500,00 no cartão final 4821 (limite 20.000). Córdoba"
        assert M.mask(data=reply) == ("Olá <NAME>, a senhora deve <AMOUNT> no cartão final "
                                      "<LAST4> (limite <AMOUNT>). <CITY>")
    assert M.mask(data="Olá María") == "Olá María"  # the run's values are gone


def test_tool_results_stored_as_json_text():
    import json
    history = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": json.dumps(PROFILE)}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hola María"}]}]
    with run_scope():
        masked = json.dumps(M.mask(data=history), ensure_ascii=False)
    assert "María" not in masked and "4821" not in masked and "1500" not in masked


def test_user_ids_are_stable_keyed_pseudonyms():
    assert M.pseudonym("auth0|1") == M.pseudonym("auth0|1") != M.pseudonym("auth0|2")
    assert Masker(b"other").pseudonym("auth0|1") != M.pseudonym("auth0|1")
    assert M.mask(data={"user_id": "auth0|1"}) == {"user_id": M.pseudonym("auth0|1")}


def test_objects_become_plain_values():
    from dataclasses import dataclass

    @dataclass
    class Conv:
        user_id: str
        title: str

    assert M.mask(data=(Conv("auth0|1", "mi email a@b.co"),)) == [
        {"user_id": M.pseudonym("auth0|1"), "title": "mi email <EMAIL>"}]
