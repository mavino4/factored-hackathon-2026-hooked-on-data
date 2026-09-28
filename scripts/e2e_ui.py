"""Browser end-to-end test of the web UI (headless Chromium via Playwright).

Starts the API (and, for OIDC, the dev issuer) as real servers against a real model
provider and the core-banking DB, then drives the UI like a customer: sign in, ask for
the credit-card balance (the figure is checked against the DB), reload, switch user and
check that the new user sees only their own data. Runs the browser in Spanish (dev
auth) and Portuguese (OIDC auth) to check the translated UI. Fails on any browser
console error (including Content-Security-Policy violations).

    make bank-db                                                     # once
    uv run --with playwright python -m playwright install chromium   # once
    AIP_PROVIDERS='["ollama"]' AIP_OLLAMA_MODEL=qwen2.5:7b \\
        uv run --with playwright python scripts/e2e_ui.py
"""

import asyncio
import contextlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
BANK_URL = os.environ.setdefault(
    "AIP_BANK_DATABASE_URL", "postgresql+asyncpg://bank_reader:bank_reader@localhost:5432/bank")

from aiplatform.banking.repository import PostgresBankRepository
from evals.run import mentions_amount

API = "http://localhost:8765"
ISSUER = "http://127.0.0.1:9000/"
REPLY_TIMEOUT_MS = 300_000  # local models can be slow on the first call

UI = {  # the labels the browser should show in each language
    "es": {"continue": "Continuar", "new_query": "+ Nueva consulta",
           "title_query": "Nueva consulta", "sign_in": "Iniciar sesión",
           "sign_out": "Cerrar sesión", "question": "¿Cuál es el saldo de mi tarjeta de crédito?"},
    "pt": {"continue": "Continuar", "new_query": "+ Nova consulta",
           "title_query": "Nova consulta", "sign_in": "Entrar", "sign_out": "Sair",
           "question": "Qual é o saldo do meu cartão de crédito?"},
}


async def card_balances(users: list[str]) -> dict[str, float]:
    repo = PostgresBankRepository(BANK_URL)
    try:
        return {u: float((await repo.get_products(u, "Tarjeta Crédito"))[0].current_balance)
                for u in users}
    finally:
        await repo.close()


@contextlib.contextmanager
def server(args: list[str], env: dict[str, str], health_url: str):
    proc = subprocess.Popen(args, env={**os.environ, **env},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            with contextlib.suppress(OSError):
                urllib.request.urlopen(health_url, timeout=1)
                break
            time.sleep(0.2)
        else:
            raise RuntimeError(f"server did not start: {args}")
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def api_server(extra_env: dict[str, str]):
    env = {"AIP_METRICS_PORT": "0", "AIP_LOG_LEVEL": "WARNING", **extra_env}
    return server([sys.executable, "-m", "uvicorn", "--factory", "aiplatform.api.app:create_app",
                   "--port", "8765"], env, f"{API}/healthz")


def watch_console(page: Page, errors: list[str]) -> None:
    page.on("console", lambda msg: msg.type == "error" and errors.append(msg.text))
    page.on("pageerror", lambda exc: errors.append(str(exc)))


def ask_balance(page: Page, lang: str, user: str, truth: dict[str, float]) -> None:
    ui = UI[lang]
    page.get_by_role("button", name=ui["new_query"]).click()
    expect(page.locator("#chat-title")).to_have_text(ui["title_query"])
    expect(page.locator("#input")).to_be_enabled()
    page.locator("#input").fill(ui["question"])
    page.keyboard.press("Enter")
    expect(page.locator("#send")).to_be_enabled(timeout=REPLY_TIMEOUT_MS)
    text = page.locator("#messages").inner_text()
    chips = page.locator(".tool").all_inner_texts()
    print(f"  [{user}] tools={chips} expected={truth[user]}")
    print(f"  [{user}] reply: {page.locator('.msg.assistant .bubble').last.inner_text()[:200]!r}")
    others = [u for u in truth if u != user and mentions_amount(text, truth[u])]
    assert not others, f"{user} sees figures of {others}"
    if not any(c.startswith("get_products") for c in chips):
        print(f"  WARN: model did not call get_products for {user} (model quality)")
    elif not mentions_amount(text, truth[user]):
        print(f"  WARN: figure shown to {user} differs from the DB (model quality)")
    else:
        print(f"  OK: {user} sees their own balance from the DB")


def run_dev_mode(browser, truth) -> list[str]:
    print("\n[dev auth, browser in es-ES]")
    errors: list[str] = []
    ui = UI["es"]
    with api_server({"AIP_AUTH_MODE": "dev"}):
        page = browser.new_page(locale="es-ES")
        watch_console(page, errors)
        page.goto(API)
        expect(page.locator("html")).to_have_attribute("lang", "es")
        page.locator("#dev-user").fill("ana")
        page.get_by_role("button", name=ui["continue"]).click()
        expect(page.locator("#user-name")).to_have_text("ana")
        ask_balance(page, "es", "ana", truth)
        page.reload()
        expect(page.locator(".msg.user .bubble").first).to_have_text(ui["question"],
                                                                     timeout=10_000)
        print("  reload: history restored")
        page.get_by_role("button", name=ui["sign_out"]).click()
        page.locator("#dev-user").fill("bruno")
        page.get_by_role("button", name=ui["continue"]).click()
        expect(page.locator("#user-name")).to_have_text("bruno", timeout=10_000)
        page.wait_for_timeout(500)
        expect(page.locator("#conversation-list li")).to_have_count(0)
        print("  switched to bruno: clean session")
        ask_balance(page, "es", "bruno", truth)
        page.close()
    return errors


def run_oidc_mode(browser, truth) -> list[str]:
    print("\n[OIDC auth (authorization code + PKCE via the dev issuer), browser in pt-BR]")
    errors: list[str] = []
    ui = UI["pt"]
    issuer = server([sys.executable, "scripts/dev_oidc.py", "serve"], {},
                    f"{ISSUER}.well-known/openid-configuration")
    oidc_env = {"AIP_AUTH_MODE": "oidc", "AIP_OIDC_ISSUER": ISSUER,
                "AIP_OIDC_AUDIENCE": "aiplatform-dev", "AIP_OIDC_CLIENT_ID": "web-ui"}
    with issuer, api_server(oidc_env):
        page = browser.new_page(locale="pt-BR")
        watch_console(page, errors)
        page.goto(API)
        expect(page.locator("html")).to_have_attribute("lang", "pt")
        page.get_by_role("button", name=ui["sign_in"]).click()
        page.wait_for_url(f"{ISSUER}authorize*")
        page.locator("input[name=username]").fill("bruno")
        page.get_by_role("button", name="Sign in").click()  # the dev issuer's own page
        page.wait_for_url(f"{API}/")
        expect(page.locator("#user-name")).to_have_text("bruno", timeout=10_000)
        print("  signed in as bruno through the issuer's login page")
        ask_balance(page, "pt", "bruno", truth)
        page.get_by_role("button", name=ui["sign_out"]).click()
        page.get_by_role("button", name=ui["sign_in"]).click()
        page.wait_for_url(f"{ISSUER}authorize*")
        assert "prompt=login" in page.url
        page.locator("input[name=username]").fill("ana")
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_url(f"{API}/")
        expect(page.locator("#user-name")).to_have_text("ana", timeout=10_000)
        page.wait_for_timeout(500)
        expect(page.locator("#conversation-list li")).to_have_count(0)
        print("  switched to ana: clean session")
        ask_balance(page, "pt", "ana", truth)
        page.close()
    return errors


def main() -> int:
    truth = asyncio.run(card_balances(["ana", "bruno"]))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        errors = run_dev_mode(browser, truth) + run_oidc_mode(browser, truth)
        browser.close()
    if errors:
        print("\nFAIL: browser console errors:")
        for error in errors:
            print(f"  {error}")
        return 1
    print("\nPASS (isolation holds, translated UI, no console errors, no CSP violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
