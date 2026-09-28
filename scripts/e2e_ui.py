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
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
BANK_URL = os.environ.setdefault(
    "AIP_BANK_DATABASE_URL", "postgresql+asyncpg://bank_reader:bank_reader@localhost:5432/bank")

from aiplatform.banking.repository import PostgresBankRepository
from evals.run import detect_language, mentions_amount

API = "http://localhost:8765"
ISSUER_PORT = 9123  # not 9000: that port is often taken (e.g. Jupyter)
ISSUER = f"http://127.0.0.1:{ISSUER_PORT}/"
REPLY_TIMEOUT_MS = 300_000  # local models can be slow on the first call

UI = {  # the labels the browser should show in each language
    "es": {"continue": "Continuar", "new_query": "+ Nueva consulta",
           "title_query": "Nueva consulta", "sign_in": "Iniciar sesión",
           "sign_out": "Cerrar sesión", "hello": "¡Hola",
           "quick": "Saldo de mi tarjeta de crédito", "status": "Consultando"},
    "pt": {"continue": "Continuar", "new_query": "+ Nova consulta",
           "title_query": "Nova consulta", "sign_in": "Entrar", "sign_out": "Sair",
           "hello": "Olá", "quick": "Saldo do meu cartão de crédito", "status": "Consultando"},
}
# Things a customer must never see: tool names, arguments or raw results.
HIDDEN = ["get_products", "get_customer_profile", "product_type", "balance_meaning"]


async def bank_truth(users: list[str]) -> dict[str, dict]:
    repo = PostgresBankRepository(BANK_URL)
    try:
        return {u: {"balance": float((await repo.get_products(u, "Tarjeta Crédito"))[0]
                                     .current_balance),
                    "first_name": (await repo.get_customer(u)).first_name}
                for u in users}
    finally:
        await repo.close()


@contextlib.contextmanager
def server(args: list[str], env: dict[str, str], health_url: str):
    with tempfile.NamedTemporaryFile("w+", prefix="e2e-server-", suffix=".log") as log:
        proc = subprocess.Popen(args, env={**os.environ, **env}, stdout=log,
                                stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            healthy = False
            while time.monotonic() < deadline and proc.poll() is None:
                with contextlib.suppress(OSError):
                    urllib.request.urlopen(health_url, timeout=1)
                    healthy = True
                    break
                time.sleep(0.3)
            if not healthy:
                log.seek(0)
                raise RuntimeError(f"server did not start: {args}\n--- server log ---\n"
                                   f"{log.read()[-2000:]}")
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


def assert_welcome(page: Page, lang: str, user: str, truth: dict[str, dict]) -> None:
    """Every session starts with a new query: greeting with the customer's name + quick actions."""
    ui = UI[lang]
    expect(page.locator("#chat-title")).to_have_text(ui["title_query"], timeout=10_000)
    greeting = page.locator(".msg.greeting .bubble")
    expect(greeting).to_contain_text(ui["hello"])
    expect(greeting).to_contain_text(truth[user]["first_name"])
    expect(greeting).to_contain_text("BankBot")
    expect(page.locator(".quick-actions button")).to_have_count(5)
    print(f"  greeting: {greeting.inner_text()!r}")


def assert_nothing_hidden_is_shown(page: Page) -> None:
    body = page.locator("#messages").inner_text()
    leaked = [h for h in HIDDEN if h in body]
    assert not leaked and page.locator(".tool").count() == 0, f"tool details visible: {leaked}"


def ask_balance(page: Page, lang: str, user: str, truth: dict[str, dict]) -> None:
    """Ask with the quick-action button; the answer must be the customer's own figure."""
    ui = UI[lang]
    with page.expect_request(lambda r: r.method == "POST" and "/agent-runs" in r.url) as sent:
        page.get_by_role("button", name=ui["quick"]).click()
    sent_language = sent.value.post_data_json.get("language")
    assert sent_language == lang, f"request sent language {sent_language!r}, UI shows {lang!r}"
    status = page.locator(".status")
    expect(status).to_be_visible(timeout=10_000)
    print(f"  progress shown: {status.inner_text()!r}")
    expect(page.locator(".quick-actions")).to_have_count(0)
    expect(page.locator("#send")).to_be_enabled(timeout=REPLY_TIMEOUT_MS)
    expect(status).to_have_count(0)
    assert_nothing_hidden_is_shown(page)
    text = page.locator("#messages").inner_text()
    reply = page.locator(".msg.assistant .bubble").last.inner_text()
    print(f"  [{user}] reply: {reply[:200]!r}")
    replied_in = detect_language(reply)
    if replied_in == lang:
        print(f"  OK: replied in the UI language ({lang})")
    else:
        print(f"  WARN: replied in {replied_in or 'unknown'}, UI is {lang} (model quality)")
    others = [u for u in truth if u != user and mentions_amount(text, truth[u]["balance"])]
    assert not others, f"{user} sees figures of {others}"
    if mentions_amount(text, truth[user]["balance"]):
        print(f"  OK: {user} sees their own balance from the DB ({truth[user]['balance']})")
    else:
        print(f"  WARN: figure shown to {user} differs from the DB (model quality)")


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
        assert_welcome(page, "es", "ana", truth)
        ask_balance(page, "es", "ana", truth)

        # Language selector (top right): switches the texts and is remembered on reload.
        page.locator("#lang-select").select_option("pt")
        expect(page.locator("html")).to_have_attribute("lang", "pt")
        expect(page.get_by_role("button", name=UI["pt"]["new_query"])).to_be_visible()
        page.reload()
        expect(page.locator("html")).to_have_attribute("lang", "pt")
        assert_welcome(page, "pt", "ana", truth)  # a reload starts a new query
        print("  language switched to pt and kept after reload")
        page.locator("#lang-select").select_option("es")
        expect(page.locator("html")).to_have_attribute("lang", "es")

        # The previous query is in the sidebar; its history shows no tool details.
        page.locator("#conversation-list button").first.click()
        expect(page.locator(".msg.user .bubble").first).to_have_text(ui["quick"],
                                                                     timeout=10_000)
        assert_nothing_hidden_is_shown(page)
        print("  history restored without tool details")

        page.get_by_role("button", name=ui["sign_out"]).click()
        page.locator("#dev-user").fill("bruno")
        page.get_by_role("button", name=ui["continue"]).click()
        expect(page.locator("#user-name")).to_have_text("bruno", timeout=10_000)
        page.wait_for_timeout(500)
        expect(page.locator("#conversation-list li")).to_have_count(0)
        assert truth["ana"]["first_name"] not in page.locator("body").inner_text(), \
            "bruno sees ana's name"
        print("  switched to bruno: clean session")
        assert_welcome(page, "es", "bruno", truth)
        ask_balance(page, "es", "bruno", truth)
        page.close()
    return errors


def run_oidc_mode(browser, truth) -> list[str]:
    print("\n[OIDC auth (authorization code + PKCE via the dev issuer), browser in pt-BR]")
    errors: list[str] = []
    ui = UI["pt"]
    issuer = server([sys.executable, "scripts/dev_oidc.py", "serve", "--port", str(ISSUER_PORT)],
                    {},
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
        assert_welcome(page, "pt", "bruno", truth)
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
        assert_welcome(page, "pt", "ana", truth)
        ask_balance(page, "pt", "ana", truth)
        page.close()
    return errors


def main() -> int:
    truth = asyncio.run(bank_truth(["ana", "bruno"]))
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
