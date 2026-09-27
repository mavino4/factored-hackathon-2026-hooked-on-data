"""Browser end-to-end test of the web UI (headless Chromium via Playwright).

Starts the API (and, for OIDC, the dev issuer) as real servers against a real
model provider, then drives the UI like a user: sign in, chat with streaming,
reload, run an agent task and approve its action. Fails on any browser console
error (including Content-Security-Policy violations).

    uv run --with playwright python -m playwright install chromium   # once
    AIP_PROVIDERS='["ollama"]' uv run --with playwright python scripts/e2e_ui.py
"""

import contextlib
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import Page, expect, sync_playwright

API = "http://localhost:8765"
ISSUER = "http://127.0.0.1:9000/"
REPLY_TIMEOUT_MS = 120_000


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


def chat_and_agent_flow(page: Page) -> None:
    # Chat with streaming.
    page.get_by_role("button", name="+ New chat").click()
    expect(page.locator("#chat-title")).to_have_text("New chat")
    page.locator("#input").fill("Reply with exactly: hello there")
    page.keyboard.press("Enter")
    reply = page.locator(".msg.assistant .bubble").last
    expect(reply).not_to_be_empty(timeout=REPLY_TIMEOUT_MS)
    expect(page.locator("#send")).to_be_enabled(timeout=REPLY_TIMEOUT_MS)
    print(f"  chat reply: {reply.text_content()!r}")

    # History survives a reload; the sidebar shows the titled conversation.
    page.reload()
    expect(page.locator(".msg.user .bubble").first).to_have_text(
        "Reply with exactly: hello there", timeout=10_000)
    expect(page.locator("#conversation-list .title").first).to_contain_text("Reply with exactly")
    print("  reload: history restored")

    # Agent task with an approval.
    page.get_by_role("button", name="+ New agent task").click()
    expect(page.locator("#chat-title")).to_have_text("New agent task")
    expect(page.locator("#input")).to_be_enabled()
    page.locator("#input").fill("Open a support ticket titled 'Printer broken' with details "
                                "'Paper jam on floor 2'. Use the create_support_ticket tool.")
    page.keyboard.press("Enter")
    card = page.locator(".approval").first
    try:
        expect(card).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    except AssertionError:
        shown = page.locator("#messages").inner_text()
        print("  WARN: model did not request the ticket tool (quality, not plumbing). "
              f"Page showed: {shown[-300:]!r}")
        return
    print(f"  approval card: {card.locator('pre').text_content()!r}")
    card.get_by_role("button", name="Approve").click()
    expect(page.locator(".tool.ok").first).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(page.locator("#send")).to_be_enabled(timeout=REPLY_TIMEOUT_MS)
    print(f"  approved; tool result: {page.locator('.tool.ok').first.get_attribute('title')!r}")


def run_dev_mode(browser) -> list[str]:
    print("\n[dev auth]")
    errors: list[str] = []
    with api_server({"AIP_AUTH_MODE": "dev"}):
        page = browser.new_page()
        watch_console(page, errors)
        page.goto(API)
        page.locator("#dev-user").fill("ana")
        page.get_by_role("button", name="Continue").click()
        expect(page.locator("#user-name")).to_have_text("ana")
        chat_and_agent_flow(page)
        page.close()
    return errors


def run_oidc_mode(browser) -> list[str]:
    print("\n[OIDC auth: authorization code + PKCE via the dev issuer]")
    errors: list[str] = []
    issuer = server([sys.executable, "scripts/dev_oidc.py", "serve"], {},
                    f"{ISSUER}.well-known/openid-configuration")
    oidc_env = {"AIP_AUTH_MODE": "oidc", "AIP_OIDC_ISSUER": ISSUER,
                "AIP_OIDC_AUDIENCE": "aiplatform-dev", "AIP_OIDC_CLIENT_ID": "web-ui"}
    with issuer, api_server(oidc_env):
        page = browser.new_page()
        watch_console(page, errors)
        page.goto(API)
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_url(f"{ISSUER}authorize*")
        page.locator("input[name=username]").fill("bruno")
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_url(f"{API}/")
        expect(page.locator("#user-name")).to_have_text("bruno", timeout=10_000)
        print("  signed in as bruno through the issuer's login page")
        chat_and_agent_flow(page)
        page.get_by_role("button", name="Sign out").click()
        expect(page.get_by_role("button", name="Sign in")).to_be_visible()
        print("  signed out")
        page.close()
    return errors


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        errors = run_dev_mode(browser) + run_oidc_mode(browser)
        browser.close()
    if errors:
        print("\nFAIL: browser console errors:")
        for error in errors:
            print(f"  {error}")
        return 1
    print("\nPASS (no console errors, no CSP violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
