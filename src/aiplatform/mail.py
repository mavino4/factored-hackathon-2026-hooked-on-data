"""Login notices. A successful sign-in emails the configured addresses.

The message names who signed in and from where. It never includes the password.
Delivery uses SMTP (AIP_SMTP_*). If that is not configured, or the server rejects
the message, sign-in still succeeds and the failure is logged.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage

from aiplatform.config import Settings

log = logging.getLogger(__name__)


def send_login_notice(settings: Settings, *, username: str, ip: str | None,
                      smtp_factory=None) -> None:
    recipients = [address for address in settings.login_notify_emails if address]
    if not recipients:
        return
    if not settings.smtp_host or not settings.smtp_from:
        log.warning("login email skipped: set AIP_SMTP_HOST and AIP_SMTP_FROM")
        return
    message = EmailMessage()
    message["Subject"] = f"BankBot sign-in: {username}"
    message["From"] = settings.smtp_from
    message["To"] = ", ".join(recipients)
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    message.set_content(
        f"User {username} signed in at {when}.\n"
        f"Client address: {ip or 'unknown'}.\n")
    try:
        factory = smtp_factory or smtplib.SMTP
        with factory(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
            smtp.ehlo()
            if settings.smtp_tls:
                smtp.starttls()
                smtp.ehlo()
            if settings.smtp_username and settings.smtp_password is not None:
                smtp.login(settings.smtp_username, settings.smtp_password.get_secret_value())
            smtp.send_message(message)
    except Exception:
        log.exception("login email to %s failed", ", ".join(recipients))
