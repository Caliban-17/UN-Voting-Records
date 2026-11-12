"""Sending the newsletter by SMTP: the confirmation message and each edition.

Configured entirely by environment (``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``,
``SMTP_PASS``, ``SMTP_FROM``; ``PUBLIC_URL`` for the links). With no SMTP host
set nothing is sent, and sign-ups stay pending until it is.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
import json
import logging
import os
import smtplib
import ssl

from src import subscribers
from src.newsletter_live import render_live
from src.pulse import UTC, atomic_json, data_dir, date_value, stamp

logger = logging.getLogger(__name__)
PUBLICATION = "UN-Scrupulous"


def configured() -> bool:
    return all(os.getenv(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_FROM"))


def public_url() -> str:
    explicit = os.getenv("PUBLIC_URL") or ""
    domain = os.getenv("RAILWAY_PUBLIC_DOMAIN") or ""
    return (explicit or (f"https://{domain}" if domain else "")).rstrip("/")


def min_days_between_editions() -> int:
    return max(1, int(os.getenv("NEWSLETTER_MIN_DAYS", "7")))


def _send(messages: list[EmailMessage]) -> list[bool]:
    """Send each message; one flag per message, so a rejected recipient can be
    retried without the others being sent the edition twice."""
    host, port = os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT") or "465")
    context = ssl.create_default_context()
    server = smtplib.SMTP_SSL(host, port, context=context, timeout=30) if port == 465 else smtplib.SMTP(host, port, timeout=30)
    sent = []
    with server:
        if port != 465:
            server.starttls(context=context)
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        for message in messages:
            try:
                server.send_message(message)
                sent.append(True)
            except smtplib.SMTPException:
                logger.exception("Could not send to one recipient; continuing")
                sent.append(False)
    return sent


def _envelope(recipient: str, subject: str) -> EmailMessage:
    message = EmailMessage()
    sender = os.environ["SMTP_FROM"]
    message["From"] = sender if "<" in sender else formataddr((PUBLICATION, sender))
    message["To"] = recipient
    message["Subject"] = subject
    message["Message-ID"] = make_msgid(domain=sender.rsplit("@", 1)[-1].strip("> "))
    return message


def confirmation_message(email: str, token: str, base: str) -> EmailMessage:
    link = f"{base}/subscribe/confirm/{token}"
    message = _envelope(email, f"Confirm your subscription to {PUBLICATION}")
    message.set_content(
        f"Someone asked to subscribe this address to {PUBLICATION}, a newsletter on what the United Nations "
        f"is voting on and who stands where.\n\nConfirm here:\n{link}\n\n"
        "If that wasn't you, ignore this message and you'll hear nothing more.\n")
    return message


def send_confirmation(email: str, token: str, base: str | None = None) -> bool:
    if not configured():
        logger.warning("SMTP is not configured; the confirmation for a new sign-up was not sent")
        return False
    return all(_send([confirmation_message(email, token, base or public_url())]))


def edition_message(edition, subscriber: dict, base: str) -> EmailMessage:
    """Plain text plus HTML, with each chart attached as an inline PNG (mail
    clients strip inline SVG) and a one-click unsubscribe header."""
    from src.newsletter_email import _replace_svgs_with_cid_images

    unsubscribe = f"{base}/unsubscribe/{subscriber['token']}"
    footer = [("Unsubscribe", unsubscribe)] + ([("Read on the web", f"{base}/newsletter")] if base else [])
    message = _envelope(subscriber["email"], edition.email_subject)
    message["List-Unsubscribe"] = f"<{unsubscribe}>"
    message["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    message.set_content(render_live(edition, "txt", footer))
    html, charts = _replace_svgs_with_cid_images(render_live(edition, "html", footer))
    message.add_alternative(html, subtype="html")
    part = message.get_payload()[-1]
    for cid, png, _width in charts:
        part.add_related(png, maintype="image", subtype="png", cid=f"<{cid}>", filename=f"{cid}.png")
    return message


def send_edition(edition, recipients: list[dict], directory=None, base: str | None = None) -> int:
    base = base or public_url()
    due = [s for s in recipients if s.get("last_sent_hash") != edition.content_hash]
    if not due or not configured():
        return 0
    outcome = _send([edition_message(edition, s, base) for s in due])
    for subscriber, delivered in zip(due, outcome):
        if delivered:
            subscribers.mark_sent(subscriber["email"], edition.content_hash, directory)
    return sum(outcome)


def deliver_if_due(edition, directory=None, now: datetime | None = None) -> int:
    """Email a changed edition to confirmed subscribers, at most once every
    ``NEWSLETTER_MIN_DAYS`` (a weekly newsletter by default, however often the
    edition itself is recomposed)."""
    directory = directory or data_dir()
    now = now or datetime.now(UTC)
    path = directory / "delivery.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    last = date_value(state.get("sent_at"))
    if state.get("content_hash") == edition.content_hash or not configured():
        return 0
    if last and now - last < timedelta(days=min_days_between_editions()):
        return 0
    recipients = subscribers.confirmed(directory)
    if not recipients:
        return 0
    sent = send_edition(edition, recipients, directory)
    # The edition counts as delivered only once nobody is still owed it. A
    # recipient the server rejected stays unmarked, so the next pass retries
    # just them; recording the hash now would skip them for good.
    owed = [s for s in subscribers.confirmed(directory) if s.get("last_sent_hash") != edition.content_hash]
    if sent and not owed:
        atomic_json(path, {"content_hash": edition.content_hash, "sent_at": stamp(now), "recipients": len(recipients)})
    if sent:
        logger.info("Edition %s emailed to %d subscribers (%d still owed)", edition.content_hash[:12], sent, len(owed))
    return sent
