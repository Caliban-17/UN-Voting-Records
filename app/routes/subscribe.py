"""The pages a visitor can reach without the password: sign in, or subscribe
to the newsletter instead (no account needed), confirm and unsubscribe."""

import json
import logging
import re

from flask import Blueprint, Response, jsonify, redirect, request, session

from app.middleware import _client_fingerprint, check_password, login_attempt_allowed, site_password

from src import mailer, subscribers
from src.pulse import data_dir

logger = logging.getLogger(__name__)
bp = Blueprint("subscribe", __name__)

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>UN-Scrupulous</title><link rel="icon" href="/favicon.svg" type="image/svg+xml"><link rel="alternate icon" href="/favicon.ico"><link rel="apple-touch-icon" href="/apple-touch-icon.png"></head>
<body style="margin:0;background:#f3f1eb;padding:40px 16px;font-family:Georgia,serif;color:#102c35">
<main style="max-width:520px;margin:0 auto;background:#fffdf8;padding:32px 28px">
<p style="font:12px Arial,sans-serif;letter-spacing:1px;text-transform:uppercase;color:#5d6f73;border-bottom:3px solid #102c35;padding-bottom:14px;margin:0">UN-Scrupulous</p>
<h1 style="font-size:30px;line-height:1.15;margin:22px 0 12px">{title}</h1>
<p style="font-size:17px;line-height:1.55;color:#243d44">{body}</p>{form}
</main></body></html>"""
FORM = """<form method="post" action="/subscribe" style="margin-top:20px">
<label for="email" style="font:13px Arial,sans-serif;color:#5d6f73;display:block;margin-bottom:6px">Email address</label>
<input id="email" name="email" type="email" required maxlength="254" autocomplete="email" placeholder="name@example.org"
 style="width:100%;box-sizing:border-box;font:16px Arial,sans-serif;padding:11px;border:1px solid #9fb3b0;background:#fff">
<input name="website" tabindex="-1" autocomplete="off" aria-hidden="true" style="position:absolute;left:-9999px">
<button type="submit" style="margin-top:12px;font:700 15px Arial,sans-serif;padding:11px 18px;border:0;background:#102c35;color:#fffdf8;cursor:pointer">Subscribe</button>
<p style="font:12.5px/1.5 Arial,sans-serif;color:#5d6f73;margin-top:14px">One email a week at most. You confirm by email first, and every issue has an unsubscribe link.</p>
</form>"""
SENT = ("Check your inbox", "If that address isn't already subscribed, a confirmation link is on its way. "
        "Nothing more is sent until you follow it.")


LOGIN = """<form method="post" action="/login" style="margin-top:20px">
<input type="hidden" name="next" value="{next}">
<label for="password" style="font:13px Arial,sans-serif;color:#5d6f73;display:block;margin-bottom:6px">Password</label>
<input id="password" name="password" type="password" required autocomplete="current-password" autofocus
 style="width:100%;box-sizing:border-box;font:16px Arial,sans-serif;padding:11px;border:1px solid #9fb3b0;background:#fff">
<button type="submit" style="margin-top:12px;font:700 15px Arial,sans-serif;padding:11px 18px;border:0;background:#102c35;color:#fffdf8;cursor:pointer">Sign in</button>
</form>
<h2 style="font:700 13px Arial,sans-serif;letter-spacing:1.2px;text-transform:uppercase;border-top:1px solid #102c35;padding-top:14px;margin:34px 0 0">No password?</h2>
<p style="font-size:17px;line-height:1.55;color:#243d44;margin:10px 0 0">Get the weekly newsletter by email instead.</p>"""


def _local(target: str) -> str:
    """Only ever redirect within this site."""
    return target if target.startswith("/") and not target.startswith("//") and "\\" not in target else "/"


def login_page(target="/", error="", status=200):
    from html import escape
    body = "The research site is private." + (f' <strong style="color:#c8553d">{error}</strong>' if error else "")
    return page("Sign in", body, LOGIN.format(next=escape(_local(target), quote=True)) + FORM, status)


@bp.get("/login")
def login_form():
    if not site_password() or session.get("signed_in"):
        return redirect(_local(request.args.get("next", "/")))
    return login_page(request.args.get("next", "/"))


@bp.post("/login")
def login_submit():
    target = request.form.get("next", "/")
    if not login_attempt_allowed():
        return login_page(target, "Too many attempts. Wait a minute and try again.", 429)
    if not check_password(request.form.get("password", "")):
        return login_page(target, "That password isn't right.", 401)
    session.clear()
    session["signed_in"] = True
    session.permanent = True
    return redirect(_local(target))


@bp.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect("/login")


def page(title, body, form="", status=200):
    return Response(PAGE.format(title=title, body=body, form=form), status=status, mimetype="text/html")


@bp.get("/subscribe")
def subscribe_form():
    return page("What the UN voted on, and who stood where",
                "A weekly reading of the General Assembly and Security Council record: the vote that mattered, "
                "how it compares with last year, and what is due next.", FORM)


@bp.post("/subscribe")
def subscribe_submit():
    if request.form.get("website"):  # a field only a script fills in
        return page(*SENT)
    email = subscribers.valid_email(request.form.get("email", ""))
    if not email:
        return page("That address doesn't look right", "Check it and try again.", FORM, 400)
    if not subscribers.allow_signup(_client_fingerprint()):
        return page("Too many attempts", "Try again in an hour.", status=429)
    # Nobody can use the form to pelt one address, or the world, with
    # confirmation messages sent in our name.
    if not subscribers.allow_signup("address:" + email, limit=2) or \
            not subscribers.allow_signup("everyone", limit=subscribers.SIGNUPS_PER_HOUR_GLOBAL):
        return page(*SENT)
    row = subscribers.add(email)
    if row:
        base = mailer.public_url() or request.url_root.rstrip("/")
        try:
            mailer.send_confirmation(email, row["token"], base)
        except Exception:
            logger.exception("Confirmation email failed")
    return page(*SENT)


@bp.get("/subscribe/confirm/<token>")
def subscribe_confirm(token):
    row = subscribers.confirm(token)
    if not row:
        return page("That link has expired", "Sign up again to get a new one.", FORM, 404)
    try:  # a new subscriber gets the current edition straight away
        from src.newsletter import edition_from_dict
        edition = edition_from_dict(json.loads((data_dir() / "newsletter.json").read_text()))
        mailer.send_edition(edition, [row], base=mailer.public_url() or request.url_root.rstrip("/"))
    except Exception:
        logger.exception("Could not send the current edition to a new subscriber")
    return page("You're subscribed", "The current edition is on its way, and the next will follow within a week.")


UNSUBSCRIBE = """<form method="post" action="/unsubscribe/{token}" style="margin-top:20px">
<button type="submit" style="font:700 15px Arial,sans-serif;padding:11px 18px;border:0;background:#102c35;color:#fffdf8;cursor:pointer">Unsubscribe</button>
</form>"""


@bp.get("/unsubscribe/<token>")
def unsubscribe_ask(token):
    """A link in an email is opened by scanners and previewers as well as by
    people, so following it only asks; the button (or a mail client's
    one-click request, which is a POST) does the unsubscribing."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", token):
        return page("That link isn't valid", "Use the unsubscribe link from a recent issue.", status=404)
    return page("Unsubscribe?", "You'll stop receiving UN-Scrupulous.", UNSUBSCRIBE.format(token=token))


@bp.post("/unsubscribe/<token>")
def unsubscribe(token):
    subscribers.unsubscribe(token)
    return page("You're unsubscribed", "You won't receive any more emails from UN-Scrupulous.")


# ── mail credentials, handed over without anyone reading them ───────────────


@bp.post("/admin/mail")
def provision_mail():
    """Store SMTP settings sent by the repository's own workflow, which holds
    them as secrets. Needs the shared PROVISION_TOKEN; never echoes a value."""
    import hmac
    import os

    token = os.getenv("PROVISION_TOKEN")
    given = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        return jsonify({"error": "Not found"}), 404
    if not _client_ok() or not hmac.compare_digest(given.encode(), token.encode()):
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(silent=True) or {}
    try:
        stored = mailer.store_settings(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    result = {"stored": stored, "configured": mailer.configured()}
    if data.get("test") and mailer.configured():
        try:
            result["test_sent"] = mailer.send_test()
        except Exception as exc:
            logger.exception("Test message failed")
            result.update(test_sent=False, test_error=type(exc).__name__)
    return jsonify(result)


def _sender_authorised() -> bool:
    import hmac
    import os

    token = os.getenv("PROVISION_TOKEN") or ""
    given = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return bool(token) and hmac.compare_digest(given.encode(), token.encode())


@bp.get("/admin/mail/outbox")
def outbox():
    """Queued messages for the send-mail workflow (base64 MIME). Token only."""
    import base64

    if not _sender_authorised():
        return jsonify({"error": "Forbidden"}), 403
    return jsonify({"messages": [{"id": m["id"], "mime": base64.b64encode(m["message"]).decode()}
                                 for m in subscribers.pending_mail()]})


@bp.post("/admin/mail/outbox")
def outbox_acknowledge():
    if not _sender_authorised():
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(silent=True) or {}
    try:
        sent, failed = [int(i) for i in data.get("sent", [])], [int(i) for i in data.get("failed", [])]
    except (TypeError, ValueError):
        return jsonify({"error": "sent and failed must be lists of ids"}), 400
    return jsonify({"queue": subscribers.acknowledge(sent, failed)})


def _client_ok() -> bool:
    return subscribers.allow_signup("provision:" + _client_fingerprint())


# ── icon (public: a browser asks for it before anyone has signed in) ─────────


@bp.get("/favicon.ico")
@bp.get("/favicon.svg")
@bp.get("/apple-touch-icon.png")
def icon():
    from flask import current_app
    response = current_app.send_static_file(request.path.lstrip("/"))
    response.headers["Cache-Control"] = "public, max-age=604800"
    return response
