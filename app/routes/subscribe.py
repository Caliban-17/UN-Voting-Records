"""Public sign-up, confirmation and unsubscribe pages (no account, no password)."""

import json
import logging

from flask import Blueprint, Response, request

from app.middleware import _client_fingerprint

from src import mailer, subscribers
from src.pulse import data_dir

logger = logging.getLogger(__name__)
bp = Blueprint("subscribe", __name__)

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>UN-Scrupulous</title></head>
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


@bp.route("/unsubscribe/<token>", methods=["GET", "POST"])
def unsubscribe(token):
    subscribers.unsubscribe(token)
    return page("You're unsubscribed", "You won't receive any more emails from UN-Scrupulous.")
