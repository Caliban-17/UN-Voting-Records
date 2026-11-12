"""Public sign-up behind a private site: double opt-in, unsubscribe, the password gate."""

import pytest

from src import mailer, subscribers


@pytest.fixture
def site(tmp_path, monkeypatch):
    from app import create_app
    monkeypatch.setenv("PULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SITE_PASSWORD", "open-sesame")
    subscribers._attempts.clear()
    sent = []
    monkeypatch.setattr(mailer, "send_confirmation", lambda email, token, base=None: sent.append((email, token, base)) or True)
    monkeypatch.setattr(mailer, "send_edition", lambda edition, recipients, directory=None, base=None: 0)
    from app import middleware
    middleware._rate_buckets.clear()
    with create_app().test_client() as client:
        yield client, sent, tmp_path


def test_the_site_is_private_but_signing_up_is_not(site):
    client, _, _ = site
    for path in ("/", "/briefing", "/newsletter", "/atlas"):
        response = client.get(path)
        assert response.status_code == 302 and response.headers["Location"].startswith("/login?next="), path
    for path in ("/api/pulse", "/newsletter/feed.xml"):
        assert client.get(path).status_code == 401, path
    for path in ("/subscribe", "/health"):
        assert client.get(path).status_code == 200, path
    login = client.get("/login?next=/briefing")
    # The sign-in page offers the newsletter to those without the password.
    assert b'name="password"' in login.data and b'action="/subscribe"' in login.data
    wrong = client.post("/login", data={"password": "wrong", "next": "/briefing"})
    assert wrong.status_code == 401 and b"isn" in wrong.data
    assert client.get("/briefing").status_code == 302
    right = client.post("/login", data={"password": "open-sesame", "next": "/briefing"})
    assert right.status_code == 302 and right.headers["Location"] == "/briefing"
    assert client.get("/briefing").status_code == 200 and client.get("/api/pulse").status_code == 200
    assert client.get("/logout").status_code == 302 and client.get("/briefing").status_code == 302


def test_sign_in_never_redirects_off_site_and_slows_guessing(site, monkeypatch):
    from app import middleware
    client, _, _ = site
    for target in ("https://evil.example/", "//evil.example", "/\\evil.example"):
        response = client.post("/login", data={"password": "open-sesame", "next": target})
        assert response.headers["Location"] == "/", target
        client.get("/logout")
    monkeypatch.setattr(middleware, "MAX_API_REQUESTS_PER_MIN", 3)
    middleware._rate_buckets.clear()
    codes = [client.post("/login", data={"password": "guess"}).status_code for _ in range(5)]
    middleware._rate_buckets.clear()
    assert codes == [401, 401, 401, 429, 429]


def test_mail_settings_arrive_by_token_and_are_never_echoed(site, monkeypatch):
    client, _, directory = site
    body = {"host": "smtp.example.org", "port": "465", "user": "u@example.org", "password": "s3cret-value",
            "sender": "editor@example.org", "test": True}
    assert client.post("/admin/mail", json=body).status_code == 404            # off until a token is set
    monkeypatch.setenv("PROVISION_TOKEN", "hand-off")
    assert client.post("/admin/mail", json=body, headers={"Authorization": "Bearer nope"}).status_code == 403
    tests = []
    monkeypatch.setattr(mailer, "_send", lambda messages: tests.append(messages[0]) or [True])
    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_FROM", "SMTP_PORT"):
        monkeypatch.delenv(key, raising=False)
    ok = client.post("/admin/mail", json=body, headers={"Authorization": "Bearer hand-off"})
    assert ok.status_code == 200 and ok.json == {"configured": True, "test_sent": True,
                                                 "stored": ["host", "password", "port", "sender", "user"]}
    assert b"s3cret-value" not in ok.data and tests[0]["To"] == "editor@example.org"
    stored = directory / "smtp.json"
    assert oct(stored.stat().st_mode & 0o777) == "0o600" and mailer.settings()["SMTP_HOST"] == "smtp.example.org"
    assert client.post("/admin/mail", json={"host": "x"}, headers={"Authorization": "Bearer hand-off"}).status_code == 400


def test_double_opt_in_and_unsubscribe(site):
    client, sent, directory = site
    assert client.post("/subscribe", data={"email": "not-an-address"}).status_code == 400
    response = client.post("/subscribe", data={"email": " Reader@Example.org "})
    assert response.status_code == 200 and b"Check your inbox" in response.data
    assert [s[0] for s in sent] == ["reader@example.org"]
    assert subscribers.confirmed(directory) == []                    # nothing is sent before confirmation
    token = sent[0][1]
    assert client.get("/subscribe/confirm/nonsense").status_code == 404
    assert b"subscribed" in client.get(f"/subscribe/confirm/{token}").data
    assert [s["email"] for s in subscribers.confirmed(directory)] == ["reader@example.org"]
    # Signing up again reveals nothing and sends nothing.
    again = client.post("/subscribe", data={"email": "reader@example.org"})
    assert again.data == response.data and len(sent) == 1
    assert client.post(f"/unsubscribe/{token}").status_code == 200
    assert subscribers.confirmed(directory) == []
    assert client.get(f"/subscribe/confirm/{token}").status_code == 404   # an unsubscribed link cannot re-subscribe


def test_scripts_and_floods_are_turned_away(site):
    client, sent, _ = site
    assert client.post("/subscribe", data={"email": "bot@example.org", "website": "http://spam"}).status_code == 200
    assert sent == []
    codes = [client.post("/subscribe", data={"email": f"r{i}@example.org"}).status_code for i in range(7)]
    assert codes == [200] * 5 + [429] * 2


def test_editions_go_out_at_most_weekly_and_never_twice(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    for key, value in (("SMTP_HOST", "smtp.example.org"), ("SMTP_USER", "u"), ("SMTP_PASS", "p"), ("SMTP_FROM", "e@example.org")):
        monkeypatch.setenv(key, value)
    row = subscribers.add("reader@example.org", tmp_path)
    subscribers.confirm(row["token"], tmp_path)
    batches = []
    monkeypatch.setattr(mailer, "edition_message", lambda edition, subscriber, base: (edition.content_hash, subscriber["email"]))
    monkeypatch.setattr(mailer, "_send", lambda messages: batches.append(messages) or [True] * len(messages))
    now = datetime(2026, 11, 12, tzinfo=timezone.utc)
    first, second = SimpleNamespace(content_hash="a" * 64), SimpleNamespace(content_hash="b" * 64)
    assert mailer.deliver_if_due(first, tmp_path, now) == 1
    assert mailer.deliver_if_due(first, tmp_path, now + timedelta(days=30)) == 0      # same edition, never twice
    assert mailer.deliver_if_due(second, tmp_path, now + timedelta(days=3)) == 0      # changed, but inside the week
    assert mailer.deliver_if_due(second, tmp_path, now + timedelta(days=7)) == 1
    assert [b[0][0][0] for b in batches] == ["a", "b"]
    monkeypatch.delenv("SMTP_HOST")
    assert mailer.deliver_if_due(first, tmp_path, now + timedelta(days=60)) == 0      # unconfigured: silent no-op


def test_a_rejected_recipient_is_retried_without_resending_to_the_rest(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    for key, value in (("SMTP_HOST", "smtp.example.org"), ("SMTP_USER", "u"), ("SMTP_PASS", "p"), ("SMTP_FROM", "e@example.org")):
        monkeypatch.setenv(key, value)
    for email in ("a@example.org", "b@example.org"):
        subscribers.confirm(subscribers.add(email, tmp_path)["token"], tmp_path)
    monkeypatch.setattr(mailer, "edition_message", lambda edition, subscriber, base: subscriber["email"])
    batches = []
    reject = {"b@example.org"}
    monkeypatch.setattr(mailer, "_send", lambda messages: batches.append(messages) or [m not in reject for m in messages])
    edition, now = SimpleNamespace(content_hash="c" * 64), datetime(2026, 11, 12, tzinfo=timezone.utc)
    assert mailer.deliver_if_due(edition, tmp_path, now) == 1
    assert not (tmp_path / "delivery.json").exists()            # not delivered while someone is owed it
    reject.clear()
    assert mailer.deliver_if_due(edition, tmp_path, now + timedelta(minutes=30)) == 1
    assert batches == [["a@example.org", "b@example.org"], ["b@example.org"]]
    assert (tmp_path / "delivery.json").exists()
    assert mailer.deliver_if_due(edition, tmp_path, now + timedelta(hours=1)) == 0


def test_outbox_mode_queues_mail_for_the_workflow_to_send(site, monkeypatch):
    import base64
    import email

    client, _, directory = site
    monkeypatch.setenv("MAIL_TRANSPORT", "outbox")
    monkeypatch.setenv("PROVISION_TOKEN", "hand-off")
    monkeypatch.setenv("SMTP_FROM", "editor@example.org")
    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"):
        monkeypatch.delenv(key, raising=False)
    assert mailer.configured()                                   # only a sender address is needed to queue
    assert mailer._send([mailer.confirmation_message("reader@example.org", "tok", "https://un.example")]) == [True]
    assert client.get("/admin/mail/outbox").status_code == 403
    auth = {"Authorization": "Bearer hand-off"}
    queue = client.get("/admin/mail/outbox", headers=auth).json["messages"]
    assert len(queue) == 1
    message = email.message_from_bytes(base64.b64decode(queue[0]["mime"]))
    assert message["To"] == "reader@example.org" and "confirm/tok" in message.get_payload()
    # A failure leaves it queued; a success empties the row of its content.
    assert client.post("/admin/mail/outbox", json={"failed": [queue[0]["id"]]}, headers=auth).json["queue"] == {"pending": 1}
    assert len(client.get("/admin/mail/outbox", headers=auth).json["messages"]) == 1
    assert client.post("/admin/mail/outbox", json={"sent": [queue[0]["id"]]}, headers=auth).json["queue"] == {"sent": 1}
    assert client.get("/admin/mail/outbox", headers=auth).json["messages"] == []
    assert subscribers.pending_mail(directory) == []
    assert client.post("/admin/mail/outbox", json={"sent": ["x"]}, headers=auth).status_code == 400


def test_stale_queued_mail_is_dropped_not_sent_late(tmp_path):
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 11, 12, tzinfo=timezone.utc)
    subscribers.enqueue(b"old", tmp_path, now - timedelta(days=8))
    fresh = subscribers.enqueue(b"fresh", tmp_path, now)
    assert [m["id"] for m in subscribers.pending_mail(tmp_path, now=now)] == [fresh]
