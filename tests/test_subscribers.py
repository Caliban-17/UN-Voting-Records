"""Public sign-up behind a private site: double opt-in, unsubscribe, the password gate."""

import base64

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
    with create_app().test_client() as client:
        yield client, sent, tmp_path


def _auth(password):
    return {"Authorization": "Basic " + base64.b64encode(f"reader:{password}".encode()).decode()}


def test_the_site_is_private_but_signing_up_is_not(site):
    client, _, _ = site
    for path in ("/", "/briefing", "/newsletter", "/atlas", "/api/pulse", "/newsletter/feed.xml"):
        response = client.get(path)
        assert response.status_code == 401 and "Basic" in response.headers["WWW-Authenticate"], path
    assert client.get("/briefing", headers=_auth("wrong")).status_code == 401
    assert client.get("/briefing", headers=_auth("open-sesame")).status_code == 200
    for path in ("/subscribe", "/health"):
        assert client.get(path).status_code == 200, path


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
