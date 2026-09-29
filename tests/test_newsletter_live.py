from copy import deepcopy
from datetime import timedelta
import json

from src.newsletter import edition_from_dict, edition_to_dict
from src.newsletter_live import build_current_edition
from src.newsletter_publisher import publish_newsletter
from src.newsletter_render import render_html, render_markdown, render_text
from src.un_decisions import current_affairs, parse_register
from tests.test_un_decisions import NOW, register


def evidence():
    decisions = {"records": parse_register(register(), 81, NOW), "data_through": "2026-09-17", "checked_at": NOW.isoformat()}
    research = {"data_through": "2026-07-28", "period": "January–28 July 2026 versus the same dates in 2023–2025"}
    return decisions, {}, research


def test_live_edition_roundtrips_and_all_formats_show_separate_coverage():
    edition = build_current_edition(current_affairs(*evidence(), now=NOW), "2026-09-29")
    restored = edition_from_dict(edition_to_dict(edition))
    assert restored == edition
    for render in (render_html, render_markdown, render_text):
        output = render(restored)
        assert output == render(edition)
        assert "2026-09-17" in output and "2026-07-28" in output
        assert "152" in output and "A/RES/81/1" in output
        assert "no country positions" in output
        assert "Substack" not in output
        assert "ga81_resolutions.html" in output


def test_polling_does_not_change_hash_but_tally_correction_does():
    live = current_affairs(*evidence(), now=NOW)
    edition = build_current_edition(live)
    second = deepcopy(live)
    second["checked_at"] = "2026-09-29T21:00:00Z"
    second["sources"] = [{"status": "error"}]
    assert build_current_edition(second).content_hash == edition.content_hash
    second["decisions"][0]["tally"]["yes"] = 151
    assert build_current_edition(second).content_hash != edition.content_hash


def test_publisher_is_idempotent_and_archives_corrections(tmp_path):
    decisions, pulse, research = evidence()
    first = publish_newsletter(None, decisions, pulse, research, tmp_path, NOW)
    original = (tmp_path / "newsletter.html").read_text()
    second = publish_newsletter(None, decisions, pulse, research, tmp_path, NOW + timedelta(hours=6))
    assert first["changed"] and not second["changed"]
    assert (tmp_path / "newsletter.html").read_text() == original
    decisions["records"][0]["tally"]["yes"] = 151
    third = publish_newsletter(None, decisions, pulse, research, tmp_path, NOW)
    assert third["changed"]
    assert len(list((tmp_path / "newsletters").glob("*.json"))) == 2
    assert json.loads((tmp_path / "newsletter.json").read_text())["content_hash"] == third["content_hash"]


def test_live_html_escapes_source_titles():
    live = current_affairs(*evidence(), now=NOW)
    live["decisions"][0]["title"] = '<script>alert("bad")</script>'
    assert "<script>" not in render_html(build_current_edition(live))


def test_daily_rollcall_refresh_is_throttled_and_retries_errors(tmp_path):
    from types import SimpleNamespace
    from src.pulse_worker import refresh_votes_if_due
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="")
    assert refresh_votes_if_due(tmp_path, NOW, runner)["status"] == "ok"
    assert "--promote" in calls[0]
    assert refresh_votes_if_due(tmp_path, NOW + timedelta(hours=6), runner) is None
    assert len(calls) == 1
    def failed(command, **kwargs):
        return SimpleNamespace(returncode=1, stderr="validation refused")
    assert refresh_votes_if_due(tmp_path, NOW + timedelta(days=1), failed)["status"] == "error"
    assert refresh_votes_if_due(tmp_path, NOW + timedelta(days=1, minutes=20), runner) is None
    assert refresh_votes_if_due(tmp_path, NOW + timedelta(days=1, hours=1), runner)["status"] == "ok"


def test_refused_voting_refresh_returns_failure_exit_code(monkeypatch):
    from scripts import refresh_data
    monkeypatch.setattr("sys.argv", ["refresh_data.py"])
    monkeypatch.setattr(refresh_data, "_run", lambda *args, **kwargs: -1)
    assert refresh_data.main() == 1


def test_published_newsletter_routes_share_the_archived_edition(tmp_path, monkeypatch):
    from app import create_app
    import xml.etree.ElementTree as ET
    monkeypatch.setenv("PULSE_DATA_DIR", str(tmp_path))
    result = publish_newsletter(None, *evidence(), directory=tmp_path, now=NOW)
    with create_app().test_client() as client:
        latest = client.get("/newsletter")
        archived = client.get("/newsletter/editions/" + result["content_hash"])
        assert latest.status_code == 200 and latest.data == archived.data
        feed = ET.fromstring(client.get("/newsletter/feed.xml").data)
        assert feed.find("channel/item/guid").text == result["content_hash"]
        assert client.get("/newsletter/editions/invalid").status_code == 404
