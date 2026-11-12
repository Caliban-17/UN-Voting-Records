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
        assert "17 September" in output and "28 July 2026" in output
        assert "2026-09-17" not in output          # dates are written, not stamped
        assert "152" in output and "docs.un.org/en/A/RES/81/1" in output
        assert "position is inferred from a total" in output
        assert "Substack" not in output
        assert "ga81_resolutions.html" in output


def test_polling_and_headlines_do_not_change_hash_but_tally_correction_does():
    live = current_affairs(*evidence(), now=NOW)
    edition = build_current_edition(live)
    second = deepcopy(live)
    second["checked_at"] = "2026-09-29T21:00:00Z"
    second["sources"] = [{"status": "error"}]
    assert build_current_edition(second).content_hash == edition.content_hash
    second["headlines"] = [{"title": "A new wire story", "url": "https://news.un.org/x",
                            "source": "UN News", "published_at": "2026-09-29T20:00:00Z"}]
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


def test_single_decision_headline_and_links_to_the_resolution():
    edition = build_current_edition(current_affairs(*evidence(), now=NOW), "2026-09-29")
    # The headline is the vote, not a count of votes.
    assert edition.headline == "Participation by the State of Palestine passes 152 to 3"
    assert edition.lede.startswith("On 17 September the General Assembly voted 152 to 3, with 4 abstaining")
    html = render_html(edition)
    assert 'href="https://docs.un.org/en/A/RES/81/1"' in html
    assert "ga81_resolutions.html" in html


def test_new_headline_alone_does_not_publish_a_new_edition(tmp_path):
    decisions, _, research = evidence()
    story = {"title": "Story", "url": "https://news.un.org/1", "source": "UN News",
             "published_at": "2026-09-29T09:00:00Z", "issues": []}
    first = publish_newsletter(None, decisions, {"items": [story]}, research, tmp_path, NOW)
    later = {**story, "title": "Later story", "url": "https://news.un.org/2",
             "published_at": "2026-09-29T11:00:00Z"}
    second = publish_newsletter(None, decisions, {"items": [later, story]}, research, tmp_path, NOW)
    assert first["changed"] and not second["changed"]
    assert len(list((tmp_path / "newsletters").glob("*.json"))) == 1


def test_newsletter_feed_reads_only_the_newest_archives(tmp_path, monkeypatch):
    import os
    import xml.etree.ElementTree as ET
    from app import create_app
    monkeypatch.setenv("PULSE_DATA_DIR", str(tmp_path))
    archive = tmp_path / "newsletters"
    archive.mkdir()
    for n in range(70):
        digest = f"{n:064x}"
        (archive / f"{digest}.json").write_text(json.dumps({
            "content_hash": digest, "edition_date": f"2026-{1 + n // 28:02d}-{1 + n % 28:02d}",
            "headline": f"Edition {n}", "lede": "Lede"}))
        os.utime(archive / f"{digest}.json", (n, n))
    (archive / f"{0:064x}.json").write_text("not json")  # oldest: never parsed
    with create_app().test_client() as client:
        feed = ET.fromstring(client.get("/newsletter/feed.xml").data)
    titles = [node.text for node in feed.findall("channel/item/title")]
    assert len(titles) == 30 and titles[0] == "Edition 69"


def test_watch_next_counts_every_recorded_decision_awaiting_roll_calls():
    decisions, pulse, research = evidence()
    older = {**decisions["records"][0], "symbol": "A/RES/80/300", "date": "2026-08-10"}
    decisions["records"].append(older)
    edition = build_current_edition(current_affairs(decisions, pulse, research, now=NOW), "2026-09-29")
    assert "Country-by-country results for 2 recent votes are not yet published" in render_text(edition)
    # An edition archived before the ledger-wide count still renders.
    del edition.current_affairs["pending_rollcalls"]
    assert "not yet published" not in render_text(edition)


def test_late_registrations_are_listed_apart_and_change_the_edition():
    decisions, pulse, research = evidence()
    late = {**decisions["records"][0], "symbol": "A/RES/80/311", "date": "2026-08-04",
            "title": "Registered weeks after adoption", "first_seen_at": "2026-09-25T10:00:00Z"}
    plain = build_current_edition(current_affairs(decisions, pulse, research, now=NOW), "2026-09-29")
    decisions["records"].append(late)
    edition = build_current_edition(current_affairs(decisions, pulse, research, now=NOW), "2026-09-29")
    assert edition.content_hash != plain.content_hash
    for render in (render_html, render_markdown, render_text):
        output = render(edition)
        assert "Registered weeks after adoption" in output
        assert "Published in the register only on 25 September" in output
    assert "Published in the register only" not in render_text(plain)
    assert edition_from_dict(edition_to_dict(edition)) == edition


# ── the edition as a story ───────────────────────────────────────────────────


def _cuba(yes=130, no=17, abstain=25):
    return {"symbol": "A/RES/81/7", "session": 81, "date": "2026-10-28", "adoption": "recorded_vote",
            "title": "Necessity of ending the economic, commercial and financial embargo imposed by the "
                     "United States of America against Cuba",
            "tally": {"yes": yes, "no": no, "abstain": abstain},
            "url": "https://docs.un.org/en/A/RES/81/7", "source_url": "https://example.org/register",
            "history": [{"year": 2024, "yes": 187, "no": 2, "abstain": 1},
                        {"year": 2025, "yes": 165, "no": 7, "abstain": 12,
                         "against": ["Argentina", "Israel", "United States"]}],
            "change": {"yes": yes - 165, "no": no - 7, "abstain": abstain - 12}}


def _live(**extra):
    quiet = {**_cuba(), "symbol": "A/RES/81/2", "title": "Multilingualism", "tally": None, "adoption": "without_vote"}
    quiet.pop("history"), quiet.pop("change")
    base = {"window_start": "2026-10-13", "window_end": "2026-11-11", "decisions": [_cuba(), quiet],
            "recorded_count": 1, "without_vote_count": 1, "pending_rollcalls": 12,
            "decision_data_through": "2026-10-28", "rollcall_data_through": "2026-07-28",
            "research_period": "x", "headlines": [], "checked_at": "x", "sources": [],
            "findings": [{"id": "alignment-USA", "title": "Agreement with the United States",
                          "finding": "Other members sided with the United States in 10.0% of comparisons.",
                          "interpretation": "i", "caveat": "A LONG CAVEAT", "evidence": {}}]}
    return {**base, **extra}


def test_headline_and_standfirst_lead_with_the_most_notable_vote_and_last_year():
    edition = build_current_edition(_live(), "2026-11-11")
    assert edition.headline == "Opposition on the Cuba embargo more than doubles: 17 vote against, up from 7"
    assert "voted 130 to 17, with 25 abstaining" in edition.lede
    assert "Last year: 165 to 7, with 12 abstaining. In 2024 it was 187 to 2." in edition.lede
    assert "The 3 against in 2025 were Argentina, Israel and United States" in edition.lede
    text = render_text(edition)
    assert "28 October · 130 for, 17 against, 25 abstaining. Last year: 165 to 7, with 12 abstaining." in text
    assert "Adopted without a vote: Multilingualism." in text
    # No statistics from a handful of items, no per-finding caveats, one methods note.
    assert "0 of " not in text and "A LONG CAVEAT" not in text
    assert text.count("HOW TO READ THIS") == 1
    assert "No country-by-country votes have been published since 28 July 2026, so this picture is unchanged." in text


def test_each_chart_states_its_finding_and_survives_every_format():
    edition = build_current_edition(_live(
        calendar={"upcoming": [{"label": "Nuclear ban treaty", "typical_date": "4 December", "days_until": 23,
                                "last": {"year": 2025, "yes": 119, "no": 45, "abstain": 12}}],
                  "committees_until": "2026-11-30"},
        vetoes_this_year=[{"date": "2026-10-30", "draft": "S/2026/868", "record": "S/PV.10260",
                           "agenda": "The situation concerning Western Sahara", "members": ["Russia"]}],
        bodies={"vetoed_drafts_by_year": {"2024": 7, "2025": 3, "2026": 1},
                "vetoes_by_member": {"2024": {"Russia": 4, "United States": 4}, "2025": {"United States": 3},
                                     "2026": {"Russia": 1}},
                "council": [], "committee": [], "committee_totals": {}}), "2026-11-11")
    html = render_html(edition)
    assert html.count("<svg") == 3
    assert "17 against, where there were 7" in html
    assert "1 veto cast on 1 draft so far in 2026; the recent peak was 8 in 2024" in html
    assert "1 recurring vote due in the next five weeks" in html
    assert 'style="max-width' not in html.split("<svg", 1)[1].split(">", 1)[0]   # the PNG converter needs plain sizes
    text = render_text(edition)
    assert "[Chart: 17 against, where there were 7." in text
    assert "1 draft resolution has been vetoed in 2026. The latest was on 30 October" in text
    assert "Nuclear ban treaty, usually around 4 December. Last time: 119 to 45, with 12 abstaining." in text
    assert edition_from_dict(edition_to_dict(edition)) == edition


def test_council_committee_and_calendar_context_never_mints_an_edition():
    plain = build_current_edition(_live(), "2026-11-11")
    busy = build_current_edition(_live(bodies={"council": [], "committee": [], "committee_totals": {}},
                                       calendar={"upcoming": []}), "2026-11-11")
    assert busy.content_hash == plain.content_hash
    veto = build_current_edition(_live(vetoes_this_year=[{"date": "2026-10-30", "draft": "S/1", "record": "r",
                                                          "agenda": "Western Sahara", "members": ["Russia"]}]),
                                 "2026-11-11")
    assert veto.content_hash != plain.content_hash


def test_an_edition_email_carries_charts_as_images_and_an_unsubscribe_link(monkeypatch):
    from src import mailer
    monkeypatch.setenv("SMTP_FROM", "editor@example.org")
    edition = build_current_edition(_live(), "2026-11-11")
    message = mailer.edition_message(edition, {"email": "reader@example.org", "token": "tok"}, "https://un.example")
    assert message["Subject"] == edition.headline and message["To"] == "reader@example.org"
    assert message["List-Unsubscribe"] == "<https://un.example/unsubscribe/tok>"
    body = message.get_body(("html",)).get_content()
    assert "<svg" not in body and 'src="cid:' in body and "https://un.example/unsubscribe/tok" in body
    assert [p.get_content_type() for p in message.walk()].count("image/png") == 1
    assert "https://un.example/unsubscribe/tok" in message.get_body(("plain",)).get_content()
