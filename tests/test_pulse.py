from datetime import datetime, timedelta
import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from src.pulse import UTC, SOURCES, briefing, parse_feed, read_snapshot, refresh, rss
from src.research_brief import build_research_brief

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def feed(title="Council adopts resolution", date="Tue, 29 Sep 2026 10:00:00 GMT", link="https://press.un.org/en/2026/sc1.doc.htm"):
    return f'''<rss><channel><item><title>{title}</title><link>{link}</link>
    <pubDate>{date}</pubDate><description><![CDATA[<p>Reported <b>facts</b>.</p><script>bad()</script>]]></description>
    </item></channel></rss>'''.encode()


def good_source(source, now):
    items, rejected = parse_feed(feed(link=f"https://example.org/{source['id']}"), source, now)
    return items, {**source, "status": "ok", "last_success_at": now.isoformat(), "rejected": rejected}


def test_dates_sanitization_and_links():
    items, rejected = parse_feed(feed(), SOURCES[0], NOW)
    assert rejected == 0
    assert items[0]["summary"] == "Reported facts ."
    assert items[0]["published_at"] == "2026-09-29T10:00:00Z"
    assert parse_feed(feed(date="invalid"), SOURCES[0], NOW)[1] == 1
    assert parse_feed(feed(date="Wed, 30 Sep 2026 10:00:00 GMT"), SOURCES[0], NOW)[1] == 1
    assert parse_feed(feed(link="javascript:alert(1)"), SOURCES[0], NOW)[1] == 1
    with pytest.raises(ValueError):
        parse_feed(b"<html>Access denied</html>", SOURCES[0], NOW)
    with pytest.raises(ValueError):
        parse_feed(b'<!DOCTYPE rss [<!ENTITY x "bad">]><rss/>', SOURCES[0], NOW)


def test_atom_and_tracking_url():
    body = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Health update</title>
    <link href="https://who.int/news/test?utm_source=rss&amp;id=2"/><published>2026-09-28T13:00:00+02:00</published>
    <summary>Health statement</summary></entry></feed>'''
    items, rejected = parse_feed(body, SOURCES[-1], NOW)
    assert not rejected
    assert items[0]["url"] == "https://who.int/news/test?id=2"
    assert items[0]["published_at"] == "2026-09-28T11:00:00Z"


def test_refresh_idempotence_failure_retention_and_staleness(tmp_path):
    first = refresh(tmp_path, NOW, good_source)
    second = refresh(tmp_path, NOW + timedelta(minutes=30), good_source)
    assert first["content_hash"] == second["content_hash"]
    assert len(second["items"]) == len(SOURCES)
    assert len(list((tmp_path / "archive").glob("*.json"))) == 1
    def failed(source, now):
        return [], {**source, "status": "error", "error": "Timeout"}
    failed_snapshot = refresh(tmp_path, NOW + timedelta(hours=1), failed)
    assert failed_snapshot["items"] == second["items"]
    assert all(s["last_success_at"] for s in failed_snapshot["sources"])
    assert briefing(failed_snapshot, NOW + timedelta(hours=1))["status"] == "partial"
    assert briefing(failed_snapshot, NOW + timedelta(hours=4))["status"] == "stale"
    assert read_snapshot(tmp_path) == failed_snapshot


def test_revision_and_exact_headline_grouping(tmp_path):
    first = refresh(tmp_path, NOW, good_source)
    assert len(briefing(first, NOW)["updates"]) == 1
    assert len(briefing(first, NOW)["updates"][0]["also_reported_by"]) == len(SOURCES) - 1
    def changed(source, now):
        items, status = good_source(source, now)
        items[0]["summary"] = "Corrected source summary."
        return items, status
    revised = refresh(tmp_path, NOW + timedelta(hours=1), changed)
    assert revised["content_hash"] != first["content_hash"]
    assert revised["items"][0]["revised_at"]
    assert revised["items"][0]["first_seen_at"] == first["items"][0]["first_seen_at"]
    parsed = ET.fromstring(rss(revised, "https://example.org/briefing"))
    assert parsed.find("channel/item/guid").text == revised["items"][0]["url"]


def votes():
    # Different topic frequencies cause raw agreement to shift even though
    # the within-topic voting pattern is identical in every year.
    rows = []
    for year, counts in ((2023, (20, 10)), (2024, (20, 10)), (2025, (10, 20)), (2026, (5, 5))):
        for topic, count in zip(("Nuclear disarmament", "Human rights"), counts):
            for n in range(count):
                rcid = year * 1000 + (100 if "Nuclear" in topic else 200) + n
                for country in ("USA", "CHN", "RUS", "FRA"):
                    vote = 1 if country == "USA" or "Nuclear" in topic else -1
                    rows.append({"year": year, "date": pd.Timestamp(year, 9, 1), "rcid": rcid,
                                 "country_identifier": country, "vote": vote, "issue": topic})
    return pd.DataFrame(rows)


def test_research_agenda_adjustment_and_stable_hash():
    first = build_research_brief(votes(), NOW)
    second = build_research_brief(votes(), NOW + timedelta(days=1))
    assert first["content_hash"] == second["content_hash"]
    assert first["recent_year"] == 2025
    assert first["baseline_years"] == [2023, 2024]
    usa = next(f for f in first["findings"] if f["id"] == "alignment-USA")
    assert usa["evidence"]["recent_pct"] < usa["evidence"]["baseline_pct"]
    assert usa["evidence"]["topic_adjusted_change_pp"] == pytest.approx(0)
    assert usa["evidence"]["recent_comparisons"] == 90
    assert "not influence" in usa["caveat"]
    assert len(first["findings"]) == 6


def test_research_deduplicates_rows_and_requires_baseline():
    frame = votes()
    normal = build_research_brief(frame, NOW)
    doubled = build_research_brief(pd.concat([frame, frame]), NOW)
    assert normal["content_hash"] == doubled["content_hash"]
    assert build_research_brief(frame[frame.year == 2025], NOW)["status"] == "insufficient"
    assert build_research_brief(None, NOW)["status"] == "insufficient"


def test_current_year_uses_same_dates_in_prior_years():
    frame = votes()
    extra = frame[frame.year == 2025].copy()
    extra["year"] = 2026
    extra["date"] = pd.Timestamp(2026, 9, 1)
    extra["rcid"] += 1000
    # An autumn bloc of votes must not enter the YTD baseline.
    autumn = frame[frame.year == 2024].copy()
    autumn["date"] = pd.Timestamp(2024, 12, 1)
    autumn["rcid"] += 500
    brief = build_research_brief(pd.concat([frame[frame.year != 2026], extra, autumn]), NOW)
    assert brief["recent_year"] == 2026
    assert "same dates" in brief["period"]
    assert brief["baseline_resolutions"] == 90
    assert brief["recent_resolutions"] == 30


def test_publication_works_without_voting_data_and_escapes_source_text(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_DATA_DIR", str(tmp_path))
    from app import create_app
    app = create_app()
    with app.test_client() as browser:
        response = browser.get("/")
        assert response.status_code == 200
        assert b"What the record reveals" in response.data
        assert browser.get("/api/pulse").json["stale"] is True
        assert browser.get("/api/research-brief").json["status"] == "insufficient"
        assert browser.get("/briefing/editions/not-an-edition").status_code == 404
        brief = build_research_brief(votes(), NOW)
        (tmp_path / "research").mkdir()
        (tmp_path / "research.json").write_text(json.dumps(brief))
        (tmp_path / "research" / f"{brief['content_hash'][:16]}.json").write_text(json.dumps(brief))
        assert browser.get("/briefing").status_code == 200
        assert browser.get(f"/briefing/editions/{brief['content_hash'][:16]}").status_code == 200
        root = ET.fromstring(browser.get("/briefing/research.xml").data)
        assert root.find("channel/item/guid").text == brief["content_hash"]


def test_worker_preserves_research_publication_date(tmp_path, monkeypatch):
    from src import pulse_worker
    monkeypatch.setattr(pulse_worker, "refresh", lambda directory: {"items": [], "sources": []})
    monkeypatch.setattr("src.un_decisions.refresh_decisions", lambda directory: {"records": [], "sources": []})
    monkeypatch.setattr("src.newsletter_publisher.publish_newsletter", lambda *args: {"changed": False})
    pulse_worker.refresh_all(tmp_path, votes())
    before = (tmp_path / "research.json").read_text()
    pulse_worker.refresh_all(tmp_path, votes())
    assert (tmp_path / "research.json").read_text() == before
    assert len(list((tmp_path / "research").glob("*.json"))) == 1


def test_health_topic_needs_the_organisation_not_the_pronoun():
    fled, _ = parse_feed(feed(title="Civilians who fled the shelling reach the border"), SOURCES[0], NOW)
    assert "Health" not in fled[0]["topics"]
    who, _ = parse_feed(feed(title="WHO warns of cholera spread"), SOURCES[0], NOW)
    assert "Health" in who[0]["topics"]


def test_worker_never_replaces_ready_research_with_a_thin_rebuild(tmp_path, monkeypatch):
    from src import pulse_worker
    monkeypatch.setattr(pulse_worker, "refresh", lambda directory: {"items": [], "sources": []})
    monkeypatch.setattr("src.un_decisions.refresh_decisions", lambda directory: {"records": [], "sources": []})
    monkeypatch.setattr("src.newsletter_publisher.publish_newsletter", lambda *args: {"changed": False})
    pulse_worker.refresh_all(tmp_path, votes())
    before = (tmp_path / "research.json").read_text()
    frame = votes()
    pulse_worker.refresh_all(tmp_path, frame[frame.year == 2025])
    assert (tmp_path / "research.json").read_text() == before


def test_named_european_time_zones_are_not_read_as_utc():
    from src.pulse import date_value
    assert date_value("Tue, 10 Nov 2026 11:59:32 CET").isoformat() == "2026-11-10T10:59:32+00:00"
    assert date_value("Fri, 10 Jul 2026 11:59:32 CEST").isoformat() == "2026-07-10T09:59:32+00:00"
    assert date_value("Tue, 10 Nov 2026 11:59:32 GMT").isoformat() == "2026-11-10T11:59:32+00:00"
    items, rejected = parse_feed(feed(date="Tue, 29 Sep 2026 13:30:00 CEST"), SOURCES[2], NOW)
    assert not rejected and items[0]["published_at"] == "2026-09-29T11:30:00Z"


def test_a_source_repeating_its_own_title_is_not_one_story():
    def item(n, source_id, day):
        return {"id": f"{source_id}{n}", "title": "Daily Press Briefing by the Office of the Spokesperson",
                "url": f"https://press.un.org/en/{source_id}{n}", "summary": "", "source_id": source_id,
                "source": source_id, "published_at": f"2026-09-{day}T17:00:00Z", "topics": ["UN affairs"], "issues": []}
    snapshot = {"checked_at": NOW.isoformat(), "sources": [],
                "items": [item(3, "un-press", 28), item(2, "un-press", 27), item(1, "un-press", 26),
                          item(9, "un-news", 28), item(8, "un-news", 23)]}
    updates = briefing(snapshot, NOW)["updates"]
    # Three days of one source's briefing stay three items; another source's
    # copy of the same day's is attached once; a copy five days older is not.
    assert len(updates) == 4
    assert [len(u["also_reported_by"]) for u in updates] == [1, 0, 0, 0]
    assert updates[0]["also_reported_by"][0]["source"] == "un-news"


def test_research_stays_seasonal_after_new_year_while_the_data_stops_midyear():
    frame = votes()
    extra = frame[frame.year == 2025].copy()
    extra["year"] = 2026
    extra["date"] = pd.Timestamp(2026, 7, 28)
    extra["rcid"] += 1000
    frame = pd.concat([frame[frame.year != 2026], extra])
    frame.loc[frame.year < 2026, "date"] = frame.loc[frame.year < 2026, "year"].map(lambda y: pd.Timestamp(y, 7, 1))
    autumn = frame[frame.year == 2024].copy()
    autumn["date"] = pd.Timestamp(2024, 12, 1)
    autumn["rcid"] += 500
    frame = pd.concat([frame, autumn])
    before = build_research_brief(frame, datetime(2026, 12, 31, tzinfo=UTC))
    after = build_research_brief(frame, datetime(2027, 1, 2, tzinfo=UTC))
    assert "same dates" in before["period"] and before["recent_year"] == 2026
    # Midnight on 1 January must not turn a part-year into a "full" 2026.
    assert after["period"] == before["period"]
    assert after["content_hash"] == before["content_hash"]
    assert after["baseline_resolutions"] == 90 and after["recent_resolutions"] == 30


def test_research_uses_whole_years_once_december_is_in_the_record():
    frame = votes()
    frame = frame[frame.year != 2026].copy()
    frame["date"] = frame["year"].map(lambda y: pd.Timestamp(y, 12, 20))
    brief = build_research_brief(frame, NOW)
    assert brief["period"] == "2025 versus 2023–2024"
    assert "Complete calendar years" in brief["scope"]


def test_topic_rules_on_real_headlines():
    from src.pulse import OTHER_TOPIC, classify
    cases = {
        "Record carbon dioxide levels will impact us for centuries": ["Climate & development"],
        "At least 56 migrants die in weekend shipwrecks off Senegal and Greece": ["Humanitarian"],
        "Russia drone strike on Ukrainian railway station kills 13": ["Peace & security"],
        "COP17: the world has declared a war on nature, biodiversity summit hears": ["Climate & development"],
        "Committee on Economic, Social and Cultural Rights Opens Eightieth Session": ["Human rights"],
        "Security Council Elects Five Judges to International Court of Justice": ["Peace & security", "UN affairs"],
        "Haiti: voters prepare for elections as electricity returns": [OTHER_TOPIC],
        "Secretary-General Urges World to Uphold International Law": [OTHER_TOPIC],
        "Necessity of ending the economic, commercial and financial embargo against Cuba": [OTHER_TOPIC],
    }
    for title, expected in cases.items():
        assert classify(title)[0] == expected, title
    assert classify("El Fasher under siege")[1] == ["Sudan"]
    assert classify("Houthi attacks on shipping")[1] == ["Yemen"]
    assert classify("South Sudan talks resume")[1] == ["South Sudan"]


def test_refresh_relabels_retained_articles_and_lists_other(tmp_path):
    def old_rules(source, now):
        items, status = good_source(source, now)
        items[0]["title"] = "Secretary-General Urges World to Uphold International Law"
        items[0]["topics"] = ["UN affairs"]          # as the previous rules labelled it
        return items, status
    first = refresh(tmp_path, NOW, old_rules)
    assert all(i["topics"] == ["Other"] for i in first["items"])
    def unreachable(source, now):
        return [], {**source, "status": "error", "error": "Timeout"}
    stored = json.loads((tmp_path / "latest.json").read_text())
    for item in stored["items"]:
        item["topics"] = ["UN affairs"]
    (tmp_path / "latest.json").write_text(json.dumps(stored))
    second = refresh(tmp_path, NOW + timedelta(minutes=30), unreachable)
    assert all(i["topics"] == ["Other"] for i in second["items"])
    names = [t["name"] for t in briefing(second, NOW + timedelta(minutes=30))["topics"]]
    assert names[-1] == "Other" and "UN affairs" in names


def test_routine_notices_do_not_lead_the_briefing():
    def item(n, title, hour):
        return {"id": str(n), "title": title, "url": f"https://press.un.org/en/{n}", "summary": "",
                "source_id": f"s{n}", "source": f"Source {n}", "published_at": f"2026-09-29T{hour:02d}:00:00Z",
                "topics": ["Other"], "issues": []}
    snapshot = {"checked_at": NOW.isoformat(), "sources": [], "items": [
        item(1, "Daily Press Briefing by the Office of the Spokesperson for the Secretary-General", 11),
        item(2, "Security Council Adopts Resolution on Haiti", 8),
        item(3, "Committee Continues General Discussion", 10)]}
    lead = [u["title"] for u in briefing(snapshot, NOW)["lead"]]
    assert lead[0] == "Security Council Adopts Resolution on Haiti"
    assert lead[-1].startswith("Daily Press Briefing")


def test_items_from_another_language_section_are_rejected():
    french = feed(title="Le nouveau Représentant permanent présente ses lettres de créance",
                  link="https://www.ungeneva.org/fr/news-media/news/2026/11/1")
    assert parse_feed(french, SOURCES[2], NOW) == ([], 1)
    english = feed(link="https://www.ungeneva.org/en/news-media/news/2026/11/1")
    assert parse_feed(english, SOURCES[2], NOW)[1] == 0
