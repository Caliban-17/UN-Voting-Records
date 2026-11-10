from datetime import datetime, timezone

import pytest

from src.pulse import atomic_json
from src.un_decisions import current_affairs, parse_register, read_decisions, refresh_decisions

NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


def register(session=81, action="with a vote (152-3-4)", date="17 September 2026"):
    return f'''<table><tr><th>Number</th><th>Title</th><th>Action</th></tr>
    <tr><th><a href="https://www.undocs.org/A/RES/{session}/1">{session}/1</a></th>
    <td>Participation by the State of Palestine</td>
    <td>A/{session}/L.2 adopted on {date} {action}</td></tr></table>'''.encode()


def test_recorded_tally_is_separate_from_country_votes():
    record = parse_register(register(), 81, NOW)[0]
    assert record["tally"] == {"yes": 152, "no": 3, "abstain": 4}
    assert record["symbol"] == "A/RES/81/1"
    assert record["date"] == "2026-09-17"
    assert "country" not in record
    assert record["source_url"].endswith("ga81_resolutions.html")


def test_without_vote_is_not_invented_unanimity():
    record = parse_register(register(action="without a vote"), 81, NOW)[0]
    assert record["tally"] is None
    assert record["adoption"] == "without_vote"


@pytest.mark.parametrize("body", [
    b"<html>Client challenge</html>",
    register(action="with a vote (194-0-0)"),
    register(action="with a vote (unknown)"),
    register(date="30 September 2026"),
])
def test_bad_source_cannot_publish_guessed_data(body):
    with pytest.raises(ValueError):
        parse_register(body, 81, NOW)


def test_failure_retains_records_but_exposes_health(tmp_path):
    refresh_decisions(tmp_path, NOW, lambda url: register(80 if "80_" in url else 81))
    def fail(url):
        raise OSError("source unavailable")
    result = refresh_decisions(tmp_path, NOW, fail)
    assert len(result["records"]) == 2
    assert all(s["status"] == "error" and s["last_success_at"] for s in result["sources"])
    assert read_decisions(tmp_path) == result


def test_unexpected_shrink_retains_prior(tmp_path):
    records = parse_register(register(), 81, NOW)
    records.append({**records[0], "symbol": "A/RES/81/2"})
    atomic_json(tmp_path / "decisions.json", {"records": records})
    result = refresh_decisions(tmp_path, NOW, lambda url: register(80 if "80_" in url else 81))
    assert result["sources"][1]["status"] == "error"
    assert any(r["symbol"] == "A/RES/81/2" for r in result["records"])


def test_digest_filters_future_and_old_records_and_diversifies_headlines():
    records = parse_register(register(), 81, NOW)
    records += [{**records[0], "date": "2026-10-01"}, {**records[0], "date": "2026-07-28"}]
    news = {"items": [dict(title=f"Item {i}", url=f"https://news.un.org/{i}", source="UN News",
                           published_at="2026-09-29T09:00:00Z", issues=[issue])
                      for i, issue in enumerate(["Sudan", "Sudan", "Ukraine"])]}
    live = current_affairs({"records": records}, news, {}, NOW)
    assert len(live["decisions"]) == 1
    assert len(live["headlines"]) == 2
    assert live["recorded_count"] == 1


def test_lettered_resolution_parts_are_distinct_adoptions():
    body = register(80).replace(b'>80/1<', b'>80/1 A<') + register(80).replace(b'>80/1<', b'>80/1 B<')
    records = parse_register(body, 80, NOW)
    assert [r["symbol"] for r in records] == ["A/RES/80/1 A", "A/RES/80/1 B"]
    # Each lettered part is its own document; the bare number does not resolve.
    assert [r["url"] for r in records] == ["https://docs.un.org/en/A/RES/80/1A",
                                           "https://docs.un.org/en/A/RES/80/1B"]


def test_decisions_are_ordered_by_number_not_by_text():
    from src.un_decisions import decision_order
    base = parse_register(register(80, date="5 December 2025"), 80, NOW)[0]
    records = [{**base, "symbol": f"A/RES/80/{n}"} for n in (99, 100, 106, 9)]
    records.append({**base, "symbol": "A/RES/80/106 B"})
    live = current_affairs({"records": records}, {}, {}, datetime(2025, 12, 6, tzinfo=timezone.utc))
    assert [r["symbol"] for r in live["decisions"]] == [
        "A/RES/80/106 B", "A/RES/80/106", "A/RES/80/100", "A/RES/80/99", "A/RES/80/9"]
    assert decision_order({"date": "2025-12-05", "symbol": "odd"}) == ("2025-12-05", 0, 0, "")


def test_pending_roll_calls_count_the_whole_ledger_not_the_window():
    recorded = parse_register(register(), 81, NOW)[0]
    older = {**recorded, "symbol": "A/RES/80/300", "date": "2026-08-10"}      # outside the 30 days
    covered = {**recorded, "symbol": "A/RES/80/250", "date": "2026-07-01"}    # already in the CSV
    quiet = {**recorded, "symbol": "A/RES/80/301", "date": "2026-08-11", "tally": None}
    live = current_affairs({"records": [recorded, older, covered, quiet]}, {},
                           {"data_through": "2026-07-28"}, NOW)
    assert len(live["decisions"]) == 1
    assert live["pending_rollcalls"] == 2
    assert current_affairs({"records": [recorded]}, {}, {}, NOW)["pending_rollcalls"] is None


def _two_rows(first_date, second_date):
    return (register(80, date=first_date).replace(b"</table>", b"")
            + register(80, date=second_date).split(b"</tr>", 1)[1].replace(b">80/1<", b">80/2<"))


def test_a_decision_registered_late_still_reaches_an_edition(tmp_path):
    from datetime import timedelta

    def session_80(body):
        return lambda url: body if "80_" in url else register(81)

    # First read of the registers: nothing is "newly seen".
    first = refresh_decisions(tmp_path, NOW, session_80(register(80, date="4 August 2026")))
    assert all(r["first_seen_at"] is None for r in first["records"])
    # Weeks later the register adds a resolution adopted back in August.
    later = NOW + timedelta(days=5)
    second = refresh_decisions(tmp_path, later, session_80(_two_rows("4 August 2026", "5 August 2026")))
    seen = {r["symbol"]: r["first_seen_at"] for r in second["records"]}
    assert seen["A/RES/80/1"] is None and seen["A/RES/80/2"].startswith("2026-10-04")
    # A further poll keeps the original sighting.
    third = refresh_decisions(tmp_path, later + timedelta(days=1),
                              session_80(_two_rows("4 August 2026", "5 August 2026")))
    assert {r["symbol"]: r["first_seen_at"] for r in third["records"]} == seen

    live = current_affairs(third, {}, {}, later)
    assert [r["symbol"] for r in live["late_decisions"]] == ["A/RES/80/2"]
    assert live["late_decisions"][0]["registered"] == "2026-10-04"
    assert all("first_seen_at" not in r for r in live["decisions"] + live["late_decisions"])
    # Thirty days on it has had its edition and drops out again.
    assert "late_decisions" not in current_affairs(third, {}, {}, later + timedelta(days=40))


def test_headlines_prefer_events_to_routine_notices():
    def item(n, title, source="UN Meetings Coverage", hour=17, issues=()):
        return {"title": title, "url": f"https://press.un.org/en/{n}", "source": source,
                "published_at": f"2026-09-28T{hour:02d}:00:00Z", "issues": list(issues)}
    news = {"items": [
        item(1, "Daily Press Briefing by the Office of the Spokesperson for the Secretary-General", hour=19),
        item(2, "New Permanent Representative of Kyrgyzstan Presents Credentials", hour=18),
        item(3, "Secretary-General Appoints Jane Doe of Ruritania Resident Coordinator in Kyrgyzstan", hour=18),
        item(4, "Second Committee Opens Eighty-First Session, Approving Organization of Work", hour=18),
        item(5, "Security Council Adopts Resolution Extending Mission Mandate", hour=10),
        item(6, "Committee Hears Briefing on Statistics", hour=16),
        item(7, "Committee Continues General Discussion", hour=15),
        item(8, "Famine confirmed in North Darfur", source="UN News", hour=12, issues=["Sudan"]),
        item(9, "Sudan: aid convoy attacked", source="UN News", hour=12, issues=["Sudan"]),
    ]}
    titles = [h["title"] for h in current_affairs({"records": []}, news, {}, NOW.replace(day=29))["headlines"]]
    # Events lead, newest first, and the 10:00 adoption outranks committee
    # business published that afternoon; no briefing, credentials, appointment
    # or organisation-of-work notice; two per source; one report per issue.
    assert titles[:2] == ["Sudan: aid convoy attacked",
                          "Security Council Adopts Resolution Extending Mission Mandate"]
    assert not [t for t in titles if "Briefing by the Office" in t or "Credentials" in t
                or "Appoints" in t or "Organization of Work" in t]
    assert sum("Committee" in t for t in titles) <= 1 and len(titles) == 3
    assert sum(t in ("Famine confirmed in North Darfur", "Sudan: aid convoy attacked") for t in titles) == 1
