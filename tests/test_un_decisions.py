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
    assert all(r["url"] == "https://docs.un.org/en/A/RES/80/1" for r in records)
