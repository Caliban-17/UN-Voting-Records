"""Security Council and committee ledger, from the veto list and meetings coverage."""

from datetime import datetime, timezone

import pytest

from src.un_bodies import body_of, classify, parse_coverage, parse_vetoes, read_bodies, refresh_bodies, summary, tally_in

NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)


def veto_table(extra=""):
    rows = "".join(f"<tr><td>{d}</td><td>S/{i}</td><td>S/PV.{i}</td><td>{a}</td><td>{m}</td></tr>" for i, (d, a, m) in enumerate(
        [("30 October 2026", "The situation concerning Western Sahara", "Russian Federation"),
         ("17 September 2026", "Non-proliferation", "China Russian Federation"),
         ("20 November 2024", "The situation in the Middle East", "USA")]
        + [("1 March 1980", "Old item", "*USSR")] * 200))
    return f"<table><tr><th>Date</th><th>Draft</th><th>Written Record</th><th>Agenda Item</th><th>Permanent Member Casting Negative Vote</th></tr>{rows}{extra}</table>"


def listing(*items):
    return "".join(f"<article><div>{d}</div><div>Meetings Coverage</div><div>{s}</div>"
                   f"<h3><a href=\"/en/2026/{s.lower().replace('/', '')}.doc.htm\">{t}</a></h3><div>{x}</div></article>"
                   for d, s, t, x in items)


def test_vetoes_are_read_with_the_librarys_spellings():
    vetoes = parse_vetoes(veto_table(), NOW)
    assert vetoes[0] == {"date": "2026-10-30", "draft": "S/0", "record": "S/PV.0",
                         "agenda": "The situation concerning Western Sahara", "members": ["Russia"]}
    assert vetoes[1]["members"] == ["Russia", "China"] and vetoes[2]["members"] == ["United States"]
    with pytest.raises(ValueError):          # an unknown caster must not be guessed
        parse_vetoes(veto_table("<tr><td>1 May 2026</td><td>S/x</td><td>r</td><td>a</td><td>Somebody</td></tr>"), NOW)
    with pytest.raises(ValueError):          # a challenge page is not an empty history
        parse_vetoes("<html>Client Challenge</html>", NOW)


def test_coverage_rows_are_classified_from_their_headlines():
    assert body_of("GA/DIS/3783") == "First Committee" and body_of("SC/16498") == "Security Council"
    assert classify("First Committee Concludes Voting, Approving 14 Texts on Regional Security") == ("approved", 14)
    assert classify("Second Committee Approves Five Texts, Including One") == ("approved", 5)
    assert classify("Security Council Fails to Adopt Resolution to Extend UN Mission") == ("defeated", None)
    assert classify("Security Council Hears Update on Yemen") == ("meeting", None)
    assert tally_in("adopted by a recorded vote of 120 in favour to 5 against, with 50 abstentions") == {"yes": 120, "no": 5, "abstain": 50}
    assert tally_in("14 members in favour, one abstaining") is None       # never guessed from prose
    rows = parse_coverage(listing(("11 November 2026", "GA/EF/3630", "Second Committee Approves Five Texts", "Summary.")), NOW)
    assert rows[0]["body"] == "Second Committee" and rows[0]["action"] == "approved" and rows[0]["texts"] == 5
    assert rows[0]["url"] == "https://press.un.org/en/2026/gaef3630.doc.htm" and rows[0]["summary"] == "Summary."


def test_ledger_accumulates_and_survives_a_failed_source(tmp_path):
    pages = {"security-council": listing(("30 October 2026", "SC/16498", "Security Council Fails to Adopt Resolution", "s")),
             "general-assembly": listing(("2 November 2026", "GA/DIS/3783", "First Committee Concludes Voting, Approving 14 Texts", "s"))}
    def page(url):
        return pages[url.split("/content/")[1].split("/")[0]]
    first = refresh_bodies(tmp_path, NOW, veto_table, page)
    assert [s["status"] for s in first["sources"]] == ["ok", "ok", "ok"]
    assert len(first["actions"]) == 2 and len(first["vetoes"]) == 203
    def down(url=None):
        raise OSError("challenge")
    second = refresh_bodies(tmp_path, NOW, down, down)
    assert [s["status"] for s in second["sources"]] == ["error"] * 3
    assert second["actions"] == first["actions"] and second["vetoes"] == first["vetoes"]
    assert read_bodies(tmp_path) == second
    digest = summary(second, NOW)
    assert [a["symbol"] for a in digest["council"]] == ["SC/16498"]
    assert digest["committee_totals"] == {"First Committee": {"meetings": 1, "texts": 14}}
    assert len(digest["vetoes_this_year"]) == 2 and digest["vetoed_drafts_by_year"][2026] == 2
    assert digest["vetoes_by_member"][2026] == {"Russia": 2, "China": 1}
