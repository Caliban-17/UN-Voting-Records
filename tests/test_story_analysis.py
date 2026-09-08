"""Tests for the big-picture (story) analysis and its API."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import story_analysis as S
from src.regional_groups import GROUP_ORDER, REGIONAL_GROUPS, lineage_codes, regional_group


@pytest.fixture(autouse=True)
def _small_years_count_as_complete(monkeypatch):
    """The synthetic record has two votes a year; don't drop its last year."""
    monkeypatch.setattr(S, "INCOMPLETE_YEAR_MIN_VOTES", 0)
    monkeypatch.setattr(S, "FULL_YEAR_MIN_VOTES", 0)


# ── synthetic record ─────────────────────────────────────────────────────────


def _frame() -> pd.DataFrame:
    """Four members, three years, six recorded votes with official tallies."""
    members = ["AAA", "BBB", "CCC", "DDD"]
    votes = [
        # rcid, year, symbol, title, subjects, votes per member (1/-1/0/None)
        (1, 2001, "A/RES/56/1", "Nuclear disarmament : resolution", "NUCLEAR DISARMAMENT", [1, 1, 1, 1]),
        (2, 2001, "A/RES/56/2", "Situation of human rights in Xanadu : resolution", "HUMAN RIGHTS--REPORTS", [1, 1, -1, -1]),
        (3, 2002, "A/RES/57/1", "Question of Palestine : resolution", "PALESTINE QUESTION", [1, 1, -1, 0]),
        (4, 2002, "A/RES/57/2", "Necessity of ending the embargo imposed by the United States of America against Cuba", "CUBA--UNITED STATES", [1, -1, 1, 1]),
        (5, 2003, "A/RES/58/1", "Necessity of ending the embargo imposed by the United States of America against Cuba", "CUBA--UNITED STATES", [1, -1, 1, None]),
        (6, 2003, "A/RES/58/2", "Programme budget for the biennium : resolution", "UN--BUDGET", [-1, -1, 1, 1]),
    ]
    rows = []
    for rcid, year, symbol, title, subject, vs in votes:
        yes = sum(1 for v in vs if v == 1)
        no = sum(1 for v in vs if v == -1)
        abst = sum(1 for v in vs if v == 0)
        for code, v in zip(members, vs):
            rows.append({
                "rcid": rcid, "year": year, "date": f"{year}-11-0{rcid}", "issue": title,
                "subjects": subject, "primary_topic": subject, "resolution": symbol,
                "total_yes": yes, "total_no": no, "total_abstentions": abst,
                "country_identifier": code, "country_name": code, "vote": v,
            })
    return pd.DataFrame(rows)


# ── themes ───────────────────────────────────────────────────────────────────


def test_classify_theme_orders_specific_before_generic():
    assert S.classify_theme("Israeli practices affecting the human rights of the Palestinian people") == "Israel & Palestine"
    assert S.classify_theme("Situation of human rights in Myanmar") == "Human rights"
    assert S.classify_theme("Establishment of a nuclear-weapon-free zone in South Asia") == "Nuclear weapons & disarmament"
    assert S.classify_theme("Question of Western Sahara") == "Decolonization & self-determination"
    assert S.classify_theme("Programme budget for the biennium 2024-2025") == "UN institutions, budget & law"
    assert S.classify_theme("Aggression against Ukraine") == "Peace, security & conflicts"
    assert S.classify_theme("") == S.OTHER_THEME


def test_agenda_counts_each_resolution_once_and_takeaway_reads():
    out = S.agenda_by_year(_frame())
    assert out["years"] == [2001, 2002, 2003]
    assert out["totals"] == [2, 2, 2]
    assert out["counts"]["Nuclear weapons & disarmament"] == [1, 0, 0]
    assert out["counts"]["Development, economy & environment"] == [0, 1, 1]
    assert "recorded votes" in out["takeaway"]
    assert "consensus" in out["caveat"]


# ── division ─────────────────────────────────────────────────────────────────


def test_pooled_agreement_matches_hand_count():
    # Two members: agree on 3 of 4 shared side-takings.
    m = pd.DataFrame([[1, 1, -1, 1], [1, 1, 1, 1]], index=["A", "B"], columns=[1, 2, 3, 4])
    assert S.pooled_agreement(m) == pytest.approx(0.75)
    assert S.pooled_agreement(pd.DataFrame()) is None


def test_division_by_year_flags_split_votes():
    out = S.division_by_year(_frame())
    by_year = {r["year"]: r for r in out["series"]}
    # 2001: one unanimous vote, one 2-2 split → half divided
    assert by_year[2001]["divided_share"] == pytest.approx(0.5)
    assert by_year[2001]["contested_share"] == pytest.approx(0.5)
    assert 0 < by_year[2001]["agreement"] < 1
    assert out["last_full_year"] in (2001, 2002, 2003)


# ── alignment ────────────────────────────────────────────────────────────────


def test_world_alignment_counts_companions_and_isolation():
    out = S.world_alignment_with(_frame(), "BBB")
    by_year = {r["year"]: r for r in out["series"]}
    # 2003: BBB voted No on rcid 5 with nobody else (isolated), and No on 6 with AAA.
    assert by_year[2003]["isolated_votes"] == 2
    assert by_year[2003]["votes_cast"] == 2
    # 2001: BBB agreed with AAA twice, CCC once, DDD once out of 6 comparisons
    assert by_year[2001]["agreement"] == pytest.approx(4 / 6, abs=1e-3)
    assert out["anchor"] == "BBB"


def test_world_alignment_unknown_anchor_raises():
    with pytest.raises(ValueError):
        S.world_alignment_with(_frame(), "ZZZ")


def test_alignment_scatter_returns_other_members_with_regions():
    out = S.alignment_scatter(_frame(), 2001, 2003, anchors=("AAA", "BBB"), min_comparisons=1)
    codes = {p["code"] for p in out["points"]}
    assert codes == {"CCC", "DDD"}
    ccc = next(p for p in out["points"] if p["code"] == "CCC")
    # CCC vs AAA: shared sides on 1,2,3,4,5,6 → same on 1,4,5 → 3/6
    assert ccc["aaa"] == pytest.approx(0.5)
    assert ccc["region"] == "Other"
    assert "members voted more often with" in out["takeaway"]


# ── landmarks & recurring ────────────────────────────────────────────────────


def test_landmark_votes_only_returns_symbols_present(monkeypatch):
    monkeypatch.setattr(S, "LANDMARK_VOTES", [
        {"symbol": "A/RES/56/2", "label": "Xanadu", "why": "because"},
        {"symbol": "A/RES/99/99", "label": "missing", "why": "never"},
    ])
    out = S.landmark_votes(_frame())
    assert [o["label"] for o in out] == ["Xanadu"]
    assert out[0]["yes"] == 2 and out[0]["no"] == 2 and out[0]["abstain"] == 0
    assert out[0]["theme"] == "Human rights"


def test_resolution_vote_map_labels_every_member():
    out = S.resolution_vote_map(_frame(), 5)
    votes = {v["code"]: v["vote"] for v in out["votes"]}
    assert votes == {"AAA": "yes", "BBB": "no", "CCC": "yes", "DDD": "absent"}
    assert out["tally"] == {"yes": 2, "no": 1, "abstain": 0, "absent": 1}
    with pytest.raises(ValueError):
        S.resolution_vote_map(_frame(), 999)


def test_recurring_series_one_row_per_year():
    out = S.recurring_vote_series(_frame(), "cuba")
    assert [(s["year"], s["yes"], s["no"]) for s in out["series"]] == [(2002, 3, 1), (2003, 2, 1)]
    assert "Support rose" in out["takeaway"]


# ── regional groups ──────────────────────────────────────────────────────────


def test_regional_groups_cover_all_193_members_plus_historical():
    sizes = {g: sum(1 for v in REGIONAL_GROUPS.values() if v == g) for g in GROUP_ORDER}
    # 54 + 54 + 23 + 33 + 29 current members, plus historical states folded in
    assert sizes["Africa"] >= 54 and sizes["Asia-Pacific"] >= 54
    assert sizes["Eastern Europe"] >= 23 and sizes["Latin America & Caribbean"] == 33
    assert sizes["Western Europe & Others"] >= 29
    assert regional_group("usa") == "Western Europe & Others"
    assert regional_group("XXX") == "Other"
    assert lineage_codes("RUS") == ["RUS", "SUN"]


# ── API (real dataset; skipped automatically when the CSV is absent) ─────────


@pytest.fixture
def client():
    from web_app import app, load_data

    app.config["TESTING"] = True
    load_data()
    with app.test_client() as c:
        yield c


def test_story_landmarks_match_the_historical_record(client):
    data = client.get("/api/story/landmarks").get_json()
    tallies = {lm["symbol"]: (lm["yes"], lm["no"], lm["abstain"]) for lm in data["landmarks"]}
    assert tallies["A/RES/2758(XXVI)"] == (76, 35, 17)
    assert tallies["A/RES/ES-11/1"] == (141, 5, 35)
    assert tallies["A/RES/47/19"] == (59, 3, 71)


def test_story_endpoints_respond(client):
    for path in ("/api/story/agenda", "/api/story/division", "/api/story/alignment?anchor=USA",
                 "/api/story/scatter?start=2021&end=2025&base_start=2001&base_end=2005",
                 "/api/story/recurring/cuba", "/api/story/recurring/nazism",
                 "/api/story/country/BRA", "/api/story/this-week"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "takeaway" in json.loads(r.get_data(as_text=True)), path
    assert client.get("/api/story/alignment?anchor=ZZZ").status_code == 400
    assert client.get("/api/story/recurring/nope").status_code == 404


# ── the newsletter carries whole-record context outside its dedup hash ────────


def test_edition_carries_big_picture_context():
    from src.newsletter import build_newsletter_edition, edition_from_dict, edition_to_dict
    from src.newsletter_render import render_html, render_markdown, render_text
    from tests.test_publish_workflow_consistency import _sparse_latest_year_df

    df = _sparse_latest_year_df()
    edition = build_newsletter_edition(df, recent_year=2025, baseline_window_years=2)
    assert edition.big_picture.get("stats"), "expected whole-record stats"
    assert edition_from_dict(edition_to_dict(edition)).big_picture == edition.big_picture
    assert "The bigger picture" in render_html(edition)
    assert "The bigger picture" in render_markdown(edition)
    assert "bigger picture" in render_text(edition).lower()
    # Older archives without the field still load, with empty context.
    payload = edition_to_dict(edition)
    payload.pop("big_picture")
    assert edition_from_dict(payload).big_picture == {}


# ── the long view, this week, regional breakdowns, the recurring catalogue ────


def test_country_story_tracks_anchor_agreement_and_winning_side():
    out = S.country_story(_frame(), "CCC", anchors=("AAA", "BBB"), min_votes=1)
    by_year = {r["year"]: r for r in out["series"]}
    # 2001: CCC vs AAA agree on rcid 1, differ on rcid 2; same against BBB.
    assert by_year[2001]["aaa"] == pytest.approx(0.5)
    assert by_year[2001]["bbb"] == pytest.approx(0.5)
    # rcid 1 was unanimous (CCC with the majority); rcid 2 tied (nobody was).
    assert by_year[2001]["with_majority"] == pytest.approx(0.5)
    assert out["first_year"] == 2001 and out["region"] == "Other"
    assert "sided with" in out["takeaway"]
    with pytest.raises(ValueError):
        S.country_story(_frame(), "ZZZ")


def test_recent_votes_names_the_dissenters():
    out = S.recent_votes(_frame(), days=14, limit=10, as_of="2003-11-10")
    assert out["sitting"] and out["count"] == 2
    assert not S.recent_votes(_frame(), days=14, as_of="2004-03-01")["sitting"]
    latest = out["votes"][0]
    # rcid 6 tied 2–2: the No side counts as the dissent.
    assert latest["rcid"] == 6 and latest["dissenters"] == ["AAA", "BBB"]
    assert "recorded votes in the fortnight" in out["takeaway"]


def test_resolution_map_breaks_the_vote_down_by_region():
    out = S.resolution_vote_map(_frame(), 5)
    assert out["regions"] == [
        {"region": "Other", "yes": 2, "no": 1, "abstain": 0, "absent": 1, "members": 4}
    ]
    assert isinstance(out["region_note"], str)


def test_recurring_catalogue_lists_every_key():
    out = S.recurring_vote_series(_frame(), "cuba")
    assert {a["key"] for a in out["available"]} == set(S.RECURRING_VOTES)
    assert len(S.RECURRING_VOTES) >= 8


def test_edition_leads_with_this_week_only_in_season():
    from src.newsletter import build_newsletter_edition, edition_from_dict, edition_to_dict
    from src.newsletter_render import render_html, render_markdown, render_text
    from tests.test_publish_workflow_consistency import _sparse_latest_year_df

    df = _sparse_latest_year_df()  # latest votes: early February 2026
    edition = build_newsletter_edition(
        df, recent_year=2025, baseline_window_years=2, edition_date="2026-02-10"
    )
    assert edition.this_week.get("sitting") and edition.this_week["votes"]
    assert edition.in_this_issue[0].anchor == "this-week"
    assert "This Week in the Assembly" in render_html(edition)
    assert "This Week in the Assembly" in render_markdown(edition)
    assert "this week in the assembly" in render_text(edition).lower()
    assert edition_from_dict(edition_to_dict(edition)).this_week == edition.this_week
    # Off-season (the latest vote is months before the edition date): no panel,
    # no TOC entry, and the hash differs only by the panel's absence.
    off = build_newsletter_edition(
        df, recent_year=2025, baseline_window_years=2, edition_date="2026-09-08"
    )
    assert off.this_week == {}
    assert off.in_this_issue[0].anchor != "this-week"
    assert off.content_hash != edition.content_hash
    again = build_newsletter_edition(
        df, recent_year=2025, baseline_window_years=2, edition_date="2026-10-01"
    )
    assert again.content_hash == off.content_hash


# ── the session calendar, the week ahead, and since-last-edition deltas ───────


def test_session_phase_covers_the_year():
    from datetime import date

    from src.session_calendar import session_phase

    assert session_phase(date(2026, 12, 10))["key"] == "plenary"
    assert session_phase(date(2026, 11, 15))["key"] == "committees"
    assert session_phase(date(2026, 11, 16))["key"] == "late_committees"
    assert session_phase(date(2027, 1, 1))["key"] == "recess"
    assert session_phase(date(2026, 7, 4))["key"] == "summer"
    assert session_phase(date(2026, 9, 8))["key"] == "opening"


def test_calendar_lists_upcoming_recurring_votes():
    from src.session_calendar import calendar_for

    # The synthetic Cuba votes fell on 4 and 5 November → typical early November.
    cal = calendar_for(_frame(), as_of="2004-10-20")
    assert [u["key"] for u in cal["upcoming"]] == ["cuba"]
    assert 14 <= cal["upcoming"][0]["days_until"] <= 16
    assert cal["upcoming"][0]["typical_date"].endswith("November")
    assert "Coming up" in cal["note"]
    assert calendar_for(_frame(), as_of="2004-12-01")["upcoming"] == []
    assert calendar_for(_frame(), as_of="2004-12-01")["phase"]["key"] == "plenary"


def test_edition_carries_the_week_ahead():
    from src.newsletter import build_newsletter_edition, edition_from_dict, edition_to_dict
    from src.newsletter_render import render_html, render_markdown, render_text
    from tests.test_publish_workflow_consistency import _sparse_latest_year_df

    df = _sparse_latest_year_df()
    edition = build_newsletter_edition(
        df, recent_year=2025, baseline_window_years=2, edition_date="2025-10-20"
    )
    assert edition.calendar.get("phase", {}).get("key") == "committees"
    assert "The week ahead" in render_html(edition)
    assert "The week ahead" in render_markdown(edition)
    assert "THE WEEK AHEAD" in render_text(edition)
    assert edition_from_dict(edition_to_dict(edition)).calendar == edition.calendar


def test_ledger_snapshot_lets_the_next_edition_report_deltas(tmp_path, monkeypatch):
    from src import newsletter as N
    from src.newsletter_ledger import last_published, record_published
    from tests.test_publish_workflow_consistency import _sparse_latest_year_df

    df = _sparse_latest_year_df()
    ledger = tmp_path / "ledger.json"
    first = N.build_newsletter_edition(
        df, recent_year=2025, baseline_window_years=2, edition_date="2026-09-01"
    )
    assert first.big_picture["year"] == 2025
    assert first.big_picture["raw"]["votes"] == 40
    # First edition ever: nothing to compare with, so no delta text.
    assert all("since the" not in s["context"] for s in first.big_picture["stats"])

    record = record_published(first, path=ledger)
    assert record["big_picture"] == {"year": 2025, "raw": first.big_picture["raw"]}

    # Next week: the composer looks the prior record up in the ledger.
    monkeypatch.setattr(N, "_prior_ledger_record", lambda focus: last_published(focus, path=ledger))
    second = N.build_newsletter_edition(
        df, recent_year=2025, baseline_window_years=2, edition_date="2026-09-08"
    )
    votes = next(s for s in second.big_picture["stats"] if s["key"] == "votes")
    assert votes["context"].endswith("unchanged since the 2026-09-01 edition")
    assert votes["delta"] == 0
    # Context never touches the dedup hash.
    assert second.content_hash == first.content_hash
