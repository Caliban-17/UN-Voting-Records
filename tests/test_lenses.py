"""Tests for the 'Through which lens?' layer: partitions, explained variance,
cohesion, and the timeline's shape."""

from __future__ import annotations

import pandas as pd
import pytest

from src import lenses as L
from src import lenses_partitions as P
from src import story_analysis as S


@pytest.fixture(autouse=True)
def _small_years_count(monkeypatch):
    monkeypatch.setattr(S, "INCOMPLETE_YEAR_MIN_VOTES", 0)
    monkeypatch.setattr(S, "FULL_YEAR_MIN_VOTES", 0)


# ── partitions ───────────────────────────────────────────────────────────────


def test_alliance_camps_follow_treaty_years():
    assert P.alliance_camp("POL", 1980) == P.SOVIET_LED
    assert P.alliance_camp("POL", 1999) == P.US_LED
    assert P.alliance_camp("POL", 1995) == P.NON_ALIGNED
    assert P.alliance_camp("SUN", 1960) == P.SOVIET_LED
    assert P.alliance_camp("RUS", 2020) == P.SOVIET_LED
    assert P.alliance_camp("NZL", 1970) == P.US_LED and P.alliance_camp("NZL", 1990) == P.NON_ALIGNED
    assert P.alliance_camp("IND", 1975) == P.NON_ALIGNED
    assert P.alliance_camp("FIN", 2022) == P.NON_ALIGNED and P.alliance_camp("FIN", 2023) == P.US_LED


def test_world_system_tiers_use_the_world_bank_series_and_backfill():
    assert P.world_system_tier("USA", 2000) == P.CORE
    assert P.world_system_tier("BRA", 2000) == P.SEMI
    assert P.world_system_tier("IND", 2000) == P.PERIPHERY
    assert P.world_system_tier("KOR", 2023) == P.CORE
    # before 1987: the first classification carried back, socialist bloc as semi-periphery
    assert P.world_system_tier("USA", 1950) == P.CORE
    assert P.world_system_tier("POL", 1975) == P.SEMI
    assert P.world_system_tier("SUN", 1960) == P.SEMI
    assert P.world_system_tier("GER", 1980) == P.CORE
    assert P.world_system_tier("XXX", 2000) is None


def test_feminist_cohort_and_partition_labels():
    assert P.feminist_foreign_policy("SWE", 2016) and not P.feminist_foreign_policy("SWE", 2023)
    assert P.feminist_foreign_policy("MEX", 2021) and not P.feminist_foreign_policy("MEX", 2019)
    labels = P.partition_labels(["USA", "SWE", "XXX"], 2016, "ffp")
    assert labels == {"USA": "Other members", "SWE": "Feminist foreign policy", "XXX": "Other members"}
    assert P.partition_labels(["USA", "XXX"], 2016, "tier") == {"USA": P.CORE}
    with pytest.raises(ValueError):
        P.partition_labels(["USA"], 2016, "nope")


# ── explained variance and cohesion ──────────────────────────────────────────


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [[1, 1, -1], [1, 1, -1], [-1, -1, 1], [-1, -1, 1], [1, -1, 0]],
        index=list("ABCDE"), columns=[1, 2, 3],
    )


def test_explained_variance_bounds():
    m = _matrix()
    assert L.explained_variance(m, {"A": "x", "B": "x", "C": "y", "D": "y"}) == pytest.approx(1.0)
    assert L.explained_variance(m, {"A": "x", "B": "y", "C": "x", "D": "y"}) == pytest.approx(0.0)
    assert L.explained_variance(m, {"A": "x", "B": "x"}) is None          # one group
    assert L.explained_variance(m, {"A": "x", "B": "y"}) is None          # too few members


def test_adjusted_explained_variance_removes_chance():
    m = _matrix()
    out = L.adjusted_explained_variance(m, {"A": "x", "B": "x", "C": "y", "D": "y"}, permutations=40, seed=1)
    assert out["raw"] == pytest.approx(1.0)
    assert 0 <= out["baseline"] < 1
    assert out["adjusted"] == pytest.approx(1.0)
    unrelated = L.adjusted_explained_variance(m, {"A": "x", "B": "y", "C": "x", "D": "y"}, permutations=40, seed=1)
    assert unrelated["adjusted"] == 0.0


def test_group_cohesion_is_pooled_pairwise_agreement():
    m = _matrix()
    coh = L.group_cohesion(m, {"A": "x", "B": "x", "C": "y", "D": "y", "E": "z"})
    assert coh["x"] == pytest.approx(1.0) and coh["y"] == pytest.approx(1.0)
    assert coh["z"] is None  # a group of one has no pairs


# ── the timeline ─────────────────────────────────────────────────────────────


def _frame() -> pd.DataFrame:
    members = ["USA", "GBR", "SUN", "POL", "IND", "BRA"]
    votes = [
        (1, 1985, "A/RES/40/1", "Nuclear disarmament", "DISARMAMENT", [1, 1, -1, -1, 1, 1]),
        (2, 1985, "A/RES/40/2", "Situation of human rights in Xanadu", "HUMAN RIGHTS", [1, 1, -1, -1, 0, 1]),
        (3, 1986, "A/RES/41/1", "The right to development", "DEVELOPMENT", [-1, -1, 1, 1, 1, 1]),
        (4, 1986, "A/RES/41/2", "Violence against women", "WOMEN", [1, 1, 1, 1, 1, 1]),
    ]
    rows = []
    for rcid, year, symbol, title, subject, vs in votes:
        for code, v in zip(members, vs):
            rows.append({
                "rcid": rcid, "year": year, "date": f"{year}-12-01", "issue": title, "subjects": subject,
                "primary_topic": subject, "resolution": symbol,
                "total_yes": sum(1 for x in vs if x == 1), "total_no": sum(1 for x in vs if x == -1),
                "total_abstentions": sum(1 for x in vs if x == 0),
                "country_identifier": code, "country_name": code, "vote": v,
            })
    return pd.DataFrame(rows)


def test_lens_timeline_has_every_lens_and_era():
    out = L.lens_timeline(_frame(), permutations=5)
    assert out["years"] == [1985, 1986]
    assert out["last_full_year"] == 1986
    assert [x["key"] for x in out["lenses"]] == L.LENSES
    for lens in L.LENSES:
        assert len(out["indices"][lens]) == 2
        for series in out["components"][lens].values():
            assert len(series) == 2
    # the Cold War split is a perfect alliance partition in 1985
    assert out["partitions"]["alliance"][0]["raw"] == pytest.approx(1.0)
    assert out["cohesion"]["soviet_led"][0] == pytest.approx(1.0)
    assert out["agenda"]["gender"][1] == pytest.approx(0.5)
    assert out["eras"][0]["decade"] == 1980 and out["eras"][0]["top"] in L.LENSES
    assert set(out["definitions"]) >= {"explained_variance", "index", "alliance_camps", "tiers"}
    assert len(out["caveats"]) >= 3


@pytest.fixture
def client():
    from web_app import app, load_data

    app.config["TESTING"] = True
    load_data()
    with app.test_client() as c:
        yield c


def test_lenses_endpoint_reads_the_record(client):
    data = client.get("/api/story/lenses").get_json()
    years = data["years"]
    assert years[0] == 1946 and years[-1] >= 2025
    assert 2025 <= data["last_full_year"] <= years[-1]
    i = years.index(1985)
    # late Cold War: the alliance partition explains far more than chance on security items
    assert data["home_turf"]["alliance"][i]["adjusted"] > 0.5
    # the Soviet camp voted as one
    assert data["cohesion"]["soviet_led"][i] > 0.95
    assert {e["label"] for e in data["eras"]} >= {"1980s", "1990s", "2020s"}
