"""Tests for the country-profile endpoint and its underlying builder."""

from __future__ import annotations

import pandas as pd
import pytest

from src.country_profile import BLOC_REFERENCE, P5_REFERENCE, build_country_profile
from web_app import app, load_data


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    load_data()
    with app.test_client() as c:
        yield c


def _toy_df() -> pd.DataFrame:
    # Three votes, two years. USA & GBR vote together except for one split.
    rows = [
        # rcid 1 — 2018 — everyone Yes
        {"rcid": 1, "country_identifier": "USA", "vote": 1, "date": "2018-01-01",
         "year": 2018, "issue": "Climate", "resolution": "A/RES/1"},
        {"rcid": 1, "country_identifier": "GBR", "vote": 1, "date": "2018-01-01",
         "year": 2018, "issue": "Climate", "resolution": "A/RES/1"},
        {"rcid": 1, "country_identifier": "RUS", "vote": -1, "date": "2018-01-01",
         "year": 2018, "issue": "Climate", "resolution": "A/RES/1"},
        # rcid 2 — 2019 — USA Yes, GBR No (the split), RUS No
        {"rcid": 2, "country_identifier": "USA", "vote": 1, "date": "2019-06-01",
         "year": 2019, "issue": "Sanctions", "resolution": "A/RES/2"},
        {"rcid": 2, "country_identifier": "GBR", "vote": -1, "date": "2019-06-01",
         "year": 2019, "issue": "Sanctions", "resolution": "A/RES/2"},
        {"rcid": 2, "country_identifier": "RUS", "vote": -1, "date": "2019-06-01",
         "year": 2019, "issue": "Sanctions", "resolution": "A/RES/2"},
        # rcid 3 — 2019 — USA & GBR Yes again
        {"rcid": 3, "country_identifier": "USA", "vote": 1, "date": "2019-11-01",
         "year": 2019, "issue": "Trade", "resolution": "A/RES/3"},
        {"rcid": 3, "country_identifier": "GBR", "vote": 1, "date": "2019-11-01",
         "year": 2019, "issue": "Trade", "resolution": "A/RES/3"},
        {"rcid": 3, "country_identifier": "RUS", "vote": -1, "date": "2019-11-01",
         "year": 2019, "issue": "Trade", "resolution": "A/RES/3"},
    ]
    return pd.DataFrame(rows)


def test_build_country_profile_toy_data_surfaces_top_ally_and_split():
    profile = build_country_profile(_toy_df(), "USA", 2018, 2019)
    assert profile["country"] == "USA"
    assert profile["totals"]["votes_cast"] == 3
    assert profile["top_allies"][0]["country"] == "GBR"
    # GBR similarity should be high but not 1.0 (they split once)
    assert 0 < profile["top_allies"][0]["similarity"] < 1.0
    # The split should be surfaced as a divergence with the top ally.
    divs = profile["biggest_divergences"]
    assert any(d["resolution"] == "A/RES/2" for d in divs)
    # P5 series exists for every reference.
    for ref in P5_REFERENCE:
        assert ref in profile["p5_alignment"]


def test_build_country_profile_raises_on_unknown_window():
    with pytest.raises(ValueError):
        build_country_profile(_toy_df(), "USA", 2030, 2031)


def test_country_profile_endpoint_returns_payload(client):
    response = client.get("/api/country/USA/profile?start_year=2018&end_year=2020")
    assert response.status_code == 200
    data = response.get_json()
    assert data["country"] == "USA"
    assert "top_allies" in data
    assert "top_opponents" in data
    assert "p5_alignment" in data
    assert set(data["references"]) >= {"USA", "GBR", "FRA", "RUS", "CHN"}


def test_country_profile_endpoint_rejects_invalid_code(client):
    response = client.get("/api/country/XXX/profile")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_country_profile_endpoint_rejects_invalid_year_range(client):
    response = client.get("/api/country/USA/profile?start_year=2030&end_year=2020")
    assert response.status_code == 400


def _bloc_test_df() -> pd.DataFrame:
    # USA votes with Western peers (GBR, FRA) and against Eastern peers (RUS, CHN).
    rows = []
    for rcid, year in [(1, 2018), (2, 2019), (3, 2020)]:
        for code, vote in [
            ("USA", 1), ("GBR", 1), ("FRA", 1), ("DEU", 1),
            ("RUS", -1), ("CHN", -1),
            ("IND", 0), ("BRA", 0),
        ]:
            rows.append({
                "rcid": rcid, "country_identifier": code, "country_name": code,
                "vote": vote, "date": f"{year}-06-01", "year": year,
                "issue": "Test", "resolution": f"A/RES/{rcid}",
            })
    return pd.DataFrame(rows)


def test_bloc_alignment_distinguishes_blocs():
    profile = build_country_profile(_bloc_test_df(), "USA", 2018, 2020)
    blocs = profile["bloc_alignment"]
    # USA votes identically with Western peers — should be high.
    assert blocs["Western"]["alignment"] is not None
    assert blocs["Western"]["alignment"] > 0.9
    # USA votes opposite to Eastern peers — should be low.
    assert blocs["Eastern"]["alignment"] is not None
    assert blocs["Eastern"]["alignment"] < 0.1
    # Non-aligned peers abstain — should land in the middle.
    assert blocs["Non-aligned"]["alignment"] is not None
    assert 0.3 < blocs["Non-aligned"]["alignment"] < 0.7


def test_name_lookup_resolves_display_names():
    df = _bloc_test_df()
    names = {"USA": "United States", "GBR": "United Kingdom"}
    profile = build_country_profile(df, "USA", 2018, 2020, name_lookup=names)
    assert profile["country_name"] == "United States"
    ally_names = {a["country"]: a["name"] for a in profile["top_allies"]}
    assert ally_names.get("GBR") == "United Kingdom"


def test_bloc_reference_includes_all_three_blocs():
    assert set(BLOC_REFERENCE.keys()) == {"Western", "Eastern", "Non-aligned"}
    for members in BLOC_REFERENCE.values():
        assert all(isinstance(m, str) and len(m) == 3 for m in members)
