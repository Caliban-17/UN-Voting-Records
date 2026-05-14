"""Tests for drift analysis — pairwise alignment, drift detection, percentiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift_analysis import (
    _humanize_topic,
    find_alignment_drifts,
    pair_percentile,
    pairwise_agreement_matrix,
)
from web_app import app, load_data


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    load_data()
    with app.test_client() as c:
        yield c


def _drift_df() -> pd.DataFrame:
    """Baseline 2017–2021: USA & RUS aligned (and GBR with them). Recent
    2022: USA & RUS diverge on 'Ukraine sovereignty' but still agree on
    'Trade'. CHN abstains throughout so it doesn't itself drift."""
    rows = []
    for yr in range(2017, 2022):
        for rcid in range(yr * 100, yr * 100 + 10):
            for code, vote in [("USA", 1), ("RUS", 1), ("GBR", 1), ("CHN", 0)]:
                rows.append({
                    "rcid": rcid, "country_identifier": code, "vote": vote,
                    "year": yr, "date": f"{yr}-06-01",
                    "primary_topic": "TRADE",
                })
    for rcid in range(220200, 220210):
        is_ukraine = rcid < 220208
        topic = "UKRAINE SOVEREIGNTY" if is_ukraine else "TRADE"
        rows.append({"rcid": rcid, "country_identifier": "USA", "vote": 1, "year": 2022, "date": "2022-06-01", "primary_topic": topic})
        rows.append({"rcid": rcid, "country_identifier": "RUS", "vote": -1 if is_ukraine else 1, "year": 2022, "date": "2022-06-01", "primary_topic": topic})
        rows.append({"rcid": rcid, "country_identifier": "GBR", "vote": 1, "year": 2022, "date": "2022-06-01", "primary_topic": topic})
        rows.append({"rcid": rcid, "country_identifier": "CHN", "vote": 0, "year": 2022, "date": "2022-06-01", "primary_topic": topic})
    return pd.DataFrame(rows)


def test_humanize_topic_titlecases_all_caps():
    assert _humanize_topic("MIDDLE EAST SITUATION") == "Middle east situation"
    # Already-cased strings are passed through.
    assert _humanize_topic("Climate finance") == "Climate finance"
    assert _humanize_topic("") == ""


def test_pairwise_agreement_matrix_handles_nan_without_warnings():
    df = _drift_df()
    with np.errstate(all="raise"):
        countries, agreement, overlap = pairwise_agreement_matrix(df, min_overlap=1)
    assert "USA" in countries
    a, b = countries.index("USA"), countries.index("RUS")
    assert 0.0 <= agreement[a, b] <= 1.0
    assert overlap[a, b] > 0


def test_find_alignment_drifts_surfaces_ukraine_shift():
    df = _drift_df()
    drifts = find_alignment_drifts(
        df, recent_year=2022, baseline_window=5, top_n=5,
        min_baseline_overlap=5, min_recent_overlap=5,
    )
    assert drifts, "expected at least one drift"
    # USA-RUS should be among the top drifts (GBR-RUS ties on magnitude).
    pairs = [{d["country_a"], d["country_b"]} for d in drifts]
    assert {"USA", "RUS"} in pairs
    usa_rus = next(d for d in drifts if {d["country_a"], d["country_b"]} == {"USA", "RUS"})
    assert usa_rus["delta"] < -0.5
    topics = {t["topic"] for t in usa_rus.get("driving_topics", [])}
    assert any("Ukraine" in t for t in topics), f"expected Ukraine-named driver, got {topics}"


def test_find_alignment_drifts_country_filter_only_returns_that_country():
    df = _drift_df()
    drifts = find_alignment_drifts(
        df, recent_year=2022, baseline_window=5, top_n=10,
        min_baseline_overlap=5, min_recent_overlap=5,
        country_filter="USA",
    )
    assert drifts
    for d in drifts:
        assert "USA" in (d["country_a"], d["country_b"])


def test_find_alignment_drifts_direction_filter():
    df = _drift_df()
    downs = find_alignment_drifts(
        df, recent_year=2022, baseline_window=5, top_n=5,
        min_baseline_overlap=5, min_recent_overlap=5, direction="down",
    )
    assert all(d["delta"] < 0 for d in downs)
    ups = find_alignment_drifts(
        df, recent_year=2022, baseline_window=5, top_n=5,
        min_baseline_overlap=5, min_recent_overlap=5, direction="up",
    )
    assert all(d["delta"] > 0 for d in ups)


def test_pair_percentile_returns_relative_rank():
    df = _drift_df()
    df_2022 = df[df["year"] == 2022]
    pct = pair_percentile(df_2022, "USA", "GBR", min_overlap=5)
    assert pct is not None
    assert pct["agreement"] == 1.0
    assert 80 < pct["percentile"] <= 100  # USA-GBR is among the highest
    pct_rus = pair_percentile(df_2022, "USA", "RUS", min_overlap=5)
    assert pct_rus["percentile"] < pct["percentile"]


def test_drift_endpoint_returns_payload(client):
    response = client.get("/api/drift?recent_year=2020&baseline_window=3&top=5")
    assert response.status_code == 200
    payload = response.get_json()
    assert "drifts" in payload
    assert payload["recent_year"] == 2020
    assert payload["baseline_window"] == {"start": 2017, "end": 2019}


def test_drift_endpoint_validates_direction(client):
    response = client.get("/api/drift?direction=sideways")
    assert response.status_code == 400


def test_drift_endpoint_rejects_unknown_recent_year(client):
    response = client.get("/api/drift?recent_year=3000&baseline_window=2")
    assert response.status_code == 400
