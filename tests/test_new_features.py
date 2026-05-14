"""Tests for the four new features: cluster auto-naming, Coalition Builder,
drift digest, and known-events endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.cluster_naming import label_clusters
from src.coalition import TIER_THRESHOLDS, build_coalition
from src.drift_analysis import compose_drift_digest
from src.similarity_utils import compute_cosine_similarity_matrix
from web_app import app, load_data


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    load_data()
    with app.test_client() as c:
        yield c


# ── Cluster auto-naming ──────────────────────────────────────────────────────


def _two_bloc_df() -> pd.DataFrame:
    """West (USA/GBR/FRA) supports human rights; East (RUS/CHN/IRN) opposes."""
    rows = []
    for rcid, topic in [(1, "HUMAN RIGHTS"), (2, "HUMAN RIGHTS"), (3, "TRADE"), (4, "TRADE")]:
        for c in ["USA", "GBR", "FRA", "RUS", "CHN", "IRN"]:
            west = c in ("USA", "GBR", "FRA")
            if topic == "HUMAN RIGHTS":
                vote = 1 if west else -1
            else:
                vote = 1  # everyone agrees on trade
            rows.append({
                "rcid": rcid, "country_identifier": c, "country_name": c,
                "vote": vote, "year": 2020, "date": "2020-06-01",
                "primary_topic": topic, "issue": topic,
            })
    return pd.DataFrame(rows)


def test_cluster_labels_pick_signature_topic_and_lead():
    df = _two_bloc_df()
    vm = df.pivot_table(index="country_identifier", columns="rcid", values="vote")
    sim = compute_cosine_similarity_matrix(vm)
    clusters = {0: ["USA", "GBR", "FRA"], 1: ["RUS", "CHN", "IRN"]}
    labels = label_clusters(
        clusters, sim, df,
        name_lookup={"USA": "United States", "GBR": "United Kingdom", "FRA": "France",
                     "RUS": "Russia", "CHN": "China", "IRN": "Iran"},
    )
    for cid in (0, 1):
        info = labels[cid]
        assert info["lead"] in clusters[cid]
        assert "Human rights" in (info["signature_topic"] or "")
        assert info["n_members"] == 3
        assert "bloc" in info["label"] or "group" in info["label"] or "Solo" in info["label"]


def test_cluster_labels_handles_solo_member():
    df = _two_bloc_df()
    vm = df.pivot_table(index="country_identifier", columns="rcid", values="vote")
    sim = compute_cosine_similarity_matrix(vm)
    labels = label_clusters({0: ["RUS"]}, sim, df, name_lookup={"RUS": "Russia"})
    assert "Solo" in labels[0]["label"]
    assert labels[0]["lead"] == "RUS"


# ── Coalition Builder ────────────────────────────────────────────────────────


def _coalition_df() -> pd.DataFrame:
    """USA/GBR back nuclear-ban resolutions, RUS opposes, IND fence-sits."""
    rows = []
    for rcid in range(1, 6):
        for c, vote in [("USA", 1), ("GBR", 1), ("FRA", 1), ("RUS", -1), ("CHN", -1), ("IND", 0)]:
            rows.append({
                "rcid": rcid, "country_identifier": c, "country_name": c,
                "vote": vote, "year": 2020, "date": "2020-06-01",
                "issue": "Nuclear weapons disarmament",
                "primary_topic": "NUCLEAR DISARMAMENT",
            })
    # Add a noise resolution that should NOT match the topic.
    for c in ["USA", "RUS"]:
        rows.append({
            "rcid": 99, "country_identifier": c, "country_name": c,
            "vote": 1, "year": 2020, "date": "2020-06-01",
            "issue": "International trade", "primary_topic": "TRADE",
        })
    return pd.DataFrame(rows)


def test_coalition_classifies_into_tiers_correctly():
    report = build_coalition(_coalition_df(), "nuclear", 2020, 2020)
    assert report["matched_resolutions"] == 5
    # USA/GBR/FRA voted Yes every time — champion supporters.
    champions = {r["country"] for r in report["tiers"]["Champion supporter"]}
    assert {"USA", "GBR", "FRA"}.issubset(champions)
    # RUS/CHN voted No every time — champion opposed.
    opposed = {r["country"] for r in report["tiers"]["Champion opposed"]}
    assert {"RUS", "CHN"}.issubset(opposed)
    # IND abstained — fence-sitter.
    fence = {r["country"] for r in report["tiers"]["Fence-sitter"]}
    assert "IND" in fence


def test_coalition_predicted_tally_counts_directions():
    report = build_coalition(_coalition_df(), "nuclear", 2020, 2020)
    tally = report["predicted_tally"]
    assert tally["yes"] >= 3  # USA, GBR, FRA
    assert tally["no"] >= 2   # RUS, CHN
    assert tally["abstain"] >= 1  # IND


def test_coalition_rejects_empty_topic():
    with pytest.raises(ValueError):
        build_coalition(_coalition_df(), "", 2020, 2020)


def test_coalition_endpoint_returns_payload(client):
    response = client.get("/api/coalition?topic=nuclear&start_year=2015&end_year=2020")
    assert response.status_code == 200
    payload = response.get_json()
    assert "predicted_tally" in payload
    assert "tiers" in payload
    assert set(payload["tiers"].keys()) == {t for t, _ in TIER_THRESHOLDS}


def test_coalition_endpoint_rejects_missing_topic(client):
    response = client.get("/api/coalition")
    assert response.status_code == 400


# ── Drift digest ─────────────────────────────────────────────────────────────


def test_compose_digest_produces_three_paragraphs():
    drifts = [
        {"country_a": "USA", "country_b": "RUS",
         "baseline_agreement": 0.6, "recent_agreement": 0.2,
         "delta": -0.4, "abs_delta": 0.4,
         "n_baseline_votes": 30, "n_recent_votes": 10,
         "driving_topics": [{"topic": "Ukraine", "count": 8}]},
        {"country_a": "USA", "country_b": "CHN",
         "baseline_agreement": 0.4, "recent_agreement": 0.6,
         "delta": 0.2, "abs_delta": 0.2,
         "n_baseline_votes": 30, "n_recent_votes": 10,
         "driving_topics": [{"topic": "Climate", "count": 5}]},
    ]
    text = compose_drift_digest(
        drifts, 2022, {"start": 2017, "end": 2021},
        name_lookup={"USA": "United States", "RUS": "Russia", "CHN": "China"},
    )
    paragraphs = text.split("\n\n")
    assert len(paragraphs) == 3
    assert "United States" in paragraphs[0]
    assert "Russia" in paragraphs[0]
    assert "Ukraine" in paragraphs[0]


def test_compose_digest_empty_input_does_not_crash():
    text = compose_drift_digest([], 2020, {"start": 2015, "end": 2019})
    assert "No alignment drifts" in text


def test_drift_digest_endpoint_returns_payload(client):
    response = client.get("/api/drift/digest?recent_year=2020&baseline_window=3")
    assert response.status_code == 200
    data = response.get_json()
    assert "digest" in data
    assert isinstance(data["digest"], str)
    assert len(data["digest"]) > 10


# ── Known events ─────────────────────────────────────────────────────────────


def test_known_events_endpoint_returns_validated_events(client):
    response = client.get("/api/events")
    assert response.status_code == 200
    payload = response.get_json()
    events = payload.get("events", [])
    assert events, "expected at least one event"
    for ev in events:
        assert "year" in ev and isinstance(ev["year"], int)
        assert "label" in ev and isinstance(ev["label"], str) and ev["label"]


def test_known_events_file_parses_and_is_sorted():
    path = Path("data/known_events.json")
    assert path.exists()
    with path.open() as f:
        data = json.load(f)
    years = [e["year"] for e in data["events"]]
    assert all(1946 <= y <= 2100 for y in years)


# ── Clustering endpoint exposes labels ───────────────────────────────────────


def test_clustering_endpoint_returns_cluster_labels(client):
    response = client.post(
        "/api/analysis/clustering",
        json={
            "start_year": 2015,
            "end_year": 2020,
            "num_clusters": 4,
            "include_stability": False,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "cluster_labels" in payload
    for _, info in payload["cluster_labels"].items():
        assert "label" in info and isinstance(info["label"], str)
        assert "lead" in info
        assert "n_members" in info
