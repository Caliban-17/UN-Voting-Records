"""Tests for the Weekly Atlas newsletter composer + renderers + endpoint."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from src.newsletter import (
    DEFAULT_WATCHED_TOPICS,
    NewsletterEdition,
    build_newsletter_edition,
    edition_to_dict,
)
from src.newsletter_render import render_html, render_markdown, render_text
from web_app import app, load_data


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    load_data()
    with app.test_client() as c:
        yield c


def _milei_df() -> pd.DataFrame:
    """Synthesize a 'Milei pivot' fixture: ARG aligned with TKM in baseline,
    breaks away in recent year on Israel/Palestine + self-determination."""
    rows = []
    # Baseline 2021-2023: ARG and TKM vote together on Palestine, Nuclear, Trade.
    for yr in (2021, 2022, 2023):
        for rcid in range(yr * 1000, yr * 1000 + 10):
            # First 4 of each year are palestine-themed so coalition has baseline data
            issue = "Palestine refugees" if rcid % 10 < 4 else "Generic resolution"
            topic = "PALESTINE QUESTION" if rcid % 10 < 4 else "TRADE"
            for code, vote in [
                ("ARG", 1), ("TKM", 1), ("USA", -1), ("ISR", -1),
                ("BFA", 1), ("SOM", 1), ("CHN", 0), ("RUS", 0),
                ("FRA", -1), ("GBR", -1),
            ]:
                rows.append({
                    "rcid": rcid, "country_identifier": code, "country_name": code,
                    "vote": vote, "year": yr, "date": f"{yr}-06-01",
                    "issue": issue, "primary_topic": topic,
                })
    # 2024: ARG flips to align with USA/ISR on Palestine + Self-determination
    palestine_rcids = list(range(20240001, 20240008))
    sd_rcids = list(range(20240020, 20240025))
    for rcid in palestine_rcids:
        for code, vote in [
            ("ARG", -1), ("TKM", 1), ("USA", -1), ("ISR", -1),
            ("BFA", 1), ("SOM", 1), ("CHN", 0), ("RUS", 0),
            ("FRA", -1), ("GBR", -1),
        ]:
            rows.append({
                "rcid": rcid, "country_identifier": code, "country_name": code,
                "vote": vote, "year": 2024, "date": "2024-11-01",
                "issue": "Palestine refugees",
                "primary_topic": "TERRITORIES OCCUPIED BY ISRAEL--SETTLEMENT POLICY",
            })
    for rcid in sd_rcids:
        for code, vote in [
            ("ARG", -1), ("TKM", 1), ("USA", -1), ("ISR", -1),
            ("BFA", 1), ("SOM", 1), ("CHN", 0), ("RUS", 0),
            ("FRA", -1), ("GBR", -1),
        ]:
            rows.append({
                "rcid": rcid, "country_identifier": code, "country_name": code,
                "vote": vote, "year": 2024, "date": "2024-11-15",
                "issue": "Self-determination",
                "primary_topic": "SELF-DETERMINATION OF PEOPLES",
            })
    # One contested vote with clear breakaways (for spotlight)
    for code, vote in [
        ("ARG", -1), ("USA", -1), ("ISR", -1),
        ("TKM", 1), ("BFA", 1), ("SOM", 1), ("CHN", 1),
        ("RUS", 1), ("FRA", -1), ("GBR", -1),
    ]:
        rows.append({
            "rcid": 99990000, "country_identifier": code, "country_name": code,
            "vote": vote, "year": 2024, "date": "2024-12-01",
            "issue": "Promoting international cooperation",
            "primary_topic": "INTERNATIONAL COOPERATION",
        })
    return pd.DataFrame(rows)


def test_build_newsletter_edition_returns_typed_object():
    df = _milei_df()
    edition = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3,
        watched_topics=("palestine", "nuclear"),
        name_lookup={"ARG": "Argentina", "TKM": "Turkmenistan", "USA": "United States",
                     "ISR": "Israel", "BFA": "Burkina Faso", "SOM": "Somalia",
                     "CHN": "China", "RUS": "Russia", "FRA": "France", "GBR": "United Kingdom"},
    )
    assert isinstance(edition, NewsletterEdition)
    assert edition.recent_year == 2024
    assert edition.baseline_window == {"start": 2021, "end": 2023}
    # The lead drift should involve ARG.
    pair = {edition.lead_story.supporting_drifts[0]["country_a"],
            edition.lead_story.supporting_drifts[0]["country_b"]} if edition.lead_story.supporting_drifts else set()
    assert "Argentina" in edition.headline or "ARG" in edition.headline


def test_newsletter_has_named_topic_drivers_not_just_numbers():
    df = _milei_df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    # Lede should mention the topic, not just a number.
    body = edition.lead_story.body
    assert "Territories occupied" in body or "Self-determination" in body, (
        f"Lead body should name driving topics: {body!r}"
    )


def test_newsletter_coalition_watch_surfaces_tier_jumpers():
    df = _milei_df()
    edition = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3,
        watched_topics=("palestine",),
    )
    palestine = next(
        (snap for snap in edition.coalition_watch if snap.topic == "palestine"),
        None,
    )
    assert palestine is not None
    # ARG should appear as a tier-jumper (flipped from supporter to opposed).
    movers = {m.country for m in palestine.movers}
    assert "ARG" in movers


def test_render_markdown_emits_all_publication_sections():
    df = _milei_df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    md = render_markdown(edition)
    # Section names come from src.newsletter_voice.SECTION_TITLES.
    from src.newsletter_voice import SECTION_TITLES
    for heading in [
        "# UN-Scrupulous",
        f"## {SECTION_TITLES['by_the_numbers']}",
        f"## {SECTION_TITLES['shift']}",
        f"## {SECTION_TITLES['movers']}",
        f"## {SECTION_TITLES['coalition']}",
        f"## {SECTION_TITLES['next']}",
        "### In this issue",
        "### Methodology",
        "### Sources",
    ]:
        assert heading in md, f"Missing heading {heading!r} in markdown"
    # Editorial connective tissue.
    assert "Why it matters" in md
    assert "Edition №" in md


def test_render_html_is_email_safe_and_self_contained():
    df = _milei_df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    html = render_html(edition)
    # Strict email-safety: NO external CSS, NO scripts, NO remote images.
    assert "<script" not in html.lower()
    assert "stylesheet" not in html.lower()
    assert "src=\"http" not in html and "src='http" not in html
    # Inline-style table layout (Outlook-friendly): every visible block has
    # an inline ``style="…"`` attribute somewhere in its hierarchy.
    assert html.count("style=\"") > 30
    # Print rules included.
    assert "@media print" in html
    # Inline SVG chart embedded when drifts exist (no <img src=…>).
    if edition.chart_payloads.get("top_drifts_raw"):
        assert "<svg" in html
    # Edition masthead present.
    assert "Edition №" in html


def test_render_text_is_plain_and_wrapped():
    """Plain-text renderer for multipart/alternative MIME emails."""
    df = _milei_df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    txt = render_text(edition)
    from src.newsletter_voice import SECTION_TITLES
    # No HTML tags.
    assert "<" not in txt or ">" not in txt or not (
        "<html" in txt.lower() or "<div" in txt.lower() or "<p" in txt.lower()
    )
    # Includes section headers as upper-case banners.
    assert SECTION_TITLES["by_the_numbers"].upper() in txt
    assert SECTION_TITLES["coalition"].upper() in txt
    assert "UN-SCRUPULOUS" in txt
    # Each line is ≤ 90 columns (allowing for one or two stretches).
    long_lines = [ln for ln in txt.split("\n") if len(ln) > 100]
    assert len(long_lines) < 5, f"too many long lines for plain-text email: {long_lines[:3]}"


def test_newsletter_endpoint_returns_json(client):
    response = client.get("/api/newsletter/weekly?recent_year=2020&baseline_window=2&format=json")
    assert response.status_code == 200
    data = response.get_json()
    for key in (
        "headline", "subhead", "lede", "nut_graf",
        "edition_number", "dateline", "byline",
        "in_this_issue", "lead_story", "coalition_watch",
        "next_to_watch", "sources", "markdown",
    ):
        assert key in data, f"Missing key {key!r}"
    assert data["markdown"].startswith("# UN-Scrupulous")


def test_newsletter_endpoint_returns_markdown(client):
    response = client.get("/api/newsletter/weekly?recent_year=2020&baseline_window=2&format=markdown")
    assert response.status_code == 200
    assert response.mimetype.startswith("text/markdown")
    body = response.get_data(as_text=True)
    assert "# UN-Scrupulous" in body
    assert "Edition №" in body


def test_newsletter_endpoint_returns_html(client):
    response = client.get("/api/newsletter/weekly?recent_year=2020&baseline_window=2&format=html")
    assert response.status_code == 200
    assert response.mimetype.startswith("text/html")
    body = response.get_data(as_text=True)
    assert "Edition №" in body
    # SVG chart only embeds when drifts exist; tolerate sparse-data fixtures.
    if "top_drifts_raw" in body or "drifts_for_chart" in body:
        assert "<svg" in body
    assert "@media print" in body


def test_newsletter_endpoint_returns_text(client):
    response = client.get("/api/newsletter/weekly?recent_year=2020&baseline_window=2&format=text")
    assert response.status_code == 200
    assert response.mimetype.startswith("text/plain")
    body = response.get_data(as_text=True)
    assert "UN-SCRUPULOUS" in body
    assert "Edition №" in body


def test_newsletter_endpoint_rejects_invalid_format(client):
    response = client.get("/api/newsletter/weekly?format=pdf")
    assert response.status_code == 400


def test_newsletter_endpoint_rejects_unknown_year(client):
    response = client.get("/api/newsletter/weekly?recent_year=3000")
    assert response.status_code == 400


def test_default_watched_topics_are_lowercase_strings():
    for topic in DEFAULT_WATCHED_TOPICS:
        assert isinstance(topic, str) and topic == topic.lower()


def test_edition_to_dict_is_json_serializable():
    df = _milei_df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    payload = edition_to_dict(edition)
    # Must round-trip through JSON cleanly.
    serialized = json.dumps(payload, default=str)
    assert len(serialized) > 100
    assert "lead_story" in serialized


def test_render_markdown_escapes_or_preserves_topic_strings():
    """Topic strings often contain '--' from UNBISnet; markdown should pass these through."""
    df = _milei_df()
    edition = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3,
        watched_topics=("palestine",),
    )
    md = render_markdown(edition)
    # We don't want any raw `{` or `}` literals from f-string bugs.
    assert "{" not in md or "}" not in md or re.search(r"\{[a-z_]+\}", md) is None
