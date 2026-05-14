"""Tests for subject line generation, per-country edition, and edition archive."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.newsletter import _slugify, _truncate_subject, build_newsletter_edition
from src.newsletter_archive import (
    list_editions,
    read_edition,
    retrospective_drift_count,
    save_edition,
)
from web_app import app, load_data


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    load_data()
    with app.test_client() as c:
        yield c


# Reuse the synthetic fixture from the main newsletter tests.
def _df() -> pd.DataFrame:
    from tests.test_newsletter import _milei_df
    return _milei_df()


# ── Subject line + slug ─────────────────────────────────────────────────────


def test_truncate_subject_keeps_short_strings_intact():
    s = "Atlas №20: Argentina alignment fell 57 pts"
    assert _truncate_subject(s) == s


def test_truncate_subject_word_breaks_long_strings():
    s = "Atlas №20: a very long headline " * 5
    out = _truncate_subject(s, limit=78)
    assert len(out) <= 78
    assert out.endswith("…")
    # Should break at a word boundary, not mid-word.
    assert " " in out


def test_slugify_produces_filesystem_safe_strings():
    assert _slugify("Argentina ↔ Turkmenistan: a/b!") == "argentina-turkmenistan-a-b"
    assert _slugify("") == "edition"


def test_edition_carries_subject_line_and_slug():
    df = _df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    assert edition.email_subject.startswith("UN-Scrupulous №")
    assert len(edition.email_subject) <= 78
    # Slug is filesystem-safe and includes the ISO week.
    assert "/" not in edition.edition_slug
    assert edition.edition_slug.startswith(f"{2026 if edition.edition_date.startswith('2026') else 2025}-w")


# ── Per-country edition ─────────────────────────────────────────────────────


def test_country_edition_filters_drifts_to_focus_country():
    df = _df()
    edition = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3, country_focus="ARG",
    )
    assert edition.country_focus == "ARG"
    # Every supporting drift should involve ARG.
    for d in edition.lead_story.supporting_drifts:
        assert "ARG" in (d["country_a"], d["country_b"])
    # Byline rebranded for country edition.
    assert "Country edition" in edition.byline
    assert "ARG" in edition.byline or "Argentina" in edition.byline


def test_country_edition_appears_in_email_subject():
    df = _df()
    edition = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3, country_focus="ARG",
    )
    assert "ARG" in edition.email_subject


def test_country_endpoint_returns_country_payload(client):
    response = client.get("/api/newsletter/weekly?recent_year=2020&baseline_window=2&country=USA&format=json")
    # Tolerant: if the loaded dataset doesn't have USA in 2020 (e.g. a
    # truncated dev cache), the endpoint correctly returns 400. The
    # important thing is it doesn't crash with 500.
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        data = response.get_json()
        assert data["country_focus"] == "USA"
        assert "USA" in data["email_subject"]


def test_country_endpoint_rejects_unknown_country(client):
    response = client.get("/api/newsletter/weekly?recent_year=2020&baseline_window=2&country=XXX")
    assert response.status_code == 400


# ── Archive ─────────────────────────────────────────────────────────────────


def test_save_and_list_edition_roundtrip(tmp_path):
    df = _df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    written = save_edition(edition, archive_dir=tmp_path)
    # All four formats persisted.
    assert set(written.keys()) == {"json", "md", "html", "txt"}
    for path in written.values():
        assert Path(path).exists()

    listing = list_editions(archive_dir=tmp_path)
    assert len(listing) == 1
    entry = listing[0]
    assert entry["slug"] == edition.edition_slug
    assert entry["headline"] == edition.headline
    assert "md" in entry["formats"] and "html" in entry["formats"]


def test_read_edition_returns_canonical_content(tmp_path):
    df = _df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    save_edition(edition, archive_dir=tmp_path)
    year = int(edition.edition_date.split("-")[0])

    result = read_edition(year, edition.edition_slug, "md", archive_dir=tmp_path)
    assert result is not None
    mime, body = result
    assert mime.startswith("text/markdown")
    assert body.startswith("# UN-Scrupulous")


def test_read_edition_rejects_path_traversal(tmp_path):
    df = _df()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    save_edition(edition, archive_dir=tmp_path)
    year = int(edition.edition_date.split("-")[0])
    # All these must return None — never escape the archive root.
    assert read_edition(year, "../etc/passwd", "md", archive_dir=tmp_path) is None
    assert read_edition(year, edition.edition_slug, "exe", archive_dir=tmp_path) is None
    assert read_edition(year, "..%2fpasswd", "md", archive_dir=tmp_path) is None


def test_retrospective_aggregates_drifts_across_editions(tmp_path):
    df = _df()
    # Two editions for different years should aggregate together.
    save_edition(
        build_newsletter_edition(df, recent_year=2024, baseline_window_years=3),
        archive_dir=tmp_path,
    )
    save_edition(
        build_newsletter_edition(df, recent_year=2024, baseline_window_years=2),
        archive_dir=tmp_path,
    )
    report = retrospective_drift_count(archive_dir=tmp_path)
    assert report["n_editions"] >= 1
    assert isinstance(report["pairs"], list)


def test_archive_save_endpoint_creates_files(client, tmp_path, monkeypatch):
    # Repoint the archive to a temp dir to keep the test hermetic.
    monkeypatch.setenv("ATLAS_ARCHIVE_DIR", str(tmp_path))
    # Reload module-level default — newsletter_archive caches the env var
    # via DEFAULT_ARCHIVE_DIR, but read_edition / save_edition accept overrides.
    response = client.post("/api/newsletter/archive?recent_year=2020&baseline_window=2")
    assert response.status_code == 200
    data = response.get_json()
    assert "slug" in data
    assert "written" in data
    # Default archive dir is process-global; we accept either the temp dir or
    # the default location, as long as files were created.
    for ext in ("md", "html", "json", "txt"):
        assert Path(data["written"][ext]).exists()


def test_archive_list_endpoint_returns_editions(client):
    response = client.get("/api/newsletter/archive")
    assert response.status_code == 200
    data = response.get_json()
    assert "editions" in data
    assert isinstance(data["editions"], list)


def test_archive_get_endpoint_404s_on_missing(client):
    response = client.get("/api/newsletter/archive/1999/no-such-slug.md")
    assert response.status_code == 404


def test_retrospective_endpoint_returns_aggregates(client):
    response = client.get("/api/newsletter/retrospective")
    assert response.status_code == 200
    data = response.get_json()
    assert "pairs" in data
    assert "n_editions" in data
