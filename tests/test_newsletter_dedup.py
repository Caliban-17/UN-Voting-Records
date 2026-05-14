"""Tests for the content-hash dedup gate that prevents repeat editions.

The gate exists because the UN voting calendar is highly seasonal — most
votes happen Sept–Dec, and Jan–Aug weeks routinely have zero new
resolutions. Without this gate, an automated weekly cron would happily
ship the same edition every week through the off-season.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.newsletter import build_newsletter_edition


def _df_2024():
    """Reuse the Milei-pivot synthetic fixture."""
    from tests.test_newsletter import _milei_df
    return _milei_df()


def test_content_hash_is_deterministic_across_runs():
    """Two runs over identical inputs produce identical hashes."""
    df = _df_2024()
    a = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    b = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    assert a.content_hash == b.content_hash
    # 64 hex chars = sha256
    assert len(a.content_hash) == 64


def test_content_hash_ignores_time_varying_fields():
    """Same data, different edition_date → same hash. This is what makes
    'skip if unchanged' work — time alone shouldn't bust the hash."""
    df = _df_2024()
    a = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3, edition_date="2026-05-14",
    )
    b = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3, edition_date="2026-05-21",
    )
    # Same data + same recent_year + same window — must collide.
    assert a.content_hash == b.content_hash


def test_content_hash_differs_when_window_changes():
    df = _df_2024()
    a = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    b = build_newsletter_edition(df, recent_year=2024, baseline_window_years=2)
    assert a.content_hash != b.content_hash


def test_content_hash_differs_when_data_changes():
    """Add a new contested vote to the dataset → hash must differ."""
    df = _df_2024()
    extra = pd.DataFrame([{
        "rcid": 88880000, "country_identifier": code, "country_name": code,
        "vote": vote, "year": 2024, "date": "2024-12-30",
        "issue": "New emergency resolution",
        "primary_topic": "EMERGENCY",
    } for code, vote in [
        ("ARG", 1), ("USA", 1), ("ISR", 1),
        ("TKM", -1), ("BFA", -1), ("CHN", -1), ("RUS", -1),
    ]])
    df_extended = pd.concat([df, extra], ignore_index=True)

    a = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    b = build_newsletter_edition(df_extended, recent_year=2024, baseline_window_years=3)
    assert a.content_hash != b.content_hash


def test_content_hash_differs_for_country_edition():
    df = _df_2024()
    global_ed = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    country_ed = build_newsletter_edition(
        df, recent_year=2024, baseline_window_years=3, country_focus="ARG",
    )
    assert global_ed.content_hash != country_ed.content_hash


def test_archive_listing_exposes_content_hash(tmp_path):
    from src.newsletter_archive import list_editions, save_edition
    df = _df_2024()
    edition = build_newsletter_edition(df, recent_year=2024, baseline_window_years=3)
    save_edition(edition, archive_dir=tmp_path)
    listing = list_editions(archive_dir=tmp_path)
    assert len(listing) == 1
    assert listing[0]["content_hash"] == edition.content_hash
