"""Tests for the rendered retrospective ("year in review") recap.

The retrospective turns the archive's aggregated drift counts
(``retrospective_drift_count``) into an editorial Markdown/text recap. These
tests cover the renderers directly (synthetic reports) plus the full chain:
build editions -> archive -> aggregate -> render.
"""

from __future__ import annotations

from src.newsletter import build_newsletter_edition
from src.newsletter_archive import retrospective_drift_count, save_edition
from src.newsletter_render import (
    render_retrospective_markdown,
    render_retrospective_text,
)


def _report() -> dict:
    return {
        "n_editions": 3,
        "pairs": [
            {"country_a": "ARG", "country_b": "TKM", "count": 3,
             "total_abs_delta_pts": 142.0, "latest_edition": "2026-05-19"},
            {"country_a": "USA", "country_b": "ISR", "count": 1,
             "total_abs_delta_pts": 11.0, "latest_edition": "2026-05-12"},
        ],
    }


def test_markdown_has_headline_drift_of_year_and_ranked_pairs():
    md = render_retrospective_markdown(
        _report(),
        name_lookup={"ARG": "Argentina", "TKM": "Turkmenistan"},
        year=2026,
    )
    assert md.startswith("# UN-Scrupulous — 2026 Retrospective")
    assert "Drift of the 2026" in md
    assert "Argentina ↔ Turkmenistan" in md          # name_lookup applied
    assert "1. **Argentina ↔ Turkmenistan**" in md   # ranked, top first
    assert "142 pts" in md
    assert "3 archived editions" in md


def test_text_variant_falls_back_to_codes():
    txt = render_retrospective_text(_report(), name_lookup=None, year=None)
    assert "RETROSPECTIVE" in txt
    assert "ARG <-> TKM" in txt
    assert "3 appearances" in txt


def test_empty_archive_is_graceful():
    empty = {"n_editions": 0, "pairs": []}
    md = render_retrospective_markdown(empty, year=2026)
    txt = render_retrospective_text(empty)
    assert "No editions archived yet" in md
    assert "No editions archived yet" in txt


def test_singular_edition_grammar():
    one = {"n_editions": 1, "pairs": [
        {"country_a": "ARG", "country_b": "TKM", "count": 1,
         "total_abs_delta_pts": 5.0, "latest_edition": "2026-05-19"},
    ]}
    md = render_retrospective_markdown(one, year=2026)
    assert "1 archived edition." in md     # not "editions"
    assert "1 appearance," in md           # not "appearances"


def test_end_to_end_from_archive(tmp_path):
    """Build real editions -> archive -> aggregate -> render Markdown."""
    from tests.test_newsletter import _milei_df

    df = _milei_df()
    # Distinct edition_date per edition so the archive slugs (which key on the
    # edition week, not recent_year) don't collide and overwrite each other.
    for yr, date in [(2023, "2023-06-06"), (2024, "2024-06-04")]:
        ed = build_newsletter_edition(
            df, recent_year=yr, baseline_window_years=2, edition_date=date
        )
        save_edition(ed, archive_dir=tmp_path)

    report = retrospective_drift_count(archive_dir=tmp_path)
    assert report["n_editions"] == 2
    md = render_retrospective_markdown(report, year=None)
    assert "All-Time Retrospective" in md
    # At least one ranked realignment line rendered.
    assert "1. **" in md
