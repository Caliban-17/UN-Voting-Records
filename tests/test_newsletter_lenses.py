"""The newsletter's 'Through which lens' section: composed from the lens
engine, outside the content hash, rendered in all three formats."""

from __future__ import annotations

import pytest

from src import story_analysis as S
from src.newsletter import _lens_panel, edition_from_dict, edition_to_dict
from src.newsletter_render import render_html, render_markdown, render_text
from tests.test_lenses import _frame
from tests.test_newsletter_roundtrip import _edition


@pytest.fixture(autouse=True)
def _small_years_count(monkeypatch):
    monkeypatch.setattr(S, "INCOMPLETE_YEAR_MIN_VOTES", 0)
    monkeypatch.setattr(S, "FULL_YEAR_MIN_VOTES", 0)


def _sample_panel() -> dict:
    return {
        "year": 2025,
        "era": {"label": "2020s", "top": "Realism", "close": "Constructivism"},
        "verdicts": [
            {"key": "alliance_loyalty", "theories": ["Realism"], "prediction": "Allies stand with their patron.",
             "verdict": "not supported", "reading": "When the United States was in a small minority, its allies voted with it 8% of the time."},
            {"key": "colonial_line", "theories": ["Post-colonial / critical"], "prediction": "Empire's line persists.",
             "verdict": "supported", "reading": "Colonial history explained 6% of the vote beyond wealth and alliance."},
        ],
        "tally": {"Realism": {"supported": 0, "not_supported": 2, "mixed": 2}},
        "character": {"year": 2025, "democracy_share": 0.5, "women_share": 26.2},
        "focus": {"code": "ARG", "name": "Argentina", "coordinates": ["electoral democracy (V-Dem)", "semi-periphery by income"]},
        "takeaway": "By deed, the 2020s read as realism, with constructivism close behind.",
        "caveat": "Recorded votes only.",
    }


def test_lens_panel_from_a_small_frame_has_the_shape():
    panel = _lens_panel(_frame(), None, None)
    assert panel["year"] == 1986
    assert panel["era"]["label"] == "1980s" and panel["era"]["top"]
    assert isinstance(panel["verdicts"], list) and isinstance(panel["tally"], dict)
    assert "caveat" in panel and "takeaway" in panel


def test_lens_panel_country_edition_names_the_coordinates():
    panel = _lens_panel(_frame(), "POL", {"POL": "Poland"})
    assert panel["focus"]["name"] == "Poland"
    coords = " ".join(panel["focus"]["coordinates"])
    assert "Russian-led alliance camp" in coords          # 1986: the Warsaw Pact
    assert "Eastern Europe" in coords


def test_lens_panel_never_sinks_an_edition():
    import pandas as pd

    assert _lens_panel(pd.DataFrame({"rcid": [], "year": []}), None, None) == {}


def test_section_renders_in_every_format_and_round_trips():
    edition = _edition()
    hash_before = edition.content_hash
    edition.lenses = _sample_panel()
    for renderer in (render_markdown, render_html, render_text):
        out = " ".join(renderer(edition).split())          # the text renderer wraps lines
        assert "through which lens" in out.lower()          # and upper-cases section titles
        assert "8% of the time" in out
        assert "Argentina in 2025 is" in out
    assert edition.content_hash == hash_before                  # context never changes the hash
    reloaded = edition_from_dict(edition_to_dict(edition))
    assert reloaded.lenses == edition.lenses
    assert render_html(reloaded) == render_html(edition)
    # older archives without the field still load
    payload = edition_to_dict(edition)
    del payload["lenses"]
    assert edition_from_dict(payload).lenses == {}


def test_section_absent_when_there_are_no_verdicts():
    edition = _edition()
    edition.lenses = {}
    assert "Through which lens" not in render_markdown(edition)
    assert "Through which lens" not in render_html(edition)
