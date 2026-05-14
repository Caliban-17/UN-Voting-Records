"""Tests for the editorial display-name map."""

from __future__ import annotations

from src.country_display import (
    COUNTRY_DISPLAY_NAMES,
    _strip_parenthetical,
    display_lookup,
    display_name,
)


def test_p5_have_clean_journalistic_names():
    assert COUNTRY_DISPLAY_NAMES["USA"] == "United States"
    assert COUNTRY_DISPLAY_NAMES["GBR"] == "United Kingdom"
    assert COUNTRY_DISPLAY_NAMES["RUS"] == "Russia"


def test_strip_parenthetical_removes_legal_suffix():
    assert _strip_parenthetical("Iran (Islamic Republic Of)") == "Iran"
    assert _strip_parenthetical("Micronesia (Federated States Of)") == "Micronesia"
    assert _strip_parenthetical("United Kingdom") == "United Kingdom"


def test_display_name_prefers_override_over_raw():
    # Override exists — use it even if raw is provided.
    assert display_name("IRN", "Iran (Islamic Republic Of)") == "Iran"
    # No override — strip the parenthetical from raw.
    assert display_name("XYZ", "Some Place (Yet Another Form)") == "Some Place"


def test_korea_disambiguation_uses_north_south():
    """The crucial case — both Koreas share 'Korea' so the override is essential."""
    assert display_name("PRK", "Korea (Democratic People's Republic Of)") == "North Korea"
    assert display_name("KOR", "Korea (Republic Of)") == "South Korea"


def test_display_name_fallbacks_gracefully():
    assert display_name(None, None) == "?"
    assert display_name("", "") == "?"
    assert display_name("ABC", None) == "ABC"  # code-only fallback


def test_display_lookup_applies_to_full_map():
    raw = {
        "IRN": "Iran (Islamic Republic Of)",
        "PRK": "Korea (Democratic People's Republic Of)",
        "USA": "United States Of America",
        "XYZ": "Somewhere (Republic Of)",
    }
    out = display_lookup(raw)
    assert out["IRN"] == "Iran"
    assert out["PRK"] == "North Korea"
    assert out["USA"] == "United States"  # override wins over raw form
    assert out["XYZ"] == "Somewhere"  # fallback strip
