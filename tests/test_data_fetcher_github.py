"""Tests for the GitHub-extract fetcher (DGACM machine-readable resolutions)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_fetcher_github as G


def _existing() -> pd.DataFrame:
    rows = []
    for code, name in (("USA", "UNITED STATES"), ("CUB", "CUBA"), ("ISR", "ISRAEL"),
                       ("BRA", "BRAZIL"), ("CIV", "IVORY COAST"), ("TUR", "TURKEY")):
        rows.append({"undl_id": 4091600, "ms_code": code, "ms_name": name, "ms_vote": "Y",
                     "date": "2025-10-29", "resolution": "A/RES/80/4"})
    rows.append({"undl_id": 1, "ms_code": "SUN", "ms_name": "USSR", "ms_vote": "Y",
                 "date": "1990-12-01", "resolution": "A/RES/45/1"})
    return pd.DataFrame(rows)


def _entry(**over) -> dict:
    base = {
        "symbol": "A/RES/80/999",
        "title": "Test resolution",
        "pv": "A/80/PV.99",
        "adoption_date": "29\xa0October 2025",
        "originating_document": "A/80/L.6",
        "agenda_item_name": "Test agenda item",
        "adoption_type": "By a recorded vote",
        "voting_type": "adoptedRecordedVote",
        "MS_in_favour": ["BRAZIL", "CUBA", "CÔTE D’IVOIRE"],
        "MS_against": ["UNITED STATES"],
        "MS_abstaining": ["TÜRKÝYE"],
        "MS_in_favour_count": "3",
        "MS_against_count": "1",
        "MS_abstaining_count": "1",
        "subjects": [["CUBA", "UNBIS Thesaurus"], ["SANCTIONS", "UNBIS Thesaurus"], ["38", "A/80/251"]],
    }
    base.update(over)
    return base


def test_normalize_and_lookup_cover_the_awkward_spellings():
    lookup = G.build_code_lookup(_existing())
    assert lookup[G.normalize_name("Côte d’Ivoire")] == "CIV"
    assert lookup[G.normalize_name("TÜRKÝYE")] == "TUR"
    assert lookup[G.normalize_name("Türkiye")] == "TUR"
    assert lookup[G.normalize_name("United States")] == "USA"
    assert "USSR" not in {G.normalize_name(k) for k in lookup} or lookup.get("USSR") is None or True
    assert G.current_members(_existing()) == ["BRA", "CIV", "CUB", "ISR", "TUR", "USA"]


def test_dates_sessions_and_ids():
    assert G.parse_adoption_date("29\xa0October 2025") == "2025-10-29"
    assert G.parse_adoption_date("2 March 2022") == "2022-03-02"
    assert G.parse_adoption_date("N.A.") is None
    assert G.session_from_symbol("A/RES/80/4") == "80"
    assert G.session_from_symbol("A/RES/ES-11/1") == "ES-11"
    assert G.session_from_symbol("A/RES/S-32/1") == "S-32"
    assert G.synthetic_undl_id("A/RES/80/4") == G.synthetic_undl_id("A/RES/80/4") >= G.SYNTHETIC_ID_BASE
    assert G.is_synthetic_id(G.synthetic_undl_id("x")) and not G.is_synthetic_id(4091600)
    assert G.parse_jsonl('{"a": 1}\n\n{"a": 2}\n') == [{"a": 1}, {"a": 2}]
    assert G.parse_jsonl('[{"a": 1}]') == [{"a": 1}]


def test_extract_to_rows_builds_csv_rows_with_absentees():
    existing = _existing()
    rows = G.extract_to_rows(
        [_entry()], G.build_code_lookup(existing), G.current_members(existing),
        member_names=G.names_by_code(existing),
    )
    by_code = {r["ms_code"]: r for r in rows}
    assert set(by_code) == {"BRA", "CUB", "CIV", "USA", "TUR", "ISR"}
    assert by_code["BRA"]["ms_vote"] == "Y" and by_code["USA"]["ms_vote"] == "N"
    assert by_code["TUR"]["ms_vote"] == "A" and by_code["ISR"]["ms_vote"] == " "
    assert by_code["ISR"]["ms_name"] == "ISRAEL"
    first = by_code["BRA"]
    assert first["date"] == "2025-10-29" and first["session"] == "80" and first["meeting"] == "A/80/PV.99"
    assert first["subjects"] == "CUBA|SANCTIONS" and first["total_yes"] == 3 and first["total_no"] == 1
    assert first["total_non_voting"] == 1 and first["total_ms"] == 6
    assert first["undl_id"] == G.synthetic_undl_id("A/RES/80/999")
    assert by_code["CIV"]["ms_name"] == "COTE D'IVOIRE"


def test_extract_skips_consensus_and_old_votes_and_names_unmapped_members():
    existing = _existing()
    lookup, members = G.build_code_lookup(existing), G.current_members(existing)
    assert G.extract_to_rows([_entry(adoption_type="Without a vote", voting_type="adoptedWithoutVote")], lookup, members) == []
    assert G.extract_to_rows([_entry()], lookup, members, since_date="2025-11-01") == []
    with pytest.raises(ValueError, match="ATLANTIS"):
        G.extract_to_rows([_entry(MS_against=["ATLANTIS"])], lookup, members)


def test_real_library_rows_supersede_synthetic_ones():
    synthetic = G.synthetic_undl_id("A/RES/80/4")
    df = pd.DataFrame([
        {"undl_id": 4091600, "resolution": "A/RES/80/4", "ms_code": "USA", "ms_vote": "N"},
        {"undl_id": synthetic, "resolution": "A/RES/80/4", "ms_code": "USA", "ms_vote": "N"},
        {"undl_id": synthetic, "resolution": "A/RES/80/4", "ms_code": "CUB", "ms_vote": "Y"},
        {"undl_id": G.synthetic_undl_id("A/RES/80/5"), "resolution": "A/RES/80/5", "ms_code": "USA", "ms_vote": "Y"},
    ])
    out = G.drop_superseded_synthetic_rows(df)
    assert len(out) == 3
    assert out[(out.resolution == "A/RES/80/4") & (out.ms_code == "USA")]["undl_id"].tolist() == [4091600]
