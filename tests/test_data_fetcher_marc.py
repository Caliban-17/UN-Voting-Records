"""Tests for the MARC-XML data fetcher.

Uses fixture XML — no live HTTP calls — so the suite runs in CI without
depending on UN DL availability.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from src.data_fetcher_marc import (
    _parse_record_date,
    parse_marc_record,
)


# Minimal MARC fixture — a real UN-DL voting record stripped to essentials.
_SAMPLE_MARC = """<?xml version="1.0" encoding="UTF-8"?>
<collection xmlns="http://www.loc.gov/MARC21/slim">
<record>
  <controlfield tag="001">4096564</controlfield>
  <datafield tag="245" ind1=" " ind2=" ">
    <subfield code="a">National human rights institutions :</subfield>
    <subfield code="b">resolution /</subfield>
    <subfield code="c">adopted by the General Assembly</subfield>
  </datafield>
  <datafield tag="269" ind1=" " ind2=" ">
    <subfield code="a">2025-12-18</subfield>
  </datafield>
  <datafield tag="791" ind1=" " ind2=" ">
    <subfield code="a">A/RES/80/214</subfield>
    <subfield code="c">80</subfield>
  </datafield>
  <datafield tag="952" ind1=" " ind2=" ">
    <subfield code="a">A/80/PV.69</subfield>
  </datafield>
  <datafield tag="991" ind1=" " ind2=" ">
    <subfield code="d">HUMAN RIGHTS ADVANCEMENT</subfield>
  </datafield>
  <datafield tag="991" ind1=" " ind2=" ">
    <subfield code="d">NATIONAL INSTITUTIONS--HUMAN RIGHTS</subfield>
  </datafield>
  <datafield tag="996" ind1=" " ind2=" ">
    <subfield code="b">167</subfield>
    <subfield code="c">2</subfield>
    <subfield code="d">11</subfield>
    <subfield code="e">13</subfield>
    <subfield code="f">193</subfield>
  </datafield>
  <datafield tag="967" ind1=" " ind2=" ">
    <subfield code="a">1</subfield>
    <subfield code="c">USA</subfield>
    <subfield code="d">N</subfield>
    <subfield code="e">UNITED STATES</subfield>
  </datafield>
  <datafield tag="967" ind1=" " ind2=" ">
    <subfield code="a">2</subfield>
    <subfield code="c">GBR</subfield>
    <subfield code="d">Y</subfield>
    <subfield code="e">UNITED KINGDOM</subfield>
  </datafield>
  <datafield tag="967" ind1=" " ind2=" ">
    <subfield code="a">3</subfield>
    <subfield code="c">RUS</subfield>
    <subfield code="e">RUSSIAN FEDERATION</subfield>
  </datafield>
</record>
</collection>"""


def _parse_record():
    root = ET.fromstring(_SAMPLE_MARC)
    return root.find("{http://www.loc.gov/MARC21/slim}record")


def test_parse_marc_record_extracts_country_votes():
    rows = parse_marc_record(_parse_record())
    assert len(rows) == 3
    by_code = {r["ms_code"]: r for r in rows}
    assert by_code["USA"]["ms_vote"] == "N"
    assert by_code["USA"]["ms_name"] == "UNITED STATES"
    assert by_code["GBR"]["ms_vote"] == "Y"
    # Country without a vote subfield → blank (sentinel for absent / not voting).
    assert by_code["RUS"]["ms_vote"].strip() == ""


def test_parse_marc_record_extracts_unbis_subjects_from_tag_991():
    """The key fix — UN DL stores subjects in custom tag 991 d, not standard 650."""
    rows = parse_marc_record(_parse_record())
    subjects = rows[0]["subjects"]
    assert "HUMAN RIGHTS ADVANCEMENT" in subjects
    assert "NATIONAL INSTITUTIONS--HUMAN RIGHTS" in subjects
    assert "|" in subjects  # multi-subject separator


def test_parse_marc_record_extracts_vote_totals_from_tag_996():
    """Tag 996 uses non-standard subfields b/c/d/e/f — verify the mapping."""
    rows = parse_marc_record(_parse_record())
    r = rows[0]
    assert r["total_yes"] == 167
    assert r["total_no"] == 2
    assert r["total_abstentions"] == 11
    assert r["total_non_voting"] == 13
    assert r["total_ms"] == 193


def test_parse_marc_record_extracts_title_resolution_meeting():
    rows = parse_marc_record(_parse_record())
    r = rows[0]
    assert r["title"].startswith("National human rights institutions")
    assert "resolution" in r["title"]
    assert r["resolution"] == "A/RES/80/214"
    assert r["session"] == "80"
    assert r["meeting"] == "A/80/PV.69"
    assert r["date"] == "2025-12-18"
    assert "4096564" in r["undl_link"]


def test_parse_marc_record_returns_empty_for_non_vote_record():
    """A MARC record without any 967 country fields isn't a vote — return []."""
    xml = """<?xml version="1.0"?>
    <collection xmlns="http://www.loc.gov/MARC21/slim">
    <record>
      <controlfield tag="001">9999</controlfield>
      <datafield tag="245"><subfield code="a">Not a vote</subfield></datafield>
    </record>
    </collection>"""
    rec = ET.fromstring(xml).find("{http://www.loc.gov/MARC21/slim}record")
    assert parse_marc_record(rec) == []


def test_parse_record_date_handles_missing_or_malformed():
    """The cutoff logic depends on this — must not raise on bad input."""
    assert _parse_record_date([]) is None
    assert _parse_record_date([{"date": ""}]) is None
    assert _parse_record_date([{"date": "not a date"}]) is None
    assert _parse_record_date([{"date": "2025-12-18"}]).strftime("%Y-%m-%d") == "2025-12-18"


def test_parser_falls_back_to_tag_650_when_991_missing():
    """If a record lacks tag 991 (older records use 650/653), we fall back."""
    xml = """<?xml version="1.0"?>
    <collection xmlns="http://www.loc.gov/MARC21/slim">
    <record>
      <controlfield tag="001">5</controlfield>
      <datafield tag="650"><subfield code="a">CLIMATE CHANGE</subfield></datafield>
      <datafield tag="967">
        <subfield code="c">USA</subfield>
        <subfield code="d">Y</subfield>
        <subfield code="e">UNITED STATES</subfield>
      </datafield>
    </record>
    </collection>"""
    rec = ET.fromstring(xml).find("{http://www.loc.gov/MARC21/slim}record")
    rows = parse_marc_record(rec)
    assert rows[0]["subjects"] == "CLIMATE CHANGE"
