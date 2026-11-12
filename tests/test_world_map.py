"""The bloc map: groups found from votes, one hue per group, flags in the captions."""

import pandas as pd

from src import world_map as W

WASHINGTON, MOSCOW = ["USA", "ISR", "ARG"], ["RUS", "PRK"]
WEST = ["GBR", "FRA", "DEU", "ITA", "ESP", "NLD", "BEL", "SWE", "NOR", "DNK", "CAN", "AUS"]
MAJORITY = ["CHN", "IND", "BRA", "ZAF", "NGA", "EGY", "IDN", "MEX", "PAK", "SAU", "KEN", "CHL",
            "PER", "VNM", "THA", "MAR", "DZA", "GHA", "COL", "BGD"]


def _frame(year=2025):
    rows = []
    for rcid in range(30):
        votes = {**{c: 1 for c in MAJORITY},
                 **{c: (-1 if rcid % 5 == 0 else 1) for c in WEST},
                 **{c: (1 if rcid % 6 == 0 else -1) for c in WASHINGTON},
                 **{c: (-1 if rcid % 3 == 1 else 1) for c in MOSCOW}}
        for code, vote in votes.items():
            rows.append({"rcid": rcid, "year": year, "date": f"{year}-12-0{1 + rcid % 9}", "issue": f"Vote {rcid}",
                         "country_identifier": code, "country_name": code, "vote": vote})
    return pd.DataFrame(rows)


def test_flags_come_from_iso_codes_and_defunct_states_have_none():
    assert W.flag("USA") == "🇺🇸" and W.flag("gbr") == "🇬🇧" and W.flag("PRK") == "🇰🇵"
    assert W.flag("SUN") == "" and W.flag("") == ""


def test_blocs_are_found_from_the_votes_and_named_for_what_they_hold():
    blocs = W.voting_blocs(_frame(), 2025)
    by_key = {b["key"]: set(b["members"]) for b in blocs["blocs"]}
    assert by_key == {"washington": set(WASHINGTON), "west": set(WEST),
                      "majority": set(MAJORITY), "moscow": set(MOSCOW)}
    assert [b["key"] for b in blocs["blocs"]] == ["washington", "west", "majority", "moscow"]
    assert blocs["members"]["CHN"] == {"bloc": "majority", "tightness": 1.0}
    assert blocs["between"]["washington|majority"] < blocs["between"]["west|majority"]
    assert W.voting_blocs(_frame()[lambda d: d.country_identifier.isin(WEST)], 2025) == {}   # too few to group


def test_map_colours_each_bloc_and_captions_carry_flags():
    blocs = W.voting_blocs(_frame(), 2025)
    names = {"USA": "United States", "ISR": "Israel", "ARG": "Argentina", "RUS": "Russia", "PRK": "North Korea"}
    title = W.bloc_title(blocs, names)
    assert title == "Only Argentina and Israel vote with Washington"
    svg = W.bloc_map_svg(blocs, title)
    assert svg.startswith("<svg") and 'style="' not in svg.split(">", 1)[0]
    for key in ("washington", "west", "majority", "moscow"):
        assert any(shade in svg for shade in W.BLOCS[key][1]), key
    assert W.NO_DATA in svg and svg.count("<path") > 150 and len(svg) < 200_000
    lines = W.bloc_lines(blocs, names)
    assert lines[0].startswith("Votes with Washington (3;") and "🇺🇸 United States" in lines[0] and "🇮🇱 Israel" in lines[0]
    assert "including 🇨🇳 CHN" in lines[2] and "more." in lines[2]
    assert lines[3].startswith("Votes with Moscow (2;") and "🇷🇺 Russia" in lines[3]


def test_no_outline_streaks_across_the_date_line():
    import re
    for code, path in W._outlines():
        xs = [float(x) for x in re.findall(r"[ML](-?[\d.]+),", path)]
        jumps = [abs(a - b) for a, b in zip(xs, xs[1:])]
        # Within one ring, neighbouring points are close; a streak would jump most of the width.
        rings = path.split("Z")
        for ring in rings:
            rx = [float(x) for x in re.findall(r"[ML](-?[\d.]+),", ring)]
            assert all(abs(a - b) < 250 for a, b in zip(rx, rx[1:])), code
        assert jumps is not None
