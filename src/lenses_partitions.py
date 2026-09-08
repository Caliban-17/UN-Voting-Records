"""
The partitions each theory of international relations says should organise
UN voting, as year-aware lookups.

* **Alliance camps** (realism): which members were bound to Washington or to
  Moscow by treaty in a given year — NATO and the United States' bilateral
  defence treaties on one side; the Warsaw Pact, the Soviet Union's aligned
  states and, after 1992, the Collective Security Treaty (Organization) on
  the other. Everyone else is non-aligned for that year.
* **World-system tiers** (world-systems theory): core, semi-periphery and
  periphery, read off the World Bank's historical income classification
  (high, upper-middle, lower-middle/low) from 1987, carried back to earlier
  years with the socialist bloc placed in the semi-periphery, as Wallerstein
  placed it.
* **The feminist-foreign-policy cohort** (feminist IR): states that adopted
  an explicit feminist foreign policy, with the year they did.
* **Regional groups** (constructivism's institutional identities) come from
  :mod:`src.regional_groups`.

Every list is curated from the public record and deliberately short; each
is a proxy and is labelled as one in the UI. Change a year here, not in
the analysis.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.regional_groups import regional_group

BASE_DIR = Path(__file__).resolve().parent.parent
INCOME_CSV = BASE_DIR / "data" / "world_bank_income_groups.csv"

US_LED, SOVIET_LED, NON_ALIGNED = "US-led", "Soviet/Russian-led", "Non-aligned"
CORE, SEMI, PERIPHERY = "Core", "Semi-periphery", "Periphery"

# (code, first year, last year or None) — membership of the US-led camp.
NATO_AND_US_TREATY_ALLIES: list[tuple[str, int, Optional[int]]] = [
    # NATO, by accession
    *[(c, 1949, None) for c in "USA CAN GBR FRA BEL NLD LUX DNK NOR ISL ITA PRT".split()],
    ("GRC", 1952, None), ("TUR", 1952, None),
    ("GER", 1955, 1990), ("DEU", 1955, None),
    ("ESP", 1982, None),
    ("POL", 1999, None), ("CZE", 1999, None), ("HUN", 1999, None),
    *[(c, 2004, None) for c in "BGR EST LVA LTU ROU SVK SVN".split()],
    ("ALB", 2009, None), ("HRV", 2009, None), ("MNE", 2017, None), ("MKD", 2020, None),
    ("FIN", 2023, None), ("SWE", 2024, None),
    # United States bilateral / regional defence treaties
    ("JPN", 1952, None), ("KOR", 1953, None), ("PHL", 1951, None),
    ("AUS", 1951, None), ("NZL", 1951, 1986), ("THA", 1954, None),
    ("ISR", 1967, None),  # no treaty, but the closest partnership in the record
]

SOVIET_AND_RUSSIAN_LED: list[tuple[str, int, Optional[int]]] = [
    ("SUN", 1946, 1991),
    # Warsaw Pact 1955–1991
    ("POL", 1955, 1991), ("CSK", 1955, 1991), ("HUN", 1955, 1991), ("ROU", 1955, 1991),
    ("BGR", 1955, 1991), ("DDR", 1956, 1990), ("ALB", 1955, 1968),
    # aligned outside the Pact
    ("MNG", 1962, 1991), ("CUB", 1972, 1991), ("VNM", 1978, 1991),
    # Collective Security Treaty 1992–2001, then the CSTO
    ("RUS", 1992, None), ("ARM", 1992, 2023), ("KAZ", 1992, None), ("KGZ", 1992, None),
    ("TJK", 1992, None), ("BLR", 1993, None), ("UZB", 1992, 1999), ("UZB", 2006, 2012),
    ("AZE", 1993, 1999), ("GEO", 1993, 1999),
]

# States that adopted an explicit feminist foreign policy, with the year.
FEMINIST_FOREIGN_POLICY: list[tuple[str, int, Optional[int]]] = [
    ("SWE", 2014, 2022), ("CAN", 2017, None), ("FRA", 2019, None), ("LUX", 2019, None),
    ("MEX", 2020, None), ("ESP", 2021, None), ("DEU", 2022, None), ("CHL", 2022, None),
    ("NLD", 2022, None), ("COL", 2022, None), ("LBR", 2022, None), ("SVN", 2023, None),
    ("MNG", 2023, None),
]

# Pre-1987 placement for states the World Bank series does not cover, or
# that Wallerstein placed differently from their later income group.
SOCIALIST_BLOC_SEMI_PERIPHERY = {"SUN", "CSK", "DDR", "POL", "HUN", "ROU", "BGR", "YUG", "ALB", "MNG", "CUB"}
HISTORICAL_TIERS = {"GER": CORE, "SUN": SEMI, "CSK": SEMI, "DDR": SEMI, "YUG": SEMI, "SCG": SEMI,
                    "YMD": PERIPHERY, "EAT": PERIPHERY, "EAZ": PERIPHERY}
INCOME_TO_TIER = {"H": CORE, "UM": SEMI, "LM": PERIPHERY, "L": PERIPHERY}


def _member(spans: list[tuple[str, int, Optional[int]]], code: str, year: int) -> bool:
    return any(c == code and start <= year <= (end or 9999) for c, start, end in spans)


def alliance_camp(code: str, year: int) -> str:
    code = str(code).strip().upper()
    if _member(SOVIET_AND_RUSSIAN_LED, code, year):
        return SOVIET_LED
    if _member(NATO_AND_US_TREATY_ALLIES, code, year):
        return US_LED
    return NON_ALIGNED


def feminist_foreign_policy(code: str, year: int) -> bool:
    return _member(FEMINIST_FOREIGN_POLICY, str(code).strip().upper(), year)


@lru_cache(maxsize=1)
def _income_table() -> dict[str, dict[int, str]]:
    table: dict[str, dict[int, str]] = {}
    if not INCOME_CSV.exists():
        return table
    with INCOME_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table.setdefault(row["code"], {})[int(row["year"])] = row["group"]
    return table


def world_system_tier(code: str, year: int) -> Optional[str]:
    """Core / Semi-periphery / Periphery for a member in a year, or None when
    the record has nothing to say (a state the World Bank never classified)."""
    code = str(code).strip().upper()
    if code in HISTORICAL_TIERS:
        return HISTORICAL_TIERS[code]
    groups = _income_table().get(code)
    if not groups:
        return None
    if year in groups:
        return INCOME_TO_TIER[groups[year]]
    years = sorted(groups)
    if year < years[0]:
        if code in SOCIALIST_BLOC_SEMI_PERIPHERY and year <= 1991:
            return SEMI
        return INCOME_TO_TIER[groups[years[0]]]
    # gaps or years after the last edition: carry the nearest earlier value
    earlier = [y for y in years if y < year]
    return INCOME_TO_TIER[groups[earlier[-1]]] if earlier else None


def partition_labels(codes: list[str], year: int, scheme: str) -> dict[str, str]:
    """``{code: group}`` for one scheme and year; codes the scheme cannot
    place are left out."""
    out: dict[str, str] = {}
    for code in codes:
        if scheme == "alliance":
            out[code] = alliance_camp(code, year)
        elif scheme == "tier":
            tier = world_system_tier(code, year)
            if tier:
                out[code] = tier
        elif scheme == "region":
            group = regional_group(code)
            if group != "Other":
                out[code] = group
        elif scheme == "ffp":
            out[code] = "Feminist foreign policy" if feminist_foreign_policy(code, year) else "Other members"
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
    return out
