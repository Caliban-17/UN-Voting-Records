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
* **Regime type** (liberalism): V-Dem's Regimes of the World (closed
  autocracy, electoral autocracy, electoral democracy, liberal democracy)
  and its liberal democracy index, via Our World in Data.
* **Women's representation** (feminist IR): the share of seats held by
  women, IPU via the World Bank from 1997, V-Dem's series before that.
* **Defence-pact communities** (realism, data-derived): connected components
  of the Correlates of War defence-pact graph, to 2012 where the data end.

All three are built by ``scripts/build_supplementary.py`` into ``data/``.

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
VDEM_CSV = BASE_DIR / "data" / "vdem_regimes.csv"
WOMEN_CSV = BASE_DIR / "data" / "women_in_parliament.csv"
COW_CSV = BASE_DIR / "data" / "cow_defense_communities.csv"
COLONIAL_CSV = BASE_DIR / "data" / "colonial_history.csv"

US_LED, SOVIET_LED, NON_ALIGNED = "US-led", "Soviet/Russian-led", "Non-aligned"
CORE, SEMI, PERIPHERY = "Core", "Semi-periphery", "Periphery"
REGIME_LABELS = {0: "Closed autocracy", 1: "Electoral autocracy", 2: "Electoral democracy", 3: "Liberal democracy"}
DEMOCRACY, AUTOCRACY = "Democracy", "Autocracy"
NO_PACT = "No defence pact"
REPRESENTATION_LABELS = ("Fewest women in parliament", "Middle third", "Most women in parliament")
COLONIAL_POWERS, SETTLER_STATES = "Former colonial powers", "Settler states"
DECOLONISED_WAVE, OLDER_EXCOLONIES, NEVER_COLONISED = (
    "Independent since 1945", "Independent before 1945", "Never colonised or other",
)
NORTH_GROUPS = {COLONIAL_POWERS, SETTLER_STATES}
# States that ruled overseas empires in the twentieth century (post-colonial
# theory's metropoles), and the settler-colonial states it groups with them.
COLONIAL_POWER_CODES = {"GBR", "FRA", "ESP", "PRT", "NLD", "BEL", "ITA", "DEU", "GER", "USA", "JPN", "DNK"}
SETTLER_STATE_CODES = {"CAN", "AUS", "NZL", "ISR"}

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

# ── Self-constituted identity groups (constructivism) ───────────────────────
# Organisations whose membership is an identity claim rather than a treaty of
# defence or a level of income: Europe, the Arab nation, the Islamic ummah,
# non-alignment. (code, joined, left or None.)

EUROPEAN_UNION: list[tuple[str, int, Optional[int]]] = [
    *[(c, 1958, None) for c in "BEL FRA DEU LUX NLD ITA".split()], ("GER", 1958, 1990),
    ("DNK", 1973, None), ("IRL", 1973, None), ("GBR", 1973, 2020),
    ("GRC", 1981, None), ("ESP", 1986, None), ("PRT", 1986, None),
    ("AUT", 1995, None), ("FIN", 1995, None), ("SWE", 1995, None),
    *[(c, 2004, None) for c in "CYP CZE EST HUN LVA LTU MLT POL SVK SVN".split()],
    ("BGR", 2007, None), ("ROU", 2007, None), ("HRV", 2013, None),
]

ARAB_LEAGUE: list[tuple[str, int, Optional[int]]] = [
    *[(c, 1945, None) for c in "EGY IRQ JOR LBN SAU SYR".split()], ("YEM", 1945, None), ("YMD", 1967, 1990),
    ("LBY", 1953, None), ("SDN", 1956, None), ("MAR", 1958, None), ("TUN", 1958, None), ("KWT", 1961, None),
    ("DZA", 1962, None), ("BHR", 1971, None), ("OMN", 1971, None), ("QAT", 1971, None), ("ARE", 1971, None),
    ("MRT", 1973, None), ("SOM", 1974, None), ("DJI", 1977, None), ("COM", 1993, None),
]

# Organisation of Islamic Cooperation, by year of accession.
ISLAMIC_COOPERATION: list[tuple[str, int, Optional[int]]] = [
    *[(c, 1969, None) for c in "AFG DZA TCD EGY GIN IDN IRN JOR KWT LBN LBY MYS MLI MRT MAR NER PAK SAU SEN SOM SDN TUN TUR YEM".split()],
    ("YMD", 1969, 1990),
    ("BHR", 1970, None), ("OMN", 1970, None), ("QAT", 1970, None), ("SYR", 1970, None), ("ARE", 1971, None),
    ("SLE", 1972, None), ("BGD", 1974, None), ("GAB", 1974, None), ("GMB", 1974, None), ("GNB", 1974, None),
    ("UGA", 1974, None), ("BFA", 1975, None), ("CMR", 1975, None), ("COM", 1976, None), ("IRQ", 1976, None),
    ("MDV", 1976, None), ("DJI", 1978, None), ("BEN", 1982, None), ("BRN", 1984, None), ("NGA", 1986, None),
    ("AZE", 1991, None), ("ALB", 1992, None), ("KGZ", 1992, None), ("TJK", 1992, None), ("TKM", 1992, None),
    ("MOZ", 1994, None), ("KAZ", 1995, None), ("UZB", 1995, None), ("SUR", 1996, None), ("TGO", 1997, None),
    ("GUY", 1998, None), ("CIV", 2001, None),
]

# Non-Aligned Movement, by year of accession (the movement's summit records
# as compiled on Wikipedia, 2026-09); former members with their leaving year.
NON_ALIGNED_MOVEMENT: list[tuple[str, int, Optional[int]]] = [
    ("AFG", 1961, None), ("COD", 1961, None), ("CUB", 1961, None), ("CYP", 1961, 2004), ("DZA", 1961, None), ("EGY", 1961, None),
    ("ETH", 1961, None), ("GHA", 1961, None), ("GIN", 1961, None), ("IDN", 1961, None), ("IND", 1961, None), ("IRQ", 1961, None),
    ("KHM", 1961, None), ("LBN", 1961, None), ("LKA", 1961, None), ("MAR", 1961, None), ("MLI", 1961, None), ("MMR", 1961, None),
    ("NPL", 1961, None), ("SAU", 1961, None), ("SDN", 1961, None), ("SOM", 1961, None), ("TUN", 1961, None), ("YEM", 1961, 1990),
    ("YUG", 1961, 1992), ("BDI", 1964, None), ("BEN", 1964, None), ("CAF", 1964, None), ("CMR", 1964, None), ("COG", 1964, None),
    ("JOR", 1964, None), ("KEN", 1964, None), ("KWT", 1964, None), ("LAO", 1964, None), ("LBR", 1964, None), ("LBY", 1964, None),
    ("MRT", 1964, None), ("MWI", 1964, None), ("NGA", 1964, None), ("SEN", 1964, None), ("SLE", 1964, None), ("SYR", 1964, None),
    ("TCD", 1964, None), ("TGO", 1964, None), ("TZA", 1964, None), ("UGA", 1964, None), ("ZMB", 1964, None), ("ARE", 1970, None),
    ("BWA", 1970, None), ("GAB", 1970, None), ("GNQ", 1970, None), ("GUY", 1970, None), ("JAM", 1970, None), ("LSO", 1970, None),
    ("MYS", 1970, None), ("RWA", 1970, None), ("SGP", 1970, None), ("SWZ", 1970, None), ("TTO", 1970, None), ("YMD", 1970, 1990),
    ("CHL", 1971, 2026), ("ARG", 1973, 1991), ("BFA", 1973, None), ("BGD", 1973, None), ("BHR", 1973, None), ("BTN", 1973, None),
    ("CIV", 1973, None), ("GMB", 1973, None), ("MDG", 1973, None), ("MLT", 1973, 2004), ("MUS", 1973, None), ("NER", 1973, None),
    ("OMN", 1973, None), ("PER", 1973, None), ("QAT", 1973, None), ("PRK", 1975, None), ("AGO", 1976, None), ("COM", 1976, None),
    ("CPV", 1976, None), ("GNB", 1976, None), ("MDV", 1976, None), ("MOZ", 1976, None), ("PAN", 1976, None), ("STP", 1976, None),
    ("SYC", 1976, None), ("VNM", 1976, None), ("BOL", 1979, None), ("GRD", 1979, None), ("IRN", 1979, None), ("NAM", 1979, None),
    ("NIC", 1979, None), ("PAK", 1979, None), ("ZWE", 1979, None), ("BLZ", 1981, None), ("BHS", 1983, None), ("BRB", 1983, None),
    ("COL", 1983, None), ("DJI", 1983, None), ("ECU", 1983, None), ("LCA", 1983, None), ("SUR", 1983, None), ("VUT", 1983, None),
    ("VEN", 1989, None), ("YEM", 1990, None), ("BRN", 1993, None), ("GTM", 1993, None), ("MNG", 1993, None), ("PHL", 1993, None),
    ("PNG", 1993, None), ("THA", 1993, None), ("UZB", 1993, None), ("ZAF", 1994, None), ("ERI", 1995, None), ("HND", 1995, None),
    ("TKM", 1995, None), ("BLR", 1998, None), ("DOM", 2000, None), ("TLS", 2003, None), ("VCT", 2003, None), ("ATG", 2006, None),
    ("DMA", 2006, None), ("HTI", 2006, None), ("KNA", 2006, None), ("AZE", 2011, None), ("FJI", 2011, None), ("SSD", 2024, None),
]

IDENTITY_EU, IDENTITY_ARAB, IDENTITY_OIC, IDENTITY_NAM, IDENTITY_NONE = (
    "European Union", "Arab League", "Islamic Cooperation", "Non-Aligned", "No identity group",
)


def identity_group(code: str, year: int) -> str:
    """One identity label per member and year, by precedence: EU, Arab League,
    OIC (non-Arab), NAM (other), none. Overlaps are real (every Arab League
    member is in the OIC and most are non-aligned); the precedence keeps the
    most specific claim."""
    code = str(code).strip().upper()
    if _member(EUROPEAN_UNION, code, year):
        return IDENTITY_EU
    if _member(ARAB_LEAGUE, code, year):
        return IDENTITY_ARAB
    if _member(ISLAMIC_COOPERATION, code, year):
        return IDENTITY_OIC
    if _member(NON_ALIGNED_MOVEMENT, code, year):
        return IDENTITY_NAM
    return IDENTITY_NONE


def identity_memberships(code: str, year: int) -> dict[str, bool]:
    code = str(code).strip().upper()
    return {
        "eu": _member(EUROPEAN_UNION, code, year),
        "arab": _member(ARAB_LEAGUE, code, year),
        "oic": _member(ISLAMIC_COOPERATION, code, year),
        "nam": _member(NON_ALIGNED_MOVEMENT, code, year),
    }


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


@lru_cache(maxsize=1)
def _vdem_table() -> dict[str, dict[int, tuple[Optional[int], Optional[float]]]]:
    table: dict[str, dict[int, tuple[Optional[int], Optional[float]]]] = {}
    if not VDEM_CSV.exists():
        return table
    with VDEM_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            regime = int(row["regime"]) if row["regime"] else None
            libdem = float(row["libdem"]) if row["libdem"] else None
            table.setdefault(row["code"], {})[int(row["year"])] = (regime, libdem)
    return table


def regime_type(code: str, year: int) -> Optional[int]:
    """Regimes of the World category (0–3) for a member in a year, or None."""
    return _vdem_table().get(str(code).strip().upper(), {}).get(year, (None, None))[0]


def liberal_democracy_index(code: str, year: int) -> Optional[float]:
    """V-Dem's liberal democracy index (0–1) for a member in a year, or None."""
    return _vdem_table().get(str(code).strip().upper(), {}).get(year, (None, None))[1]


@lru_cache(maxsize=1)
def _women_table() -> dict[str, dict[int, float]]:
    table: dict[str, dict[int, float]] = {}
    if not WOMEN_CSV.exists():
        return table
    with WOMEN_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table.setdefault(row["code"], {})[int(row["year"])] = float(row["share"])
    return table


def women_in_parliament(code: str, year: int) -> Optional[float]:
    """Share of parliamentary seats held by women (percent), or None."""
    return _women_table().get(str(code).strip().upper(), {}).get(year)


@lru_cache(maxsize=1)
def _cow_table() -> dict[str, dict[int, str]]:
    table: dict[str, dict[int, str]] = {}
    if not COW_CSV.exists():
        return table
    with COW_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table.setdefault(row["code"], {})[int(row["year"])] = row["community"]
    return table


def cow_years() -> tuple[Optional[int], Optional[int]]:
    """First and last year the defence-pact data cover."""
    years = {y for groups in _cow_table().values() for y in groups}
    return (min(years), max(years)) if years else (None, None)


def defense_pact_community(code: str, year: int) -> Optional[str]:
    """The defence-pact component a member sits in that year; ``NO_PACT``
    when it has none; None outside the years the data cover."""
    first, last = cow_years()
    if first is None or not first <= year <= last:
        return None
    return _cow_table().get(str(code).strip().upper(), {}).get(year, NO_PACT)


@lru_cache(maxsize=1)
def _colonial_table() -> dict[str, dict]:
    table: dict[str, dict] = {}
    if not COLONIAL_CSV.exists():
        return table
    with COLONIAL_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[row["code"]] = {
                "ruler": row["ruler"] or None,
                "independence_year": int(row["independence_year"]) if row["independence_year"] else None,
                "independence_type": row["independence_type"],
                "overseas_empire": row["overseas_empire"] == "1",
            }
    return table


def former_ruler(code: str) -> Optional[str]:
    """ISO-3 of the state's former colonial ruler (an overseas empire), or None."""
    row = _colonial_table().get(str(code).strip().upper())
    return row["ruler"] if row and row["overseas_empire"] else None


def colonial_group(code: str, year: int) -> Optional[str]:
    """Post-colonial theory's line: former colonial powers, settler states,
    the post-1945 decolonisation wave, older ex-colonies (Latin America and
    the like), and states never colonised by an overseas empire. Apartheid
    South Africa counts as a settler state to 1993."""
    code = str(code).strip().upper()
    if code in COLONIAL_POWER_CODES:
        return COLONIAL_POWERS
    if code in SETTLER_STATE_CODES or (code == "ZAF" and year <= 1993):
        return SETTLER_STATES
    row = _colonial_table().get(code)
    if row is None:
        return None
    # ICOW types: 2 decolonisation, 4 partition of a decolonised territory (Korea, Viet Nam)
    if row["overseas_empire"] and row["independence_type"] in ("2", "4") and row["independence_year"]:
        return DECOLONISED_WAVE if row["independence_year"] >= 1945 else OLDER_EXCOLONIES
    return NEVER_COLONISED


def representation_terciles(codes: list[str], year: int) -> dict[str, str]:
    """Members split into thirds by women's share of parliament that year."""
    shares = {c: women_in_parliament(c, year) for c in codes}
    known = sorted((v, c) for c, v in shares.items() if v is not None)
    if len(known) < 6:
        return {}
    third = len(known) // 3
    out: dict[str, str] = {}
    for i, (_, code) in enumerate(known):
        out[code] = REPRESENTATION_LABELS[min(2, i // third)] if third else REPRESENTATION_LABELS[1]
    return out


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
        elif scheme == "regime":
            regime = regime_type(code, year)
            if regime is not None:
                out[code] = REGIME_LABELS[regime]
        elif scheme == "democracy":
            regime = regime_type(code, year)
            if regime is not None:
                out[code] = DEMOCRACY if regime >= 2 else AUTOCRACY
        elif scheme == "pact":
            community = defense_pact_community(code, year)
            if community is not None:
                out[code] = community
        elif scheme == "representation":
            return representation_terciles(codes, year)
        elif scheme == "identity":
            out[code] = identity_group(code, year)
        elif scheme == "colonial":
            group = colonial_group(code, year)
            if group:
                out[code] = group
        elif scheme == "north_south":
            # the metropoles and settler states against the ex-colonies; the
            # never-colonised (the Soviet bloc, Ottoman successors) sit out
            group = colonial_group(code, year)
            if group in NORTH_GROUPS:
                out[code] = "North"
            elif group in (DECOLONISED_WAVE, OLDER_EXCOLONIES):
                out[code] = "South"
        elif scheme in ("eu", "arab", "oic", "nam"):
            out[code] = "Member" if identity_memberships(code, year)[scheme] else "Not a member"
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
    return out
