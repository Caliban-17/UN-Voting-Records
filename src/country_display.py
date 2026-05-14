"""
Hand-curated short display names for UN-listed countries.

UN Digital Library uses legal long-form names like
``Iran (Islamic Republic Of)`` and ``Korea (Democratic People's Republic Of)``.
Those read awkwardly in journalistic headlines and pull-quotes. This module
maps each ISO-3 code to the form an editor would actually use in a sentence
(e.g. "Iran", "North Korea"), while preserving the legal form for
methodology / footnotes elsewhere in the dataset (``country_name`` column).

Entries are only listed where the override differs from a naive
parenthetical-strip — countries whose UN long form is the same as their
common name fall through to the raw ``country_name``.
"""

from __future__ import annotations

import re

# Each entry: ISO-3 → "the name a journalist would type"
COUNTRY_DISPLAY_NAMES: dict[str, str] = {
    # P5
    "USA": "United States",
    "GBR": "United Kingdom",
    "RUS": "Russia",
    "CHN": "China",
    "FRA": "France",
    # G7 / G20 misc
    "DEU": "Germany",
    "KOR": "South Korea",
    "PRK": "North Korea",
    "JPN": "Japan",
    "ITA": "Italy",
    "BRA": "Brazil",
    "IND": "India",
    "MEX": "Mexico",
    "ZAF": "South Africa",
    "TUR": "Turkey",
    "SAU": "Saudi Arabia",
    # Middle East / North Africa
    "IRN": "Iran",
    "IRQ": "Iraq",
    "SYR": "Syria",
    "ISR": "Israel",
    "PSE": "Palestine",
    "ARE": "United Arab Emirates",
    "YEM": "Yemen",
    "LBN": "Lebanon",
    "LBY": "Libya",
    "EGY": "Egypt",
    # Disambiguating-needed states
    "VEN": "Venezuela",
    "BOL": "Bolivia",
    "FSM": "Micronesia",
    "MDA": "Moldova",
    "TZA": "Tanzania",
    "LAO": "Laos",
    "MMR": "Myanmar",
    "CIV": "Côte d'Ivoire",
    "CZE": "Czechia",
    "MKD": "North Macedonia",
    "SWZ": "Eswatini",
    "BRN": "Brunei",
    "PRY": "Paraguay",
    "TLS": "Timor-Leste",
    "COD": "DR Congo",
    "COG": "Republic of Congo",
    # Other long-form annoyances
    "VAT": "Vatican City",
    "ATG": "Antigua and Barbuda",
    "TTO": "Trinidad and Tobago",
    "VCT": "Saint Vincent and the Grenadines",
    "KNA": "Saint Kitts and Nevis",
    "BIH": "Bosnia and Herzegovina",
}

# Patterns we strip from a raw UN long-form when no explicit override is given.
# Order matters — strip the parenthetical first.
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")
_TRAILING_OF = re.compile(r",\s+the\s+former\s+yugoslav\s+republic\s+of\b", re.IGNORECASE)


def _strip_parenthetical(name: str) -> str:
    """Strip a trailing parenthetical clause from a UN long-form name."""
    return _PARENTHETICAL.sub("", str(name or "")).strip()


def display_name(code: str | None, raw_name: str | None) -> str:
    """Return the short editorial-quality name for one country.

    Falls back to a parenthetical-stripped version of ``raw_name`` when no
    explicit override is set; falls back to ``code`` if even that is empty.
    """
    if code:
        code = str(code).strip().upper()
        if code in COUNTRY_DISPLAY_NAMES:
            return COUNTRY_DISPLAY_NAMES[code]
    if raw_name:
        cleaned = _strip_parenthetical(raw_name)
        if cleaned:
            return cleaned
    return code or "?"


def display_lookup(raw_lookup: dict[str, str] | None) -> dict[str, str]:
    """Apply ``display_name`` to every entry in a ``{code: raw_name}`` map."""
    if not raw_lookup:
        return {}
    return {code: display_name(code, name) for code, name in raw_lookup.items()}
