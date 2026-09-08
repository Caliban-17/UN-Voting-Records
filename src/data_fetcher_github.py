"""
Voting records from DGACM's machine-readable resolutions on GitHub.

Since September 2026 the UN Digital Library fronts every page with an AWS
WAF bot challenge (``x-amzn-waf-action: challenge``), and its robots.txt has
always disallowed ``/search``, so the MARC-XML route in
``src/data_fetcher_marc.py`` is closed to scripts and stays opt-in only.

The Department for General Assembly and Conference Management publishes
the same resolutions as Akoma Ntoso XML on GitHub —
https://github.com/UNxml/GAresolutions — with a JSON-lines extract per
session under ``data_extract/`` carrying, for every resolution adopted by
recorded vote: the members in favour, against and abstaining, the tallies,
the adoption date, the plenary meeting and the UNBIS subjects. It is
unofficial ("for informational purposes only"), requires attribution to the
United Nations, and is uploaded in batches — weeks to months behind the
plenary — which README states plainly.

What the extract lacks and how this module fills it:

* ISO-3 codes: names are mapped through the existing CSV's own name→code
  pairs, with a short alias table for renamed or re-spelled states.
* Absentees: every current member not in any list is recorded with a
  blank vote, the historical CSV's convention for absent / not voting.
* Record ids: the library's ``undl_id`` is unknown here, so a stable
  synthetic id is derived from the symbol (2 000 000 000 + CRC32 — far above
  any real id), and the refresh drops a synthetic row whenever a real one
  for the same resolution and member exists.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import zlib
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

REPO = "UNxml/GAresolutions"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/main/data_extract/"
API_CONTENTS = f"https://api.github.com/repos/{REPO}/contents/data_extract"
SYNTHETIC_ID_BASE = 2_000_000_000
SOURCE_NOTE = "DGACM machine-readable resolutions (github.com/UNxml/GAresolutions)"

# Raw or normalised spellings the extract uses that the CSV's own names do
# not cover. Keys are matched after ``normalize_name``; a few pre-normalised
# forms are listed too, for mojibake that normalisation cannot repair.
NAME_ALIASES: dict[str, str] = {
    "COTE D'IVOIRE": "CIV",
    "IVORY COAST": "CIV",
    "NETHERLANDS": "NLD",
    "NETHERLANDS (KINGDOM OF THE)": "NLD",
    "TURKEY": "TUR",
    "TURKIYE": "TUR",
    "TURKYYE": "TUR",       # "TÜRKÝYE" after accent stripping
    "CZECH REPUBLIC": "CZE",
    "CZECHIA": "CZE",
    "SWAZILAND": "SWZ",
    "ESWATINI": "SWZ",
    "CAPE VERDE": "CPV",
    "CABO VERDE": "CPV",
    "NORTH MACEDONIA": "MKD",
    "THE FORMER YUGOSLAV REPUBLIC OF MACEDONIA": "MKD",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "UNITED KINGDOM": "GBR",
    "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND": "GBR",
    "RUSSIAN FEDERATION": "RUS",
    "IRAN (ISLAMIC REPUBLIC OF)": "IRN",
    "BOLIVIA (PLURINATIONAL STATE OF)": "BOL",
    "VENEZUELA (BOLIVARIAN REPUBLIC OF)": "VEN",
    "MICRONESIA (FEDERATED STATES OF)": "FSM",
    "SYRIAN ARAB REPUBLIC": "SYR",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "LAO",
    "VIET NAM": "VNM",
    "REPUBLIC OF KOREA": "KOR",
    "DEMOCRATIC PEOPLE'S REPUBLIC OF KOREA": "PRK",
    "REPUBLIC OF MOLDOVA": "MDA",
    "UNITED REPUBLIC OF TANZANIA": "TZA",
    "BRUNEI DARUSSALAM": "BRN",
    "DEMOCRATIC REPUBLIC OF THE CONGO": "COD",
    "CONGO": "COG",
    "TIMOR-LESTE": "TLS",
    "MYANMAR": "MMR",
}

_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", " ": " "})


def normalize_name(name: str) -> str:
    """Upper-case, straight quotes, single spaces, accents stripped."""
    text = str(name or "").translate(_QUOTES)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().upper()


def build_code_lookup(existing_df: Optional[pd.DataFrame], years_back: int = 3) -> dict[str, str]:
    """``{normalised name: ISO-3}`` from the CSV's recent rows plus aliases."""
    lookup: dict[str, str] = {}
    if existing_df is not None and not existing_df.empty and {"ms_name", "ms_code"} <= set(existing_df.columns):
        frame = existing_df[["ms_name", "ms_code", "date"]].dropna(subset=["ms_name", "ms_code"])
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.notna().any():
            frame = frame[dates >= dates.max() - pd.Timedelta(days=365 * years_back)]
        for name, code in zip(frame["ms_name"], frame["ms_code"]):
            code = str(code).strip().upper()
            if len(code) == 3:
                lookup.setdefault(normalize_name(name), code)
    for name, code in NAME_ALIASES.items():
        lookup.setdefault(normalize_name(name), code)
    return lookup


def names_by_code(existing_df: Optional[pd.DataFrame], years_back: int = 3) -> dict[str, str]:
    """``{ISO-3: name as the CSV writes it}`` from recent rows, for absentees."""
    out: dict[str, str] = {}
    if existing_df is None or existing_df.empty or not {"ms_name", "ms_code"} <= set(existing_df.columns):
        return out
    frame = existing_df[["ms_name", "ms_code", "date"]].dropna(subset=["ms_name", "ms_code"])
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.notna().any():
        frame = frame[dates >= dates.max() - pd.Timedelta(days=365 * years_back)]
    for name, code in zip(frame["ms_name"], frame["ms_code"]):
        code = str(code).strip().upper()
        if len(code) == 3 and str(name).strip():
            out.setdefault(code, str(name).strip())
    return out


def current_members(existing_df: Optional[pd.DataFrame], years_back: int = 2) -> list[str]:
    """ISO-3 codes that cast or were recorded for a vote in the last
    ``years_back`` years of the CSV — the members to mark as not voting."""
    if existing_df is None or existing_df.empty or "ms_code" not in existing_df.columns:
        return []
    dates = pd.to_datetime(existing_df["date"], errors="coerce")
    recent = existing_df[dates >= dates.max() - pd.Timedelta(days=365 * years_back)]
    return sorted(str(c).strip().upper() for c in recent["ms_code"].dropna().unique() if len(str(c).strip()) == 3)


def parse_adoption_date(text: str) -> Optional[str]:
    """'29\\xa0October 2025' → '2025-10-29'; None when unparseable."""
    cleaned = normalize_name(text).title()
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def session_from_symbol(symbol: str) -> str:
    """'A/RES/80/4' → '80'; 'A/RES/ES-11/1' → 'ES-11'; 'A/RES/S-32/1' → 'S-32'."""
    m = re.search(r"A/RES/((?:ES-|S-)?\d+)/", str(symbol or ""))
    return m.group(1) if m else ""


def synthetic_undl_id(symbol: str) -> int:
    return SYNTHETIC_ID_BASE + zlib.crc32(str(symbol).strip().encode("utf-8"))


def is_synthetic_id(value) -> bool:
    try:
        return int(value) >= SYNTHETIC_ID_BASE
    except (TypeError, ValueError):
        return False


def parse_jsonl(text: str) -> list[dict]:
    """The extracts are JSON lines; accept a JSON array too."""
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        return json.loads(stripped)
    return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def _is_recorded_vote(entry: dict) -> bool:
    return (
        str(entry.get("adoption_type") or "").strip().lower() == "by a recorded vote"
        or str(entry.get("voting_type") or "").strip() == "adoptedRecordedVote"
    )


def _count(entry: dict, key: str, fallback: int) -> int:
    value = str(entry.get(key) or "").strip()
    return int(value) if value.isdigit() else fallback


def extract_to_rows(
    entries: Iterable[dict],
    code_lookup: dict[str, str],
    members: Iterable[str],
    since_date: Optional[str] = None,
    member_names: Optional[dict[str, str]] = None,
) -> list[dict]:
    """One CSV-schema row per (resolution, member) for every recorded vote in
    ``entries`` on or after ``since_date``. Raises ``ValueError`` naming any
    member the lookup cannot map, rather than dropping votes silently."""
    member_set = {str(m).strip().upper() for m in members}
    unmapped: set[str] = set()
    rows: list[dict] = []
    for entry in entries:
        if not _is_recorded_vote(entry):
            continue
        symbol = str(entry.get("symbol") or "").strip()
        date = parse_adoption_date(entry.get("adoption_date") or "")
        if not symbol or not date:
            logger.warning("Skipping extract entry without symbol/date: %r", entry.get("symbol"))
            continue
        if since_date and date < since_date:
            continue
        lists = {"Y": entry.get("MS_in_favour") or [], "N": entry.get("MS_against") or [], "A": entry.get("MS_abstaining") or []}
        totals = {
            "total_yes": _count(entry, "MS_in_favour_count", len(lists["Y"])),
            "total_no": _count(entry, "MS_against_count", len(lists["N"])),
            "total_abstentions": _count(entry, "MS_abstaining_count", len(lists["A"])),
        }
        subjects = "|".join(
            str(term).strip()
            for pair in (entry.get("subjects") or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[1] == "UNBIS Thesaurus"
            for term in [pair[0]]
            if str(term).strip()
        )
        base = {
            "undl_id": synthetic_undl_id(symbol),
            "date": date,
            "session": session_from_symbol(symbol),
            "resolution": symbol,
            "draft": str(entry.get("originating_document") or ""),
            "committee_report": "",
            "meeting": str(entry.get("pv") or ""),
            "title": str(entry.get("title") or "").strip(),
            "agenda_title": str(entry.get("agenda_item_name") or "").strip(),
            "subjects": subjects,
            **totals,
            "total_ms": len(member_set) if member_set else None,
            "undl_link": f"https://digitallibrary.un.org/search?p={symbol}",
            "source": SOURCE_NOTE,
        }
        voted: dict[str, tuple[str, str]] = {}
        for letter, names in lists.items():
            for name in names:
                code = code_lookup.get(normalize_name(name))
                if not code:
                    unmapped.add(str(name))
                    continue
                voted[code] = (letter, str(name))
        base["total_non_voting"] = max(0, len(member_set - set(voted))) if member_set else None
        for code, (letter, name) in sorted(voted.items()):
            rows.append({**base, "ms_code": code, "ms_name": normalize_name(name), "ms_vote": letter})
        for code in sorted(member_set - set(voted)):
            rows.append({**base, "ms_code": code, "ms_name": (member_names or {}).get(code, code), "ms_vote": " "})
    if unmapped:
        raise ValueError(
            "Extract names with no ISO-3 mapping (add them to NAME_ALIASES): "
            + ", ".join(sorted(unmapped))
        )
    return rows


def list_extract_files(session: Optional[requests.Session] = None, timeout: int = 60) -> list[str]:
    """Names of the JSON extracts in the repository, via the GitHub API."""
    s = session or requests.Session()
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = s.get(API_CONTENTS, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return sorted(item["name"] for item in resp.json() if str(item.get("name", "")).endswith(".json"))


def fetch_extract(name: str, session: Optional[requests.Session] = None, timeout: int = 120) -> list[dict]:
    s = session or requests.Session()
    resp = s.get(RAW_BASE + name, timeout=timeout)
    resp.raise_for_status()
    return parse_jsonl(resp.text)


def fetch_recent_votes(
    since_date: Optional[str],
    existing_df: Optional[pd.DataFrame],
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Every recorded vote in the extracts dated on or after ``since_date``,
    in the historical CSV's schema. Needs the existing CSV for the name→code
    map and the current member list."""
    s = session or requests.Session()
    names = list_extract_files(session=s)
    logger.info("GitHub extracts available: %s", ", ".join(names))
    entries: list[dict] = []
    for name in names:
        items = fetch_extract(name, session=s)
        logger.info("  %s: %d entries", name, len(items))
        entries.extend(items)
    lookup = build_code_lookup(existing_df)
    members = current_members(existing_df)
    rows = extract_to_rows(
        entries, lookup, members, since_date=since_date, member_names=names_by_code(existing_df)
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    logger.info(
        "GitHub extracts yielded %d rows across %d resolutions since %s.",
        len(df), df["resolution"].nunique(), since_date or "the beginning",
    )
    return df


def drop_superseded_synthetic_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a synthetic-id row when a real-id row exists for the same
    (resolution, member): the library's own record always wins."""
    if df is None or df.empty or not {"undl_id", "resolution", "ms_code"} <= set(df.columns):
        return df
    synthetic = df["undl_id"].map(is_synthetic_id)
    if not synthetic.any():
        return df
    real_keys = set(zip(df.loc[~synthetic, "resolution"].astype(str), df.loc[~synthetic, "ms_code"].astype(str)))
    keys = list(zip(df["resolution"].astype(str), df["ms_code"].astype(str)))
    superseded = pd.Series([synthetic.iat[i] and keys[i] in real_keys for i in range(len(df))], index=df.index)
    if superseded.any():
        logger.info("Dropping %d synthetic rows superseded by library records.", int(superseded.sum()))
    return df[~superseded]
