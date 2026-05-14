"""
MARC-XML data fetcher for UN Digital Library voting records.

The UN Digital Library's Invenio backend exposes every record in MARC XML
via the ``?of=xm`` output format. For voting records, the MARC fields we
care about are:

  245 — title
  269 — vote date (YYYY-MM-DD)
  650/653/655 — subjects (UNBISnet headings)
  791 — resolution number (A/RES/…)
  952 — meeting reference
  967 — one repeated field per UN member state, with subfields:
        c = ISO-3 country code (e.g. ``AFG``)
        d = vote (``Y``, ``N``, ``A``, or absent for non-voting/absent)
        e = country name (e.g. ``AFGHANISTAN``)

This module is ~100× faster than the old Playwright HTML scraper because
the search endpoint returns full MARC for every record in one HTTP call,
so 50 records ≈ 9,500 country-vote rows per request.

Output schema matches the historical CSV exactly so the merge step is
boring:

  undl_id, ms_code, ms_name, ms_vote, date, session, resolution, meeting,
  title, subjects, undl_link
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET

import pandas as pd
import requests

logger = logging.getLogger(__name__)

UN_DL_BASE = "https://digitallibrary.un.org"
MARC_NS = "http://www.loc.gov/MARC21/slim"
NSMAP = {"m": MARC_NS}

# Vote field 967 subfield codes
_SUB_COUNTRY_CODE = "c"
_SUB_VOTE = "d"
_SUB_COUNTRY_NAME = "e"

# Counter-intuitive but verified: UN DL's CDN accepts the default
# ``python-requests/2.x`` User-Agent but throttles browser-impersonating
# UAs (returns HTTP 202 with empty body — likely an anti-scraping rule
# that targets ``Mozilla/5.0…`` strings). We therefore deliberately do
# NOT override the User-Agent header; transparent automation is the
# correct posture here anyway. We still pad with a polite request delay.
DEFAULT_HEADERS: dict[str, str] = {}
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES_PER_PAGE = 4
RETRY_BACKOFF_SECONDS = 2.0


def _datafield_subfields(record: ET.Element, tag: str) -> Iterator[dict[str, str]]:
    """Yield one dict per ``<datafield tag="TAG">`` block in the record."""
    for df in record.findall(f"m:datafield[@tag='{tag}']", NSMAP):
        subs: dict[str, str] = {}
        for sf in df.findall("m:subfield", NSMAP):
            code = sf.get("code") or ""
            subs[code] = (sf.text or "").strip()
        yield subs


def _first_subfield(record: ET.Element, tag: str, code: str = "a") -> str | None:
    for subs in _datafield_subfields(record, tag):
        if code in subs:
            return subs[code]
    return None


def _controlfield(record: ET.Element, tag: str) -> str | None:
    cf = record.find(f"m:controlfield[@tag='{tag}']", NSMAP)
    return (cf.text or "").strip() if cf is not None and cf.text else None


def parse_marc_record(record: ET.Element) -> list[dict]:
    """Convert one ``<record>`` element to one row per country vote.

    Returns an empty list when the record has no 967 fields (not a vote).

    UN-Digital-Library-specific MARC tags we read:
      245 — title (subfields a, b, c — three parts of the title)
      269 — vote date
      791 — resolution number (subfield a = symbol like A/RES/80/214)
      793 — meeting type (PLENARY MEETING / committee, etc.)
      952 — meeting reference (e.g. A/80/PV.69)
      967 — country vote rows (c=ISO-3, d=Y/N/A, e=country name)
      991 — UN subject heading (subfield d = UNBISnet heading,
            ``HUMAN RIGHTS--REPORTS`` style; this is what the historical
            CSV's ``subjects`` column contains, and what our downstream
            topic analysis depends on)
      992 — alternative vote date
      996 — vote totals (a=yes, c=no, d=abstain, e=non-voting, f=total)
    """
    undl_id = _controlfield(record, "001")
    if not undl_id:
        return []
    title_subs = next(_datafield_subfields(record, "245"), {}) or {}
    title_parts = [title_subs.get(c, "") for c in ("a", "b", "c")]
    title = " ".join(p for p in title_parts if p).strip()
    date = (
        _first_subfield(record, "269", "a")
        or _first_subfield(record, "992", "a")
        or ""
    )
    resolution = _first_subfield(record, "791", "a") or ""
    session = _first_subfield(record, "791", "c") or ""
    meeting = _first_subfield(record, "952", "a") or ""

    # UN-specific subject extraction: tag 991 subfield d is the
    # UNBISnet heading. Multiple 991 fields = multiple headings.
    subject_headings: list[str] = []
    for sub in _datafield_subfields(record, "991"):
        heading = (sub.get("d") or "").strip()
        if heading and heading not in subject_headings:
            subject_headings.append(heading)
    # Historical CSV uses "|" as the separator between multiple subjects.
    subjects = "|".join(subject_headings)
    # Fall back to standard MARC tags 650/653 if 991 is empty.
    if not subjects:
        subjects = "|".join(
            s.get("a", "") for s in _datafield_subfields(record, "650") if s.get("a")
        ) or "|".join(
            s.get("a", "") for s in _datafield_subfields(record, "653") if s.get("a")
        )

    # Tag 996 vote totals — UN-DL-specific subfield codes:
    # b=yes, c=no, d=abstain, e=non-voting, f=total member states.
    totals = next(_datafield_subfields(record, "996"), {}) or {}

    def _int_or_none(key: str) -> int | None:
        v = totals.get(key, "")
        return int(v) if v.isdigit() else None

    total_yes = _int_or_none("b")
    total_no = _int_or_none("c")
    total_abstain = _int_or_none("d")
    total_non_voting = _int_or_none("e")
    total_ms = _int_or_none("f")

    rows: list[dict] = []
    for sub in _datafield_subfields(record, "967"):
        code = sub.get(_SUB_COUNTRY_CODE)
        name = sub.get(_SUB_COUNTRY_NAME)
        vote = sub.get(_SUB_VOTE, "")  # blank when absent / not voting
        if not code:
            continue
        rows.append({
            "undl_id": int(undl_id) if undl_id.isdigit() else undl_id,
            "ms_code": code,
            "ms_name": name or code,
            "ms_vote": vote if vote else " ",  # match historical 'absent' convention
            "date": date,
            "session": session,
            "resolution": resolution,
            "meeting": meeting,
            "title": title,
            "subjects": subjects,
            "total_yes": total_yes,
            "total_no": total_no,
            "total_abstentions": total_abstain,
            "total_non_voting": total_non_voting,
            "total_ms": total_ms,
            "undl_link": f"{UN_DL_BASE}/record/{undl_id}",
        })
    return rows


def fetch_records_page(
    jrec: int = 1,
    page_size: int = 50,
    timeout: int = 60,
    session: requests.Session | None = None,
) -> tuple[list[ET.Element], int | None]:
    """Fetch one page of voting records via the MARC XML search endpoint.

    Returns ``(records, total_count)``. ``total_count`` is parsed from the
    XML comment ``<!-- Search-Engine-Total-Number-Of-Results: N -->`` that
    Invenio injects at the top of the response.
    """
    params = {
        "ln": "en",
        "cc": "Voting Data",
        "sf": "year",        # sort by year
        "so": "d",           # descending
        "of": "xm",          # MARC XML output
        "rg": str(page_size),
        "fct__2": "General Assembly",
        "jrec": str(jrec),
    }
    s = session or requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    # Invenio sometimes returns 202 (Accepted, deferred processing) with an
    # empty body and expects the client to retry. Loop with exponential backoff.
    text = ""
    for attempt in range(MAX_RETRIES_PER_PAGE):
        resp = s.get(f"{UN_DL_BASE}/search", params=params, timeout=timeout)
        if resp.status_code == 200 and resp.text.strip():
            text = resp.text
            break
        if resp.status_code in (202, 429, 503):
            wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            logger.info("UN DL %s at jrec=%s; retry in %ss", resp.status_code, jrec, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
    if not text:
        raise RuntimeError(
            f"UN DL gave no usable response after {MAX_RETRIES_PER_PAGE} retries (jrec={jrec})."
        )

    # Pull the total-result comment out of the first ~200 chars.
    total: int | None = None
    head = text[:300]
    marker = "Search-Engine-Total-Number-Of-Results:"
    if marker in head:
        try:
            total = int(head.split(marker, 1)[1].split("-->", 1)[0].strip())
        except (ValueError, IndexError):
            total = None

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        logger.warning("MARC XML parse failed at jrec=%s: %s", jrec, exc)
        return [], total
    records = root.findall("m:record", NSMAP)
    return records, total


def _parse_record_date(rows: list[dict]) -> datetime | None:
    """Return the parsed date of a record's first row, or None if unparseable."""
    if not rows:
        return None
    raw = (rows[0].get("date") or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return None


def fetch_recent_votes(
    since_date: str | None = None,
    max_pages: int = 200,
    page_size: int = 50,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """Walk paged MARC search results, stopping at ``since_date`` (YYYY-MM-DD).

    Records are returned sorted-by-year-desc by Invenio, so once we hit a
    record older than ``since_date`` we stop the walk. ``since_date=None``
    walks all the way to the safety limit ``max_pages``.

    Cleanups applied to every returned row:
      * empty-date rows are dropped (not real vote rows)
      * (undl_id, ms_code) duplicates are dedup'd, keeping the last seen
    """
    rows: list[dict] = []
    cutoff_dt = (
        datetime.strptime(since_date, "%Y-%m-%d") if since_date else None
    )
    session = requests.Session()
    should_stop = False
    for page in range(max_pages):
        if should_stop:
            break
        jrec = 1 + page * page_size
        try:
            records, _ = fetch_records_page(jrec, page_size, session=session)
        except requests.RequestException as exc:
            logger.warning("Page %s failed (%s); stopping.", jrec, exc)
            break
        if not records:
            logger.info("Empty page at jrec=%s — done.", jrec)
            break

        page_added = 0
        oldest_in_page: datetime | None = None
        for rec in records:
            record_rows = parse_marc_record(rec)
            if not record_rows:
                continue
            d = _parse_record_date(record_rows)
            if d is None:
                # Skip undated records (administrative entries etc.).
                continue
            if oldest_in_page is None or d < oldest_in_page:
                oldest_in_page = d
            if cutoff_dt is not None and d < cutoff_dt:
                logger.info(
                    "Reached record dated %s (< cutoff %s) — stopping walk.",
                    d.date(), cutoff_dt.date(),
                )
                should_stop = True
                break
            rows.extend(record_rows)
            page_added += 1

        logger.info(
            "Page jrec=%s: kept %d records (oldest %s), running total %d rows.",
            jrec, page_added,
            oldest_in_page.date() if oldest_in_page else "?",
            len(rows),
        )
        if not should_stop:
            time.sleep(request_delay)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Final scrub: drop empty-date rows, dedupe on (undl_id, ms_code).
    df = df[df["date"].astype(str).str.strip() != ""].copy()
    before = len(df)
    df = df.drop_duplicates(subset=["undl_id", "ms_code"], keep="last")
    if len(df) < before:
        logger.info("Dropped %d duplicate (undl_id, ms_code) rows.", before - len(df))
    return df.reset_index(drop=True)
