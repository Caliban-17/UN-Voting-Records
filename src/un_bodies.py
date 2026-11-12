"""The Security Council and the Assembly's committees, from public UN pages.

Two sources, both read as published and never guessed at:

* the Dag Hammarskjöld Library's veto list (every veto since 1946: date,
  draft, agenda item, the permanent members that cast it);
* the UN's meetings-coverage listings, which give each meeting's date,
  document symbol, headline and summary.

Neither carries country-by-country votes. The article pages that do are behind
a browser challenge, which is not worked around: what is recorded here is that
a text was approved, adopted, defeated or vetoed, with a vote count only when
the headline or summary states one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
import os
import re

from bs4 import BeautifulSoup
import requests

from src.pulse import UTC, atomic_json, data_dir, stamp

logger = logging.getLogger(__name__)

VETO_PAGE = "https://research.un.org/en/docs/sc/quick/veto"
COVERAGE = "https://press.un.org/en/content/{section}/meetings-coverage"
SECTIONS = ("security-council", "general-assembly")
AGENT = {"User-Agent": "UN-Scrupulous/1.0 (research newsletter; public pages)"}
RETAIN_DAYS = 150
# Spellings the library uses -> one name per permanent member.
PERMANENT = {"Russian Federation": "Russia", "USSR": "USSR", "United States": "United States",
             "USA": "United States", "United Kingdom": "United Kingdom", "UK": "United Kingdom",
             "China": "China", "France": "France"}
BODIES = {
    "SC": "Security Council", "GA/DIS": "First Committee", "GA/EF": "Second Committee",
    "GA/SHC": "Third Committee", "GA/SPD": "Fourth Committee", "GA/AB": "Fifth Committee",
    "GA/L": "Sixth Committee", "GA": "General Assembly plenary",
}
COMMITTEE_REMIT = {
    "First Committee": "disarmament", "Second Committee": "economy and development",
    "Third Committee": "human rights", "Fourth Committee": "decolonization and peacekeeping",
    "Fifth Committee": "budget", "Sixth Committee": "legal",
}
NUMBER_WORDS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


def _get(url: str) -> str:
    response = requests.get(url, headers=AGENT, timeout=(5, 25))
    response.raise_for_status()
    return response.text


def fetch_veto_table() -> str:
    """The library page embeds the table in a frame; follow it."""
    frame = re.search(r'<iframe[^>]+src="([^"]*sc_veto_list[^"]*)"', _get(VETO_PAGE))
    if not frame:
        raise ValueError("Veto list frame not found on the library page")
    src = frame[1]
    return _get("https:" + src if src.startswith("//") else src)


def parse_vetoes(body: str, now: datetime) -> list[dict]:
    records = []
    for row in BeautifulSoup(body, "html.parser").select("tr"):
        cells = [" ".join(c.stripped_strings) for c in row.find_all(["td", "th"])]
        if len(cells) != 5:
            continue
        try:
            when = datetime.strptime(cells[0], "%d %B %Y").date()
        except ValueError:
            continue
        members = list(dict.fromkeys(
            name for spelling, name in PERMANENT.items() if re.search(rf"\b{re.escape(spelling)}\b", cells[4])))
        if not members or when > now.date():
            raise ValueError(f"Unrecognised veto record: {cells[0]}")
        records.append({"date": when.isoformat(), "draft": cells[1], "record": cells[2],
                        "agenda": cells[3], "members": members})
    if len(records) < 200:
        raise ValueError("Veto list is implausibly short; keeping the previous copy")
    return sorted(records, key=lambda r: (r["date"], r["draft"]), reverse=True)


def body_of(symbol: str) -> str:
    prefix = re.sub(r"/\d+.*$", "", symbol.strip().upper())
    return BODIES.get(prefix, "Other")


def classify(title: str) -> tuple[str, int | None]:
    """What the meeting did, from its headline: (action, number of texts)."""
    text = title.lower()
    if re.search(r"fails? to adopt|rejects?\b|not adopted", text):
        return "defeated", None
    if re.search(r"\bveto", text):
        return "veto", None
    match = re.search(r"\b(approv\w+|adopt\w+|pass\w+)\s+(\w+)\s+(?:draft\s+)?(?:resolutions?|texts?|decisions?|drafts?)", text)
    if match:
        word = match[2]
        count = int(word) if word.isdigit() else NUMBER_WORDS.get(word)
        return ("approved" if match[1].startswith("approv") else "adopted"), count
    if re.search(r"\b(adopt\w+|approv\w+)\b", text):
        return ("approved" if "approv" in text else "adopted"), None
    return "meeting", None


def tally_in(text: str) -> dict | None:
    """A vote count, only when the text states one outright."""
    match = re.search(r"(\d+)\s+in favou?r\s+to\s+(\d+|none|no)\s+against,?\s+with\s+(\d+|no)\s+abstentions?", text, re.I)
    if not match:
        return None
    number = lambda v: 0 if v.lower() in ("none", "no") else int(v)  # noqa: E731
    return {"yes": number(match[1]), "no": number(match[2]), "abstain": number(match[3])}


def council_note(title: str) -> str | None:
    """How the Council voted, when the headline says so. With fifteen members
    the headline often names everyone who did not vote in favour."""
    if re.search(r"\bunanimous", title, re.I):
        return "unanimous"
    match = re.search(r"with (\d+) members? in favou?r(?:,\s*(.+?)\s+abstaining)?", title, re.I)
    if match:
        return f"{match[1]} in favour" + (f"; {match[2].strip()} abstaining" if match[2] else "")
    match = re.search(r"vote of (\d+) in favou?r to (\d+|none) against(?:,? with (\d+|no) abstentions?)?", title, re.I)
    if match:
        return f"{match[1]} in favour, {match[2]} against" + (f", {match[3]} abstaining" if match[3] else "")
    return None


MIRROR = "https://github.com/Caliban-17/UN-Voting-Records/releases/download/newsletter-current/bodies.json"


def fetch_mirror() -> dict | None:
    """The ledger as last published by the scheduled workflow. Some networks
    are served a browser challenge instead of a listing; the workflow's copy
    fills the gap rather than leaving a section empty."""
    try:
        response = requests.get(os.getenv("BODIES_MIRROR_URL", MIRROR), headers=AGENT, timeout=(5, 25))
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data.get("actions"), list) else None
    except (requests.RequestException, ValueError):
        return None


def parse_coverage(body: str, now: datetime) -> list[dict]:
    """One record per meeting on a coverage listing page."""
    records = []
    for article in BeautifulSoup(body, "html.parser").select("article"):
        link = article.select_one("a[href*='.doc.htm']")
        parts = list(article.stripped_strings)
        if not link or len(parts) < 3:
            continue
        try:
            when = datetime.strptime(parts[0], "%d %B %Y").date()
        except ValueError:
            continue
        symbol = next((p for p in parts if re.fullmatch(r"[A-Z]{2,3}(/[A-Z]{1,4})?/\d+[A-Z]?", p)), "")
        title = " ".join(link.get_text(" ", strip=True).split())
        after = parts[parts.index(link.get_text(strip=True)) + 1:] if link.get_text(strip=True) in parts else []
        summary = " ".join(" ".join(after).split())[:500]
        if not symbol or not title or when > now.date():
            continue
        action, count = classify(title)
        records.append({
            "date": when.isoformat(), "symbol": symbol, "body": body_of(symbol), "title": title,
            "summary": summary, "action": action, "texts": count,
            "tally": tally_in(title + " " + summary),
            "vote_note": council_note(title) if body_of(symbol) == "Security Council" else None,
            "url": "https://press.un.org" + link["href"] if link["href"].startswith("/") else link["href"],
        })
    return records


def read_bodies(directory=None) -> dict:
    try:
        return json.loads(((directory or data_dir()) / "bodies.json").read_text())
    except (OSError, ValueError):
        return {"actions": [], "vetoes": [], "sources": []}


def refresh_bodies(directory=None, now=None, get_vetoes=fetch_veto_table, get_page=_get, mirror=None) -> dict:
    """Merge today's listings into the ledger; a failed source keeps its history."""
    now = now or datetime.now(UTC)
    directory = directory or data_dir()
    prior = read_bodies(directory)
    actions = {a["symbol"]: a for a in prior.get("actions", [])}
    vetoes, sources = prior.get("vetoes", []), []

    def attempt(name, url, work):
        previous = next((s for s in prior.get("sources", []) if s["name"] == name), {})
        status = {"name": name, "url": url, "checked_at": stamp(now),
                  "last_success_at": previous.get("last_success_at")}
        try:
            status.update(status="ok", count=work(), last_success_at=stamp(now))
        except Exception as exc:  # a source failing must not lose the ledger
            logger.warning("%s could not be read: %s", name, exc)
            status.update(status="error", error=str(exc)[:200])
        sources.append(status)

    def read_vetoes():
        nonlocal vetoes
        fresh = parse_vetoes(get_vetoes(), now)
        if len(fresh) < len(vetoes):
            raise ValueError("Veto list shrank; keeping the previous copy")
        vetoes = fresh
        return len(fresh)

    attempt("Security Council veto list (Dag Hammarskjöld Library)", VETO_PAGE, read_vetoes)
    for section in SECTIONS:
        url = COVERAGE.format(section=section)
        # The first read walks back a few pages to fill the window; after that
        # the front page is enough.
        pages = range(4) if not any(a for a in actions.values() if a.get("section") == section) else range(1)

        def read_section(section=section, url=url, pages=pages):
            found = 0
            for page in pages:
                for record in parse_coverage(get_page(url + (f"?page={page}" if page else "")), now):
                    actions[record["symbol"]] = {**record, "section": section}
                    found += 1
            if not found:
                raise ValueError("Listing contained no recognisable meetings")
            return found

        attempt(f"UN meetings coverage: {section.replace('-', ' ')}", url, read_section)
    if mirror is None and get_page is _get:
        mirror = fetch_mirror
    if mirror and any(s["status"] == "error" for s in sources):
        published = mirror()
        if published:
            for record in published.get("actions", []):
                actions.setdefault(record["symbol"], record)
            if len(published.get("vetoes", [])) > len(vetoes):
                vetoes = published["vetoes"]
            for status in sources:
                if status["status"] == "error":
                    status["mirror"] = "filled from the published ledger"
    cutoff = (now - timedelta(days=RETAIN_DAYS)).date().isoformat()
    kept = sorted((a for a in actions.values() if a["date"] >= cutoff),
                  key=lambda a: (a["date"], a["symbol"]), reverse=True)
    result = {"checked_at": stamp(now), "sources": sources, "actions": kept, "vetoes": vetoes}
    atomic_json(directory / "bodies.json", result)
    return result


def summary(bodies: dict, now: datetime, days: int = 30) -> dict:
    """What the Council and the committees did in the window, for an edition."""
    start, end = (now - timedelta(days=days - 1)).date().isoformat(), now.date().isoformat()
    recent = [a for a in bodies.get("actions", []) if start <= a["date"] <= end]
    council = [a for a in recent if a["body"] == "Security Council" and a["action"] in ("adopted", "defeated", "veto")]
    committee = [a for a in recent if a["body"] in COMMITTEE_REMIT and a["action"] == "approved"]
    vetoes = bodies.get("vetoes", [])
    year = now.year
    by_year: dict[int, dict[str, int]] = {}
    for veto in vetoes:
        y = int(veto["date"][:4])
        if y < year - 14:
            continue
        slot = by_year.setdefault(y, {})
        for member in veto["members"]:
            slot[member] = slot.get(member, 0) + 1
    drafts = {y: sum(1 for v in vetoes if int(v["date"][:4]) == y) for y in range(year - 14, year + 1)}
    per_committee: dict[str, dict] = {}
    for item in committee:
        slot = per_committee.setdefault(item["body"], {"meetings": 0, "texts": 0})
        slot["meetings"] += 1
        slot["texts"] += item["texts"] or 0
    return {
        "window_start": start, "window_end": end,
        "council": council[:8], "committee": committee[:8], "committee_totals": per_committee,
        "vetoes_window": [v for v in vetoes if start <= v["date"] <= end],
        "vetoes_this_year": [v for v in vetoes if v["date"].startswith(str(year))],
        "vetoed_drafts_by_year": drafts, "vetoes_by_member": by_year,
        "sources": bodies.get("sources", []),
    }
