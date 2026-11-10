"""Automated, source-attributed UN briefing. No model or editorial gate required.

Only published feed text is used. Categories and priority are navigation aids,
not claims that a statement is a decision or that similar articles corroborate it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime, format_datetime
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UTC = timezone.utc
SOURCES = (
    {"id": "un-news", "name": "UN News", "url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml", "scope": "UN system"},
    {"id": "un-press", "name": "UN Meetings Coverage", "url": "https://press.un.org/en/rss.xml", "scope": "Decisions & diplomacy"},
    {"id": "geneva", "name": "UN Geneva", "url": "https://www.ungeneva.org/news-media/press-items-list/rss.xml", "scope": "Rights & Geneva briefings"},
    {"id": "who", "name": "WHO", "url": "https://www.who.int/rss-feeds/news-english.xml", "scope": "Global health"},
)
# Keyword rules, checked against a month of real items (405 articles, November
# 2026). Each exclusion below removed a measured false match: "war on nature",
# the "Committee on Economic, Social and Cultural Rights", the Cuba embargo's
# "economic, commercial and financial", "recent developments in", the
# "International Court of Justice", national elections and "electricity".
# "Secretary-General" alone says nothing about the subject and is not a rule.
TOPICS = {
    "Peace & security": (
        r"\b(wars?(?! on (nature|drugs|plastic|waste))|warfare|conflicts?|ceasefire|security council|"
        r"peacekeep\w*|disarmament|nuclear|missile\w*|terroris\w*|attack\w*|drones?|airstrike\w*|"
        r"strikes? (on|kill|hit)|shelling|bomb\w*|hostilit\w*|armed (group|forces|conflict)\w*|"
        r"militar\w+|troops|invasion|killed|killing\w*|violence)\b"
    ),
    "Humanitarian": (
        r"\b(humanitarian|famine|hunger|starv\w*|malnutrition|displac\w*|refugees?|migrants?|"
        r"relief|food insecurity|aid (workers?|deliver\w*|convoy\w*|agenc\w*|access)|lifesaving aid|"
        r"floods?|flooding|earthquake\w*|cyclone\w*|hurricane\w*|shelter\w*|shipwreck\w*)\b"
    ),
    "Human rights": (
        r"\b(human rights|rights (council|committee|experts?|chief|office|officials?|body|defenders?|violations?)|"
        r"(cultural|political|children'?s|women'?s|indigenous) rights|rights of (the )?\w+|"
        r"torture|discrimination|racism|women|gender|detention|detained|arbitrar\w+|"
        r"freedom of (expression|the press|religion|assembly)|journalists?|accountability|impunity|"
        r"war crimes?|genocide)\b"
    ),
    # WHO, HIV and AIDS are matched case-sensitively: "who" is a pronoun, "aids" a verb.
    "Health": (
        r"\b(health|(?-i:WHO|HIV|AIDS)|diseases?|pandemic|epidemic|vaccin\w*|cholera|mpox|ebola|"
        r"malaria|tuberculosis|measles|polio)\b"
    ),
    "Climate & development": (
        r"\b(climate|environment\w*|emissions?|biodiversity|carbon|drought\w*|deforestation|"
        r"(extreme|record|global) (temperatures?|heat)|global warming|heatwaves?|"
        r"sustainable development|development (goals?|financ\w+|cooperation|agenda|bank|aid)|"
        r"financing for development|developing countr\w+|least developed|poverty|"
        r"econom(y|ies|ic growth|ic recovery|ic crisis|ic outlook)|trade|education|sdgs?)\b"
    ),
    "UN affairs": (
        r"\b(budget\w*|reform\w*|appoint\w*|elects|elected (to|as)|general assembly|"
        r"(fifth|second|third|fourth|sixth|first) committee|funding|financial (situation|crisis)|"
        r"assessed contributions?|liquidity|un80|credentials|resident coordinator)\b"
    ),
}
OTHER_TOPIC = "Other"
# Issue dossiers are deliberately distinct from event deduplication.
ISSUES = {
    "Sudan": r"(?<!south )\bsudan\b|\b(darfur|el.fasher|khartoum)\b", "South Sudan": r"\bsouth sudan\b",
    "Ukraine": r"\bukrain\w*\b",
    "Israel & Palestine": r"\b(gaza|palestin\w*|israel\w*|west bank)\b",
    "Haiti": r"\bhaiti\w*\b", "Myanmar": r"\b(myanmar|rohingya)\b",
    "Afghanistan": r"\bafghan\w*\b", "Yemen": r"\b(yemen\w*|houthi\w*)\b",
    "Syria": r"\bsyri\w*\b", "Lebanon": r"\bleban\w*\b",
    "DR Congo": r"\b(congo|drc)\b", "Iran": r"\b(iran\w*|hormuz)\b",
}

# Notices that are published every day or week and report no event.
ROUTINE = re.compile(
    r"\b(daily press briefing|noon briefing|presents? credentials|note to correspondents|"
    r"media advisory|activities of (the )?secretary-general|approving organization of work|"
    r"appoints\b.*\b(resident coordinator|representative|envoy|adviser|advisor|head|director|"
    r"commander|coordinator|secretary|chief))\b", re.I)
CONSEQUENCE = re.compile(
    r"\b(adopts?|adopted|veto\w*|ceasefire|famine|emergency|outbreak|cuts|condemn\w*|kill\w*|"
    r"attack\w*|sanctions?|resolution|treaty|agreement|dies?|dead|deaths?)\b", re.I)


def classify(text: str) -> tuple[list[str], list[str]]:
    """Topic labels and issue dossiers for a title plus summary."""
    topics = [k for k, pattern in TOPICS.items() if re.search(pattern, text, re.I)]
    issues = [k for k, pattern in ISSUES.items() if re.search(pattern, text, re.I)]
    return topics or [OTHER_TOPIC], issues


def is_routine(title: str) -> bool:
    return bool(ROUTINE.search(title))


def headline_priority(item: dict, now: datetime) -> float:
    """Higher first: an event outranks a routine notice published later the
    same day, and recency decides among equals. UN News stamps every item
    12:00 UTC, so time of day alone would rank any afternoon press release
    above all of that day's reporting."""
    age = max(0, (now - date_value(item["published_at"])).total_seconds() / 3600)
    return 36 * bool(CONSEQUENCE.search(item["title"])) - 72 * is_routine(item["title"]) - age


def stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


# Zone names RFC 2822 parsers do not know; left alone they are read as UTC.
ZONES = {"CET": "+0100", "CEST": "+0200", "EET": "+0200", "EEST": "+0300",
         "WET": "+0000", "WEST": "+0100", "BST": "+0100"}


def date_value(value: str | None) -> datetime | None:
    if not value:
        return None
    zone = re.search(r"\s([A-Z]{3,4})$", value.strip())
    if zone and zone[1] in ZONES:
        value = value.strip()[:zone.start()] + " " + ZONES[zone[1]]
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        try:
            dt = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def clean_text(value: str) -> str:
    if "<" not in (value or ""):
        return " ".join(unescape(value or "").split())
    soup = BeautifulSoup(value or "", "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if parts.scheme not in ("https", "http") or not parts.hostname or parts.username:
        return ""
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), ""))


def _child(node, names):
    for child in node:
        if child.tag.split("}")[-1] in names and child.text:
            return child.text.strip()
    return ""


def parse_feed(body: bytes, source: dict, now: datetime) -> tuple[list[dict], int]:
    # Refuse entity declarations; feeds do not need DTDs.
    if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
        raise ValueError("Feed contains an unsupported XML declaration")
    root = ET.fromstring(body)
    if root.tag.split("}")[-1].lower() not in ("rss", "feed", "rdf"):
        raise ValueError("Source returned a page instead of an RSS/Atom feed")
    items, rejected = [], 0
    for node in root.iter():
        if node.tag.split("}")[-1] not in ("item", "entry"):
            continue
        title = clean_text(_child(node, {"title"}))[:500]
        link = _child(node, {"link"})
        if not link:
            link = next((c.get("href", "") for c in node if c.tag.split("}")[-1] == "link" and c.get("rel", "alternate") == "alternate"), "")
        link = canonical_url(link)
        published = date_value(_child(node, {"pubDate", "published", "date"}))
        updated = date_value(_child(node, {"updated"}))
        published = published or updated
        # The English feeds occasionally carry an item from another language's
        # section (UN Geneva's /fr/); it does not belong in an English briefing.
        other_language = re.match(r"/(fr|es|ru|ar|zh)/", urlsplit(link).path or "")
        if not title or not link or not published or other_language or published > now + timedelta(minutes=15):
            rejected += 1
            continue
        summary = clean_text(_child(node, {"description", "summary", "content"}))[:600]
        text = title + " " + summary
        topics, issues = classify(text)
        items.append({
            "id": hashlib.sha256(link.encode()).hexdigest()[:20],
            "title": title, "url": link, "summary": summary,
            "source_id": source["id"], "source": source["name"],
            "published_at": stamp(published),
            "source_updated_at": stamp(updated) if updated and updated <= now + timedelta(minutes=15) else None,
            "topics": topics, "issues": issues,
        })
    return items, rejected


def fetch_source(source: dict, now: datetime) -> tuple[list[dict], dict]:
    status = {**source, "checked_at": stamp(now)}
    try:
        with requests.Session() as session:
            retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), respect_retry_after_header=False)
            session.mount("https://", HTTPAdapter(max_retries=retry))
            with session.get(source["url"], timeout=(5, 15), stream=True, headers={"User-Agent": "UN-Scrupulous-Pulse/1.0 (public RSS reader)"}) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > 4_000_000:
                        raise ValueError("Feed exceeds size limit")
        items, rejected = parse_feed(bytes(body), source, now)
        if not items:
            raise ValueError("Feed contains no valid dated items")
        status.update(status="ok", last_success_at=stamp(now), items=len(items), rejected=rejected,
                      latest_item_at=max(i["published_at"] for i in items))
        if now - date_value(status["latest_item_at"]) > timedelta(days=30):
            status["status"] = "quiet"
        return items, status
    except (requests.RequestException, ValueError, ET.ParseError) as exc:
        status.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:180]}", items=0)
        return [], status


def data_dir() -> Path:
    return Path(os.getenv("PULSE_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data" / "pulse")))


def read_snapshot(directory: Path | None = None) -> dict:
    try:
        return json.loads(((directory or data_dir()) / "latest.json").read_text())
    except (OSError, ValueError):
        return {"schema_version": 1, "checked_at": None, "items": [], "sources": []}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def refresh(directory: Path | None = None, now: datetime | None = None, fetcher=fetch_source) -> dict:
    """Merge corrections and preserve verified history through partial failures."""
    now, directory = now or datetime.now(UTC), directory or data_dir()
    previous = read_snapshot(directory)
    cutoff = now - timedelta(days=30)
    items = {i["id"]: i for i in previous["items"] if (date_value(i["published_at"]) or now) >= cutoff}
    prior_sources = {s["id"]: s for s in previous["sources"]}
    statuses = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda source: fetcher(source, now), SOURCES))
    for source, (incoming, status) in zip(SOURCES, results):
        if status["status"] == "error":
            status["last_success_at"] = prior_sources.get(source["id"], {}).get("last_success_at")
        statuses.append(status)
        for item in incoming:
            if date_value(item["published_at"]) < cutoff:
                continue
            prior = items.get(item["id"])
            item["first_seen_at"] = prior.get("first_seen_at", stamp(now)) if prior else stamp(now)
            changed = prior and any(item.get(k) != prior.get(k) for k in ("title", "summary", "source_updated_at"))
            item["revised_at"] = stamp(now) if changed else (prior.get("revised_at") if prior else None)
            items[item["id"]] = item
    # Retained items are re-labelled on every pass, so a change to the rules
    # reaches the whole 30-day window at once instead of only newly fetched items.
    for item in items.values():
        item["topics"], item["issues"] = classify(item["title"] + " " + item["summary"])
    ordered = sorted(items.values(), key=lambda i: (i["published_at"], i["id"]), reverse=True)[:2500]
    content_hash = hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest()
    snapshot = {"schema_version": 1, "checked_at": stamp(now), "content_hash": content_hash,
                "sources": statuses, "items": ordered}
    atomic_json(directory / "latest.json", snapshot)
    if content_hash != previous.get("content_hash") and ordered:
        atomic_json(directory / "archive" / f"{now:%Y-%m-%d}.json", snapshot)
    # Bounded local archive. A daily file contains that day's latest changed edition.
    for path in (directory / "archive").glob("*.json"):
        if path.stem < (now - timedelta(days=90)).strftime("%Y-%m-%d"):
            path.unlink()
    return snapshot


def briefing(snapshot: dict, now: datetime | None = None, topic: str = "", issue: str = "", query: str = "") -> dict:
    now = now or datetime.now(UTC)
    checked = date_value(snapshot.get("checked_at"))
    stale = checked is None or now - checked > timedelta(hours=2)
    statuses = snapshot.get("sources", [])
    cutoff = now - timedelta(days=7)
    recent = [i for i in snapshot["items"] if (date_value(i["published_at"]) or datetime.min.replace(tzinfo=UTC)) >= cutoff]
    matching = [i for i in recent if (not topic or topic in i["topics"]) and (not issue or issue in i["issues"])
                and (not query or query.casefold() in (i["title"] + " " + i["summary"]).casefold())]
    # Exact normalised headlines from DIFFERENT sources within three days are one
    # update; topical similarity is NOT evidence that reports describe the same
    # event. A source repeating its own title (each day's "Daily Press Briefing")
    # is a new item every time. Retain every attributed source link.
    groups = {}
    for item in matching:
        key = re.sub(r"\W+", " ", item["title"].casefold()).strip()
        lead = groups.get(key)
        same_story = (
            lead is not None and lead["source_id"] != item["source_id"]
            and all(o["source"] != item["source"] for o in lead["also_reported_by"])
            and abs(date_value(lead["published_at"]) - date_value(item["published_at"])) <= timedelta(days=3)
        )
        if same_story:
            lead["also_reported_by"].append({"source": item["source"], "url": item["url"]})
        else:
            groups[key if lead is None else f"{key} {item['id']}"] = {**item, "also_reported_by": []}
    updates = list(groups.values())

    ranked = sorted(updates, key=lambda item: headline_priority(item, now), reverse=True)
    # Give the lead five a mix of sources; retain chronological updates below.
    leads, counts = [], {}
    for item in ranked:
        source_id = item["source_id"]
        if counts.get(source_id, 0) < 2:
            leads.append(item)
            counts[source_id] = counts.get(source_id, 0) + 1
        if len(leads) == 5:
            break
    for item in ranked:
        if len(leads) < 5 and item not in leads:
            leads.append(item)
    return {
        "checked_at": snapshot.get("checked_at"), "stale": stale,
        "status": "stale" if stale else ("partial" if any(s["status"] != "ok" for s in statuses) else "current"),
        "sources": statuses, "lead": leads, "updates": updates[:100],
        "topics": [{"name": t, "count": sum(t in i["topics"] for i in recent)} for t in TOPICS]
        + [{"name": OTHER_TOPIC, "count": n} for n in [sum(OTHER_TOPIC in i["topics"] for i in recent)] if n],
        "issues": [{"name": t, "count": sum(t in i["issues"] for i in recent)} for t in ISSUES if any(t in i["issues"] for i in recent)],
        "total": len(updates), "latest_published_at": max((i["published_at"] for i in recent), default=None),
        "methodology": "Automatically collected official feeds. Excerpts are source text; topic labels and headline priority use rules. Last seven days, with 30 days retained. Exact matching headlines are grouped; issue dossiers can contain separate events. Publication dates are not meeting dates. Coverage is limited to the listed feeds.",
    }


def rss(snapshot: dict, public_url: str) -> bytes:
    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    for key, value in {"title": "UN-Scrupulous | UN Pulse", "link": public_url,
                       "description": "Automated updates from official UN sources", "language": "en"}.items():
        ET.SubElement(channel, key).text = value
    # Stable GUID and source date: polling never republishes unchanged stories.
    for item in snapshot["items"][:100]:
        node = ET.SubElement(channel, "item")
        for key, value in {"title": item["title"], "link": item["url"], "description": f"{item['source']}: {item['summary']}",
                           "pubDate": format_datetime(date_value(item["published_at"]))}.items():
            ET.SubElement(node, key).text = value
        ET.SubElement(node, "guid", isPermaLink="true").text = item["url"]
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
