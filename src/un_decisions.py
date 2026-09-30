"""Faster official GA decisions, kept separate from country-level roll calls."""

from datetime import datetime, timedelta
import json
import re

from bs4 import BeautifulSoup

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.pulse import UTC, atomic_json, data_dir, stamp

REGISTER = "https://public.e-delegate.un.org/reports/ga{session}_resolutions.html"


def fetch_register(url):
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
            respect_retry_after_header=False)))
        with session.get(url, timeout=(5, 20), stream=True) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > 4_000_000:
                    raise ValueError("Register exceeds size limit")
            return bytes(body)


def parse_register(body, session, now):
    soup = BeautifulSoup(body.decode("utf-8-sig") if isinstance(body, bytes) else body, "html.parser")
    records = []
    for row in soup.select("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) != 3 or not cells[0].find("a"):
            continue
        number = " ".join(cells[0].stripped_strings)
        if not re.fullmatch(rf"{session}/\d+(?: [A-Z])?", number):
            continue
        title = " ".join(cells[1].stripped_strings)
        action = " ".join(cells[2].stripped_strings)
        date_match = re.search(r"adopted on (\d{1,2} \w+ \d{4})", action)
        tally_match = re.search(r"with a vote\s*\((\d+)\s*[-–]\s*(\d+)\s*[-–]\s*(\d+)\)", action)
        no_vote = "without a vote" in action
        if not title or not date_match or (not tally_match and not no_vote):
            raise ValueError(f"Unrecognised adoption record: {number}")
        adopted = datetime.strptime(date_match[1], "%d %B %Y").date()
        if adopted > now.date():
            raise ValueError(f"Future adoption date: {number}")
        tally = dict(zip(("yes", "no", "abstain"), map(int, tally_match.groups()))) if tally_match else None
        if tally and not 0 < sum(tally.values()) <= 193:
            raise ValueError(f"Invalid tally: {number}")
        records.append({"symbol": f"A/RES/{number}", "session": session, "title": title,
                        "date": adopted.isoformat(), "tally": tally,
                        "adoption": "recorded_vote" if tally else "without_vote",
                        "url": f"https://docs.un.org/en/A/RES/{number.split()[0]}",
                        "source_url": REGISTER.format(session=session), "source": "UN General Assembly resolutions register"})
    if not records:
        raise ValueError("Register contained no recognised resolutions; preserving previous coverage")
    if len({r["symbol"] for r in records}) != len(records):
        raise ValueError("Duplicate resolution symbols in register")
    return records


def read_decisions(directory=None):
    try:
        return json.loads(((directory or data_dir()) / "decisions.json").read_text())
    except (OSError, ValueError):
        return {"records": [], "sources": []}


def refresh_decisions(directory=None, now=None, fetcher=fetch_register):
    now = now or datetime.now(UTC)
    directory = directory or data_dir()
    prior = read_decisions(directory)
    records = {r["symbol"]: r for r in prior.get("records", [])}
    # Around the September changeover poll both adjacent regular sessions.
    session = now.year - (1945 if now.month >= 9 else 1946)
    sources = []
    for number in (session - 1, session):
        url = REGISTER.format(session=number)
        previous = next((s for s in prior.get("sources", []) if s["url"] == url), {})
        status = {"url": url, "session": number, "checked_at": stamp(now),
                  "last_success_at": previous.get("last_success_at")}
        try:
            fetched = parse_register(fetcher(url), number, now)
            old_count = sum(r["session"] == number for r in records.values())
            if len(fetched) < old_count:
                raise ValueError("Register unexpectedly shrank; retaining previous records")
            records.update({r["symbol"]: r for r in fetched})
            status.update(status="ok", count=len(fetched), last_success_at=stamp(now))
        except Exception as exc:
            status.update(status="error", error=str(exc)[:200])
        sources.append(status)
    result = {"checked_at": stamp(now), "sources": sources,
              "records": sorted(records.values(), key=lambda r: (r["date"], r["symbol"]), reverse=True)}
    result["data_through"] = max((r["date"] for r in result["records"]), default=None)
    atomic_json(directory / "decisions.json", result)
    return result


def current_affairs(decisions, pulse, research, now=None):
    """Editorial input: recent decisions, bounded research, attributed headlines."""
    now = now or datetime.now(UTC)
    start = (now - timedelta(days=29)).date().isoformat()
    end = now.date().isoformat()
    records = [r for r in decisions.get("records", []) if start <= r["date"] <= end]
    records.sort(key=lambda r: (r["date"], r["symbol"]), reverse=True)
    recorded = sum(r["tally"] is not None for r in records)
    # A digest needs breadth; avoid letting several reports of one issue dominate.
    headlines, used_issues = [], set()
    for item in sorted(pulse.get("items", []), key=lambda x: x["published_at"], reverse=True):
        if "/blog/" in item["url"]:
            continue
        if not start <= item["published_at"][:10] <= end:
            continue
        issues = set(item.get("issues", []))
        if issues & used_issues:
            continue
        headlines.append({k: item[k] for k in ("title", "url", "source", "published_at")})
        used_issues.update(issues)
        if len(headlines) == 4:
            break
    return {"window_start": start, "window_end": end, "decisions": records,
            "recorded_count": recorded, "without_vote_count": len(records) - recorded,
            "decision_data_through": decisions.get("data_through"),
            "rollcall_data_through": research.get("data_through"),
            "research_period": research.get("period"),
            "findings": [f for f in research.get("findings", []) if f["id"] in ("alignment-USA", "division", "abstention")],
            "headlines": headlines, "checked_at": decisions.get("checked_at"),
            "sources": decisions.get("sources", []) + pulse.get("sources", [])}
