"""
Edition archive — disk-backed store for composed newsletter editions.

Each edition is written to ``ATLAS_ARCHIVE_DIR/{year}/{slug}.{md|html|json}``
so a directory listing reads like a publication archive:

    editions/
      2024/
        2024-w20-argentina-turkmenistan.md
        2024-w20-argentina-turkmenistan.html
        2024-w20-argentina-turkmenistan.json
        2024-w21-…
      2023/

The archive supports:
  * ``save_edition()`` — write all three formats for one edition (idempotent)
  * ``list_editions()`` — index suitable for a ``/api/newsletter/archive`` GET
  * ``read_edition()`` — fetch a specific archived edition by slug + format
  * ``retrospective_drift_count()`` — aggregate stats across the archive
    (the building block of a "year in review" retrospective)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.newsletter import NewsletterEdition, edition_to_dict
from src.newsletter_render import render_html, render_markdown, render_text

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_DIR = Path(os.getenv("ATLAS_ARCHIVE_DIR", "data/editions"))

# Formats we persist per edition. The .json is the canonical source — the
# other two are renderings. Keeping all three on disk avoids re-running the
# composer every time someone browses the archive.
_FORMATS = ("json", "md", "html", "txt")
_VALID_FORMAT_PATTERN = re.compile(r"^(?:json|md|html|txt)$")


def _archive_root(archive_dir: Optional[Path] = None) -> Path:
    return Path(archive_dir or DEFAULT_ARCHIVE_DIR)


def save_edition(
    edition: NewsletterEdition,
    archive_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Persist all formats of ``edition`` to the archive. Returns paths written.

    The year directory is derived from the edition_date so retrospective
    queries can simply walk ``archive_dir/{year}``.
    """
    root = _archive_root(archive_dir)
    edition_year = int(edition.edition_date.split("-")[0])
    year_dir = root / str(edition_year)
    year_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    json_path = year_dir / f"{edition.edition_slug}.json"
    json_payload = edition_to_dict(edition)
    # Include the rendered markdown alongside JSON so retrospective tooling
    # can grep through edition prose without re-rendering.
    json_payload["markdown"] = render_markdown(edition)
    json_path.write_text(json.dumps(json_payload, default=str, indent=2), encoding="utf-8")
    written["json"] = str(json_path)

    (year_dir / f"{edition.edition_slug}.md").write_text(
        render_markdown(edition), encoding="utf-8"
    )
    written["md"] = str(year_dir / f"{edition.edition_slug}.md")

    (year_dir / f"{edition.edition_slug}.html").write_text(
        render_html(edition), encoding="utf-8"
    )
    written["html"] = str(year_dir / f"{edition.edition_slug}.html")

    (year_dir / f"{edition.edition_slug}.txt").write_text(
        render_text(edition), encoding="utf-8"
    )
    written["txt"] = str(year_dir / f"{edition.edition_slug}.txt")

    logger.info("Archived edition %s (%s files)", edition.edition_slug, len(written))
    return written


def list_editions(archive_dir: Optional[Path] = None) -> list[dict]:
    """Return all archived editions, newest first.

    Each entry: ``{year, slug, edition_date, edition_number, headline,
    country_focus, email_subject, formats: {fmt: relpath}}``.
    """
    root = _archive_root(archive_dir)
    if not root.exists():
        return []

    out: list[dict] = []
    for year_dir in sorted(root.iterdir(), reverse=True):
        if not year_dir.is_dir():
            continue
        if not year_dir.name.isdigit():
            continue
        for json_path in sorted(year_dir.glob("*.json"), reverse=True):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable edition %s: %s", json_path, exc)
                continue
            slug = json_path.stem
            formats: dict[str, str] = {}
            for ext in ("md", "html", "txt", "json"):
                p = year_dir / f"{slug}.{ext}"
                if p.exists():
                    formats[ext] = f"{year_dir.name}/{slug}.{ext}"
            out.append({
                "year": int(year_dir.name),
                "slug": slug,
                "edition_date": payload.get("edition_date"),
                "edition_number": payload.get("edition_number"),
                "headline": payload.get("headline"),
                "country_focus": payload.get("country_focus"),
                "email_subject": payload.get("email_subject"),
                "content_hash": payload.get("content_hash"),
                "period_label": payload.get("period_label"),
                "formats": formats,
            })
    out.sort(key=lambda e: (e.get("edition_date") or "", e.get("slug") or ""), reverse=True)
    return out


def read_edition(
    year: int,
    slug: str,
    fmt: str = "md",
    archive_dir: Optional[Path] = None,
) -> Optional[tuple[str, str]]:
    """Read an archived edition; return (mimetype, body) or None if not found.

    ``fmt`` must be one of ``json|md|html|txt`` — anything else is rejected
    rather than mapped, to prevent path-traversal via crafted extensions.
    """
    if not _VALID_FORMAT_PATTERN.match(fmt):
        return None
    # Defense in depth: slug must not contain path separators.
    if "/" in slug or ".." in slug or "\\" in slug:
        return None
    root = _archive_root(archive_dir)
    path = root / str(int(year)) / f"{slug}.{fmt}"
    if not path.exists() or not path.is_file():
        return None
    mimetype = {
        "json": "application/json",
        "md": "text/markdown; charset=utf-8",
        "html": "text/html; charset=utf-8",
        "txt": "text/plain; charset=utf-8",
    }[fmt]
    return mimetype, path.read_text(encoding="utf-8")


def retrospective_drift_count(
    archive_dir: Optional[Path] = None,
    year: Optional[int] = None,
) -> dict:
    """Aggregate how often each country pair appears in the archive's top-drifts.

    Returns ``{pair: {count, total_abs_delta_pts, latest_edition}}`` — the
    raw material for a "drift of the year" retrospective. Pairs are stored
    sorted (alphabetical) so direction-symmetric counts merge correctly.
    """
    root = _archive_root(archive_dir)
    if not root.exists():
        return {"pairs": [], "n_editions": 0}

    pair_stats: dict[tuple[str, str], dict] = {}
    n_editions = 0
    year_filter = str(year) if year else None
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        if year_filter and year_dir.name != year_filter:
            continue
        for json_path in sorted(year_dir.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            n_editions += 1
            chart_payloads = payload.get("chart_payloads") or {}
            drifts = chart_payloads.get("top_drifts_raw") or []
            for d in drifts:
                a, b = d.get("country_a"), d.get("country_b")
                if not a or not b:
                    continue
                key = tuple(sorted((str(a), str(b))))
                rec = pair_stats.setdefault(
                    key, {"count": 0, "total_abs_delta_pts": 0.0, "latest_edition": None}
                )
                rec["count"] += 1
                rec["total_abs_delta_pts"] += abs(float(d.get("delta", 0.0))) * 100.0
                rec["latest_edition"] = payload.get("edition_date") or rec["latest_edition"]

    rows = sorted(
        (
            {
                "country_a": a,
                "country_b": b,
                "count": v["count"],
                "total_abs_delta_pts": round(v["total_abs_delta_pts"], 1),
                "latest_edition": v["latest_edition"],
            }
            for (a, b), v in pair_stats.items()
        ),
        key=lambda r: (r["count"], r["total_abs_delta_pts"]),
        reverse=True,
    )
    return {"pairs": rows, "n_editions": n_editions}
