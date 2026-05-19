"""
Published-edition ledger — the persistent memory behind "publish only when
data moves".

The full edition archive (``data/editions/``) is deliberately gitignored:
editions are reproducible from the dataset, so committing the HTML/MD/JSON
each week would just bloat the repo. The trade-off is that the archive does
*not* survive between CI runs — every GitHub Actions job starts with an
empty checkout, so the skip-if-unchanged gate had nothing to compare
against and could never actually fire.

This module is the fix. It maintains a tiny, append-only JSON ledger that
**is** committed to the repo, recording one line per *actually published*
edition:

    {
      "slug": "2026-w20-arg-grd",
      "edition_date": "2026-05-19",
      "edition_number": 20,
      "country_focus": null,
      "content_hash": "9f3a…",
      "published_at": "2026-05-19T09:01:33Z"
    }

The publish workflow reads :func:`last_published` before sending: if the
freshly composed edition's ``content_hash`` matches the last published one
for the same ``country_focus``, the send is skipped. After a successful
send it calls :func:`record_published` and commits the ledger back, so the
next run remembers. Result: an edition only goes out when the underlying
voting data has actually changed.

The ledger is intentionally minimal (hashes + metadata, no prose) so it
stays small forever — one short record per edition that ships, and
editions only ship when the data moves.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.newsletter import NewsletterEdition

logger = logging.getLogger(__name__)

# Tracked path (NOT under the gitignored data/editions/ tree, and not a
# *.csv, so .gitignore leaves it alone). Override in tests via the env var.
DEFAULT_LEDGER_PATH = Path(os.getenv("ATLAS_LEDGER_PATH", "data/published_ledger.json"))


def _ledger_path(path: Optional[Path] = None) -> Path:
    return Path(path or DEFAULT_LEDGER_PATH)


def load_ledger(path: Optional[Path] = None) -> list[dict]:
    """Return all ledger records, oldest first. Missing/corrupt → empty list.

    A corrupt ledger must not crash the publish pipeline — we log loudly and
    treat it as empty, which fails *open* (an edition will be sent) rather
    than silently never publishing again.
    """
    p = _ledger_path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Published ledger unreadable (%s) — treating as empty.", exc)
        return []
    if not isinstance(data, list):
        logger.warning("Published ledger is not a list — treating as empty.")
        return []
    return data


def last_published(
    country_focus: Optional[str] = None,
    path: Optional[Path] = None,
) -> Optional[dict]:
    """Most recent ledger entry matching ``country_focus`` (None == global).

    Matching on ``country_focus`` keeps per-country editions independent of
    the global edition — a country-focused run shouldn't be suppressed just
    because the global edition was unchanged, and vice versa.
    """
    records = load_ledger(path)
    focus = country_focus or None
    for rec in reversed(records):
        if (rec.get("country_focus") or None) == focus:
            return rec
    return None


def is_duplicate(
    edition: NewsletterEdition,
    path: Optional[Path] = None,
) -> bool:
    """True if ``edition`` has the same content_hash as the last published
    edition for its country_focus — i.e. the data hasn't moved, don't send.
    """
    prior = last_published(edition.country_focus, path)
    return bool(prior and prior.get("content_hash") == edition.content_hash)


def _append(
    *,
    slug: str,
    edition_date: str,
    edition_number: int,
    country_focus: Optional[str],
    content_hash: str,
    path: Optional[Path] = None,
) -> dict:
    """Append one record to the ledger and persist. Returns the record.

    Idempotent on (slug, content_hash): re-recording the identical edition
    is a no-op so a re-run of the publish job can't double-write.
    """
    p = _ledger_path(path)
    records = load_ledger(path)

    for rec in records:
        if rec.get("slug") == slug and rec.get("content_hash") == content_hash:
            logger.info("Edition %s already in ledger — not re-recording.", slug)
            return rec

    record = {
        "slug": slug,
        "edition_date": edition_date,
        "edition_number": int(edition_number),
        "country_focus": country_focus or None,
        "content_hash": content_hash,
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    records.append(record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Recorded edition %s (hash %s…) to ledger %s",
        slug,
        content_hash[:12],
        p,
    )
    return record


def record_published(
    edition: NewsletterEdition,
    path: Optional[Path] = None,
) -> dict:
    """Append ``edition`` to the ledger and persist. Returns the new record."""
    return _append(
        slug=edition.edition_slug,
        edition_date=edition.edition_date,
        edition_number=int(edition.edition_number),
        country_focus=edition.country_focus or None,
        content_hash=edition.content_hash,
        path=path,
    )


def record_published_dict(
    edition_dict: dict,
    path: Optional[Path] = None,
) -> dict:
    """Like :func:`record_published` but takes the archived edition JSON
    (``edition_to_dict`` output) rather than a live ``NewsletterEdition``.

    The publish workflow only has the serialized JSON on disk after the
    compose step, so this keeps the YAML free of fragile object shims.
    """
    return _append(
        slug=edition_dict["edition_slug"],
        edition_date=edition_dict["edition_date"],
        edition_number=int(edition_dict["edition_number"]),
        country_focus=edition_dict.get("country_focus") or None,
        content_hash=edition_dict["content_hash"],
        path=path,
    )
