"""Idempotent automatic newsletter composition, archive and local publication."""

from datetime import datetime
import json
import logging
import os
import tempfile

from src.newsletter import build_newsletter_edition, edition_from_dict, edition_to_dict
from src.newsletter_render import render_html, render_markdown, render_text
from src.country_display import display_lookup
from src.newsletter_story import enrich
from src.pulse import UTC, atomic_json, data_dir, read_snapshot
from src.session_calendar import calendar_for
from src.un_bodies import summary
from src.un_decisions import current_affairs, read_decisions

logger = logging.getLogger(__name__)


def atomic_text(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cached_affairs(directory=None, now=None):
    directory = directory or data_dir()
    try:
        research = json.loads((directory / "research.json").read_text())
    except (OSError, ValueError):
        research = {}
    decisions, snapshot = read_decisions(directory), read_snapshot(directory)
    if not decisions.get("records") and not snapshot.get("items"):
        return {}
    return current_affairs(decisions, snapshot, research, now)


def compose_evidence(dataframe, decisions, pulse, research, bodies=None, now=None):
    """Everything an edition is written from: the 30-day digest, each vote's
    earlier results from the record, the Council and committee ledger, and the
    calendar of votes still to come."""
    now = now or datetime.now(UTC)
    live = current_affairs(decisions, pulse, research, now)
    if dataframe is not None:
        raw = dataframe.drop_duplicates("country_identifier").set_index("country_identifier")["country_name"].to_dict()
        live["decisions"] = enrich(live["decisions"], dataframe, display_lookup(raw))
        try:
            calendar = calendar_for(dataframe, as_of=now.date().isoformat())
            live["calendar"] = {
                "upcoming": calendar["upcoming"], "phase": calendar["phase"]["label"],
                # Main Committees wrap up by the end of November.
                "committees_until": f"{now.year}-11-30" if now.month in (10, 11) else None,
            }
        except ValueError:
            logger.warning("Calendar unavailable for this edition", exc_info=True)
    if bodies and (bodies.get("actions") or bodies.get("vetoes")):
        digest = summary(bodies, now)
        live["vetoes_this_year"] = digest.pop("vetoes_this_year")
        live["bodies"] = digest
    return live


def publish_newsletter(dataframe, decisions, pulse, research, directory=None, now=None, bodies=None):
    directory = directory or data_dir()
    live = compose_evidence(dataframe, decisions, pulse, research, bodies, now)
    if not live["decisions"] and not live["headlines"]:
        raise ValueError("No current evidence available; retaining last newsletter")
    edition = build_newsletter_edition(dataframe, live_updates=live, edition_date=now.date().isoformat() if now else None)
    path = directory / "newsletter.json"
    try:
        prior = json.loads(path.read_text())
    except (OSError, ValueError):
        prior = {}
    changed = prior.get("content_hash") != edition.content_hash
    # Reuse the exact archived edition if the editorial evidence is unchanged.
    archive = directory / "newsletters" / f"{edition.content_hash}.json"
    if not changed:
        edition = edition_from_dict(prior)
    elif archive.exists():
        edition = edition_from_dict(json.loads(archive.read_text()))
    else:
        atomic_json(archive, edition_to_dict(edition))
    for suffix, renderer in (("html", render_html), ("md", render_markdown), ("txt", render_text)):
        body = renderer(edition)
        atomic_text(directory / "newsletters" / f"{edition.content_hash}.{suffix}", body)
        atomic_text(directory / f"newsletter.{suffix}", body)
    atomic_json(path, edition_to_dict(edition))
    try:
        from src.mailer import deliver_if_due
        deliver_if_due(edition, directory, now)
    except Exception:  # delivery must never undo a publication
        logger.exception("Edition published but not emailed")
    return {"changed": changed, "content_hash": edition.content_hash, "edition_date": edition.edition_date}
