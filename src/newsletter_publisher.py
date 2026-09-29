"""Idempotent automatic newsletter composition, archive and local publication."""

import json
import os
import tempfile

from src.newsletter import build_newsletter_edition, edition_from_dict, edition_to_dict
from src.newsletter_render import render_html, render_markdown, render_text
from src.pulse import atomic_json, data_dir, read_snapshot
from src.un_decisions import current_affairs, read_decisions


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


def publish_newsletter(dataframe, decisions, pulse, research, directory=None, now=None):
    directory = directory or data_dir()
    live = current_affairs(decisions, pulse, research, now)
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
    return {"changed": changed, "content_hash": edition.content_hash, "edition_date": edition.edition_date}
