"""Single-writer automatic collection and research publication.

The web process only reads the voting CSV. Complete roll calls are refreshed and
promoted outside it (refresh-data.yml, or the compose ``un-voting-refresher``
service); the worker notices the promoted file and reloads it.
"""

from datetime import datetime
import fcntl
import json
import logging
import os
import threading

from src.pulse import UTC, atomic_json, data_dir, refresh

logger = logging.getLogger(__name__)


def refresh_all(directory=None, dataframe=None):
    directory = directory or data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # OS releases the lock after a crash. Gunicorn workers and a CLI can share it.
    with (directory / ".refresh.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        snapshot = refresh(directory)
        from src.un_decisions import refresh_decisions
        decisions = refresh_decisions(directory)
        if dataframe is not None:
            from src.research_brief import build_research_brief
            brief = build_research_brief(dataframe)
            path = directory / "research.json"
            try:
                prior = json.loads(path.read_text()) if path.exists() else {}
            except ValueError:
                prior = {}
            if brief.get("status") != "ready" and prior.get("status") == "ready":
                # A thin or failed rebuild must not unpublish good research.
                logger.warning("Research rebuild not ready (%s); keeping the published brief",
                               brief.get("message"))
                brief = prior
            elif brief.get("content_hash") != prior.get("content_hash") or not prior or brief.get("status") != "ready":
                if brief.get("status") == "ready":
                    archive = directory / "research" / f"{brief['content_hash'][:16]}.json"
                    if archive.exists():
                        brief = json.loads(archive.read_text())
                    else:
                        atomic_json(archive, brief)
                atomic_json(path, brief)
            if brief.get("status") == "ready":
                from src.newsletter_publisher import publish_newsletter
                publish_newsletter(dataframe, decisions, snapshot, brief, directory)
        return snapshot


def start_worker(dataframe_provider, reload_data=None):
    """Refresh without a visitor, request-time fetch, model key or editor."""
    if os.getenv("PULSE_AUTO_REFRESH", "1") != "1":
        return
    interval = max(300, int(os.getenv("PULSE_REFRESH_SECONDS", "1800")))

    def source_version():
        from src.config import UN_VOTES_CSV_PATH
        try:
            stat = UN_VOTES_CSV_PATH.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def work():
        loaded_version = source_version()
        # A missing edition is retried as soon as data first arrives, then only
        # at the normal interval: a state that cannot publish (no evidence, thin
        # research) must not poll every source once a minute.
        attempted_with_data = False
        while True:
            try:
                # Also avoid duplicate fetches by processes that acquire the lock
                # just after another worker finishes.
                from src.pulse import read_snapshot, date_value
                version = source_version()
                changed = version is not None and version != loaded_version
                if changed and reload_data is not None:
                    if not reload_data():
                        raise RuntimeError("Changed voting data could not be loaded")
                    loaded_version = version
                    attempted_with_data = False
                checked = date_value(read_snapshot().get("checked_at"))
                has_data = dataframe_provider() is not None
                missing_research = has_data and not attempted_with_data and (
                    not (data_dir() / "research.json").exists()
                    or not (data_dir() / "newsletter.json").exists())
                if checked is None or changed or missing_research or (datetime.now(UTC) - checked).total_seconds() >= interval:
                    attempted_with_data = attempted_with_data or has_data
                    try:
                        refresh_all(dataframe=dataframe_provider())
                    except Exception:
                        logger.exception("Newsletter refresh failed; retaining the last edition")
            except Exception:
                logger.exception("Automatic briefing refresh failed; retaining the last published edition")
            threading.Event().wait(min(interval, 60))
    threading.Thread(target=work, name="un-pulse", daemon=True).start()
