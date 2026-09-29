"""Single-writer automatic collection and research publication."""

from datetime import datetime
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
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
            if brief.get("content_hash") != prior.get("content_hash") or not prior or brief.get("status") != "ready":
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


def refresh_votes_if_due(directory=None, now=None, runner=subprocess.run):
    """Daily complete-roll-call catch-up while the app is running."""
    from src.pulse import date_value, stamp
    directory = directory or data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(UTC)
    with (directory / ".votes-refresh.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        path = directory / "votes-refresh.json"
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
        checked = date_value(previous.get("checked_at"))
        interval = 86400 if previous.get("status") == "ok" else 3600
        if checked and (now - checked).total_seconds() < interval:
            return None
        result = {"checked_at": stamp(now), "status": "error"}
        root = Path(__file__).resolve().parents[1]
        try:
            process = runner([sys.executable, str(root / "scripts" / "refresh_data.py"),
                              "--days", "90", "--promote"], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=600)
            if process.returncode:
                raise RuntimeError(f"Voting refresh exited {process.returncode}: {process.stderr[-1000:]}")
            result["status"] = "ok"
        except Exception as exc:
            result["error"] = str(exc)[:1200]
            logger.warning("Automatic complete-roll-call refresh failed: %s", exc)
        atomic_json(path, result)
        return result


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
                checked = date_value(read_snapshot().get("checked_at"))
                missing_research = (not (data_dir() / "research.json").exists() or not (data_dir() / "newsletter.json").exists()) and dataframe_provider() is not None
                if checked is None or changed or missing_research or (datetime.now(UTC) - checked).total_seconds() >= interval:
                    try:
                        refresh_all(dataframe=dataframe_provider())
                    except Exception:
                        logger.exception("Newsletter refresh failed; retaining the last edition")
                if dataframe_provider() is not None:
                    refresh_votes_if_due()
            except Exception:
                logger.exception("Automatic briefing refresh failed; retaining the last published edition")
            threading.Event().wait(min(interval, 60))
    threading.Thread(target=work, name="un-pulse", daemon=True).start()
