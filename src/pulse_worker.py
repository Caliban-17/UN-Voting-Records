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


def refresh_all(directory=None, dataframe=None, collect=True):
    """One pass: collect the sources, then (given the voting frame) rebuild the
    research and the edition. ``collect=False`` composes from what the last
    collection left on disk, without fetching anything again."""
    directory = directory or data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # OS releases the lock after a crash. Gunicorn workers and a CLI can share it.
    with (directory / ".refresh.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        from src.un_bodies import read_bodies, refresh_bodies
        from src.un_decisions import refresh_decisions
        if collect:
            snapshot, decisions, bodies = refresh(directory), refresh_decisions(directory), refresh_bodies(directory)
        else:
            from src.pulse import read_snapshot
            from src.un_decisions import heal_ledger
            snapshot, decisions, bodies = read_snapshot(directory), heal_ledger(directory), read_bodies(directory)
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
                publish_newsletter(dataframe, decisions, snapshot, brief, directory, None, bodies)
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
                # The first pass after a start runs before the data has loaded,
                # so it can only collect. As soon as the data is there, do one
                # full pass: otherwise a deploy that changes what the research
                # or the edition contains shows nothing new for half an hour.
                missing_research = has_data and not attempted_with_data
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


# ── lazy mode: the web process holds no voting data between uses ─────────────


def compose_key(directory, version) -> str:
    """What an edition is composed from, apart from headlines: when this is
    unchanged there is nothing new to say and no reason to load the data."""
    import hashlib
    from datetime import timedelta
    from src.un_bodies import read_bodies
    from src.un_decisions import read_decisions

    since = (datetime.now(UTC) - timedelta(days=35)).date().isoformat()
    recent = [(r["symbol"], r["date"], r.get("tally"), r.get("first_seen_at"))
              for r in read_decisions(directory).get("records", []) if r["date"] >= since]
    vetoes = read_bodies(directory).get("vetoes", [])
    parts = [version, datetime.now(UTC).date().isoformat(), recent, len(vetoes), vetoes[:1],
             os.getenv("RAILWAY_DEPLOYMENT_ID", "")]
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def compose_in_subprocess(runner=None) -> bool:
    """Rebuild the research and the edition in a short-lived process, so the
    gigabyte it needs is gone again a minute later."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = (runner or subprocess.run)([sys.executable, "-m", "src.pulse_worker", "compose"], cwd=root, timeout=900)
    return result.returncode == 0


def start_lazy_worker(idle_hook, drop_data, compose=compose_in_subprocess):
    """The worker for a web process that loads voting data only on demand.

    Every ``interval`` it collects the sources (no voting data needed). It
    composes, in a subprocess, only when what an edition is built from has
    changed: new data, a new decision or veto, a new day, a new deploy. Every
    minute it lets the web process release data nobody is using."""
    if os.getenv("PULSE_AUTO_REFRESH", "1") != "1":
        return
    interval = max(300, int(os.getenv("PULSE_REFRESH_SECONDS", "1800")))

    def source_version():
        from src.config import UN_VOTES_CSV_PATH
        try:
            stat = UN_VOTES_CSV_PATH.stat()
            return [stat.st_mtime_ns, stat.st_size]
        except OSError:
            return None

    def work():
        from src.pulse import date_value, read_snapshot
        seen_version = source_version()
        collected = False        # collect once per start, whatever the last pass's age
        while True:
            try:
                idle_hook()
                version = source_version()
                if version != seen_version:
                    seen_version = version
                    drop_data()          # the next request loads the new file
                directory = data_dir()
                checked = date_value(read_snapshot(directory).get("checked_at"))
                if not collected or checked is None or (datetime.now(UTC) - checked).total_seconds() >= interval:
                    collected = refresh_all(directory) is not None
                state = directory / "compose-state.json"
                try:
                    done = json.loads(state.read_text()).get("key")
                except (OSError, ValueError):
                    done = None
                key = compose_key(directory, version)
                if version is not None and key != done:
                    # Record the attempt either way: a compose that cannot
                    # succeed must not be retried every minute.
                    atomic_json(state, {"key": key, "ok": bool(compose())})
            except Exception:
                logger.exception("Automatic briefing refresh failed; retaining the last published edition")
            threading.Event().wait(60)
    threading.Thread(target=work, name="un-pulse", daemon=True).start()


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["compose"]:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
        from src.config import UN_VOTES_CSV_PATH
        from src.data_processing import load_and_preprocess_data

        frame, _ = load_and_preprocess_data(str(UN_VOTES_CSV_PATH))
        raise SystemExit(0 if refresh_all(dataframe=frame, collect=False) is not None else 1)
    raise SystemExit("usage: python -m src.pulse_worker compose")
