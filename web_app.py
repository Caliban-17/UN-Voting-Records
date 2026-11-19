"""
UN Voting Intelligence Platform — entry point.

Run locally:
    python web_app.py

Production (Gunicorn):
    gunicorn -w 2 -b 0.0.0.0:5001 web_app:app
"""

from __future__ import annotations

from app import create_app
from app.services import drop_data, get_df, idle_minutes, load_data, memory_mb, unload_if_idle
import logging
import os
import threading

logger = logging.getLogger(__name__)

# Build the Flask application.
# Data is loaded once here so it is available before the first request
# regardless of how the process is started (dev server or Gunicorn).
app = create_app()

# Current-context collection starts immediately, even if the historical CSV is
# unavailable or still loading. The worker retries research once data is ready.
from src.pulse_worker import start_worker  # noqa: E402

LAZY_DATA = idle_minutes() > 0

if LAZY_DATA:
    # Hosted, mostly idle: hold no voting data until a request needs it, let
    # it go when idle, and compose editions in a short-lived process.
    from src.pulse_worker import start_lazy_worker  # noqa: E402

    def _release_when_idle() -> None:
        """Unload idle data; if the C library still will not give the memory
        back, let gunicorn replace this worker with a fresh, small one."""
        import signal

        if not unload_if_idle():
            return
        held, limit = memory_mb(), int(os.getenv("DATA_RSS_LIMIT_MB", "600"))
        if held and held > limit and "gunicorn" in os.getenv("SERVER_SOFTWARE", ""):
            logger.info("Still holding %d MB after unloading; recycling the worker", held)
            os.kill(os.getpid(), signal.SIGTERM)

    start_lazy_worker(_release_when_idle, drop_data)
else:
    start_worker(get_df, load_data)

if not LAZY_DATA and not load_data():
    logger.critical(
        "startup: data loading failed — check %s",
        os.getenv("UN_VOTING_DATA_PATH", "data/"),
    )


def _warm_lenses() -> None:
    """Compute the 'Through which lens' payload once, off the request path:
    the whole-record scorecard takes the better part of a minute, and the
    first visitor should not pay it."""
    try:
        from src.lenses import lens_timeline_cached

        df = get_df()
        if df is not None:
            lens_timeline_cached(df)
            logger.info("startup: lens timeline warmed")
    except Exception as exc:  # noqa: BLE001 — a warm-up must never take the app down
        logger.warning("startup: lens warm-up skipped: %s", exc)


if not LAZY_DATA and get_df() is not None and os.getenv("LENS_WARMUP", "1") != "0":
    threading.Thread(target=_warm_lenses, name="lens-warmup", daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    logger.info("Starting Flask dev server on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
