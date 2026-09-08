"""
UN Voting Intelligence Platform — entry point.

Run locally:
    python web_app.py

Production (Gunicorn):
    gunicorn -w 2 -b 0.0.0.0:5001 web_app:app
"""

from __future__ import annotations

from app import create_app
from app.services import get_df, load_data
import logging
import os
import threading

logger = logging.getLogger(__name__)

# Build the Flask application.
# Data is loaded once here so it is available before the first request
# regardless of how the process is started (dev server or Gunicorn).
app = create_app()

if not load_data():
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


if get_df() is not None and os.getenv("LENS_WARMUP", "1") != "0":
    threading.Thread(target=_warm_lenses, name="lens-warmup", daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    logger.info("Starting Flask dev server on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
