"""
UN Voting Intelligence Platform — entry point.

Run locally:
    python web_app.py

Production (Gunicorn):
    gunicorn -w 2 -b 0.0.0.0:5001 web_app:app
"""

from __future__ import annotations

from app import create_app
from app.services import load_data
import logging
import os

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

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    logger.info("Starting Flask dev server on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
