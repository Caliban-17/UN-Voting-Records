"""
UN Voting Intelligence Platform — Flask application package.

Create and configure the Flask app:

    from app import create_app
    flask_app = create_app()

All routes are registered via blueprints in app/routes/.
Middleware (rate limiting, security headers, concurrency guard)
is registered in app/middleware.py.
Shared services (job queue, caches, globals) live in app/services.py.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flask import Flask
from flask_cors import CORS

from src.cache_utils import LRUCache, model_registry  # noqa: F401 – re-exported

# ── single root logger configured once ──────────────────────────────────────


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        force=True,  # override any earlier basicConfig calls
    )


_configure_logging()
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory — builds and returns the configured Flask app."""
    flask_app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # ── runtime config ───────────────────────────────────────────────────────
    flask_app.config["MAX_CONTENT_LENGTH"] = int(
        os.getenv("MAX_CONTENT_LENGTH_BYTES", "1048576")
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins = [
        o.strip()
        for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
        if o.strip()
    ]
    CORS(flask_app, resources={r"/api/*": {"origins": allowed_origins or "*"}})

    # ── middleware ────────────────────────────────────────────────────────────
    from app.middleware import register_middleware

    register_middleware(flask_app)

    # ── blueprints ────────────────────────────────────────────────────────────
    from app.routes.core import bp as core_bp
    from app.routes.analysis import bp as analysis_bp
    from app.routes.visualization import bp as viz_bp
    from app.routes.prediction import bp as pred_bp
    from app.routes.jobs import bp as jobs_bp

    flask_app.register_blueprint(core_bp)
    flask_app.register_blueprint(analysis_bp, url_prefix="/api/analysis")
    flask_app.register_blueprint(viz_bp, url_prefix="/api/visualization")
    flask_app.register_blueprint(pred_bp, url_prefix="/api/prediction")
    flask_app.register_blueprint(jobs_bp, url_prefix="/api/jobs")

    return flask_app
