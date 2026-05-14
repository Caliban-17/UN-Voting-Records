"""
Middleware: rate limiting, security headers, concurrency guard.
Registered by create_app() via register_middleware().
"""

from __future__ import annotations

import logging
import os
import time
import threading
from collections import defaultdict, deque
from functools import wraps
from typing import Callable

from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)

# ── Rate limiting ─────────────────────────────────────────────────────────────

MAX_API_REQUESTS_PER_MIN = max(20, int(os.getenv("MAX_API_REQUESTS_PER_MIN", "180")))
RATE_LIMIT_WINDOW_SEC = max(5, int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60")))
_rate_lock = threading.RLock()
_rate_buckets: dict[str, deque] = defaultdict(deque)


def _client_fingerprint() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    remote = forwarded.split(",")[0].strip() if forwarded else ""
    return remote or request.remote_addr or "unknown"


def _is_rate_limited(key: str) -> bool:
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[key]
        cutoff = now - RATE_LIMIT_WINDOW_SEC
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= MAX_API_REQUESTS_PER_MIN:
            return True
        bucket.append(now)
        return False


# ── Concurrency guard ─────────────────────────────────────────────────────────

MAX_CONCURRENT_ANALYSIS = max(1, int(os.getenv("MAX_CONCURRENT_ANALYSIS", "2")))
ANALYSIS_SLOT_TIMEOUT = float(os.getenv("ANALYSIS_SLOT_TIMEOUT_SEC", "0.25"))
_analysis_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSIS)


def with_analysis_slot(func: Callable) -> Callable:
    """Decorator: limits concurrent heavy analyses."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        acquired = _analysis_semaphore.acquire(timeout=ANALYSIS_SLOT_TIMEOUT)
        if not acquired:
            return (
                jsonify(
                    {
                        "error": "Server is busy processing other analyses. Try again shortly."
                    }
                ),
                429,
            )
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _analysis_semaphore.release()
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("%s completed in %.1f ms", func.__name__, elapsed_ms)

    return wrapper


# ── Registration ──────────────────────────────────────────────────────────────


def register_middleware(app: Flask) -> None:
    @app.before_request
    def enforce_rate_limits():
        if not request.path.startswith("/api/"):
            return None
        key = f"{_client_fingerprint()}:{request.method}:{request.path}"
        if _is_rate_limited(key):
            return (
                jsonify({"error": "Rate limit exceeded. Please wait before retrying."}),
                429,
            )
        return None

    @app.after_request
    def add_security_headers(response):
        csp = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.plot.ly https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "font-src 'self' https://cdnjs.cloudflare.com data:; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        return response
