"""
Middleware: rate limiting, security headers, concurrency guard.
Registered by create_app() via register_middleware().
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import threading
from collections import defaultdict, deque
from functools import wraps
from typing import Callable

from urllib.parse import quote

from flask import Flask, redirect, request, jsonify, session

logger = logging.getLogger(__name__)

# ── Rate limiting ─────────────────────────────────────────────────────────────

MAX_API_REQUESTS_PER_MIN = max(20, int(os.getenv("MAX_API_REQUESTS_PER_MIN", "180")))
RATE_LIMIT_WINDOW_SEC = max(5, int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60")))
_rate_lock = threading.RLock()
_rate_buckets: dict[str, deque] = defaultdict(deque)


# Reverse proxies of our own in front of the app. X-Forwarded-For is written
# by whoever sends the request, so with none (the default) it is ignored: a
# client could otherwise claim a new address on every request and never be
# limited. With N proxies, the Nth entry from the right is the one they added.
TRUSTED_PROXY_COUNT = max(0, int(os.getenv("TRUSTED_PROXY_COUNT", "0")))


def _client_fingerprint() -> str:
    if TRUSTED_PROXY_COUNT:
        hops = [h.strip() for h in request.headers.get("X-Forwarded-For", "").split(",") if h.strip()]
        if len(hops) >= TRUSTED_PROXY_COUNT:
            return hops[-TRUSTED_PROXY_COUNT]
    return request.remote_addr or "unknown"


def _route_key() -> str:
    """The matched route's pattern, not the raw path: /api/story/resolution/1/map
    and /2/map are one endpoint and share one bucket, and unknown paths cannot
    mint a bucket each."""
    return request.url_rule.rule if request.url_rule is not None else "unmatched"


_last_sweep = 0.0


def _sweep_stale_buckets(now: float) -> None:
    """Drop buckets with no hits inside the window (at most once per window).
    Without this the dict grows by one entry per (client, method, path) forever."""
    global _last_sweep
    if now - _last_sweep < RATE_LIMIT_WINDOW_SEC:
        return
    _last_sweep = now
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    for key in [k for k, b in _rate_buckets.items() if not b or b[-1] < cutoff]:
        del _rate_buckets[key]


def _is_rate_limited(key: str) -> bool:
    now = time.time()
    with _rate_lock:
        _sweep_stale_buckets(now)
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


# ── Site password ─────────────────────────────────────────────────────────────

# Everything is private when SITE_PASSWORD is set, except what a would-be
# reader needs before they have it: the sign-in page, signing up to the
# newsletter, confirming and unsubscribing.
PUBLIC_PREFIXES = ("/login", "/logout", "/subscribe", "/unsubscribe", "/health", "/admin/mail")


def site_password() -> str | None:
    return os.getenv("SITE_PASSWORD") or None


def session_secret() -> bytes:
    """Signing key for the sign-in cookie, derived from the password so that
    changing the password signs everyone out and no second secret is needed."""
    seed = os.getenv("SECRET_KEY") or f"un-scrupulous-session:{site_password() or 'open'}"
    return hashlib.sha256(seed.encode()).digest()


def check_password(given: str) -> bool:
    password = site_password()
    return bool(password) and hmac.compare_digest((given or "").encode(), password.encode())


def login_attempt_allowed() -> bool:
    return not _is_rate_limited(f"{_client_fingerprint()}:login")


def _password_required():
    if not site_password() or request.path.startswith(PUBLIC_PREFIXES) or request.method == "OPTIONS":
        return None
    if session.get("signed_in"):
        return None
    if request.path.startswith("/api/") or request.path.endswith(".xml"):
        return jsonify({"error": "Sign in at /login to use this."}), 401
    return redirect("/login?next=" + quote(request.full_path.rstrip("?"), safe="/?=&"))


def register_middleware(app: Flask) -> None:
    app.secret_key = session_secret()
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=bool(os.getenv("RAILWAY_ENVIRONMENT")),
        PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,
    )
    app.before_request(_password_required)

    @app.before_request
    def enforce_rate_limits():
        if not request.path.startswith("/api/"):
            return None
        key = f"{_client_fingerprint()}:{request.method}:{_route_key()}"
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
            # Plotly and axios are vendored under static/vendor/, so no CDN is
            # allowed to run script here. Add a host only if you add a CDN tag.
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "font-src 'self' https://cdnjs.cloudflare.com data:; "
            "img-src 'self' data: blob:; "
            # The choropleth's world topojson is self-hosted under static/vendor/,
            # so no external connect-src is needed — the map has no runtime CDN
            # dependency and same-origin 'self' covers the fetch.
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
