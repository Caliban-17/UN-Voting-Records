"""Newsletter subscribers: a small SQLite table beside the published state.

Sign-up is public and needs no account. It is double opt-in: an address
receives nothing but one confirmation message until its owner follows the
link in it, and every edition carries a one-click unsubscribe link.
"""

from __future__ import annotations

from datetime import datetime
import re
import secrets
import sqlite3
import threading
import time

from src.pulse import UTC, data_dir, stamp

EMAIL = re.compile(r"[^@\s<>\"',;]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}")
SIGNUPS_PER_HOUR = 5
_attempts: dict[str, list[float]] = {}
_lock = threading.Lock()


def _connect(directory=None) -> sqlite3.Connection:
    directory = directory or data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "subscribers.db", timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS subscribers (
        email TEXT PRIMARY KEY, token TEXT UNIQUE NOT NULL, status TEXT NOT NULL,
        created_at TEXT NOT NULL, confirmed_at TEXT, last_sent_hash TEXT, last_sent_at TEXT)""")
    return db


def valid_email(value: str) -> str | None:
    value = (value or "").strip().lower()
    return value if len(value) <= 254 and EMAIL.fullmatch(value) else None


def allow_signup(client: str, now: float | None = None) -> bool:
    """At most ``SIGNUPS_PER_HOUR`` sign-up attempts per client address."""
    now = now or time.time()
    with _lock:
        recent = [t for t in _attempts.get(client, []) if now - t < 3600]
        allowed = len(recent) < SIGNUPS_PER_HOUR
        if allowed:
            recent.append(now)
        _attempts[client] = recent
        for key in [k for k, v in _attempts.items() if not v]:
            del _attempts[key]
        return allowed


def add(email: str, directory=None, now: datetime | None = None) -> dict | None:
    """Register an address as pending and return its row; None if it is already
    confirmed (the caller must answer the same way in both cases, so that the
    form never reveals who subscribes)."""
    with _connect(directory) as db:
        row = db.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
        if row and row["status"] == "confirmed":
            return None
        token = row["token"] if row else secrets.token_urlsafe(24)
        db.execute("INSERT INTO subscribers (email, token, status, created_at) VALUES (?, ?, 'pending', ?) "
                   "ON CONFLICT(email) DO UPDATE SET status = 'pending'",
                   (email, token, stamp(now or datetime.now(UTC))))
        return dict(db.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone())


def confirm(token: str, directory=None, now: datetime | None = None) -> dict | None:
    with _connect(directory) as db:
        row = db.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone()
        if not row or row["status"] == "unsubscribed":
            return None
        db.execute("UPDATE subscribers SET status = 'confirmed', confirmed_at = COALESCE(confirmed_at, ?) "
                   "WHERE token = ?", (stamp(now or datetime.now(UTC)), token))
        return dict(db.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone())


def unsubscribe(token: str, directory=None) -> bool:
    with _connect(directory) as db:
        return db.execute("UPDATE subscribers SET status = 'unsubscribed' WHERE token = ?", (token,)).rowcount > 0


def confirmed(directory=None) -> list[dict]:
    with _connect(directory) as db:
        return [dict(r) for r in db.execute("SELECT * FROM subscribers WHERE status = 'confirmed' ORDER BY email")]


def mark_sent(email: str, content_hash: str, directory=None, now: datetime | None = None) -> None:
    with _connect(directory) as db:
        db.execute("UPDATE subscribers SET last_sent_hash = ?, last_sent_at = ? WHERE email = ?",
                   (content_hash, stamp(now or datetime.now(UTC)), email))


def counts(directory=None) -> dict:
    with _connect(directory) as db:
        return {r["status"]: r["n"] for r in db.execute("SELECT status, COUNT(*) AS n FROM subscribers GROUP BY status")}
