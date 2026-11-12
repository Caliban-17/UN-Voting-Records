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
    db.execute("""CREATE TABLE IF NOT EXISTS outbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, message BLOB NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, sent_at TEXT,
        recipient TEXT)""")
    if "recipient" not in {row["name"] for row in db.execute("PRAGMA table_info(outbox)")}:
        db.execute("ALTER TABLE outbox ADD COLUMN recipient TEXT")
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
    """Unsubscribe, and cancel anything still queued for that address: the
    page promises no more emails, including ones waiting to be sent."""
    with _connect(directory) as db:
        row = db.execute("SELECT email FROM subscribers WHERE token = ?", (token,)).fetchone()
        if not row:
            return False
        db.execute("UPDATE subscribers SET status = 'unsubscribed' WHERE token = ?", (token,))
        db.execute("UPDATE outbox SET status = 'cancelled', message = x'' WHERE status = 'pending' AND recipient = ?",
                   (row["email"],))
        return True


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


# ── outbox: mail waiting for a sender that can reach the mail server ─────────

OUTBOX_KEEP_DAYS = 7
OUTBOX_MAX_ATTEMPTS = 20


def enqueue(raw: bytes, directory=None, now: datetime | None = None, recipient: str | None = None) -> int:
    with _connect(directory) as db:
        return db.execute("INSERT INTO outbox (created_at, message, recipient) VALUES (?, ?, ?)",
                          (stamp(now or datetime.now(UTC)), raw, (recipient or "").strip().lower() or None)).lastrowid


def pending_mail(directory=None, limit: int = 40, now: datetime | None = None) -> list[dict]:
    """Queued messages, oldest first. A message nobody could send for a week,
    or that failed twenty times, is dropped rather than sent stale."""
    from datetime import timedelta
    cutoff = stamp((now or datetime.now(UTC)) - timedelta(days=OUTBOX_KEEP_DAYS))
    with _connect(directory) as db:
        db.execute("UPDATE outbox SET status = 'expired' WHERE status = 'pending' AND (created_at < ? OR attempts >= ?)",
                   (cutoff, OUTBOX_MAX_ATTEMPTS))
        # Belt and braces: never hand out mail for an address that has unsubscribed.
        db.execute("UPDATE outbox SET status = 'cancelled', message = x'' WHERE status = 'pending' AND recipient IN "
                   "(SELECT email FROM subscribers WHERE status = 'unsubscribed')")
        return [dict(r) for r in db.execute(
            "SELECT id, message FROM outbox WHERE status = 'pending' ORDER BY id LIMIT ?", (limit,))]


def acknowledge(sent: list[int], failed: list[int], directory=None, now: datetime | None = None) -> dict:
    """Record what the sender did. A sent message is kept only as a row
    without its body, so no address lingers in the queue."""
    with _connect(directory) as db:
        for ident in sent:
            db.execute("UPDATE outbox SET status = 'sent', sent_at = ?, message = x'' WHERE id = ? AND status = 'pending'",
                       (stamp(now or datetime.now(UTC)), int(ident)))
        for ident in failed:
            db.execute("UPDATE outbox SET attempts = attempts + 1 WHERE id = ? AND status = 'pending'", (int(ident),))
        return {r["status"]: r["n"] for r in db.execute("SELECT status, COUNT(*) AS n FROM outbox GROUP BY status")}
