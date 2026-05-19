"""Tests for the published-edition ledger (publish-only-when-data-moves)."""

import json
from types import SimpleNamespace

import pytest

from src.newsletter_ledger import (
    is_duplicate,
    last_published,
    load_ledger,
    record_published,
    record_published_dict,
)


def _edition(slug, content_hash, *, number=1, date="2026-05-19", focus=None):
    """Duck-typed stand-in for NewsletterEdition — ledger only reads attrs."""
    return SimpleNamespace(
        edition_slug=slug,
        edition_date=date,
        edition_number=number,
        country_focus=focus,
        content_hash=content_hash,
    )


def test_empty_ledger_returns_no_prior(tmp_path):
    p = tmp_path / "ledger.json"
    assert load_ledger(p) == []
    assert last_published(None, p) is None
    assert is_duplicate(_edition("2026-w20-x", "abc"), p) is False


def test_record_then_detect_duplicate(tmp_path):
    p = tmp_path / "ledger.json"
    ed = _edition("2026-w20-arg-grd", "hash-frozen-2025")
    record_published(ed, p)

    # Same hash next week → duplicate → must be suppressed.
    assert is_duplicate(_edition("2026-w21-arg-grd", "hash-frozen-2025"), p) is True
    # Data moved → different hash → not a duplicate.
    assert is_duplicate(_edition("2026-w21-arg-grd", "hash-NEW"), p) is False


def test_record_is_idempotent(tmp_path):
    p = tmp_path / "ledger.json"
    ed = _edition("2026-w20-arg-grd", "h1")
    record_published(ed, p)
    record_published(ed, p)  # re-run of the publish job
    assert len(load_ledger(p)) == 1


def test_country_focus_is_independent(tmp_path):
    p = tmp_path / "ledger.json"
    record_published(_edition("2026-w20-global", "gh", focus=None), p)
    record_published(_edition("2026-w20-arg", "ah", focus="ARG"), p)

    # Global edition unchanged shouldn't suppress a per-country edition.
    assert last_published(None, p)["content_hash"] == "gh"
    assert last_published("ARG", p)["content_hash"] == "ah"
    assert is_duplicate(_edition("2026-w21-arg", "ah", focus="ARG"), p) is True
    assert is_duplicate(_edition("2026-w21-arg", "gh", focus="ARG"), p) is False


def test_record_published_dict_matches_object(tmp_path):
    p = tmp_path / "ledger.json"
    edition_obj = {
        "edition_slug": "2026-w20-arg-grd",
        "edition_date": "2026-05-19",
        "edition_number": 20,
        "country_focus": None,
        "content_hash": "deadbeef",
    }
    rec = record_published_dict(edition_obj, p)
    assert rec["slug"] == "2026-w20-arg-grd"
    assert rec["content_hash"] == "deadbeef"
    assert is_duplicate(_edition("2026-w21-arg-grd", "deadbeef"), p) is True


def test_corrupt_ledger_fails_open(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    # Must not raise — corrupt ledger → treated empty → edition will send
    # (fail open) rather than silently never publishing again.
    assert load_ledger(p) == []
    assert is_duplicate(_edition("2026-w20-x", "abc"), p) is False


def test_persisted_ledger_is_valid_json_list(tmp_path):
    p = tmp_path / "ledger.json"
    record_published(_edition("2026-w20-arg-grd", "h1"), p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 1
    assert set(data[0]) == {
        "slug", "edition_date", "edition_number",
        "country_focus", "content_hash", "published_at",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
