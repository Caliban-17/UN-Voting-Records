"""Guard the publish-newsletter workflow's "compose once, reuse" contract.

The duplicate-email bug had a single root cause: the workflow composed the
edition *twice* from the CSV — once in the **Compose** step (whose hash the
skip-gate checks and the ledger records) and again in the **Send** step (the
one actually emailed) — and the two composes chose ``recent_year`` differently.
Compose used ``df["year"].max()`` (a sparse, churning in-progress year) while
Send used the ``pick_recent_year`` auto-pick. The gate kept firing on the
churning latest-year edition while the stable emailed edition went out
unchanged week after week.

The fix removes the second compose entirely: Send now loads the archived
edition via ``edition_from_dict`` and emails it verbatim, so the email, the
gate, and the ledger describe one identical edition by construction. These
tests lock that architecture in place:

  * Compose must not anchor ``recent_year`` on ``df["year"].max()``.
  * Send must reuse the archived edition (``edition_from_dict``) and must NOT
    recompose (``build_newsletter_edition``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.newsletter import build_newsletter_edition, pick_recent_year

WORKFLOW_PATH = Path(".github/workflows/publish-newsletter.yml")


def _step_run(name_contains: str) -> str:
    """Return the shell/python body of the workflow step whose name contains
    ``name_contains`` (case-insensitive)."""
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["compose-and-publish"]["steps"]
    for step in steps:
        if name_contains.lower() in str(step.get("name", "")).lower():
            return step.get("run", "")
    raise AssertionError(f"No workflow step matching {name_contains!r}")


def _strip_comments(script: str) -> str:
    """Drop ``#`` comment lines so prose explaining a banned pattern doesn't
    trip a substring guard — we only assert against live code."""
    return "\n".join(
        ln for ln in script.splitlines() if not ln.strip().startswith("#")
    )


def test_compose_step_does_not_anchor_recent_year_on_max_year():
    code = _strip_comments(_step_run("Compose edition"))
    assert 'df["year"].max()' not in code, (
        "Compose anchors recent_year on df['year'].max(); on a sparse latest "
        "year this diverges from pick_recent_year and re-sends weekly. Pass the "
        "env override or None so build_newsletter_edition auto-picks."
    )


def test_compose_step_uses_env_or_none_recent_year():
    code = _step_run("Compose edition")
    assert "int(recent_year_env) if recent_year_env else None" in code


def test_send_step_reuses_archived_edition_and_does_not_recompose():
    """Send must email the archived artifact, never rebuild it from the CSV.

    Recomposing in Send is precisely what allowed the emailed edition to drift
    away from the gated/recorded one. If this assertion fails, the duplicate
    bug's root cause has been reintroduced.
    """
    code = _step_run("Send to Substack")
    assert "edition_from_dict(" in code, (
        "Send must reconstruct the archived edition via edition_from_dict."
    )
    assert "build_newsletter_edition" not in code, (
        "Send recomposes the edition (build_newsletter_edition); it must reuse "
        "the archived artifact the gate checked instead."
    )


# ── function-level dedup contract ────────────────────────────────────────────


def _sparse_latest_year_df() -> pd.DataFrame:
    """A frame whose latest year (2026) is too sparse to anchor analysis,
    with full prior years that should be the real recent_year."""
    rows = []
    countries = ["ARG", "USA", "ISR", "TKM", "BFA", "CHN", "RUS", "GRD"]
    for year in (2023, 2024, 2025):
        for r in range(40):
            rcid = year * 1000 + r
            for i, c in enumerate(countries):
                rows.append({
                    "rcid": rcid, "country_identifier": c, "country_name": c,
                    "vote": 1 if (i + r) % 2 == 0 else -1,
                    "year": year, "date": f"{year}-09-{(r % 27) + 1:02d}",
                    "issue": "Topic", "primary_topic": "GENERAL",
                })
    for r in range(2):  # 2026: sparse, in-progress
        rcid = 2026_000 + r
        for i, c in enumerate(countries):
            rows.append({
                "rcid": rcid, "country_identifier": c, "country_name": c,
                "vote": 1 if i % 2 == 0 else -1,
                "year": 2026, "date": f"2026-02-{r + 1:02d}",
                "issue": "Topic", "primary_topic": "GENERAL",
            })
    return pd.DataFrame(rows)


def test_autopick_skips_sparse_latest_year():
    df = _sparse_latest_year_df()
    assert int(df["year"].max()) == 2026
    assert pick_recent_year(df) == 2025


def test_composed_edition_anchors_on_full_year_not_sparse_max():
    """The single composed edition (recent_year=None auto-pick) anchors on the
    prior full year, NOT df["year"].max(). On sparse-latest data the two hashes
    must differ — proof that anchoring on max() would describe a different
    edition than the one published.
    """
    df = _sparse_latest_year_df()

    composed = build_newsletter_edition(df, recent_year=None, baseline_window_years=3)
    explicit = build_newsletter_edition(
        df, recent_year=pick_recent_year(df), baseline_window_years=3
    )
    max_year = build_newsletter_edition(
        df, recent_year=int(df["year"].max()), baseline_window_years=3
    )

    assert composed.recent_year == 2025
    assert composed.content_hash == explicit.content_hash
    assert composed.content_hash != max_year.content_hash
