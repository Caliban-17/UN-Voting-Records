#!/usr/bin/env python3
"""
Refresh the UN voting CSV with the most recent records from UN Digital Library.

Usage
-----
    # Pull anything voted since the CSV's latest date (auto-extends window),
    # write to a date-stamped sibling file (source CSV is never touched):
    python scripts/refresh_data.py

    # Override the window
    python scripts/refresh_data.py --days 90

    # Atomically swap the new file in as the source (DANGEROUS):
    python scripts/refresh_data.py --promote

    # Print what would happen but don't write
    python scripts/refresh_data.py --dry-run

How it works
------------
1. Reads the current CSV at ``UN_VOTING_DATA_PATH``.
2. Pulls UN Digital Library voting records via the MARC-XML API — no
   Playwright / Chromium / browser overhead. Walks back paged search
   results until it hits a record older than the CSV's latest date.
   See ``src/data_fetcher_marc.py``.
3. Merges new rows into the existing CSV using a column-aware dedup
   (handles both the legacy ``undl_id/ms_code`` schema and the fetcher's
   ``rcid/country_code`` — see the regression tests in
   ``tests/test_data_fetcher_merge.py``).
4. Refuses to write if the merged frame would be smaller than the input
   (the safety gate that catches column-mismatch bugs before they destroy
   anything).
5. By default writes to a date-stamped sibling file; pass ``--promote``
   to atomically swap in. The source CSV is never touched without that
   explicit flag, and it lives as ``r--r--r--`` on disk besides.

Notes
-----
* Requires ``requests`` (a base dependency) — no Playwright. The fetch
  is purely structured-XML over plain HTTP; runs cleanly in GH Actions.
* The fetcher hits UN DL at ~1.5 s per page-of-50; backfilling 9 months
  takes ~10 minutes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure repo root is on the path when invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import UN_VOTES_CSV_PATH  # noqa: E402
from src.data_fetcher import UNVotingDataFetcher  # noqa: E402  (for merge logic)
from src.data_fetcher_marc import fetch_recent_votes  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def _load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("No existing CSV at %s — will create on save.", path)
        return pd.DataFrame()
    logger.info("Loading existing CSV from %s …", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d existing rows.", len(df))
    return df


def _latest_vote_date(df: pd.DataFrame) -> datetime | None:
    if df.empty or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().to_pydatetime()


def _run(
    days: int,
    dry_run: bool,
    output: Path | None,
    promote: bool = False,
) -> int:
    """Refresh logic.

    Safety design:
      * The script NEVER overwrites the source CSV in-place unless the caller
        passes ``--promote``. Default behavior is to write to a date-stamped
        sibling file (``…<base>.refreshed-<ts>.csv``) and leave the source
        file untouched.
      * The user (or a separate ``promote`` step) is responsible for the
        final rename. This guarantees the source can never be destroyed by
        a buggy refresh run.
    """
    source_path = Path(UN_VOTES_CSV_PATH)
    existing = _load_existing_csv(source_path)
    latest = _latest_vote_date(existing)

    end = datetime.utcnow()
    if latest is not None:
        # Pull a little overlap so we catch any backfilled records.
        since_latest_days = max(days, (end - latest).days + 7)
    else:
        since_latest_days = days
    start = end - timedelta(days=since_latest_days)

    # MARC XML fetcher — pulls structured records from UN DL.
    cutoff_date = start.strftime("%Y-%m-%d")
    logger.info(
        "Fetching UN DL records since %s (%d-day window via MARC XML).",
        cutoff_date, since_latest_days,
    )
    new_df = fetch_recent_votes(since_date=cutoff_date)
    if new_df.empty:
        logger.info("No new records returned. Nothing to do.")
        return 0

    logger.info("Fetched %d new rows across %d resolutions.",
                len(new_df), new_df["undl_id"].nunique())
    # Reuse the merge logic + dedup safety gates from the original fetcher.
    fetcher = UNVotingDataFetcher()
    merged = (
        fetcher.merge_with_local_data(new_df, existing)
        if not existing.empty
        else new_df
    )
    delta = len(merged) - len(existing)
    logger.info("Merged dataset has %d rows (Δ %+d).", len(merged), delta)

    if dry_run:
        logger.info("Dry-run mode — skipping write.")
        return delta

    # SAFETY GATE 1: never shrink the dataset by accident.
    if not existing.empty and len(merged) < len(existing):
        logger.error(
            "Refusing to write: merged dataset has %d rows but existing had %d. "
            "This indicates a column-naming mismatch between fetcher and CSV. "
            "Inspect the merged DataFrame before any further action.",
            len(merged), len(existing),
        )
        return -1

    # SAFETY GATE 2: SOURCE-CSV IS READ-ONLY by default. We always write the
    # merged result to a date-stamped sibling file, never in-place over the
    # source. A subsequent --promote step (or manual rename) is required to
    # adopt the new file. This is the architectural guarantee that a buggy
    # merge can never destroy your source data again.
    from datetime import datetime as _dt

    out_path = (output if output else source_path).resolve()
    ts = _dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    candidate_path = source_path.with_name(
        f"{source_path.stem}.refreshed-{ts}.csv"
    )

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(candidate_path, index=False)
    logger.info(
        "Wrote refresh candidate → %s (%d rows).",
        candidate_path, len(merged),
    )

    if not promote:
        logger.info(
            "Source file %s was NOT modified. To adopt the refresh, run:\n"
            "  mv %s %s.archived-%s.csv && mv %s %s",
            source_path, source_path, source_path.with_suffix(""), ts,
            candidate_path, source_path,
        )
        logger.info("Or re-run with --promote to swap automatically.")
        return delta

    # --promote: swap atomically with a timestamped archive of the old file.
    if source_path.exists():
        import shutil
        # The source file is intentionally chmod 444; remove the lock for the
        # archive step then re-lock the new source at the end.
        try:
            source_path.chmod(0o644)
        except (OSError, PermissionError):
            pass
        archive = source_path.with_name(
            f"{source_path.stem}.archived-{ts}.csv"
        )
        shutil.move(str(source_path), str(archive))
        logger.info("Archived previous source → %s.", archive)
    candidate_path.rename(source_path)
    try:
        source_path.chmod(0o444)
    except (OSError, PermissionError):
        pass
    logger.info("Promoted refresh → %s.", source_path)

    parquet = source_path.with_suffix(".parquet")
    if parquet.exists():
        parquet.unlink()
        logger.info("Removed stale parquet cache %s.", parquet)

    # Emit a result marker for CI: the publish workflow reads this to decide
    # whether to create a new GitHub Release for the refreshed CSV.
    import json as _json
    out_dir = Path(".github/atlas-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refresh-result.json").write_text(_json.dumps({
        "delta": delta,
        "source_path": str(source_path),
        "source_rows": int(len(merged)),
        "latest_vote_date": (
            merged["date"].astype(str).max() if "date" in merged.columns else None
        ),
        "promoted": True,
        "timestamp_utc": ts,
    }, indent=2))
    logger.info("Wrote .github/atlas-out/refresh-result.json (delta=%d).", delta)
    return delta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=30, help="Window of days back to fetch (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and merge but do not write")
    parser.add_argument("--output", type=Path, default=None, help="Override output CSV path")
    parser.add_argument(
        "--promote", action="store_true",
        help="DANGEROUS: after a successful refresh, atomically swap the new file "
             "in as the source CSV. Default behaviour writes to a date-stamped "
             "sibling and leaves the source untouched.",
    )
    args = parser.parse_args()

    try:
        delta = _run(args.days, args.dry_run, args.output, args.promote)
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130
    if delta == 0:
        return 0
    logger.info("Done — %+d net rows.", delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
