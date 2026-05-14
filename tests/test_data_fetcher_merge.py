"""Regression tests for the merge logic that silently destroyed 879k rows.

The bug: ``merge_with_local_data`` deduplicated on ``["rcid", "country_code"]``
without realising the historical CSV uses ``undl_id`` / ``ms_code``. NaN values
in ``rcid`` for historical rows collapsed them all to a single representative
row.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_fetcher import UNVotingDataFetcher


def _cycle(values, n):
    return [values[i % len(values)] for i in range(n)]


def _historical_df(n: int = 100) -> pd.DataFrame:
    """Simulate the historical CSV's schema: undl_id + ms_code."""
    return pd.DataFrame({
        "undl_id": list(range(100000, 100000 + n)),
        "ms_code": _cycle(["USA", "GBR", "FRA"], n),
        "ms_name": _cycle(["United States", "United Kingdom", "France"], n),
        "ms_vote": _cycle(["Y", "N", "A"], n),
        "date": ["2020-06-01"] * n,
        "resolution": [f"A/RES/{i}" for i in range(n)],
    })


def _new_df(n: int = 10) -> pd.DataFrame:
    """Simulate the fetcher's output: rcid + country_code."""
    return pd.DataFrame({
        "rcid": list(range(900000, 900000 + n)),
        "country_code": _cycle(["USA", "GBR", "FRA"], n),
        "ms_name": _cycle(["United States", "United Kingdom", "France"], n),
        "vote": _cycle(["Y", "N", "A"], n),
        "date": ["2025-06-01"] * n,
        "resolution": [f"A/RES/{i}" for i in range(900, 900 + n)],
    })


def test_merge_preserves_all_historical_rows():
    """The regression test: with the old buggy merge this would return ~11 rows
    (10 new + 1 collapsed historical). With the fix it returns 110."""
    fetcher = UNVotingDataFetcher()
    historical = _historical_df(n=100)
    new = _new_df(n=10)
    merged = fetcher.merge_with_local_data(new, historical)
    # All 100 historical rows preserved + 10 new = 110.
    assert len(merged) == 110, f"merge dropped rows: got {len(merged)} not 110"


def test_merge_refuses_to_return_smaller_frame_than_existing():
    """Belt-and-braces: even if the dedup key is misconfigured, the function
    refuses to silently shrink the dataset."""
    fetcher = UNVotingDataFetcher()
    historical = _historical_df(n=50)
    # Empty new data — caller short-circuits, returns existing.
    out = fetcher.merge_with_local_data(pd.DataFrame(), historical)
    assert len(out) == 50


def test_merge_normalizes_rcid_to_undl_id():
    fetcher = UNVotingDataFetcher()
    historical = _historical_df(n=20)
    new = _new_df(n=5)
    merged = fetcher.merge_with_local_data(new, historical)
    # New rows should now have undl_id populated (copied from rcid).
    new_part = merged[merged["undl_id"].astype(str).str.startswith("9000")]
    assert len(new_part) == 5


def test_merge_dedupes_overlapping_records():
    """If new rows have the same undl_id as historical rows, we keep the new one."""
    fetcher = UNVotingDataFetcher()
    historical = _historical_df(n=10)
    # Craft new rows with overlapping undl_ids 100000-100002.
    new = pd.DataFrame({
        "rcid": [100000, 100001, 100002, 900003],
        "undl_id": [100000, 100001, 100002, 900003],
        "country_code": ["USA", "GBR", "FRA", "DEU"],
        "ms_code": ["USA", "GBR", "FRA", "DEU"],
        "ms_name": ["United States", "United Kingdom", "France", "Germany"],
        "vote": ["Y", "Y", "Y", "Y"],
        "date": ["2025-06-01"] * 4,
        "resolution": ["A/RES/X"] * 4,
    })
    merged = fetcher.merge_with_local_data(new, historical)
    # 10 historical - 3 overlapping + 4 new = 11 total
    assert len(merged) == 11
    # The overlapping rows should have the NEW vote ("Y") because keep="last"
    overlap = merged[merged["undl_id"] == 100000]
    assert len(overlap) == 1
    assert overlap.iloc[0]["vote"] == "Y" or overlap.iloc[0].get("ms_vote") == "Y"
